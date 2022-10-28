
# Meta-training baseline

import os
import pdb
import sys
import time
import yaml
import random
import argparse
import numpy as np
from collections import defaultdict

import torch
import torch.backends.cudnn as cudnn

from src.exp import modify_config
from src.model import get_backbone, get_meta_model
from src.optimizer import get_optimizer, get_scheduler
from src.dataset.dataset import get_val_loader, get_train_loader
from src.utils import (
    AverageMeter,
    intersectionAndUnionGPU,
    load_cfg_from_cfg_file,
    merge_cfg_from_list,
    compute_metrics,
    log_uniform,
    init_path,
    curr_lr,
    log
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Training classifier weight transformer')
    parser.add_argument('--config', type=str, required=True, help='base config file')
    parser.add_argument('--exp_id', type=str, required=True, help='exp settings')
    parser.add_argument('--debug', type=bool, default=False, help='debug mode')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = load_cfg_from_cfg_file(args.config)
    cfg.config_path = args.config
    cfg.debug_flag = args.debug
    cfg.exp_id = args.exp_id
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    return cfg

def main(args: argparse.Namespace) -> None:

    # init dump dir
    if args.exp_id != 'test':
        date, meta_cfg, exp_cfg = tuple(args.exp_id.split('_'))
        save_path = f'./results/{date}/{meta_cfg}_{exp_cfg}'
    else:
        date = meta_cfg = exp_cfg = None
        save_path = './results/test'

    # apply experiment settings
    args, exp_dict = modify_config(args, date, meta_cfg, exp_cfg)
    if not args.debug_flag:
        init_path(save_path, remove_flag=True)
        yaml.dump(args, open(os.path.join(save_path, 'config.yaml'), 'w'))    
    log(f'save_path: {save_path}\n\nexp_dict: {exp_dict}\n\nargs: {args}')

    # backup logging function
    args.log_func = log

    # set random seed as needed
    if args.manual_seed is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)

    # init backbone
    backbone = get_backbone(args).cuda()
    backbone.eval()
    # log(f'\nBackbone: {backbone}')

    # load pretrain weights
    if args.resume_weight:
        lines = []
        args.resume_weight = f'pretrained/{args.train_name}/split{args.train_split}/pspnet_resnet{args.layers}/best.pth'
        if os.path.isfile(args.resume_weight):
            lines.append(f'==> loading backbone weight from: {args.resume_weight}')
            pre_dict, cur_dict = torch.load(args.resume_weight)['state_dict'], backbone.state_dict()
            for key1, key2 in zip(pre_dict.keys(), cur_dict.keys()):
                if pre_dict[key1].shape != cur_dict[key2].shape:
                    lines.append(f'Pre-trained {key1} shape and model {key2} shape: {pre_dict[key1].shape}, {cur_dict[key2].shape}')
                    continue
                cur_dict[key2] = pre_dict[key1] 
            msg = backbone.load_state_dict(cur_dict, strict=True)
            lines.append(f"==> {msg}")
        else:
            lines.append(f"==> no weight found at '{args.resume_weight}'")
        log('\n'.join(lines))

        # freeze backbone
        for p in backbone.parameters():
            p.requires_grad = False

    # init dataloaders
    train_loader, _ = get_train_loader(args, episodic=True)
    val_loader, _ = get_val_loader(args, episodic=True)

    # init meta model
    meta_model = get_meta_model(args).cuda()
    log(f'\nMeta Model: {meta_model}')

    # init meta optimizer & scheduler
    if isinstance(args.lr_meta, list):
        args.lr_meta = log_uniform(*args.lr_meta)
        log(f'\n==> Using base meta_lr = {args.lr_meta:.5f}')
    meta_optimizer = get_optimizer(args, [dict(params=meta_model.meta_params(), lr=args.lr_meta)])
    scheduler = get_scheduler(args, meta_optimizer, len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp) if args.use_amp else None

    # init variables
    global max_val_mIoU, val_mIoU_dict, val_FBIoU_dict
    max_val_mIoU = 0.0
    val_mIoU_dict = {}
    val_FBIoU_dict = {}
    start_time = time.time()
    args.use_amp = args.get('use_amp', False)
    if args.use_amp:
        log('\n==> Using automatic mixed precision for training')
    if args.get('iter_per_epoch', None) is None:
        args.iter_per_epoch = float('inf')
    iter_per_epoch = min(args.iter_per_epoch, len(train_loader))
    if args.debug_flag:
        args.epochs = 2
        args.log_freq = 5
        args.test_num = 20
        args.per_epoch_val = 2
        iter_per_epoch = 20

    # main loop
    for epoch in range(1, args.epochs+1):

        train_mIoU, train_loss = train_epoch(
            args=args,
            epoch=epoch,
            iter_per_epoch=iter_per_epoch,
            backbone=backbone,
            meta_model=meta_model,
            train_loader=iter(train_loader),
            val_loader=val_loader,
            meta_optimizer=meta_optimizer,
            scheduler=scheduler,
            scaler=scaler,
            save_path=save_path
        )
    
    runtime = time.time() - start_time
    log(f"\n==> Timer: {runtime / (3600 * args.epochs):.1f} hrs/epoch | {runtime / 3600:.1f} hrs in total")

def train_epoch(
    args,
    epoch,
    iter_per_epoch,
    backbone,
    meta_model,
    train_loader,
    val_loader,
    meta_optimizer,
    scheduler,
    scaler,
    save_path
):

    log(f"\n==> Epoch {epoch}: training start")

    # init variables
    global max_val_mIoU, val_mIoU_dict, val_FBIoU_dict
    val_freq = iter_per_epoch // args.per_epoch_val

    for iter in range(1, iter_per_epoch+1):

        # loading samples
        # q: [1, 3, im_size, im_size]
        # s: [1, n_shots, 3, im_size, im_size]
        img_q, label_q, img_s, label_s, _, _, _ = train_loader.next()
        if torch.cuda.is_available():
            img_q = img_q.cuda()        # [1, 3, im_size, im_size]
            label_q = label_q.cuda()    # [1, im_size, im_size]
            img_s = img_s.cuda()        # [1, n_shots, 3, im_size, im_size]
            label_s = label_s.cuda()    # [1, n_shots, im_size, im_size]
        img_s = img_s.squeeze(0)        # [n_shots, 3, im_size, im_size]
        label_s = label_s.squeeze(0)    # [n_shots, im_size, im_size]

        # set module mode
        for p in meta_model.meta_params():
            p.requires_grad = True
        meta_model.train()

        # forward meta model
        pred_q, loss_q = meta_model(backbone, img_s, img_q, label_s, label_q, use_amp=args.use_amp)

        # update weights & scheduler
        try:
            meta_optimizer.zero_grad(set_to_none=True)
            loss_q[args.loss_idx].backward()
            meta_optimizer.step()
            if args.scheduler == 'cosine':
                scheduler.step()
        except Exception as err:
            if args.debug_flag:
                log(f'[Error]: {err}')

        # get num of predictions
        try:
            n_pred = len(loss_meter)
        except:
            n_pred = len(loss_q)
            loss_meter = [AverageMeter() for _ in range(n_pred)]
            mIoU_meter = [AverageMeter() for _ in range(n_pred)]
        
        # update losses & metrics
        for i in range(n_pred):
            intersection, union, target = intersectionAndUnionGPU(pred_q[i].argmax(1), label_q, 2, 255)
            IoUf, IoUb = (intersection / (union + 1e-10)).cpu().numpy()
            loss_meter[i].update(loss_q[i].item())
            mIoU_meter[i].update((IoUf + IoUb) / 2)

        # testing
        if iter % val_freq == 0:

            # logging
            info_list = [f'loss{i} {loss_meter[i].avg:.2f} mIoU{i} {mIoU_meter[i].avg:.4f}' for i in range(n_pred)]
            log(f"Epoch {epoch} Iter {iter}/{iter_per_epoch}: {' | '.join(info_list)} | lr {curr_lr(meta_optimizer)}")

            # model evaluation
            val_mIoU, val_FBIoU, val_loss = validate_epoch(
                args=args,
                epoch=epoch,
                backbone=backbone,
                meta_model=meta_model,
                val_loader=val_loader
            )
            for i in range(n_pred):
                val_mIoU_dict[i] = val_mIoU_dict.get(i, []) + [np.mean(list(val_mIoU[i].values()))]
                val_FBIoU_dict[i] = val_FBIoU_dict.get(i, []) + [val_FBIoU[i].item()]

            # log metrics
            lines = ['']
            max_mIoU, avg_mIoU, smt_mIoU = compute_metrics(val_mIoU_dict, size=args.miou_grp_size)
            max_FBIoU, avg_FBIoU, smt_FBIoU = compute_metrics(val_FBIoU_dict, size=args.miou_grp_size)
            lines.append(f"==> Max mIoU:        {' | '.join([f'pred{k} {v:.4f}' for k, v in max_mIoU.items()])}")
            lines.append(f"==> Max FB-IoU:      {' | '.join([f'pred{k} {v:.4f}' for k, v in max_FBIoU.items()])}")
            lines.append(f"==> Average mIoU:    {' | '.join([f'pred{k} {v:.4f}' for k, v in avg_mIoU.items()])}")
            lines.append(f"==> Average FB-IoU:  {' | '.join([f'pred{k} {v:.4f}' for k, v in avg_FBIoU.items()])}")
            lines.append(f"==> Smoothed mIoU:   {' | '.join([f'pred{k} {v:.4f}' for k, v in smt_mIoU.items()])}")
            lines.append(f"==> Smoothed FB-IoU: {' | '.join([f'pred{k} {v:.4f}' for k, v in smt_FBIoU.items()])}")
            log('\n'.join(lines))

            # save checkpoint
            if val_mIoU_dict[n_pred-1][-1] > max_val_mIoU:
                max_val_mIoU = val_mIoU_dict[n_pred-1][-1]
                if args.get('save_flag', None) and os.path.exists(save_path):
                    ckpt_dict = {
                        'epoch': epoch,
                        'val_mIoU': max_val_mIoU,
                        'state_dict': meta_model.state_dict(),
                        'optimizer': meta_optimizer.state_dict()
                    }
                    ckpt_path = os.path.join(save_path, 'best.pth')
                    torch.save(ckpt_dict, ckpt_path)
                    log(f'==> Epoch {epoch}: saving checkpoint as {ckpt_path}')
            
            # resume training
            if iter < iter_per_epoch:
                log(f"\n==> Epoch {epoch}: training resume")
        
        # logging
        elif iter % args.log_freq == 0:
            info_list = [f'loss{i} {loss_meter[i].avg:.2f} mIoU{i} {mIoU_meter[i].avg:.4f}' for i in range(n_pred)]
            log(f"Epoch {epoch} Iter {iter}/{iter_per_epoch}: {' | '.join(info_list)} | lr {curr_lr(meta_optimizer)}")

    return loss_meter, mIoU_meter

def validate_epoch(args, epoch, backbone, meta_model, val_loader):

    log(f"\n==> Epoch {epoch}: testing start")

    start_time = time.time()

    for episode in range(1, args.test_num+1):

        # loading samples
        # q: [1, 3, im_size, im_size]
        # s: [1, n_shots, 3, im_size, im_size]
        try:
            img_q, label_q, img_s, label_s, subcls, _, _ = iter_loader.next()
        except:
            iter_loader = iter(val_loader)
            img_q, label_q, img_s, label_s, subcls, _, _ = iter_loader.next()
        if torch.cuda.is_available():
            img_q = img_q.cuda()        # [1, 3, im_size, im_size]
            label_q = label_q.cuda()    # [1, im_size, im_size]
            img_s = img_s.cuda()        # [1, n_shots, 3, im_size, im_size]
            label_s = label_s.cuda()    # [1, n_shots, im_size, im_size]
        img_s = img_s.squeeze(0)        # [n_shots, 3, im_size, im_size]
        label_s = label_s.squeeze(0)    # [n_shots, im_size, im_size]

        # set module mode
        for p in meta_model.meta_params():
            p.requires_grad = False
        meta_model.eval()

        # forward meta model
        pred_q, loss_q = meta_model(backbone, img_s, img_q, label_s, label_q, use_amp=False)

        # get num of predictions
        try:
            n_pred = len(loss_meter)
        except:
            n_pred = len(loss_q)
            all_intrs = [0.0 for _ in range(n_pred)]
            all_union = [0.0 for _ in range(n_pred)]
            cls_intrs = [defaultdict(int) for _ in range(n_pred)]
            cls_union = [defaultdict(int) for _ in range(n_pred)]
            mIoU_meter = [defaultdict(float) for _ in range(n_pred)]
            loss_meter = [AverageMeter() for _ in range(n_pred)]
        
        # update losses & metrics
        for i in range(n_pred):
            curr_cls = subcls[0].item()
            intrs, union, target = intersectionAndUnionGPU(pred_q[i].argmax(1), label_q, 2, 255)
            intrs, union, target = intrs.cpu(), union.cpu(), target.cpu()
            all_intrs[i] += intrs
            all_union[i] += union
            cls_intrs[i][curr_cls] += intrs[1]
            cls_union[i][curr_cls] += union[1]
            mIoU_meter[i][curr_cls] = cls_intrs[i][curr_cls] / (cls_union[i][curr_cls] + 1e-10)
            loss_meter[i].update(loss_q[i].item())

        # logging & saving
        if episode % args.log_freq == 0:
            info_list = [f'loss{i} {loss_meter[i].avg:.2f} mIoU{i} {np.mean([mIoU_meter[i][j] for j in mIoU_meter[i]]):.4f}' for i in range(n_pred)]
            log(f"Episode {episode}/{args.test_num}: {' | '.join(info_list)}")

    # compute FB-IoU metrics
    FBIoU_meter = [(all_intrs[i] / (all_union[i] + 1e-10)).mean() for i in range(n_pred)]

    # logging
    runtime = time.time() - start_time
    info_list = []
    info_list.append(f"==> Testing: inference speed {args.test_num / runtime * 60:.0f} iters/min")
    for c in sorted(mIoU_meter[-1].keys()):
        info_list.append(f"==> Testing: class {c} mIoU {mIoU_meter[-1][c]:.4f}")
    info_list.append(f"==> Testing: {' | '.join([f'FB-IoU{i} {FBIoU_meter[i]:.4f}' for i in range(n_pred)])}")
    log('\n'.join(info_list))

    return mIoU_meter, FBIoU_meter, loss_meter

if __name__ == "__main__":
    args = parse_args()
    main(args)