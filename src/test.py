
# Meta-testing baseline

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
from src.dataset.dataset import get_val_loader
from src.utils import (
    AverageMeter,
    intersectionAndUnionGPU,
    load_cfg_from_cfg_file,
    merge_cfg_from_list,
    compute_metrics,
    init_path,
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
        save_path = f'./results/test_only/{meta_cfg}_{exp_cfg}'
        os.makedirs(save_path, exist_ok=True)
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
            lines.append(f'\n==> loading backbone weight from: {args.resume_weight}')
            pre_dict, cur_dict = torch.load(args.resume_weight)['state_dict'], backbone.state_dict()
            if 'gamma' in pre_dict.keys():
                del pre_dict['gamma']
            for key1, key2 in zip(pre_dict.keys(), cur_dict.keys()):
                if pre_dict[key1].shape != cur_dict[key2].shape:
                    lines.append(f'Pre-trained {key1} shape and model {key2} shape: {pre_dict[key1].shape}, {cur_dict[key2].shape}')
                    continue
                cur_dict[key2] = pre_dict[key1] 
            msg = backbone.load_state_dict(cur_dict, strict=True)
            lines.append(f"==> {msg}")
        else:
            lines.append(f"\n==> no weight found at '{args.resume_weight}'")
        log('\n'.join(lines))

        # freeze backbone
        for p in backbone.parameters():
            p.requires_grad = False

    # init meta model
    meta_model = get_meta_model(args).cuda()
    log(f'\nMeta Model: {meta_model}')

    # load meta weights
    if args.meta_weight:
        lines = []
        args.meta_weight = f'./results/{date}/{meta_cfg}_{exp_cfg}/best.pth'
        if os.path.isfile(args.meta_weight):
            lines.append(f'==> loading meta weight from: {args.meta_weight}')
            meta_weight = torch.load(args.meta_weight, map_location='cpu')['state_dict']
            msg = meta_model.load_state_dict(meta_weight, strict=True)
            lines.append(f"==> {msg}\n")
        else:
            lines.append(f"==> no weight found at '{args.meta_weight}'\n")
        log('\n'.join(lines))

        # freeze meta model
        for p in meta_model.meta_params():
            p.requires_grad = False
        meta_model.eval()

    # init dataloaders
    val_loader, _ = get_val_loader(args, episodic=True)

    # init variables
    global max_val_mIoU, val_mIoU_dict, val_FBIoU_dict
    max_val_mIoU = 0.0
    val_mIoU_dict = {}
    val_FBIoU_dict = {}
    start_time = time.time()
    if args.debug_flag:
        args.n_runs = 5
        args.log_freq = 5
        args.test_num = 20

    # main loop
    for val_idx in range(1, args.n_runs+1):

        # model evaluation
        val_mIoU, val_FBIoU, val_loss = validate_epoch(
            args=args,
            val_idx=val_idx,
            backbone=backbone,
            meta_model=meta_model,
            val_loader=val_loader
        )
        for i in range(args.n_pred):
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
    
    runtime = time.time() - start_time
    log(f"\n==> Timer: {runtime / (3600 * args.epochs):.1f} hrs/epoch | {runtime / 3600:.1f} hrs in total")

def validate_epoch(args, val_idx, backbone, meta_model, val_loader):

    log(f"\n==> Run {val_idx}: testing start")

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