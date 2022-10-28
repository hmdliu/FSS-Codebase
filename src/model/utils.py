
import pdb
import math
import random
from einops import rearrange
from collections import OrderedDict
from timm.models.layers import trunc_normal_

import torch
import torch.nn as nn
import torch.nn.functional as F

def select_shot(x, idx=None, ref=None):
    B, C, H, W = x.shape
    x = x.view(B // 2, 2, C, H, W)
    if idx is not None:
        out = x[:, idx, :, :, :]
    elif ref is not None:
        ref = ref.reshape((len(ref) // 2, 2))
        indices = torch.argmax(ref, dim=-1)
        out = x[torch.arange(B // 2, device=ref.device), indices]
    return out

def ce_loss(pred, label, ignore_index=255, reduction='mean'):
    criterion = nn.CrossEntropyLoss(
        ignore_index=ignore_index,
        reduction=reduction
    )
    return criterion(pred, label)

def weighted_ce_loss(pred, label, weight=None, ignore_index=255, reduction='mean'):
    if weight is None:
        count = torch.bincount(label.view(-1))
        weight = torch.tensor([1.0, count[0]/count[1]])
    if torch.cuda.is_available():
        weight = weight.cuda()
    criterion = nn.CrossEntropyLoss(
        weight=weight,
        ignore_index=ignore_index,
        reduction=reduction
    )
    return criterion(pred, label)

def weighted_dice_loss(prediction, target_seg, weighted_val=1.0, reduction='sum', eps=1e-8):
    """
    Weighted version of Dice Loss
    Args:
        prediction: prediction
        target_seg: segmentation target
        weighted_val: values of k positives,
        reduction: 'none' | 'mean' | 'sum'
        eps: the minimum eps
    """
    target_seg_fg = target_seg == 1                                             # [B, h, w]
    target_seg_bg = target_seg == 0                                             # [B, h, w]
    target_seg = torch.stack([target_seg_bg, target_seg_fg], dim=1).float()     # [B, 2, h, w] get rid of ignore pixels 255

    n, _, h, w = target_seg.shape

    prediction = prediction.reshape(-1, h, w)       # [B*2, h, w]
    target_seg = target_seg.reshape(-1, h, w)       # [B*2, h, w]
    prediction = torch.sigmoid(prediction)          # [B*2, h, w]
    prediction = prediction.reshape(-1, h * w)      # [B*2, h*w]
    target_seg = target_seg.reshape(-1, h * w)      # [B*2, h*w]

    # calculate dice loss
    loss_part = (prediction ** 2).sum(dim=-1) + (target_seg ** 2).sum(dim=-1)               # [B*2]
    loss = 1 - 2 * (target_seg * prediction).sum(dim=-1) / torch.clamp(loss_part, min=eps)  # [B*2]
    # normalize the loss
    loss = loss * weighted_val

    if reduction == 'sum':
        loss = loss.sum() / n
    elif reduction == 'mean':
        loss = loss.mean()
    return loss

def mask_kmeans(data, n_clusters=5, max_iter=5, log=None, init_centers=None):

    # init data & sample size
    data = data.float()
    n_samples = data.shape[0]

    # init cluster-related tensors
    if init_centers is not None:
        centers = init_centers
    else:
        centers = data[random.choices(list(range(n_samples)), k=n_clusters)]
    cluster_idx = torch.empty((n_samples,), dtype=torch.long)
    cluster_dist = torch.empty((n_samples,))
    centers_dist = torch.empty((n_samples, n_clusters))

    # main loop
    for i in range(max_iter):

        # init new cluster-related tensors
        centers_ = torch.empty_like(centers)
        cluster_idx_ = torch.empty_like(cluster_idx)
        cluster_dist_ = torch.empty_like(cluster_dist)
        centers_dist_ = torch.empty_like(centers_dist)

        # compute distance & assign cluster
        for j, sample in enumerate(data):
            dist = (sample - centers).pow(2).sum(1)
            cluster_idx_[j] = dist.argmin()
            cluster_dist_[j] = dist.min()
            centers_dist_[j] = dist
        variation = (centers_dist_ - centers_dist).mean().abs()

        # compute new centers
        for j in range(n_clusters):
            samples = data[cluster_idx_ == j]
            samples_dist = (samples.mean(0) - samples).pow(2).sum(1)
            try:
                centers_[j] = samples[samples_dist.argmin()]
            except:
                return mask_kmeans(data, n_clusters-1, max_iter, log)

        # update cluster-related tensors
        centers = centers_
        cluster_idx = cluster_idx_
        cluster_dist = cluster_dist_
        centers_dist = centers_dist_

        # log as needed
        if log is not None:
            log('\niter', i)
            log('variation = %.2f' % variation)
            log('bin_count =', torch.bincount(cluster_idx))
            log(centers)
    
    return centers, cluster_idx

def activation_map(f_q, proto_list):

    # init feats size
    B, C, H, W = f_q.size()
    N = len(proto_list)

    # rearrange inputs
    f_q = f_q.view(B, 1, C, -1)                                             # [B, 1, C, H*W]
    proto_list = list(map(lambda p: p.unsqueeze(1), proto_list))            # List[Tensor(B, 1, C)]
    proto = torch.cat(proto_list, dim=1).unsqueeze(-1)                      # [B, N, C, 1]

    # compute activation maps
    actv_map = F.cosine_similarity(f_q, proto, dim=2)                       # [B, N, H*W]
    actv_map = actv_map.view(B, N, H, W)                                    # [B, N, H, W]

    return actv_map

def init_cls_weights(n_cls, n_feats):
    weights = nn.Parameter(torch.empty(n_cls, n_feats, 1, 1))
    trunc_normal_(weights, std=0.02)
    return weights

def seg_head_weights(ckpt_path, aux_idx=1):
    weights = OrderedDict()
    aux_dict = {1: 'seg2.', 2: 'seg3.'}
    ckpt = torch.load(ckpt_path, map_location='cpu')['state_dict']
    for k, v in ckpt.items():
        if k.find(aux_dict[aux_idx]) != -1:
            weights[k.replace(aux_dict[aux_idx], '')] = v
    return weights

def freeze_module(module):
    for p in module.parameters():
        p.requires_grad = False

def interpolate(feats, size, mode='bilinear', align_corners=True):
    return F.interpolate(feats, size=size, mode=mode, align_corners=align_corners)

def resize_mask(mask, size, mode='bilinear', align_corners=True):
    mask = mask.unsqueeze(1).float()
    mask = F.interpolate(mask, size=size, mode=mode, align_corners=align_corners)
    return mask.long().squeeze(1)

def mask_avg_pool(feats, mask, label=1, eps=1e-10):
    B, C, _, _ = feats.size()
    mask = torch.where(mask == label, 1, 0)
    feats, mask = feats.view(B, C, -1), mask.view(B, 1, -1)
    feats_sum = torch.sum(feats * mask, dim=2)
    feats_num = torch.count_nonzero((mask == 1), dim=2)
    return feats_sum / (feats_num + eps)

def convert_feats(feats, feats_type):
    assert feats_type in ('cnn', 'vit')
    cnn_flag = (len(feats.size()) == 4 and feats_type == 'cnn')
    vit_flag = (len(feats.size()) == 3 and feats_type == 'vit')
    if cnn_flag or vit_flag:
        pass
    elif len(feats.size()) == 4:
        feats = rearrange(feats, 'b c h w -> b (h w) c', h=feats.size(2))
    else:
        feats = rearrange(feats, 'b (h w) c -> b c h w', h=int(feats.size(1) ** 0.5))
    return feats

def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

def resize_pos_embed(posemb, grid_old_shape, grid_new_shape, num_extra_tokens):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    posemb_tok, posemb_grid = (
        posemb[:, :num_extra_tokens],
        posemb[0, num_extra_tokens:],
    )
    if grid_old_shape is None:
        gs_old_h = int(math.sqrt(len(posemb_grid)))
        gs_old_w = gs_old_h
    else:
        gs_old_h, gs_old_w = grid_old_shape

    gs_h, gs_w = grid_new_shape
    posemb_grid = posemb_grid.reshape(1, gs_old_h, gs_old_w, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bilinear")
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb

def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if "model" in state_dict:
        # For deit models
        state_dict = state_dict["model"]
    num_extra_tokens = 1 + ("dist_token" in state_dict.keys())
    patch_size = model.patch_size
    image_size = model.patch_embed.image_size
    for k, v in state_dict.items():
        if k == "pos_embed" and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v,
                None,
                (image_size[0] // patch_size, image_size[1] // patch_size),
                num_extra_tokens,
            )
        out_dict[k] = v
    return out_dict

def padding(im, patch_size, fill_value=0):
    # make the image sizes divisible by patch_size
    H, W = im.size(2), im.size(3)
    pad_h, pad_w = 0, 0
    if H % patch_size > 0:
        pad_h = patch_size - (H % patch_size)
    if W % patch_size > 0:
        pad_w = patch_size - (W % patch_size)
    im_padded = im
    if pad_h > 0 or pad_w > 0:
        im_padded = F.pad(im, (0, pad_w, 0, pad_h), value=fill_value)
    return im_padded

def unpadding(y, target_size):
    H, W = target_size
    H_pad, W_pad = y.size(2), y.size(3)
    # crop predictions on extra pixels coming from padding
    extra_h = H_pad - H
    extra_w = W_pad - W
    if extra_h > 0:
        y = y[:, :, :-extra_h]
    if extra_w > 0:
        y = y[:, :, :, :-extra_w]
    return y

def resize(im, smaller_size):
    h, w = im.shape[2:]
    if h < w:
        ratio = w / h
        h_res, w_res = smaller_size, ratio * smaller_size
    else:
        ratio = h / w
        h_res, w_res = ratio * smaller_size, smaller_size
    if min(h, w) < smaller_size:
        im_res = F.interpolate(im, (int(h_res), int(w_res)), mode="bilinear")
    else:
        im_res = im
    return im_res

def sliding_window(im, flip, window_size, window_stride):
    B, C, H, W = im.shape
    ws = window_size

    windows = {"crop": [], "anchors": []}
    h_anchors = torch.arange(0, H, window_stride)
    w_anchors = torch.arange(0, W, window_stride)
    h_anchors = [h.item() for h in h_anchors if h < H - ws] + [H - ws]
    w_anchors = [w.item() for w in w_anchors if w < W - ws] + [W - ws]
    for ha in h_anchors:
        for wa in w_anchors:
            window = im[:, :, ha : ha + ws, wa : wa + ws]
            windows["crop"].append(window)
            windows["anchors"].append((ha, wa))
    windows["flip"] = flip
    windows["shape"] = (H, W)
    return windows

def merge_windows(windows, window_size, ori_shape):
    ws = window_size
    im_windows = windows["seg_maps"]
    anchors = windows["anchors"]
    C = im_windows[0].shape[0]
    H, W = windows["shape"]
    flip = windows["flip"]

    logit = torch.zeros((C, H, W), device=im_windows.device)
    count = torch.zeros((1, H, W), device=im_windows.device)
    for window, (ha, wa) in zip(im_windows, anchors):
        logit[:, ha : ha + ws, wa : wa + ws] += window
        count[:, ha : ha + ws, wa : wa + ws] += 1
    logit = logit / count
    logit = F.interpolate(
        logit.unsqueeze(0),
        ori_shape,
        mode="bilinear",
    )[0]
    if flip:
        logit = torch.flip(logit, (2,))
    result = F.softmax(logit, 0)
    return result

def num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_params = sum([torch.prod(torch.tensor(p.size())) for p in model_parameters])
    return n_params.item()

def get_corr(q, k):
    bs, ch, height, width = q.shape
    proj_q = q.view(bs, ch, height * width).permute(0, 2, 1)    # [1, ch, hw] -> [1, hw, ch]
    proj_k = k.view(bs, -1, height * width)                     # [1, ch, hw]

    proj_q = F.normalize(proj_q, dim=-1)
    proj_k = F.normalize(proj_k, dim=-2)
    sim = torch.bmm(proj_q, proj_k)                             # [1, 3600 (q_hw), 3600(k_hw)]
    return sim

def get_ig_mask(sim, s_label, q_label, pd_q0, pd_s):
    h, w = pd_q0.shape[-2:]
    # mask ignored support pixels
    s_mask = F.interpolate(s_label.unsqueeze(1).float(), size=(h,w), mode='nearest')  # [1,1,h,w]
    s_mask = (s_mask > 1).view(s_mask.shape[0], -1)  # [n_shot, hw]

    # ignore misleading points
    pd_q_mask0 = pd_q0.argmax(dim=1)
    q_mask = F.interpolate(q_label.unsqueeze(1).float(), size=(h,w), mode='nearest').squeeze(1)  # [1,1,h,w]
    qf_mask = (q_mask != 255.0) * (pd_q_mask0 == 1)  # predicted qry FG
    qb_mask = (q_mask != 255.0) * (pd_q_mask0 == 0)  # predicted qry BG
    qf_mask = qf_mask.view(qf_mask.shape[0], -1, 1).expand(sim.shape)  # 保留query predicted FG
    qb_mask = qb_mask.view(qb_mask.shape[0], -1, 1).expand(sim.shape)  # 保留query predicted BG

    sim_qf = sim[qf_mask].reshape(1, -1, 3600)
    if sim_qf.numel() > 0:
        th_qf = torch.quantile(sim_qf.flatten(), 0.8)
        sim_qf = torch.mean(sim_qf, dim=1)  # 取平均 对应support img 与Q前景相关 所有pixel  [B, 3600_s]
        qf_mask = sim_qf
    else:
        print('------ pred qf mask is empty! ------')
        qf_mask = torch.zeros([1, 3600], dtype=torch.float).cuda()

    sim_qb = sim[qb_mask].reshape(1, -1, 3600)
    if sim_qb.numel() > 0:
        th_qb = torch.quantile(sim_qb.flatten(), 0.8)
        sim_qb = torch.mean(sim_qb, dim=1)  # 取平均 对应support img 与Q背景相关 所有pixel, [B, 3600_s]
        qb_mask = sim_qb
    else:
        print('------ pred qb mask is empty! ------')
        qb_mask = torch.zeros([1, 3600], dtype=torch.float).cuda()

    sf_mask = pd_s.argmax(dim=1).view(1, 3600)
    null_mask = torch.zeros([1, 3600], dtype=torch.bool)
    null_mask = null_mask.cuda() if torch.cuda.is_available() else null_mask
    ig_mask1 = (sim_qf > th_qf) & (sf_mask == 0) if sim_qf.numel() > 0 else null_mask
    ig_mask3 = (sim_qb > th_qb) & (sf_mask == 1) if sim_qb.numel() > 0 else null_mask
    ig_mask2 = (sim_qf > th_qf) & (sim_qb > th_qb) if sim_qf.numel() > 0 and sim_qb.numel() > 0 else null_mask
    ig_mask = ig_mask1 | ig_mask2 | ig_mask3 | s_mask

    return ig_mask  # [B, hw_s]

LOSS_DICT = {
    'ce': ce_loss,
    'wce': weighted_ce_loss,
    'wdc': weighted_dice_loss
}
