"""
Adapted from 2020 Ross Wightman
https://github.com/rwightman/pytorch-image-models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_

from src.model.utils import convert_feats, init_weights, interpolate

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, out_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        if out_dim is None:
            out_dim = dim
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.attn = None

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.heads, C // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn

class Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, mask=None, return_attention=False):
        y, attn = self.attn(self.norm1(x), mask)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class DecoderSimple(nn.Module):
    def __init__(self, n_cls, d_encoder, bias=False, groups=1):
        super().__init__()
        self.n_cls = n_cls
        self.d_encoder = d_encoder
        self.classifier = nn.Conv2d(
            in_channels=d_encoder * groups,
            out_channels=n_cls * groups,
            kernel_size=1,
            groups=groups,
            bias=bias
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def reset_parameters(self):
        self.classifier.reset_parameters()
    
    def weights(self):
        return self.classifier.weight.data
    
    def bias(self):
        return self.classifier.bias.data

    def load_weights(self, weights):
        assert self.classifier.weight.data.size() == weights.size()
        self.classifier.weight.data = weights.clone()

    def load_bias(self, bias):
        assert self.classifier.bias.data.size() == bias.size()
        self.classifier.bias.data = bias.clone()

    def forward(self, x, im_size=None):
        if len(x.size()) == 3:
            x = convert_feats(x, 'cnn')
        x = self.classifier(x)
        if im_size is not None:
            x = interpolate(x, size=im_size)
        return x

class DecoderLinear(nn.Module):
    def __init__(self, n_cls, patch_size, d_encoder):
        super().__init__()
        self.n_cls = n_cls
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.classifier = nn.Linear(self.d_encoder, n_cls)
        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def reset_parameters(self):
        self.classifier.reset_parameters()

    def weights(self):
        return self.classifier.weight.data

    def forward(self, x, im_size):
        H, W = im_size
        GS = H // self.patch_size
        x = self.classifier(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=GS)
        return x

class MaskTransformer(nn.Module):
    def __init__(
        self,
        n_cls,
        patch_size,
        d_encoder,
        n_layers,
        n_heads,
        d_model,
        d_ff,
        drop_path_rate,
        dropout,
    ):
        super().__init__()
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.n_cls = n_cls
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model ** -0.5

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )

        self.cls_emb = nn.Parameter(torch.randn(1, n_cls, d_model))
        self.proj_dec = nn.Linear(d_encoder, d_model)

        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.proj_classes = nn.Parameter(self.scale * torch.randn(d_model, d_model))

        self.decoder_norm = nn.LayerNorm(d_model)
        self.mask_norm = nn.LayerNorm(n_cls)

        self.apply(init_weights)
        trunc_normal_(self.cls_emb, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_emb"}

    def forward(self, x, im_size):
        H, W = im_size
        GS = H // self.patch_size

        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        patches, cls_seg_feat = x[:, : -self.n_cls], x[:, -self.n_cls :]
        patches = patches @ self.proj_patch
        cls_seg_feat = cls_seg_feat @ self.proj_classes

        patches = patches / patches.norm(dim=-1, keepdim=True)
        cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)

        masks = patches @ cls_seg_feat.transpose(1, 2)
        masks = self.mask_norm(masks)
        masks = rearrange(masks, "b (h w) n -> b n h w", h=int(GS))

        return masks

    def get_attention_map(self, x, layer_id):
        if layer_id >= self.n_layers or layer_id < 0:
            raise ValueError(
                f"Provided layer_id: {layer_id} is not valid. 0 <= {layer_id} < {self.n_layers}."
            )
        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for i, blk in enumerate(self.blocks):
            if i < layer_id:
                x = blk(x)
            else:
                return blk(x, return_attention=True)

class SegHead(nn.Module):
    def __init__(self, d_in, d_mid, d_out, dropout=0.1):
        super().__init__()
        self.res1 = nn.Sequential(
            nn.Conv2d(d_in, d_mid, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),                          
        )              
        self.res2 = nn.Sequential(
            nn.Conv2d(d_mid, d_mid, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),   
            nn.Conv2d(d_mid, d_mid, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),                             
        ) 
        self.cls = nn.Sequential(
            nn.Conv2d(d_mid, d_mid, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(d_mid, d_out, kernel_size=1)
        )

    def forward(self, x, im_size=None):
        x = self.res1(x)
        x = self.res2(x) + x
        x = self.cls(x)
        if im_size is not None:
            x = interpolate(x, size=im_size)
        return x

class ProjectionHead(nn.Module):
    def __init__(self, dim_in, dim_proj=256, linear_flag=False):
        super().__init__()
        if linear_flag:
            self.proj = nn.Conv2d(dim_in, dim_proj, kernel_size=1)
        else:
            self.proj = nn.Sequential(
                nn.Conv2d(dim_in, dim_proj, kernel_size=1),
                nn.BatchNorm2d(dim_proj),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim_proj, dim_proj, kernel_size=1)
            )

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=1)
