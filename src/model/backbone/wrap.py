
import pdb
import timm
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import FeaturePyramidNetwork

from src.model.backbone.pspnet import PSPNet
from src.model.backbone.resnet import get_resnet
from src.model.utils import convert_feats

class ResNetWrap(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.arch = args.arch
        self.feats_type = args.feats_type
        self.no_relu = args.resnet_kwargs.get('no_relu', False)
        self.resnet = get_resnet(args.arch, args.pretrained, args.resnet_kwargs)
        self.out_layer = nn.MaxPool2d(2, stride=2) if args.final_pool else nn.Identity()

    def extract_features(self, x):
        return self.forward(x)

    def forward(self, x):
        if self.no_relu:
            _, x = self.resnet(x)
        else:
            x = self.resnet(x)
        x = self.out_layer(x)
        x = convert_feats(x, self.feats_type)
        return x

class PSPNetWrap(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.arch = args.arch
        self.rmid = args.rmid
        self.feats_type = args.feats_type
        self.pspnet = PSPNet(args)
        self.out_layer = nn.MaxPool2d(2, stride=2) if args.final_pool else nn.Identity()

    def extract_features(self, x):
        return self.forward(x)

    def forward(self, x):

        # fix feats_type in convert_feats func
        cf = partial(convert_feats, feats_type=self.feats_type)

        # output feats only
        if not self.rmid:
            x = self.pspnet.extract_features(x)
            x = self.out_layer(x)
            return cf(x)

        # feats of all stages
        x, mid_feats = self.pspnet.extract_features(x)
        x = cf(self.out_layer(x))
        mid_feats = [cf(f) for f in mid_feats]
        return x, mid_feats

class ViTWrap(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.arch = args.arch
        self.patch_size = args.patch_size
        self.feats_type = args.feats_type
        self.extra_tokens_num = args.extra_tokens_num
        self.vit = timm.create_model(args.arch, pretrained=args.pretrained, img_size=args.image_size)

    def extract_features(self, x):
        return self.forward(x)

    def forward(self, x):
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.vit.pos_drop(x + self.vit.pos_embed)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)
        x = x[:, self.extra_tokens_num:]
        x = convert_feats(x, self.feats_type)
        return x
