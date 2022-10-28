
import pdb
import timm

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.segmenter import Segmenter
from src.model.backbone.wrap import ResNetWrap, PSPNetWrap, ViTWrap
from src.model.blocks import DecoderSimple, DecoderLinear, MaskTransformer
from src.model.meta import META_MODEL_DICT

def get_pretrain(args):
    
    assert args.decoder in ('simple', 'linear', 'mask')

    if args.arch.find('vit') != -1:
        encoder = ViTWrap(args)
    elif args.arch.find('psp') != -1:
        encoder = PSPNetWrap(args)
    else:
        encoder = ResNetWrap(args)
    
    if args.decoder == 'simple':
        decoder = DecoderSimple(
            n_cls=args.num_classes_tr,
            d_encoder=args.encoder_dim
        )
    elif args.decoder == 'linear':
        decoder = DecoderLinear(
            n_cls=args.num_classes_tr,
            patch_size=args.patch_size,
            d_encoder=args.encoder_dim
        )
    else:
        decoder = MaskTransformer(
            n_cls=args.num_classes_tr,
            patch_size=args.patch_size,
            d_encoder=args.encoder_dim,
            n_layers=2,
            n_heads=args.decoder_dim // 64,
            d_model=args.decoder_dim,
            d_ff=args.decoder_dim * 4,
            drop_path_rate=0.0,
            dropout=0.1
        )
    model = Segmenter(
        encoder=encoder,
        decoder=decoder,
        n_cls=args.num_classes_tr,
        patch_size=args.patch_size,
        extra_tokens_num=args.extra_tokens_num
    )
    return model

def get_backbone(args):

    if args.arch.find('vit') != -1:
        backbone = ViTWrap(args)
    elif args.arch.find('psp') != -1:
        backbone = PSPNetWrap(args)
    else:
        backbone = ResNetWrap(args)

    return backbone

def get_meta_model(args):
    return META_MODEL_DICT[args.meta_model](args)
