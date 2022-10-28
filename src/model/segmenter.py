
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

class Segmenter(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        n_cls,
        patch_size=16,
        extra_tokens_num=1
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = patch_size
        self.encoder = encoder
        self.decoder = decoder
        self.extra_tokens_num = extra_tokens_num

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("encoder.", self.encoder).union(
            append_prefix_no_weight_decay("decoder.", self.decoder)
        )
        return nwd_params

    def forward(self, im):

        # forward encoder
        x = self.encoder(im)
        if isinstance(x, tuple):
            x = x[0]

        # forward decoder
        H, W = im.size(2), im.size(3)
        masks = self.decoder(x, (H, W))

        # up-sampling
        masks = F.interpolate(masks, size=(H, W), mode="bilinear", align_corners=True)

        return masks

    def get_attention_map_enc(self, im, layer_id):
        assert self.arch.find('vit') != -1
        return self.encoder.get_attention_map(im, layer_id)

    def get_attention_map_dec(self, im, layer_id):
        assert self.arch.find('vit') != -1
        x = self.forward_encoder(im)
        x = x[:, self.extra_tokens_num:]
        return self.decoder.get_attention_map(x, layer_id)
