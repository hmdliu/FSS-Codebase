
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.blocks import Attention, DecoderSimple
from src.model.utils import interpolate, convert_feats, LOSS_DICT
from src.utils import yield_params

class SampleModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.norm_s = args.norm_s
        self.norm_q = args.norm_q
        self.im_size = (args.image_size, args.image_size)

        self.inner_loss = LOSS_DICT[args.inner_loss]
        self.meta_loss = LOSS_DICT[args.meta_loss]

        self.classifier = DecoderSimple(
            n_cls=2,
            d_encoder=args.encoder_dim
        )
        self.att = Attention(
            dim=args.encoder_dim,
            heads=args.heads,
            dropout=args.att_dropout
        )
        self.norm = nn.LayerNorm(args.encoder_dim)

    def meta_params(self):
        return yield_params([self.att, self.norm])

    @staticmethod
    def compute_weight(label, n_cls):
        try:
            count = torch.bincount(label.flatten())
            weight = torch.tensor([count[0]/count[i] for i in range(n_cls)])
        except:
            weight = torch.ones(n_cls, device=label.device)
        return weight

    def inner_loop(self, f_s, label_s, weight_s):

        # reset classifier
        self.classifier.reset_parameters()
        self.classifier.train()

        # init optimizer
        optimizer = torch.optim.SGD(self.classifier.parameters(), lr=self.args.lr_cls)

        # adapt the classifier to current task
        for _ in range(self.args.adapt_iter):

            # make prediction
            pred_s = self.classifier(f_s, self.im_size)
            
            # compute loss & update classifier weights
            loss_s = self.inner_loss(pred_s, label_s, weight=weight_s)
            optimizer.zero_grad()
            loss_s.backward()
            optimizer.step()

    def forward(self, backbone, img_s, img_q, label_s, label_q, use_amp=False):

        # extract feats
        with torch.no_grad():
            f_s = backbone.extract_features(img_s)
            f_q = backbone.extract_features(img_q)

        # init variables
        B, C, _, _ = f_q.size()
        pred_q, loss_q = [], []
        weight_s = self.compute_weight(label_s, n_cls=2)
        weight_q = self.compute_weight(label_q, n_cls=2)

        # normalize feats as needed
        if self.norm_s:
            f_s = F.normalize(f_s, dim=1)
        if self.norm_q:
            f_q = F.normalize(f_q, dim=1)

        # pred0: inner loop baseline
        self.inner_loop(f_s, label_s, weight_s)
        self.classifier.eval()
        pred_q.append(self.classifier(f_q, self.im_size))
        loss_q.append(self.meta_loss(pred_q[-1], label_q, weight=weight_q))

        # pred1: self-attention refinement
        cls_weights = self.classifier.weights().squeeze()                   # [2, C]
        cls_weights = cls_weights.unsqueeze(0).expand(B, 2, C)              # [B, 2, C]
        f = torch.cat([convert_feats(f_q, 'vit'), cls_weights], dim=1)      # [B, L+2, C]
        f = self.norm(f + self.att(f)[0])                                   # [B, L+2, C]
        f_q, cls_weights = f[:, :-2, :], f[:, -2:, :]                       # [B, L, C], [B, 2, C]
        pred_1 = torch.einsum('blc,bkc->blk', f_q, cls_weights)             # [B, L, 2]
        pred_1 = convert_feats(pred_1, 'cnn')                               # [B, 2, H/8, W/8]
        pred_q.append(interpolate(pred_1, size=self.im_size))               # [B, 2, H, W]
        loss_q.append(self.meta_loss(pred_q[-1], label_q, weight=weight_q))
        
        # pred2: ensemble
        pred_q.append(pred_q[0] + pred_q[1])
        loss_q.append(self.meta_loss(pred_q[-1], label_q, weight=weight_q))
        
        return pred_q, loss_q            
