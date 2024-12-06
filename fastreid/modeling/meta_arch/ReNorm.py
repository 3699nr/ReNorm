# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import time
import torch
from torch import nn
import logging
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_heads
from fastreid.modeling.losses import *
from fastreid.modeling.losses.cross_entroy_loss import CrossEntropyLabelSmooth
from fastreid.modeling.losses.triplet_loss import TripletLoss
import numpy as np
from .build import META_ARCH_REGISTRY

import torch.nn.functional as F

@META_ARCH_REGISTRY.register()
class ARCH_RENORM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1))

        # backbone
        self.backbone = build_backbone(cfg)

        # head
        self.heads = build_heads(cfg)
        self.headsEN = build_heads(cfg)

        self.criterion_ce_1 = CrossEntropyLabelSmooth(cfg.MODEL.HEADS.NUM_CLASSES[0]).cuda()
        self.criterion_ce_2 = CrossEntropyLabelSmooth(cfg.MODEL.HEADS.NUM_CLASSES[1]).cuda()
        self.criterion_ce_3 = CrossEntropyLabelSmooth(cfg.MODEL.HEADS.NUM_CLASSES[2]).cuda()
        self.criterion_ce_list = [self.criterion_ce_1, self.criterion_ce_2, self.criterion_ce_3]  

        self.criterion_triple = TripletLoss(margin=cfg.MODEL.LOSSES.TRI_RN.MARGIN, normalize_feature=cfg.MODEL.LOSSES.TRI_RN.NORM_FEAT).cuda()
        self.criterion_triple_EN = TripletLoss(margin=cfg.MODEL.LOSSES.TRI_EN.MARGIN, normalize_feature=cfg.MODEL.LOSSES.TRI_EN.NORM_FEAT).cuda()
        self.count = 0

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs, cur_idx, epoch, use_EN):
        images = self.preprocess_image(batched_inputs)
     
        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"].to(self.device)


            s_features_1 = self.backbone(images, cur_idx, use_EN = False)

            if cur_idx == 0:
                if self.count % 2 == 0:
                    input_idx = 1
                else:
                    input_idx = 2
            elif cur_idx == 1:
                if self.count % 2 == 0:
                    input_idx = 0
                else:
                    input_idx = 2
            elif cur_idx == 2:
                if self.count % 2 == 0:
                    input_idx = 0
                else:
                    input_idx = 1
            self.count +=1

            s_features_EN = self.backbone(images, input_idx, use_EN = True)

            if targets.sum() < 0: targets.zero_()

            outputs_RN = self.heads(s_features_1, cur_idx, targets)    
            outputs_EN = self.headsEN(s_features_EN, cur_idx, targets) 
            losses = self.losses(outputs_RN, outputs_EN, targets, cur_idx)    
            
            return losses
        else:

            s_features_1 = self.backbone(images, cur_idx, use_EN)

            if use_EN:
                outputs = self.headsEN(s_features_1, cur_idx)
            else:
                outputs = self.heads(s_features_1, cur_idx)

            return outputs, s_features_1


    def preprocess_image(self, batched_inputs):
        r"""
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            images = batched_inputs["images"].to(self.device)
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs.to(self.device)
        else:
            raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs)))

        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images



    def losses(self, outputs, outputs_EN, gt_labels, cur_idx):
        r"""
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # model predictions
        # fmt: off
        pred_class_logits = outputs['pred_class_logits'].detach()
        cls_outputs       = outputs['cls_outputs']
        pred_features     = outputs['features']
        cls_outputs_EN    = outputs_EN['cls_outputs']
        pred_features_EN  = outputs_EN['features']
        # fmt: on

        # Log prediction accuracy
        log_accuracy(pred_class_logits, gt_labels)

        loss_dict = {}
        loss_names = self._cfg.MODEL.LOSSES.NAME

        if "CrossEntropyLoss" in loss_names:
            loss_dict["loss_cls"] = self.criterion_ce_list[cur_idx](cls_outputs, gt_labels)
            loss_dict["loss_cls_EN"] = self.criterion_ce_list[cur_idx](cls_outputs_EN, gt_labels)

        if "TripletLoss" in loss_names:
            loss_dict["loss_triplet"] = self.criterion_triple(pred_features, gt_labels) 
            loss_dict["loss_triplet_EN"] = self.criterion_triple_EN(pred_features_EN, gt_labels)



        return loss_dict
