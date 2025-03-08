import copy
import logging

import torch.nn as nn

from copy import deepcopy
from easydict import EasyDict

from tools.models.heads.dcn_sep_head import DCNSepHead
from tools.models.heads.sep_head import SepHead
from tools.models.losses.fast_focal_loss import FastFocalLoss
from tools.models.losses.reg_loss import RegLoss


class CenterHead(nn.Module):
    def __init__(self, cfg: EasyDict):
        super(CenterHead, self).__init__()
        self.in_channels = cfg.in_channels # [128,]
        self.class_names = [task.class_names for task in cfg.tasks]
        self.code_weights = cfg.code_weights
        self.weight = cfg.weight # weight between heatmap loss and loc los
        self.dataset = cfg.dataset

        self.in_channels = cfg.in_channels
        num_classes = [len(task.class_names) for task in cfg.tasks]
        self.num_classes = num_classes

        # TODO: Understand this code
        self.crit = FastFocalLoss()  # detect classes
        self.crit_reg = RegLoss()

        # TODO: Check this cfg param and why 7 and 9
        self.box_n_dim = 9 if 'vel'in cfg.common_heads else 7
        self.use_direction_classifier = False

        if not cfg.logger:
            logger = logging.getLogger("RPN")
        self.logger = logger
        logger.info(f"num_classes: {num_classes}")

        # TODO: Check this layer as I dont know what is this
        # a shared convolution layer
        self.shared_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, cfg.share_conv_channel,
            kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(cfg.share_conv_channel),
            nn.ReLU(inplace=True)
        )

        self.tasks = nn.ModuleList()
        print("Use HM Bias: ", cfg.init_bias)

        # TODO: Check this code from original CenterPoint
        if cfg.dcn_head:
           print("Use Deformable Convolution in the CenterHead!")

        for num_cls in num_classes:
            heads = deepcopy(cfg.common_heads)
            if not cfg.dcn_head:
                heads.update(dict(hm=(num_cls, cfg.num_hm_conv)))
                self.tasks.append(
                    SepHead(cfg.share_conv_channel, heads, bn=True, init_bias=cfg.init_bias, final_kernel=3)
                )
            else:
                self.tasks.append(
                    DCNSepHead(cfg.share_conv_channel, num_cls, heads, bn=True, init_bias=cfg.init_bias, final_kernel=3)
                )

        logger.info("FInish CenterHead Initialization")

    def forward(self, x, *kwargs):
        red_dict = []
        x = self.shared_conv(x)
        for task in self.tasks:
            red_dict.append(task(x))

        return red_dict, x

