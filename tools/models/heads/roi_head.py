import torch.nn as nn

from easydict import EasyDict


class ROIHead(nn.Module):
    def __init__(self, cfg: EasyDict):
        super(ROIHead, self).__init__()

    def forward(self, x):
        pass