import torch.nn as nn

from easydict import EasyDict


class BEVFeatureExtractor(nn.Module):
    def __init__(self, cfg: EasyDict):
        super(BEVFeatureExtractor, self).__init__()

    def forward(self, x):
        pass