import torch.nn as nn

from easydict import EasyDict


class DynamicVoxelEncoder(nn.Module):
    def __init__(self, cfg: EasyDict):
        super(DynamicVoxelEncoder, self).__init__()

    def forward(self, x):
        pass