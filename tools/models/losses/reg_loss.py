import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.models.losses.utils import transpose_and_gather_feat

# TODO: Understand this class
class RegLoss(nn.Module):
    """Regression loss for an output tensor. """
    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, output, mask, ind, target):
        '''Regression loss for an output tensor
        Arguments:
            output (batch x dim x h x w)
            mask (batch x max_objects)
            ind (batch x max_objects)
            target (batch x max_objects x dim)
        '''
        pred = transpose_and_gather_feat(output, ind)
        mask = mask.float().unsqueeze(2)

        loss = F.l1_loss(pred*mask, target*mask, reduction='none')
        loss = loss / (mask.sum() + 1e-4)
        loss = loss.transpose(2 ,0).sum(dim=2).sum(dim=1)
        return loss

