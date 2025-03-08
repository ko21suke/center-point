import torch
import torch.nn as nn

from easydict import EasyDict


"""
NOTE: This module has two sub networks
    - top-down network: produces features at increasingly small spatial resolution
        - (SLF)
            S (measured relative to the original input pseudo-image). A block has L 3x3 2D conv-layers with F output channels,each followed by BatchNorm and a ReLU. T
    - bottom-up network: upsample and concatenate of the top-down features
"""

class PointPillarsScatter(nn.Module):
    def __init__(self, config: EasyDict):
        """ Point Pillar's Scatter.
        Explanation:
        Converts learned features from dense tensor to sparse pseudo image.
        This replaces SECOND's second.pytorch.voxelnet.SparseMiddleExtractor. (https://github.com/traveller59/second.pytorch/blob/master/second/pytorch/models/pointpillars.py#L421)
        Args:
            config: configuration of the Pillar Feature Net.
                - output_shape: : ([int]: 4). Required output shape of features.
                - num_input_features:  <int>. Number of input features.
        # TODO: Try to replace this code to be original code (SECOND's code().
        """
        super(PointPillarsScatter, self).__init__()
        self.in_channels = config.num_input_features

    def forward(self, voxel_features, coords, batch_size, input_shape):
        self.nx = input_shape[0]
        self.ny = input_shape[1]
        # batch_canvas will be the final output
        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(
                self.in_channels,
                self.nx * self.ny,
                dtype=voxel_features.dtype,
                device=voxel_features.device,
            )
            # Only include non-empty pillars
            batch_mask = coords[:, 0] == batch_itt
            this_coords = coords[batch_mask, :]
            # create indices as [indices, 1]
            indices = this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :]
            voxels = voxels.t()

            # Now scatter the blob back to the canvas.
            canvas[:, indices] = voxels

            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch size, input_channels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor
        # B, channels, H*W -> B, channels, H, W as pseudo image
        return batch_canvas.view(batch_size, self.in_channels, self.ny, self.nx)




