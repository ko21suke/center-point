# TODO: Check the license of the code.

"""
PointPillars fork from SECOND.
Code written by Alex Lang and Oscar Beijbom, 2018.
Licensed under MIT License [see LICENSE].
"""

import torch
import torch.nn as nn
from easydict import EasyDict
from torch.nn import functional as F

from tools.models.normalizations.normalization import build_norm_layer
from tools.models.utils.misc import get_paddings_indicator


def has_filters(num_filers):
    return len(num_filers) > 0


class PFNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg=None, last_layer=False):
        """ Pillar Feature Net Layer.
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            norm_cfg: normalization configuration.
            last_layer: If last_layer, there is no concatenation of features.
        """
        super(PFNLayer, self).__init__()
        self.last_vef = last_layer
        self.units = out_channels if self.last_vef else out_channels // 2
        self.norm_cfg = norm_cfg if norm_cfg else dict(type="BN1d", eps=1e-3, momentum=0.01)
        self.linear = nn.Linear(in_channels, self.units, bias=False)
        self.norm = build_norm_layer(self.norm_cfg, self.units)[1]

    # https://arxiv.org/pdf/2007.00493 のp3を読めばわかる
    def forward(self, inputs):
        # input: (P, N, D)
        x = self.linear(inputs)
        # TODO:: Why cudnn.benchmark is set to False?
        torch.backends.cudnn.benchmark = False
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        torch.backends.cudnn.benchmark = True
        x = F.relu(x)
        # x: (P, N, C)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        # x_max: (P, 1, C)
        if self.last_vef:
            return x_max
        else:
            # # x_max: (P, 1, C)
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            # x_repeat: (P, N, C)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            # x_concatenated: (P, N, Cx2)
            return x_concatenated


class PillarFeatureNet(nn.Module):
    def __init__(self, config: EasyDict):
        """ Pillar Feature Net.
        Explanation:
            The network prepares the pillar features and performs forward pass through PFNLayers.
            This net performs a similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        Args:
            config: configuration of the Pillar Feature Net.
                - num_input_features=4: <int>. Number of input features, either x, y, z or x, y, z, r.
                - num_filters=(64,): Number of features in each of the N PFNLayers.
                - with_distance=False: Whether to include Euclidean distance to points.
                - voxel_size=(0.2, 0.2, 4): ize of voxels, only utilize x and y size.
                - pc_range=(0, -40, -3, 70.4, 40, 1): Point cloud range, only utilize x and y min.
                - norm_cfg=None:
                - virtual=False:
        """
        super(PillarFeatureNet, self).__init__()

        if not has_filters(config.num_filters):
            raise ValueError("Number of filters must be greater than 0")

        # TODO: Confirm from this line
        self.num_input = config.num_input_features

        # x, y, z, intensity, x_c, y_c, z_c, , x_p, y_p
        num_input_features = 5 + config.num_input_features
        # Euclidean distance from the point to the origin
        if config.with_distance:
            num_input_features += 1
        self._with_distance = config.with_distance

        # Create PillarFeatureNet layers
        # will be [9or10, filter, filter, ...] the elements indicate in and out channels of each layer
        num_filters = [num_input_features] + list(config.num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            # num_filter: [ 9, 64, 64], len(num_filter) = 3, index 1 is last layer
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, norm_cfg=config.norm_cfg, last_layer=last_layer)
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)
        # basically False
        self.virtual = config.virtual

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        self.vx = config.voxel_size[0]  # size of voxel x
        self.vy = config.voxel_size[1]  # size of voxel y
        self.x_offset = self.vx / 2 + config.pc_range[0]  # center of voxel x - min_x
        self.y_offset = self.vy / 2 + config.pc_range[1]  # center of voxel y - min_y


    def forward(
        self,
        features, # voxels features having points (index_voxels(num_voxel): row, num_points: col, num_features)
        num_points_in_voxel,  #  origianl is num_voxels
        coors  # The coordinates in grid of the each voxel ordered voxel index. [[grid_x, grid_y, grid_z] <- voxeindex 0, [grid_x, grid_y, grid_z] <- voxeindex 1, ...].
    ) -> torch.Tensor:
        if self.virtual:
            virtual_point_mask = features[..., -2] == -1
            virtual_points = features[virtual_point_mask]
            virtual_points[..., -2] = 1
            features[..., -2] = 0
            features[virtual_point_mask] = virtual_points
        dtype = features.dtype
        # Find distance of x, y, and z from cluster center
        # features = features[:, :, :self.num_input]
        # Calculate the mean of the points in the each voxel
        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_points_in_voxel.type_as(features).view(-1, 1, 1)
        # Get the distance of each point from the mean of the points in the voxel
        f_cluster = features[:, :, :3] - points_mean

        # Find distance of x, y, and z from pillar center
        # f_center = features[:, :, :2]
        f_center = torch.zeros_like(features[:, :, :2])  # shape is (num_voxels, num_points, 2)

        # Get the x and y coordinates of the voxel as center of the voxel
        f_center[:, :, 0] = features[:, :, 0] - (coors[:, 3].to(dtype).unsqueeze(1) * self.vx + self.x_offset)
        f_center[:, :, 1] = features[:, :, 1] - (coors[:, 2].to(dtype).unsqueeze(1) * self.vy + self.y_offset)
        # TODO: Understand below code
        # Combine together feature decorations
        # [x, y, z, x_c, y_c, z_c, x_p, y_p] in the paper
        features_ls = [features, f_cluster, f_center]
        if self._with_distance:
            # add distance from origin
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)
        # shape is (num_voxels, num_points, 9 or 10)
        features = torch.cat(features_ls, dim=-1)

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.

        # not voxel count, this is point count in each voxel
        # voxel_count = features.shape[1]
        num_points = features.shape[1]

        # mask shape is (num_voxels, num_points, ), num_voxel is generated by voxel generator with limit of max_voxels
        # show each voxel has how many points have as true and false
        mask = get_paddings_indicator(num_points_in_voxel, num_points, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)

        # 不要なfeaturesを0にする by mask
        # shape[num_voxels, num_points(max_points), 9 or 10], set 0 to the features where mask is False
        features *= mask
        """
        (P, N, D)
            - P: on the number of non-empty pillars per sample
            - N: on the number of points per pillar (N) to create a
            - D: on the number of features per point
        """

        # Forward pass through PFNLayers
        # Crate persedu image
        for pfn in self.pfn_layers:
            features = pfn(features)
        # features: (P, 1, C)
        return features.squeeze()

