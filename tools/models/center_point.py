# import inspect
import torch.nn as nn
from easydict import EasyDict

from tools.models import backbones
from tools.models import feature_extractors
from tools.models import heads
from tools.models import necks
from tools.models import pcd_encoders

# from tools import torchie


"""
TODO:
    - Add trainf_config and test_config refer to the models in det3d,
    - The parameter used in xxx-head should.
"""


class CenterPoint(nn.Module):
    def __init__(self, cfg):
        super(CenterPoint, self).__init__()
        self.cfg = cfg
        self._build_modules(cfg.modules)

    def _get_module(modules: dict, module: EasyDict) -> bool:
        try:
            return modules.__all__[module.name](module)
        except KeyError:
            message = f"{module.name} is not found in __all__ of {modules.__name__}"
            raise KeyError(message)

    def _build_modules(self, module_configs: EasyDict) -> nn.ModuleDict:
        """ Build modules from config.
        Args:
            module_configs: module configurations.
        Returns:
            modules to train or inference.
        """

        # NOTE: Don't use getattr to get the module as we cannot access the module by IDE.
        self.pcd_encoder = self._get_module(pcd_encoders, module_configs.reader)
        self.backbone = self._get_module(backbones, module_configs.backbone)
        self.neck_enabled = module_configs.neck.enabled
        if self.neck_enabled:
            self.neck = self._get_module(necks, module_configs.neck)

        # modules["center_head"] = get_module(heads, module_configs.center_head)

        # self.second_stage_enabled = module_configs.second_stage_modules.enabled
        # if self.second_stage_enabled:
        #     print(f"{'**'*5} Build second stage modules as second stage is enabled. {'**'*5}")
        #     modules["bev_feature_extractor"] = get_module(feature_extractors, module_configs.second_stage_modules.bev_feature_extractor)
        #     modules["roi_head"] = get_module(heads, module_configs.second_stage_modules.roi_head)
        # self.modules = modules

    def extract_feat(self, data):
        """
        data = dict(
            features=voxels (M, max_points, ndim) M: num_voxels
            num_voxels=num_points_in_voxel,
            coors=coordinates, (M, 4) as voxel coordinates (zyx) in grid level. 4 of shape is added in collect_kitti function.
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )
        """
        input_features = self.pcd_encoder(data["features"], data["num_voxels"], data["coors"]
)

        # print(input_features.shape) # 7278, 64
        # print(data["coors"].shape) # 7278, 4
        # print(data["batch_size"]) 1
        # print(data["input_shape"]) 512, 512 ,1

        x = self.backbone(input_features, data["coors"], data["batch_size"], data["input_shape"])
        x = self.neck(x) if self.neck_enabled else x
        return x

    def forward(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"] # num_points_per_voxel
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        ｘ = self.extract_feat(data)

        # predictions, _ = self.modules["bbox_head"](x)

        # if return_loss:
        #     return self.bbox_head.loss(example, predictions, self.test_cfg)
        # else:
        #     return self.bbox_head.predict(example, predictions, self.test_cfg)
        return x

    def forward_two_stage(self, x):
        raise NotImplementedError
