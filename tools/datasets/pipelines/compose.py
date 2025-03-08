import collections
import inspect

from tools import torchie
from tools.datasets.pipelines.formating import Reformat
from tools.datasets.pipelines.preprocess import Preprocess, Voxelization, AssignLabel

from tools.datasets.pipelines.loading import LoadPointCloudFromFile, LoadPointCloudAnnotations
from tools.datasets.pipelines.test_aug import DoubleFlip


__all__ = {
    # Assign labels
    "AssignLabel": AssignLabel,
    "DoubleFlip": DoubleFlip,
    # Get ground truth
    "LoadPointCloudAnnotations": LoadPointCloudAnnotations,
    # Get point cloud with sweeps included in time lag.
    "LoadPointCloudFromFile": LoadPointCloudFromFile,
    # Filter by class names and min num point, and make train data with ground truth data if the mode is train.
    "Preprocess": Preprocess,
    # Format the data bundle (dict) with keys: voxels, shape, num_points, num_voxels, coordinates
    "Reformat": Reformat,
    # Assign each points in the voxel following the point of range and the voxel size.
    "Voxelization": Voxelization,
}


# FIXME: Refactoring as this sorouce code rewirte to not use registry from det3d.
def build_pipelines(cfg, default_args=None):
    assert isinstance(cfg, dict) and "type" in cfg
    assert isinstance(default_args, dict) or default_args is None
    args = cfg.copy()
    obj_type = args.pop("type")
    if torchie.is_str(obj_type):
        obj_cls = __all__[obj_type]
        if obj_cls is None:
            raise KeyError(
                "{} is not in the {} registry".format(obj_type, "pipeline")
            )
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(
            "type must be a str or valid type, but got {}".format(type(obj_type))
        )
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    return obj_cls(**args)

class Compose(object):
    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                if transform['type'] == 'Empty':
                    continue
                transform = build_pipelines(transform)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError("transform must be callable or a dict")

    def __call__(self, res, info):
        for t in self.transforms:
            res, info = t(res, info)
            if res is None:
                return None
        return res, info

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

