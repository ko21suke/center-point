import copy
import inspect

# from tools.utils import build_from_cfg

from tools.datasets.dataset_wrappers import ConcatDataset, RepeatDataset
from tools.datasets.nuscenes.nuscenes import NuScenesDataset
from tools import torchie


__all__ = {
    "ConcatDataset": ConcatDataset,
    "RepeatDataset": RepeatDataset,
    "NuScenesDataset": NuScenesDataset,
}

def _concat_dataset(cfg, default_args=None):
    ann_files = cfg["ann_file"]
    img_prefixes = cfg.get("img_prefix", None)
    seg_prefixes = cfg.get("seg_prefixes", None)
    proposal_files = cfg.get("proposal_file", None)

    datasets = []
    num_dset = len(ann_files)
    for i in range(num_dset):
        data_cfg = copy.deepcopy(cfg)
        data_cfg["ann_file"] = ann_files[i]
        if isinstance(img_prefixes, (list, tuple)):
            data_cfg["img_prefix"] = img_prefixes[i]
        if isinstance(seg_prefixes, (list, tuple)):
            data_cfg["seg_prefix"] = seg_prefixes[i]
        if isinstance(proposal_files, (list, tuple)):
            data_cfg["proposal_file"] = proposal_files[i]
        datasets.append(build_dataset(data_cfg, default_args))

    return ConcatDataset(datasets)


def build_dataset(cfg, default_args=None):
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg["type"] == "RepeatDataset":
        dataset = RepeatDataset(
            build_dataset(cfg["dataset"], default_args), cfg["times"]
        )
    # elif isinstance(cfg['ann_file'], (list, tuple)):
    #     dataset = _concat_dataset(cfg, default_args)
    else:
        assert isinstance(cfg, dict) and "type" in cfg
        assert isinstance(default_args, dict) or default_args is None
        args = cfg.copy()
        obj_type = args.pop("type")
        if torchie.is_str(obj_type):
            obj_cls = __all__[cfg["type"]]
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                "type must be a str or valid type, but got {}".format(type(obj_type))
            )
        if default_args is not None:
            for name, value in default_args.items():
                args.setdefault(name, value)
        dataset = obj_cls(**args)
    return dataset
