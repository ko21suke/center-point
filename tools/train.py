import argparse
from copy import deepcopy
import os
from pathlib import Path

import torch.nn as nn

from tools.configs.config import load_config
from tools.models.center_point import CenterPoint
from tools.datasets.nuscenes.nuscenes import NuScenesDataset
from tools.datasets.config import Config
from tools.datasets.builder import build_dataset
from tools.torchie.apis import set_random_seed, train_detector, get_root_logger


def main(args: argparse.Namespace) -> None:
    work_dir = Path(args.work_dir)
    # FIXME: split config to manage each values like dataset, model and so on
    cfg = Config.fromfile("./tools/configs/model/default_point_pillar.py")

    # TODO: Manage configs
    distributed = int(os.environ["WORLD_SIZE"]) > 1 if "WORLD_SIZE" in os.environ else False

    if distributed:
        raise NotImplementedError("Distributed training is not supported yet.")
        # if args.launcher == "pytorch":
        #     torch.cuda.set_device(args.local_rank)
        #     torch.distributed.init_process_group(backend="nccl", init_method="env://")
        #     cfg.local_rank = args.local_rank
    else:
        cfg.local_rank = args.local_rank

    # if args.autoscale_lr:
    #     cfg.lr_config.lr_max = cfg.lr_config.lr_max * cfg.gpus

    # init logger before other steps
    # logger = get_root_logger(cfg.log_level)
    logger = get_root_logger()  # Set log level in config
    # logger.info("Distributed training: {}".format(distributed))
    # logger.info(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")

    # if args.local_rank == 0:
        # copy important files to backup
        # backup_dir = os.path.join(cfg.work_dir, "det3d")
        # os.makedirs(backup_dir, exist_ok=True)
        # # os.system("cp -r * %s/" % backup_dir)
        # # logger.info(f"Backup source files to {cfg.work_dir}/det3d")

    # # set random seeds
    # if args.seed is not None:
    #     logger.info("Set random seed to {}".format(args.seed))
    #     set_random_seed(args.seed)

    # model_cfg = load_config(args.model_cfg)
    # model = CenterPoint(model_cfg)
    # cfg
    model = CenterPoint(cfg)
    dataset_root = Path(args.dataset_cfg).absolute()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    # dataset_config = load_config(dataset_root)
    # dataset_config = Config.fromfile(dataset_root)
    # datasets = [build_dataset(dataset_config.data.train)]
    datasets = [build_dataset(cfg.data.train)]

    # if len(cfg.workflow) == 2:
    #     datasets.append(build_dataset(cfg.data.val))

    if cfg.checkpoint_config is not None:
        # save det3d version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(config=cfg.text, CLASSES=datasets[0].CLASSES)

    model.CLASSES = datasets[0].CLASSES
    train_detector(model, datasets, cfg, distributed=distributed, validate=args.validate, logger=logger)

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_cfg",
        type=str,
        default=Path(__file__).parent / "configs" / "model" / "default_point_pillar.yaml",
        help="Path to the config file",
    )
    parser.add_argument(
        "--dataset_cfg",
        type=str,
        default=Path(__file__).parent / "configs" / "dataset" / "default_nuscenes.py",
        help="Path to the dataset config file"
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default=Path(__file__).parent.parent / "work_dir",
        help="The dir to save logs and models"
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--validate",
        action="store_true",
        help="whether to evaluate the checkpoint during training",
    )
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


if __name__ == "__main__":
    args: argparse.Namespace = init_args()
    set_random_seed(21)
    main(args)