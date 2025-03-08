from pathlib import Path

import fire

from datasets.nuscenes import common as nu_ds
from datasets.utils.create_gt_database import create_gt_database


def nuscenes_data_prep(root_path, version, num_sweeps=10, filter_zero=True, virtual=False):
    # nu_ds.create_nuscenes_infos(root_path, version=version, num_sweeps=num_sweeps, filter_zero=filter_zero)
    if version == 'v1.0-trainval' or version == 'v1.0-mini':
        create_gt_database(
            "NUSC",
            root_path,
            Path(root_path) / "infos_train_{:02d}sweeps_withvelo_filter_{}.pkl".format(num_sweeps, filter_zero),
            num_sweeps=num_sweeps,
            virtual=virtual
        )
    else:
        raise NotImplementedError("NuScenes version {} not supported".format(version))


if __name__ == "__main__":
    fire.Fire()