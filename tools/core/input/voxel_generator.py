import numpy as np
from tools.ops.point_cloud.point_cloud_ops import points_to_voxel


class VoxelGenerator:
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels=20000):
        # [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        # [0.2, 0.2, 8]
        voxel_size = np.array(voxel_size, dtype=np.float32)
        # This is axis grid size of each axis (x, y, z).
        grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)

        self._voxel_size = voxel_size
        self._point_cloud_range = point_cloud_range
        self._max_num_points = max_num_points
        self._max_voxels = max_voxels
        self._grid_size = grid_size

    def generate(self, points, max_voxels=-1):
        if max_voxels == -1:
            max_voxels=self._max_voxels

        """
        points: (N, 3 + C) [x, y, z, ...]
        voxel_size: [0.2, 0.2, 8] as xyz
        point_cloud_range: [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0] as min x, min y, min z, max x, max y, max z
        max_num_points: 20
        max_voxels: 20000
        """
        return points_to_voxel(
            points,
            self._voxel_size,
            self._point_cloud_range,
            self._max_num_points,
            True,
            max_voxels,
        )

    @property
    def voxel_size(self):
        return self._voxel_size

    @property
    def max_num_points_per_voxel(self):
        return self._max_num_points

    @property
    def point_cloud_range(self):
        return self._point_cloud_range

    @property
    def grid_size(self):
        return self._grid_size
