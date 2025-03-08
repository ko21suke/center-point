import time

import numba
import numpy as np


@numba.jit(nopython=True)
def _points_to_voxel_reverse_kernel(
    points,
    voxel_size,
    coors_range, # [min_x, min_y, min_z, max_x, max_y, max_z]
    num_points_per_voxel, # (max_voxels,) as 0
    coor_to_voxelidx, # (voxel_shapes) as -1
    voxels, # shape is (max_voxels: row, max_points; col, num_features) all values are 0
    coors,
    max_points=35,
    max_voxels=20000,
):
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise
    # reduce performance
    # N = num of points
    N = points.shape[0]
    # ndim = points.shape[1] - 1
    ndim = 3
    ndim_minus_1 = ndim - 1
    # num of grid in each axis (x)
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # np.round(grid_size)
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
    # (0, 0, 0) as shape is (3,)
    coor = np.zeros(shape=(3,), dtype=np.int32)
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        # Create the the voxel indices (xyz) of a points.
        # j is 0 or 1 or 2
        for j in range(ndim):
            # Get the coordinate of the voxel of the points
            # ex: x coordinates of the point - min_x / voxel size of x
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])  # 切り下げ
            # Confirm the c is in the grid area
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            # 2 - j if j is x (0), coor[2] = gird index() <- reverse
            coor[ndim_minus_1 - j] = c
        if failed:
            continue
        # Get the voxel index (-1 means not to be set)
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]  # xyzのボクセルのindexはすべて-1であるが、点が含まれることがわかりしだい該当のグリッドの位置に合わせてindexが付与される
        # If the voxel index is not set, set the voxel index
        if voxelidx == -1:
            voxelidx = voxel_num
            # voxelの数が最大になった場合は、以降すでにindexが付与されているボックスのみに点を追加する
            if voxel_num >= max_voxels:
                continue
            voxel_num += 1
            # Set the voxel index
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            # Set the coordinates of the voxel in the voxel coordinates
            coors[voxelidx] = coor
        # Get the number of points in the voxel
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            # Set hte points in the voxel
            voxels[voxelidx, num] = points[i]
            # Increase the number of points in the voxel
            num_points_per_voxel[voxelidx] += 1
    return voxel_num


@numba.jit(nopython=True)
def _points_to_voxel_kernel(
    points,
    voxel_size,
    coors_range,
    num_points_per_voxel,
    coor_to_voxelidx,
    voxels,
    coors,
    max_points=35,
    max_voxels=20000,
):
    # need mutex if write in cuda, but numba.cuda don't support mutex.
    # in addition, pytorch don't support cuda in dataloader(tensorflow support this).
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise
    # decrease performance
    N = points.shape[0]
    # ndim = points.shape[1] - 1
    ndim = 3
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)

    lower_bound = coors_range[:3]
    upper_bound = coors_range[3:]
    coor = np.zeros(shape=(3,), dtype=np.int32)
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                continue
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1
    return voxel_num


def points_to_voxel(
    points, voxel_size, coors_range, max_points=35, reverse_index=True, max_voxels=20000
):

    """convert kitti points(N, >=3) to voxels. This version calculate
    everything in one loop. now it takes only 4.2ms(complete point cloud)
    with jit and 3.2ghz cpu.(don't calculate other features)
    Note: this function in ubuntu seems faster than windows 10.

    Args:
        points: [N, ndim] float tensor. points[:, :3] contain xyz points and
            points[:, 3:] contain other information such as reflectivity.
        voxel_size: [3] list/tuple or array, float. xyz, indicate voxel size
        coors_range: [6] list/tuple or array, float. indicate voxel range.
            format: xyzxyz, minmax
        max_points: int. indicate maximum points contained in a voxel.
        reverse_index: boolean. indicate whether return reversed coordinates.
            if points has xyz format and reverse_index is True, output
            coordinates will be zyx format, but points in features always
            xyz format.
        max_voxels: int. indicate maximum voxels this function create.
            for second, 20000 is a good choice. you should shuffle points
            before call this function because max_voxels may drop some points.

    Returns:
        voxels: [M, max_points, ndim] float tensor. only contain points.
        coordinates: [M, 3] int32 tensor.
        num_points_per_voxel: [M] int32 tensor.
    """
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.dtype)
    # [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0] / [0.2, 0.2, 8]
    # voxel shape (num voxels along with x, y, z axis)
    voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
    if reverse_index:
        # Reverse the order of the shape
        voxelmap_shape = voxelmap_shape[::-1]
    # don't create large array in jit(nopython=True) code.
    # (max_voxels,) as 0
    num_points_per_voxel = np.zeros(shape=(max_voxels,), dtype=np.int32)
    # (max_voxels,) as -1
    coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)
    # (max_voxels, max_points, num_features) as 0
    voxels = np.zeros(
        shape=(max_voxels, max_points, points.shape[-1]), dtype=points.dtype
    )
    coors = np.zeros(shape=(max_voxels, 3), dtype=np.int32)
    # Get the number of voxels with setting the points in the voxels doing  set coors, num_points_per_voxel, coor_to_voxelidx
    if reverse_index:
        voxel_num = _points_to_voxel_reverse_kernel(
            points,
            voxel_size,
            coors_range,
            num_points_per_voxel,
            coor_to_voxelidx,
            voxels,
            coors,
            max_points,
            max_voxels,
        )

    else:
        voxel_num = _points_to_voxel_kernel(
            points,
            voxel_size,
            coors_range,
            num_points_per_voxel,
            coor_to_voxelidx,
            voxels,
            coors,
            max_points,
            max_voxels,
        )

    # The coordinates of the each points based on the voxel grid scale,
    coors = coors[:voxel_num]
    # The Voxel have the points
    voxels = voxels[:voxel_num]
    # this instance show how may points in each voxel
    num_points_per_voxel = num_points_per_voxel[:voxel_num]
    return voxels, coors, num_points_per_voxel


@numba.jit(nopython=True)
def bound_points_jit(points, upper_bound, lower_bound):
    # to use nopython=True, np.bool is not supported. so you need
    # convert result to np.bool after this function.
    N = points.shape[0]
    ndim = points.shape[1]
    keep_indices = np.zeros((N,), dtype=np.int32)
    success = 0
    for i in range(N):
        success = 1
        for j in range(ndim):
            if points[i, j] < lower_bound[j] or points[i, j] >= upper_bound[j]:
                success = 0
                break
        keep_indices[i] = success
    return keep_indices
