import os
import numpy as np
import open3d as o3d
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

from pathlib import Path

data_root = Path(__file__).parent / "dataset" / "nuscenes"
version = "v1.0-mini"
nusc = NuScenes(version=version, dataroot=data_root, verbose=True)


colors = [
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (0, 1, 1),
    (1, 0, 1),
]

# LiDAR点群を読み込む関数
def load_lidar_points(lidar_sd):
    """ LiDARの点群を読み込む """
    pcl_path = data_root / lidar_sd["filename"]
    pc = LidarPointCloud.from_file(str(pcl_path))
    return pc.points.T[:, :3]  # (N, 3)



# LiDAR点群をワールド座標系へ変換
def transform_lidar_to_world(points, lidar_to_ego, ego_to_world):
    """ LiDAR点群をワールド座標系に変換 """
    if points is None:
        return None

    # 1. LiDAR → 自車座標系
    points_hom = np.hstack((points, np.ones((points.shape[0], 1))))  # (N, 4)
    points_ego = (lidar_to_ego @ points_hom.T).T[:, :3]  # (N, 3)

    # 2. 自車座標系 → ワールド座標系
    points_hom = np.hstack((points_ego, np.ones((points_ego.shape[0], 1))))  # (N, 4)
    points_world = (ego_to_world @ points_hom.T).T[:, :3]  # (N, 3)

    return points_world


# Open3Dで可視化
def visualize_point_cloud(points_list, color_list, bbox):
    """ LiDAR点群とBBoxを可視化 """
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 各点群をOpen3Dの点群オブジェクトに変換
    for points, color in zip(points_list, color_list):  # 現在の点群は緑、prevは青
        if points is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.paint_uniform_color(color)
            vis.add_geometry(pcd)

    # BBoxを追加
    vis.add_geometry(bbox)

    vis.run()
    vis.destroy_window()

# BBoxをワールド座標系のままで可視化
def get_bbox_in_world_frame(ann):
    """ BBoxをワールド座標系のままで取得 """
    size = np.array(ann["size"])
    size = size[[1, 0, 2]]  # (h, w, l) -> (w, h, l)
    translation = np.array(ann["translation"])
    rotation = Quaternion(np.array(ann["rotation"]))

    # BBoxの回転行列
    rot_matrix = transform_matrix([0, 0, 0], rotation, inverse=False)[:3, :3]

    # Open3Dのバウンディングボックスを作成
    bbox = o3d.geometry.OrientedBoundingBox(center=translation, R=rot_matrix, extent=size)
    bbox.color = (1, 0, 0)  # 赤色
    return bbox

def generate_colors(num_colors):
    """ 色のリストを生成 """
    colors = []
    for i in range(num_colors):
        color = np.zeros(3)
        color[i % 3] = 1
        colors.append(color)
    return colors

scene = nusc.scene[1]
scene_token = scene["token"]
last_sample_token = scene["last_sample_token"]
last_sample = nusc.get("sample", last_sample_token)

ann_token = last_sample["anns"][15]
ann = nusc.get("sample_annotation", ann_token)

points = []
cur_lidar_sd_token = last_sample["data"]["LIDAR_TOP"]
cur_lidar_sd = nusc.get("sample_data", cur_lidar_sd_token)
cur_lidar_calib = nusc.get("calibrated_sensor", cur_lidar_sd["calibrated_sensor_token"])
cur_lidar_to_ego = transform_matrix(np.asarray(cur_lidar_calib["translation"]), Quaternion(np.asarray(cur_lidar_calib["rotation"])), inverse=False)
cur_ego_pose = nusc.get("ego_pose", cur_lidar_sd["ego_pose_token"])
cur_ego_to_world = transform_matrix(np.asarray(cur_ego_pose["translation"]), Quaternion(np.asarray(cur_ego_pose["rotation"])), inverse=False)
cur_points_lidar = load_lidar_points(cur_lidar_sd)
cur_points_world = transform_lidar_to_world(cur_points_lidar, cur_lidar_to_ego, cur_ego_to_world)

points.append(cur_points_world)

# 5. prevのサンプルデータがある場合は取得
# for _ in range(2):
#     prev_lidar_sd = nusc.get("sample_data", cur_lidar_sd["prev"])
#     prev_lidar_calib = nusc.get("calibrated_sensor", prev_lidar_sd["calibrated_sensor_token"])
#     prev_lidar_to_ego = transform_matrix(np.asarray(prev_lidar_calib["translation"]), Quaternion(np.asarray(prev_lidar_calib["rotation"])), inverse=False)
#     prev_ego_pose = nusc.get("ego_pose", prev_lidar_sd["ego_pose_token"])
#     prev_ego_to_world = transform_matrix(np.asarray(prev_ego_pose["translation"]), Quaternion(np.asarray(prev_ego_pose["rotation"])), inverse=False)
#     points_prev_lidar = load_lidar_points(prev_lidar_sd)
#     points_prev_world = transform_lidar_to_world(points_prev_lidar, prev_lidar_to_ego, prev_ego_to_world)
#     points.append(points_prev_world)
#     cur_lidar_sd = prev_lidar_sd


bbox_world = get_bbox_in_world_frame(ann)
visualize_point_cloud(points, colors[:len(points)], bbox_world)

# def get_calibrated_sensor(nusc, sample, sensor_channel):
#     sd_record = nusc.get("sample_data", sample["data"][sensor_channel])
#     return nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])

# for scene in nusc.scene:
#     first_sample_token = scene["first_sample_token"]
#     sample = nusc.get("sample", first_sample_token)

#     lidar_top = nusc.get("ego_pose", sample["data"]["LIDAR_TOP"])
#     cam_front = nusc.get("ego_pose", sample["data"]["CAM_FRONT"])
#     cam_back = nusc.get("ego_pose", sample["data"]["CAM_BACK"])
#     lidar_top = get_calibrated_sensor(nusc, sample, "LIDAR_TOP")
#     cam_front = get_calibrated_sensor(nusc, sample, "CAM_FRONT")
#     cam_back = get_calibrated_sensor(nusc, sample, "CAM_BACK")

#     cam_front_translation = cam_front["translation"]
#     cam_front_rotation = cam_front["rotation"]
#     print(cam_front_translation)
#     cam_front_matrix = transform_matrix(cam_front_translation, Quaternion(cam_front_rotation))
#     print(cam_front_matrix)

#     cam_back_translation = cam_back["translation"]
#     cam_back_rotation = cam_back["rotation"]
#     print(cam_back_translation)
#     cam_back_matrix = transform_matrix(cam_back_translation, Quaternion(cam_back_rotation))

#     print(cam_back_matrix)

#     exit()


