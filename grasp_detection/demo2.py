import os
import argparse
import torch
import numpy as np
import open3d as o3d
from PIL import Image
import time
import pyrealsense2 as rs


from gsnet import AnyGrasp
from graspnetAPI import GraspGroup

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--max_gripper_width', type=float, default=0.1, help='Maximum gripper width (<=0.1m)')
parser.add_argument('--gripper_height', type=float, default=0.03, help='Gripper height')
parser.add_argument('--top_down_grasp', action='store_true', help='Output top-down grasps.')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
cfgs = parser.parse_args()
cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))

def run_realtime_grasping(cfgs):
    if not rs:
        return

    # 1. 初始化AnyGrasp模型
    anygrasp = AnyGrasp(cfgs)
    anygrasp.load_net()

    # 2. 配置并启动RealSense管道
    pipe = rs.pipeline()
    config = rs.config()
    # 启用深度、彩色和加速度数据流
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
    config.enable_stream(rs.stream.accel) # 启用加速度流
    
    profile = pipe.start(config)
    
    # 获取相机内参
    intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy
    
    # 获取深度传感器的缩放因子
    depth_sensor = profile.get_device().first_depth_sensor()
    scale = depth_sensor.get_depth_scale()

    # 定义工作空间
    lims = [-0.5, 0.5, -0.5, 0.5, 0.1, 1.0] # 示例工作区

    # 定义世界坐标系下的竖直向量 (在相机坐标系中动态计算)
    vertical_vector = np.array([0, -1, 0]) # 默认值，会被IMU数据覆盖

    vis = o3d.visualization.Visualizer()
    vis.create_window("AnyGrasp Consumer")
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
    first_frame = True

    try:
        while True:
            frames = pipe.wait_for_frames(10000)
            
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            accel_frame = frames.first_or_default(rs.stream.accel)

            if not depth_frame or not color_frame or not accel_frame:
                continue

            # 4. 从IMU获取当前的重力向量
            accel_data = accel_frame.as_motion_frame().get_motion_data()
            gravity_vec = np.array([accel_data.x, accel_data.y, accel_data.z])
            # 归一化得到方向
            vertical_vector = gravity_vec / np.linalg.norm(gravity_vec)
            # 根据RealSense标准，Y轴向下，静止时读数约为+9.8。我们需要指向天空的向量，所以如果Y为正，则反转。
            if vertical_vector[1] > 0:
                vertical_vector = -vertical_vector

            # 5. 提取图像和生成点云
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            color_image_rgb = color_image[:, :, ::-1]
            
            points_z = depth_image * scale
            xmap, ymap = np.arange(depth_image.shape[1]), np.arange(depth_image.shape)
            xmap, ymap = np.meshgrid(xmap, ymap)
            points_x = (xmap - cx) / fx * points_z
            points_y = (ymap - cy) / fy * points_z
            
            mask = (points_z > 0.1) & (points_z < 1.0) # 过滤近点和远点
            points = np.stack([points_x, points_y, points_z], axis=-1)[mask].astype(np.float32)
            colors = (color_image_rgb / 255.0)[mask].astype(np.float32)

            gg, cloud = anygrasp.get_grasp(points, colors, lims=lims, apply_object_mask=True, dense_grasp=False, collision_detection=True)

            if len(gg) == 0:
                print('No Grasp detected after collision detection!')
                continue
            
            vis.clear_geometries()
            vis.add_geometry(axis_pcd)
            vis.add_geometry(cloud) # Always add the point cloud

            # 7. 使用同步的重力向量进行筛选
            horizontal_grasps = GraspGroup()
            for grasp in gg:
                approach_vector = grasp.rotation_matrix[:, 2]
                dot_product = np.abs(np.dot(approach_vector, vertical_vector))
                
                if dot_product < 0.26: # 阈值 (cos(75°))
                    horizontal_grasps.add(grasp)
            
            print(f"Found {len(horizontal_grasps)} grasps parallel to the ground.")

            # 8. 后续处理与可视化
            gg_to_use = horizontal_grasps if len(horizontal_grasps) > 0 else gg
            gg = gg_to_use.nms().sort_by_score()
            
            if len(gg) > 0 and cfgs.debug:
                # 可视化代码
                grippers = gg[0:5].to_open3d_geometry_list() # 只显示前5个
                for gripper in grippers:
                    vis.add_geometry(gripper)

            if first_frame:
                vis.reset_view_point(True)
                first_frame = False

    finally:
        # 停止管道
        pipe.stop()
        if vis:
            vis.destroy_window()

if __name__ == '__main__':
    run_realtime_grasping(cfgs)