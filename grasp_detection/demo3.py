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

# --- [argparse 部分保持不变] ---
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
        print("错误: pyrealsense2 未安装。")
        return

    # 1. 初始化AnyGrasp模型
    anygrasp = AnyGrasp(cfgs)
    anygrasp.load_net()

    # --- 关键修改：设备发现与显式选择 ---
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("错误: 未检测到RealSense设备。")
        return
    
    # 获取第一个找到的设备的序列号
    device_serial_number = '147322071074'
    print(f"找到设备，序列号: {device_serial_number}")

    # 2. 配置并启动两个独立的、但指向同一设备的管道
    pipe_cam = rs.pipeline(ctx)
    config_cam = rs.config()
    config_cam.enable_device(device_serial_number) # 显式指定设备
    config_cam.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    config_cam.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

    pipe_imu = rs.pipeline(ctx)
    config_imu = rs.config()
    config_imu.enable_device(device_serial_number) # 显式指定设备
    config_imu.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 100)

    # 启动两个管道 (IMU管道通常启动更快)
    pipe_imu.start(config_imu)
    profile_cam = pipe_cam.start(config_cam)
    print("相机和IMU管道已启动。")
    
    # --- 引入对齐功能 ---
    align_to = rs.stream.color
    align = rs.align(align_to)

    # 从相机管道获取内参
    intr = profile_cam.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy
    
    depth_sensor = profile_cam.get_device().first_depth_sensor()
    scale = depth_sensor.get_depth_scale()
    print(f"Depth Scale: {scale}")

    lims = [-0.5, 0.5, -0.5, 0.5, 0.1, 1.0]
    vertical_vector = np.array([0, -1, 0])

    #... [可视化初始化部分保持不变]...
    vis = o3d.visualization.Visualizer()
    vis.create_window("AnyGrasp Realtime Demo")
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    first_frame = True

    try:
        while True:
            # 独立地从两个管道尝试获取帧
            imu_frames = pipe_imu.wait_for_frames(10000)
            print("等待新帧...")
            cam_frames = pipe_cam.wait_for_frames()
            print("获取到相机帧。")
            

            if not cam_frames:
                vis.poll_events()
                vis.update_renderer()
                continue
            
            # 如果有新的IMU数据，则更新重力向量
            if imu_frames:
                accel_frame = imu_frames.first_or_default(rs.stream.accel)
                if accel_frame:
                    accel_data = accel_frame.as_motion_frame().get_motion_data()
                    gravity_vec = np.array([accel_data.x, accel_data.y, accel_data.z])
                    if np.linalg.norm(gravity_vec) > 0:
                        vec = gravity_vec / np.linalg.norm(gravity_vec)
                        if vec[1] > 0: vec = -vec
                        vertical_vector = vec

            # --- 使用对齐后的帧 ---
            aligned_frames = align.process(cam_frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            #... [点云生成和后续处理逻辑保持不变]...
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            color_image_rgb = color_image[:, :, ::-1]
            
            # 注意：现在点云是基于对齐后的深度图生成的，与彩色图完美对应
            points_z = depth_image * scale
            xmap, ymap = np.arange(depth_image.shape[1]), np.arange(depth_image.shape)
            xmap, ymap = np.meshgrid(xmap, ymap)
            points_x = (xmap - cx) / fx * points_z
            points_y = (ymap - cy) / fy * points_z
            
            mask = (points_z > 0.1) & (points_z < 1.0)
            points = np.stack([points_x, points_y, points_z], axis=-1)[mask].astype(np.float32)
            colors = (color_image_rgb / 255.0)[mask].astype(np.float32)

            if len(points) < 500: continue

            gg, cloud = anygrasp.get_grasp(points, colors, lims=lims, apply_object_mask=True, dense_grasp=False, collision_detection=True)

            #... [筛选和可视化逻辑保持不变]...
            vis.clear_geometries()
            vis.add_geometry(axis_pcd)
            vis.add_geometry(cloud)

            if len(gg) > 0:
                horizontal_grasps = GraspGroup()
                for grasp in gg:
                    approach_vector = grasp.rotation_matrix[:, 2]
                    dot_product = np.abs(np.dot(approach_vector, vertical_vector))
                    if dot_product < 0.26:
                        horizontal_grasps.add(grasp)
                
                print(f"检测到 {len(gg)} 个总抓取, 其中 {len(horizontal_grasps)} 个是水平的。")

                gg_to_use = horizontal_grasps if len(horizontal_grasps) > 0 else gg
                gg = gg_to_use.nms().sort_by_score()
                
                if len(gg) > 0 and cfgs.debug:
                    grippers = gg[0:5].to_open3d_geometry_list()
                    for gripper in grippers:
                        vis.add_geometry(gripper)
            else:
                print('未检测到抓取!')

            vis.poll_events()
            vis.update_renderer()

            if first_frame:
                vis.reset_view_point(True)
                first_frame = False

    finally:
        # 停止两个管道
        pipe_cam.stop()
        pipe_imu.stop()
        vis.destroy_window()

if __name__ == '__main__':
    run_realtime_grasping(cfgs)