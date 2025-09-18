# inference_from_video.py
import os
import argparse
import torch
import numpy as np
import open3d as o3d
from PIL import Image
import time
import glob

from gsnet import AnyGrasp
from graspnetAPI import GraspGroup

FRAME_RATE = 15

# --- Setup ---
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--max_gripper_width', type=float, default=0.1, help='Maximum gripper width (<=0.1m)')
parser.add_argument('--gripper_height', type=float, default=0.03, help='Gripper height')
parser.add_argument('--top_down_grasp', action='store_true', help='Output top-down grasps.')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
# Add other AnyGrasp cfgs here if needed
cfgs = parser.parse_args()

def run_inference(data_dir):
    anygrasp = AnyGrasp(cfgs)
    anygrasp.load_net()

    # --- Get Camera Intrinsics and Workspace ---
    # IMPORTANT: Use the exact values from your camera!
    fx, fy = 388.16845703125, 388.16845703125
    cx, cy = 325.3074645996094, 234.87106323242188
    depth_scale = 0.0010000000474974513
    lims = [-0.2, 0.2, -0.2, 0.2, 0.3, 0.7] # Tune this box for your scene

    # --- Find all saved color images and sort them ---
    color_files = sorted(glob.glob(os.path.join(data_dir, 'color', '*.png')))
    depth_files = sorted(glob.glob(os.path.join(data_dir, 'depth', '*.png')))
    
    if len(color_files) == 0:
        print(f"Error: No images found in {os.path.join(data_dir, 'color')}")
        return

    # --- Setup Non-Blocking Visualizer ---
    vis = o3d.visualization.Visualizer()
    vis.create_window("Inference from Video")
    first_frame = True

    # --- Main Inference Loop ---
    for i in range(len(color_files)):
        color_path = color_files[i]
        depth_path = depth_files[i]

        print(f"Processing frame {i+1}/{len(color_files)}: {os.path.basename(color_path)}")

        colors_pil = Image.open(color_path)
        depths_pil = Image.open(depth_path)
        
        colors = np.array(colors_pil, dtype=np.float32) / 255.0
        depths = np.array(depths_pil)

        # --- Point Cloud Generation ---
        xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
        xmap, ymap = np.meshgrid(xmap, ymap)
        points_z = depths * depth_scale
        points_x = (xmap - cx) / fx * points_z
        points_y = (ymap - cy) / fy * points_z
        
        mask = (points_z > lims[4]) & (points_z < lims[5])
        points = np.stack([points_x, points_y, points_z], axis=-1)
        points = points[mask].astype(np.float32)
        colors = colors[mask].astype(np.float32)

        # --- Run Grasp Detection ---
        gg, cloud = anygrasp.get_grasp(points, colors, lims=lims, apply_object_mask=True, dense_grasp=False, collision_detection=True)

        if len(gg) == 0:
            print('No Grasp detected after collision detection!')

        # --- Update Visualization ---
        vis.clear_geometries()
        vis.add_geometry(cloud)

        print(f"Initially detected {len(gg)} grasps.")
    
        # 在相机坐标系中，通常Y轴是垂直方向[0, -1, 0]
        vertical_vector = np.array([0, -1, 0]) 
        
        # 新的GraspGroup存储筛选后的水平抓取
        horizontal_grasps = GraspGroup()
        
        for grasp in gg:
            # 抓取的接近方向是其旋转矩阵的第三列
            approach_vector = grasp.rotation_matrix[:, 2]

            # 如果接近方向与地面平行，那么它应该与垂直方向成90度角，点积接近0
            dot_product = np.abs(np.dot(approach_vector, vertical_vector))
            
            if dot_product < 0.26:
                horizontal_grasps.add(grasp)
                
        print(f"Found {len(horizontal_grasps)} grasps parallel to the ground.")
        
        # --- 使用筛选后的抓取进行后续处理 ---
        if len(horizontal_grasps) == 0:
            print('No horizontal grasps detected after filtering!')
            # 使用原始的gg
            gg_to_use = gg 
        else:
            gg_to_use = horizontal_grasps

        gg = gg_to_use.nms().sort_by_score()
        
        if len(gg) == 0:
            print('No Grasp detected after NMS and sorting!')

        if gg is not None and len(gg) > 0:
            gg_pick = gg[0:5] # Show only the best grasp
            print('grasp score:', gg_pick[0].score)
            
            grippers = gg_pick.to_open3d_geometry_list() # Red
            for gripper in grippers:
                vis.add_geometry(gripper)
        
        if first_frame:
            vis.reset_view_point(True)
            first_frame = False
            
        vis.poll_events()
        vis.update_renderer()
        time.sleep(1/FRAME_RATE) # Control playback speed

    print("Inference finished.")
    vis.destroy_window()

if __name__ == '__main__':
    run_inference('./realsense_capture/video')