import os
import argparse
import torch
import numpy as np
import open3d as o3d
from PIL import Image

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

def demo(data_dir):
    anygrasp = AnyGrasp(cfgs)
    anygrasp.load_net()

    # get data
    # colors = np.array(Image.open(os.path.join(data_dir, 'rpg', '0018.png')), dtype=np.float32) / 255.0
    # depths = np.array(Image.open(os.path.join(data_dir, 'depth', '0018.png')))
    colors = np.array(Image.open(os.path.join(data_dir, 'color_1.png')), dtype=np.float32) / 255.0
    depths = np.array(Image.open(os.path.join(data_dir, 'depth_1.png')))
    # get camera intrinsics
    fx, fy = 388.16845703125, 388.16845703125  # Example values, use your camera's
    cx, cy = 325.3074645996094, 234.87106323242188  # Example values, use your camera's
    depth_scale = 0.0010000000474974513
    # set workspace to filter output grasps
    xmin, xmax = -0.19, 0.12
    ymin, ymax = 0.02, 0.15
    zmin, zmax = 0.0, 1.0
    lims = [xmin, xmax, ymin, ymax, zmin, zmax]

    # get point cloud
    xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depths * depth_scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    # set your workspace to crop point cloud
    mask = (points_z > 0) & (points_z < 1)
    points = np.stack([points_x, points_y, points_z], axis=-1)
    points = points[mask].astype(np.float32)
    colors = colors[mask].astype(np.float32)
    print(points.min(axis=0), points.max(axis=0))

    gg, cloud = anygrasp.get_grasp(points, colors, lims=lims, apply_object_mask=True, dense_grasp=False, collision_detection=True)

    if len(gg) == 0:
        print('No Grasp detected after collision detection!')

    gg = gg.nms().sort_by_score()

    if len(gg) > 0:
        # Get the single best grasp
        best_grasp = gg[0]

        # Extract and print its details
        print("\n--- Details of the Best Grasp ---")
        
        # 1. Grasp Score
        score = best_grasp.score
        print(f"Score: {score:.4f}")

        # 2. Gripper Width
        width_m = best_grasp.width
        print(f"Gripper Width: {width_m*1000:.2f} mm")

        # 3. Grasp Pose: Position (Translation)
        position = best_grasp.translation
        print(f"Position (x, y, z): {position}")

        # 4. Grasp Pose: Orientation (Rotation Matrix)
        orientation = best_grasp.rotation_matrix
        print(f"Orientation (Rotation Matrix):\n{orientation}")
        print("------------------------------------")

    gg_pick = gg[0:20]
    print(gg_pick.scores)
    print('grasp score:', gg_pick[0].score)

    # visualization
    if cfgs.debug:
        trans_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
        cloud.transform(trans_mat)
        grippers = gg.to_open3d_geometry_list()
        for gripper in grippers:
            gripper.transform(trans_mat)
        o3d.visualization.draw_geometries([*grippers, cloud])
        o3d.visualization.draw_geometries([grippers[0], cloud])


if __name__ == '__main__':
    
    # demo('./example_data/') 
    demo('./realsense_capture/')