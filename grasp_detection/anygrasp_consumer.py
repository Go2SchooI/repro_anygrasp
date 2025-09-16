import argparse
import numpy as np
import open3d as o3d
import cv2
from multiprocessing import shared_memory
import time
import atexit

from gsnet import AnyGrasp
from graspnetAPI import GraspGroup

# --- 1. AnyGrasp and Argument Setup ---
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--max_gripper_width', type=float, default=0.1, help='Maximum gripper width (<=0.1m)')
parser.add_argument('--gripper_height', type=float, default=0.03, help='Gripper height')
parser.add_argument('--top_down_grasp', action='store_true', help='Output top-down grasps.')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
cfgs = parser.parse_args()
anygrasp = AnyGrasp(cfgs)
anygrasp.load_net()
print('AnyGrasp model loaded.')

# --- 2. Shared Memory Setup ---
# Note: These shapes and dtypes MUST match the producer's exactly
color_shape = (480, 640, 3)
depth_shape = (480, 640)
color_dtype = np.uint8
depth_dtype = np.uint16

# Attach to existing shared memory blocks
try:
    shm_color = shared_memory.SharedMemory(name='realsense_color')
    shm_depth = shared_memory.SharedMemory(name='realsense_depth')
    print("Attached to shared memory blocks.")
except FileNotFoundError:
    print("Error: Shared memory not found. Is the producer script running?")
    exit()

# Create NumPy arrays that use the shared memory buffers
color_image = np.ndarray(color_shape, dtype=color_dtype, buffer=shm_color.buf)
depth_image = np.ndarray(depth_shape, dtype=depth_dtype, buffer=shm_depth.buf)

# --- 3. Camera Intrinsics and Workspace ---
# These must match the camera used in the producer
fx, fy = 388.16845703125, 388.16845703125
cx, cy = 325.3074645996094, 234.87106323242188
depth_scale = 0.0010000000474974513

lims = [-0.3, 0.3, -0.3, 0.3, 0.15, 0.75] # Workspace [xmin, xmax, ymin, ymax, zmin, zmax]

# --- 4. Visualization Setup ---
vis = o3d.visualization.Visualizer()
vis.create_window("AnyGrasp Consumer")
axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
first_frame = True

def cleanup():
    print("Closing consumer.")
    shm_color.close()
    shm_depth.close()
    vis.destroy_window()
atexit.register(cleanup)

# --- 5. Main Processing and Visualization Loop ---
print("Consumer is running... Press Ctrl+C in the producer's terminal to stop.")
while True:
    # --- Create Point Cloud from Shared Memory Data ---
    colors_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    
    xmap, ymap = np.arange(depth_image.shape[1]), np.arange(depth_image.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth_image * depth_scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z
    
    mask = (points_z > 0.15) # Basic filtering
    points = np.stack([points_x, points_y, points_z], axis=-1)
    points = points[mask].astype(np.float32)
    colors_normalized = colors_rgb[mask].astype(np.float32) / 255.0

    # --- Run Grasp Detection ---
    gg, cloud = anygrasp.get_grasp(points, colors_normalized, lims=lims, apply_object_mask=True, dense_grasp=False, collision_detection=True)

    # --- **CORRECTED VISUALIZATION LOGIC** ---
    vis.clear_geometries()
    vis.add_geometry(axis_pcd)
    vis.add_geometry(cloud) # Always add the point cloud
    
    if len(gg) == 0:
        print('No Grasp detected after collision detection!')

    # Only add grippers if grasps were found
    if len(gg) > 0:
        best_grasp = gg[0]

        # Extract and print its details
        print("\n--- Details of the Best Grasp ---")
                # 3. Grasp Pose: Position (Translation)
        position = best_grasp.translation
        print(f"Position (x, y, z): {position}")

        # 4. Grasp Pose: Orientation (Rotation Matrix)
        orientation = best_grasp.rotation_matrix
        print(f"Orientation (Rotation Matrix):\n{orientation}")
        print("------------------------------------")

        gg = gg.nms().sort_by_score()
        gg_pick = gg[0:5]
        grippers = gg_pick.to_open3d_geometry_list() 
        for gripper in grippers:
            vis.add_geometry(gripper)
    
    # --- Update Window ---
    if first_frame:
        vis.reset_view_point(True)
        first_frame = False
        
    vis.poll_events()
    vis.update_renderer()