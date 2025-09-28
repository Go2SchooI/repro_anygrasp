import os, json, argparse 
import numpy as np 
from PIL import Image

def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--data_dir', required=True) 
    args = parser.parse_args()

    color_path = os.path.join(args.data_dir, 'color_1.png')
    depth_path = os.path.join(args.data_dir, 'depth_1.png')
    intr_path  = os.path.join(args.data_dir, 'intrinsics.json')

    with open(intr_path, 'r') as f:
        intr = json.load(f)
    fx, fy, cx, cy = intr['fx'], intr['fy'], intr['cx'], intr['cy']
    depth_scale = intr['depth_scale']

    colors = np.array(Image.open(color_path), dtype=np.float32) / 255.0  # HxWx3 RGB
    depths = np.array(Image.open(depth_path))  
    print(f"Depth shape: {depths.shape}")
    print(f"Color shape: {colors.shape}")                          # 16UC1 mm
    print(f"Depth data range: min={depths.min()}, max={depths.max()}")

    points_z = depths * depth_scale  # 转米
    h, w = depths.shape
    xmap = np.arange(w); ymap = np.arange(h)
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    mask = (points_z > 0) & (points_z < 3.5)  # 简单阈值
    pts = np.stack([points_x, points_y, points_z], axis=-1)[mask].astype(np.float32)
    if pts.size == 0:
        print("No valid points found for grasp generation.")
    else:
        print(f"Valid points found: {pts.shape[0]}")

    grasps = []
    if pts.size > 0:
        center = pts.mean(axis=0)              # 用点云中心当一个假抓取位置
        quat = [0.0, 0.0, 0.0, 1.0]            # 无旋转的单位四元数
        grasps.append({'translation': center.tolist(), 'quaternion': quat, 'score': 1.0, 'width': 0.05})

    out_path = os.path.join(args.data_dir, 'grasps.json')
    with open(out_path, 'w') as f:
        json.dump({'grasps': grasps}, f)

if __name__ == '__main__': 
    print("Starting anygrasp_wrapper.py")
    main()

