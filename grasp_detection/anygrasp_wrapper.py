import os
import json
import argparse
import numpy as np
from PIL import Image, ImageDraw
from gsnet import AnyGrasp
from graspnetAPI import GraspGroup

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
    parser.add_argument('--max_gripper_width', type=float, default=0.1)
    parser.add_argument('--gripper_height', type=float, default=0.03)
    parser.add_argument('--top_down_grasp', action='store_true')
    parser.add_argument('--collision_detection', action='store_true')
    parser.add_argument('--workspace', nargs=6, type=float,
                        help='xmin xmax ymin ymax zmin zmax')
    
    args = parser.parse_args()
    if args.max_gripper_width is not None:
        args.max_gripper_width = max(0, min(0.1, args.max_gripper_width))

    color_path = os.path.join(args.data_dir, 'color_1.png')
    depth_path = os.path.join(args.data_dir, 'depth_1.png')
    intr_path  = os.path.join(args.data_dir, 'intrinsics.json')

    with open(intr_path, 'r') as f:
        intr = json.load(f)
    fx, fy = intr['fx'], intr['fy']
    cx, cy = intr['cx'], intr['cy']
    depth_scale = intr.get('depth_scale', 0.001)

    colors = np.array(Image.open(color_path), dtype=np.float32) / 255.0  # HxWx3
    depths = np.array(Image.open(depth_path))

    # 构建点云
    h, w = depths.shape
    xmap, ymap = np.meshgrid(np.arange(w), np.arange(h))
    points_z = depths * depth_scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    mask = (points_z > 0) & (points_z < np.inf)
    if args.workspace:
        xmin, xmax, ymin, ymax, zmin, zmax = args.workspace
        mask = mask & \
               (points_x >= xmin) & (points_x <= xmax) & \
               (points_y >= ymin) & (points_y <= ymax) & \
               (points_z >= zmin) & (points_z <= zmax)

    pts = np.stack([points_x, points_y, points_z], axis=-1)[mask].astype(np.float32)
    cols = colors[mask].astype(np.float32)

    if pts.shape[0] == 0:
        print("No valid points after mask / workspace cropping")
        out = {'grasps': []}
        write_output(args.data_dir, out, debug_img=None)
        return

    # init anygrasp
    anygrasp = AnyGrasp(args)
    anygrasp.load_net()

    gg, cloud = anygrasp.get_grasp(pts, cols,
                                   lims=args.workspace if args.workspace else None,
                                   apply_object_mask=True,
                                   dense_grasp=False,
                                   collision_detection=args.collision_detection)
    print(f"Initially detected {len(gg)} grasps.")

    if len(gg) == 0:
        print("No grasp found")
        out = {'grasps': []}
        write_output(args.data_dir, out, debug_img=None)
        return

    gg = gg.nms().sort_by_score()
    best = gg[0]

    # 输出抓取列表
    grasps = []
    for g in gg[:10]:
        t = g.translation.tolist()
        # 从 g.rotation_matrix 或 g.quaternion 获取四元数（graspnetAPI 通常有方法）
        # 假设它有 `quaternion` 属性
        def rot_to_quat_wxyz(R):
            m00,m01,m02 = R[0,0],R[0,1],R[0,2]
            m10,m11,m12 = R[1,0],R[1,1],R[1,2]
            m20,m21,m22 = R[2,0],R[2,1],R[2,2]
            tr = m00+m11+m22
            if tr > 0:
                S = np.sqrt(tr+1.0)*2.0
                w = 0.25*S; x = (m21 - m12)/S; y = (m02 - m20)/S; z = (m10 - m01)/S
            elif (m00 > m11) and (m00 > m22):
                S = np.sqrt(1.0 + m00 - m11 - m22)*2.0
                w = (m21 - m12)/S; x = 0.25*S; y = (m01 + m10)/S; z = (m02 + m20)/S
            elif m11 > m22:
                S = np.sqrt(1.0 + m11 - m00 - m22)*2.0
                w = (m02 - m20)/S; x = (m01 + m10)/S; y = 0.25*S; z = (m12 + m21)/S
            else:
                S = np.sqrt(1.0 + m22 - m00 - m11)*2.0
                w = (m10 - m01)/S; x = (m02 + m20)/S; y = (m12 + m21)/S; z = 0.25*S
            q = np.array([w,x,y,z], dtype=np.float32)
            q /= max(1e-12, np.linalg.norm(q))
            return q

        R = np.asarray(g.rotation_matrix, dtype=np.float32)
        q = rot_to_quat_wxyz(R)  # ← 仍然写回同一个字段名 'quaternion'

        grasps.append({
            'translation': t,
            'quaternion': q.tolist(),  # ← 还是 'quaternion'，但顺序是 wxyz
            'score': float(g.score),
            'width': float(g.width)
        })

    out = {'grasps': grasps}

    debug_img = None
    if args.debug:
        # 在 color 图上画夹爪投影
        debug_img = draw_grasp_on_image(colors, pts, gg, fx, fy, cx, cy, depth_scale)

    write_output(args.data_dir, out, debug_img)

def draw_grasp_on_image(colors, pts, gg, fx, fy, cx, cy, depth_scale, top_k=1, line_px=2,
                        finger_len_m=0.04,  # 指爪在图上显示的“长度”（米，沿 binormal）
                        center_mark_px=4):
    """
    将 3D 抓取姿态投影到 RGB 图上，画出“官方 demo 风格”的两指夹爪：
    - 以 grasp 的局部坐标系：假设 R 的列向量分别是 (x=closing axis, y=binormal, z=approach)。
    - 在中心 t 上，沿 x 方向偏移 ±width/2 得到两指中心，再沿 y 方向画一条长度 finger_len_m 的线段。
    - 若方向和你观察到的 demo 有镜像，可把某个轴取反（例如把 y 改成 -y）。
    参考：anygrasp_sdk demo 里用 open3d 画 gripper（to_open3d_geometry_list），并做了 z 翻转以显示。 
    """

    import numpy as np
    from PIL import Image, ImageDraw

    h, w, _ = colors.shape
    img8 = (np.clip(colors * 255.0, 0, 255)).astype(np.uint8)
    pil = Image.fromarray(img8)
    draw = ImageDraw.Draw(pil)

    def proj(P):
        """3D 点（米，camera系）投影到像素（u,v）。"""
        X, Y, Z = P
        if Z <= 1e-6:
            return None
        u = fx * (X / Z) + cx
        v = fy * (Y / Z) + cy
        return (float(u), float(v))

    # 只画 top-k，避免太乱
    gg_viz = gg.nms().sort_by_score()[:top_k]

    for g in gg_viz:
        t = np.array(g.translation, dtype=np.float32)        # [3], 中心（m）
        R = np.array(g.rotation_matrix, dtype=np.float32)     # [3,3]
        width = float(g.width)                                # (m)

        # 约定：R 的三列分别是 x(夹爪闭合轴)、y(夹爪 binormal)、z(approach)
        x_axis = R[:, 0]
        y_axis = R[:, 1]
        z_axis = R[:, 2]

        # 两个指爪中心（从 grasp 中心沿 x 轴 ±width/2 偏移）
        half_w = 0.5 * width
        left_center  = t - half_w * x_axis
        right_center = t + half_w * x_axis

        # 指爪在线上显示一段长度 finger_len_m（沿 y 轴方向），画成线段
        half_len = 0.5 * finger_len_m
        left_a  = left_center  - half_len * y_axis
        left_b  = left_center  + half_len * y_axis
        right_a = right_center - half_len * y_axis
        right_b = right_center + half_len * y_axis

        # 同时在中心画一个小十字，便于对准
        c_u_v = proj(t)
        la = proj(left_a); lb = proj(left_b)
        ra = proj(right_a); rb = proj(right_b)

        # 如有投影失败（Z<=0），跳过
        if None in (c_u_v, la, lb, ra, rb):
            continue

        # 画两条指爪（两条线段）
        draw.line([la, lb], fill=(0,255,0), width=line_px)
        draw.line([ra, rb], fill=(0,255,0), width=line_px)

        # 画中心十字
        u, v = c_u_v
        draw.line((u - center_mark_px, v, u + center_mark_px, v), fill=(255,0,0), width=1)
        draw.line((u, v - center_mark_px, u, v + center_mark_px), fill=(255,0,0), width=1)

        # 可选：画出“闭合方向”连线（两指中心连线，表示夹爪开口）
        lcen = proj(left_center); rcen = proj(right_center)
        if None not in (lcen, rcen):
            draw.line([lcen, rcen], fill=(0,200,255), width=1)

    return pil


def write_output(data_dir, out_dict, debug_img):
    # 写 grasps.json
    path = os.path.join(data_dir, 'grasps.json')
    with open(path, 'w') as f:
        json.dump(out_dict, f)

    if debug_img is not None:
        debug_path = os.path.join(data_dir, 'overlay.png')
        debug_img.save(debug_path)

if __name__ == '__main__':
    main()
