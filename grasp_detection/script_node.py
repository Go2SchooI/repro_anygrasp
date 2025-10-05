import json
import math
import numpy as np
import omni.usd
from pxr import Usd, UsdGeom

# ====== 用户可切换的小开关 ======
USE_TFIX = True  # 先关闭。若确认相机系->Isaac需要180°@X，再改成 True

# ====== helpers（保持你原来的 xyzw 实现不变） ======
def quat_xyzw_to_rot(qx, qy, qz, qw):
    # 归一化
    n = math.sqrt(qx*qx+qy*qy+qz*qz+qw*qw) or 1.0
    x,y,z,w = qx/n, qy/n, qz/n, qw/n
    xx,yy,zz,ww = x*x,y*y,z*z,w*w
    xy,xz,yz = x*y,x*z,y*z
    wx,wy,wz = w*x,w*y,w*z
    R = np.array([
        [ww+xx-yy-zz,   2*(xy-wz),     2*(xz+wy)],
        [2*(xy+wz),     ww-xx+yy-zz,   2*(yz-wx)],
        [2*(xz-wy),     2*(yz+wx),     ww-xx-yy+zz]
    ], dtype=np.float64)
    return R

def mat_to_quat_xyzw(R):
    # 从旋转矩阵取四元数（xyzw）
    m00,m01,m02 = R[0,0],R[0,1],R[0,2]
    m10,m11,m12 = R[1,0],R[1,1],R[1,2]
    m20,m21,m22 = R[2,0],R[2,1],R[2,2]
    tr = m00+m11+m22
    if tr > 0.0:
        S = math.sqrt(tr+1.0)*2.0
        qw = 0.25*S
        qx = (m21 - m12)/S
        qy = (m02 - m20)/S
        qz = (m10 - m01)/S
    elif (m00 > m11) and (m00 > m22):
        S = math.sqrt(1.0 + m00 - m11 - m22)*2.0
        qw = (m21 - m12)/S
        qx = 0.25*S
        qy = (m01 + m10)/S
        qz = (m02 + m20)/S
    elif m11 > m22:
        S = math.sqrt(1.0 + m11 - m00 - m22)*2.0
        qw = (m02 - m20)/S
        qx = (m01 + m10)/S
        qy = 0.25*S
        qz = (m12 + m21)/S
    else:
        S = math.sqrt(1.0 + m22 - m00 - m11)*2.0
        qw = (m10 - m01)/S
        qx = (m02 + m20)/S
        qy = (m12 + m21)/S
        qz = 0.25*S
    # 返回 xyzw
    return [qx,qy,qz,qw]

def pose_to_mat(tx,ty,tz,qx,qy,qz,qw):
    M = np.eye(4, dtype=np.float64)
    M[:3,:3] = quat_xyzw_to_rot(qx,qy,qz,qw)
    M[:3, 3] = [tx,ty,tz]
    return M

def world_matrix_of(prim_path):
    # 优先用 Kit 提供的便捷函数；失败则用 Xformable 计算
    try:
        gfM = omni.usd.get_world_transform_matrix(prim_path)  # Gf.Matrix4d
        return np.array(gfM, dtype=np.float64)
    except Exception:
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"Prim not found: {prim_path}")
        gfM = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        return np.array(gfM, dtype=np.float64)

# ====== Script Node 主逻辑 ======
def compute(db):
    # 输入 pins
    toks = db.inputs.poses             # token[]，每个元素是一条 JSON 字符串
    cam_path  = str(db.inputs.cam_prim)   # token
    base_path = str(db.inputs.base_prim)  # token

    # 基本检查
    n = 0 if toks is None else len(toks)
    if n == 0 or not cam_path or not base_path:
        db.outputs.has_pose = False
        return True

    # 解析第一条抓取（JSON 字符串）
    first = toks[0]
    if not isinstance(first, str):
        first = str(first)
    try:
        p0 = json.loads(first)
        db.log_error(f"PickPose got first pose: {p0}")
    except Exception as e:
        db.log_error(f"[PickPose] json parse failed: {e}")
        db.outputs.has_pose = False
        return True

    # --- 兼容两种字段格式 ---
    # 位置：优先 translation[list]，否则 position{xyz}
    if "translation" in p0 and isinstance(p0["translation"], (list, tuple)) and len(p0["translation"]) >= 3:
        tx,ty,tz = map(float, p0["translation"][:3])
    else:
        pos = p0.get("position", {})
        tx = float(pos.get("x", 0.0)); ty = float(pos.get("y", 0.0)); tz = float(pos.get("z", 0.0))

    # 四元数：优先 quaternion[list=wxyz]，否则 orientation{wxyz}
    # 读入为 wxyz，然后转成 xyzw，供本文件的数学函数使用
    qx=qy=qz=0.0; qw=1.0
    if "quaternion" in p0 and isinstance(p0["quaternion"], (list, tuple)) and len(p0["quaternion"]) >= 4:
        w,x,y,z = map(float, p0["quaternion"][:4])  # 输入 wxyz
        qx,qy,qz,qw = x,y,z,w                        # 转 xyzw
    else:
        ori = p0.get("orientation", {})
        w = float(ori.get("w", 1.0))
        x = float(ori.get("x", 0.0))
        y = float(ori.get("y", 0.0))
        z = float(ori.get("z", 0.0))
        qx,qy,qz,qw = x,y,z,w  # 转 xyzw

    # 可选：把相机系位姿留作调试输出（xyzw）
    db.outputs.grasp_pose = [tx,ty,tz,qx,qy,qz,qw]

    # 相机系位姿 → 齐次矩阵
    T_cam_obj = pose_to_mat(tx,ty,tz,qx,qy,qz,qw)  # 这里吃 xyzw
    db.log_error(f"[cam_obj]\n{T_cam_obj}")

    # 取世界矩阵
    try:
        T_world_cam  = world_matrix_of(cam_path).T
        T_world_base = world_matrix_of(base_path).T
        db.log_error(f"[world_cam]\n{T_world_cam}")
        db.log_error(f"[world_base]\n{T_world_base}")
    except Exception as e:
        db.log_error(f"[PickPose] get world matrices failed: {e}")
        db.outputs.has_pose = False
        return True

    # 调试：相机与基座的相对位姿
    try:
        T_bc = np.linalg.inv(T_world_base) @ T_world_cam
        db.log_error(f"[PickPose] ||T_base_cam - I||_F = {np.linalg.norm(T_bc - np.eye(4)):.6f}")
    except Exception as e:
        db.log_error(f"[PickPose] debug delta failed: {e}")

    # 组合：base ← world ← cam ← (可选 T_fix) ← grasp_in_cam
    try:
        T_fix = np.eye(4, dtype=np.float64)
        if USE_TFIX:
            # USD camera(+Y up, -Z fwd) -> ROS optical(Z fwd, X right, Y down)
            # 如需修正，则相当于绕 X 轴 180 度：diag(1, -1, -1)
            R_isaac_to_cam = np.array([
                [0, -1,  0],   # X_isaac
                [0,  0, -1],   # Y_isaac
                [1,  0,  0],  # Z_isaac
            ], dtype=float)
            T_fix[1,1] = -1.0
            T_fix[2,2] = -1.0

        T_world_obj = T_world_cam @ T_fix.T @ T_cam_obj
        db.log_error(f"[world_obj]\n{T_world_obj}")
        T_base_obj = np.linalg.inv(T_world_base) @ T_world_cam @ T_fix @ T_cam_obj
    except Exception as e:
        db.log_error(f"[PickPose] compose failed: {e}")
        db.outputs.has_pose = False
        return True

    # 分解：输出为 wxyz（与下游 IK 的 wxyz 约定对齐）
    R = T_base_obj[:3,:3]
    t = T_base_obj[:3, 3]
    qx,qy,qz,qw = mat_to_quat_xyzw(R)  # 这里得到的是 xyzw
    wxyz = [float(qw), float(qx), float(qy), float(qz)]  # 转成 wxyz

    db.outputs.grasp_pose_base = [float(t[0]), float(t[1]), float(t[2]), *wxyz]  # [x,y,z,w,x,y,z]
    db.outputs.has_pose = True
    return True
