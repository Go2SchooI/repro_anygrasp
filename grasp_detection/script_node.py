import json
import math
import numpy as np
import omni.usd
from pxr import Usd, UsdGeom

# --- helpers ---
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
    return [qx,qy,qz,qw]

def pose_to_mat(tx,ty,tz,qx,qy,qz,qw):
    M = np.eye(4, dtype=np.float64)
    M[:3,:3] = quat_xyzw_to_rot(qx,qy,qz,qw)
    M[:3, 3] = [tx,ty,tz]
    return M

def decompose_mat(M):
    R = M[:3,:3]
    t = M[:3, 3]
    qx,qy,qz,qw = mat_to_quat_xyzw(R)
    return [float(t[0]), float(t[1]), float(t[2]), float(qx), float(qy), float(qz), float(qw)]

def world_matrix_of(prim_path):
    # 优先用 Kit 提供的便捷函数；失败则用 Xformable 计算
    try:
        import omni.usd
        gfM = omni.usd.get_world_transform_matrix(prim_path)  # Gf.Matrix4d
        return np.array(gfM, dtype=np.float64)
    except Exception:
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"Prim not found: {prim_path}")
        gfM = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        return np.array(gfM, dtype=np.float64)

def compute(db):
    # 读取输入
    toks = db.inputs.poses            # token[]
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
    except Exception as e:
        db.log_error(f"[PickPose] json parse failed: {e}")
        db.outputs.has_pose = False
        return True

    pos = p0.get("position", {})
    ori = p0.get("orientation", {})
    tx,ty,tz = float(pos.get("x",0.0)), float(pos.get("y",0.0)), float(pos.get("z",0.0))
    qx,qy,qz,qw = float(ori.get("x",0.0)), float(ori.get("y",0.0)), float(ori.get("z",0.0)), float(ori.get("w",1.0))
    T_cam_obj = pose_to_mat(tx,ty,tz,qx,qy,qz,qw)

    # 取世界矩阵
    try:
        T_world_cam  = world_matrix_of(cam_path)
        T_world_base = world_matrix_of(base_path)
    except Exception as e:
        db.log_error(f"[PickPose] get world matrices failed: {e}")
        db.outputs.has_pose = False
        return True
    
    # 调试：打印并比较世界矩阵与基座-相机相对位姿
    try:
        import numpy as np, omni.usd
        T_wc = world_matrix_of(str(db.inputs.cam_prim))
        T_wb = world_matrix_of(str(db.inputs.base_prim))
        T_bc = np.linalg.inv(T_wb) @ T_wc
        db.log_error(f"[PickPose] ||T_world_cam - T_world_base||_F = {np.linalg.norm(T_wc - T_wb):.6f}")
        db.log_error(f"[PickPose] ||T_base_cam - I||_F = {np.linalg.norm(T_bc - np.eye(4)):.6f}")
    except Exception as e:
        db.log_error(f"[PickPose] debug delta failed: {e}")

    try:
        T_base_obj = np.linalg.inv(T_world_base) @ T_world_cam @ T_cam_obj
    except Exception as e:
        db.log_error(f"[PickPose] compose failed: {e}")
        db.outputs.has_pose = False
        return True

    out = decompose_mat(T_base_obj)
    db.outputs.grasp_pose_base = out   # float[]: [x,y,z,qx,qy,qz,qw] in base frame
    db.outputs.has_pose = True
    return True
