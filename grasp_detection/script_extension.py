# === User config ===
ARM_PRIM_PATH = "/World/Franka/panda_link0"
EE_FRAME_NAME = "right_gripper"
OG_GOAL_ATTR  = "/ActionGraph/PickPose.outputs:grasp_pose_base"  # [x,y,z,w,x,y,z] (世界系 / base系按你脚本产出)

# 放置位姿（世界系，给一个简单位置与朝向）
PLACE_POS_W      = (0.3, 0.50, 0.25)     # 你可按需修改
PLACE_QUAT_WXYZ  = (0.7071, 0.0, 0.7071, 0.0)    # 无旋转

# 到达判定阈值
POS_TOL = 0.010      # 位置 1 cm
ANG_TOL_DEG = 7.0    # 姿态 7°

# Franka 两根手指的 DOF 索引（默认 panda_finger_joint1/2 在 7,8）
FRANKA_FINGER_IDXS = (7, 8)  # 若你的关节顺序不同，请改成实际索引


# === Imports ===
import numpy as np
import time
import carb, omni
import omni.graph.core as og
from pxr import Usd

from omni.isaac.core.articulations import Articulation
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver, ArticulationKinematicsSolver
from isaacsim.robot_motion.motion_generation import interface_config_loader

# Isaac Sim 5.0: ArticulationAction 的官方导入位置
from isaacsim.core.utils.types import ArticulationAction  # 5.0 文档示例路径

# === Globals ===
_initialized = False
_after_play_frames = 0
_arm = _aks = _lks = None

# FSM
STATE_OPEN = 0
STATE_MOVE_TO_GRASP = 1
STATE_CLOSE = 2
STATE_MOVE_TO_PLACE = 3
STATE_OPEN_AT_PLACE = 4
STATE_DONE = 5
_state = STATE_OPEN

# === Helpers ===
def _get_grasp_from_graph(attr_path: str):
    try:
        attr = og.Controller.attribute(attr_path)
        if not attr:
            return None
        val = attr.get()
        if val is None:
            return None
        arr = np.array(val, dtype=float).reshape(-1)
        return arr if arr.size == 7 else None
    except Exception as e:
        carb.log_error(f"[LulaIK] read graph attr failed: {e}")
        return None

def _prim_is_valid(path: str) -> bool:
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(path)
    return bool(prim and prim.IsValid())

def _rotmat_to_quat_wxyz(R: np.ndarray):
    m00,m01,m02 = R[0,0],R[0,1],R[0,2]
    m10,m11,m12 = R[1,0],R[1,1],R[1,2]
    m20,m21,m22 = R[2,0],R[2,1],R[2,2]
    tr = m00+m11+m22
    if tr > 0.0:
        S = np.sqrt(tr+1.0)*2.0
        w = 0.25*S
        x = (m21 - m12)/S
        y = (m02 - m20)/S
        z = (m10 - m01)/S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22)*2.0
        w = (m21 - m12)/S; x = 0.25*S; y = (m01 + m10)/S; z = (m02 + m20)/S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22)*2.0
        w = (m02 - m20)/S; x = (m01 + m10)/S; y = 0.25*S; z = (m12 + m21)/S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11)*2.0
        w = (m10 - m01)/S; x = (m02 + m20)/S; y = (m12 + m21)/S; z = 0.25*S
    return np.array([w,x,y,z], dtype=np.float64)

def _ee_pose_world():
    # 5.0 推荐：用 AKS 做 FK，直接拿末端世界位姿（pos, R）
    ee_pos, ee_R = _aks.compute_end_effector_pose()
    ee_q = _rotmat_to_quat_wxyz(ee_R)
    return ee_pos, ee_q

def _ang_err_deg(q_curr_wxyz, q_tgt_wxyz):
    a = q_curr_wxyz / (np.linalg.norm(q_curr_wxyz) + 1e-12)
    b = q_tgt_wxyz  / (np.linalg.norm(q_tgt_wxyz)  + 1e-12)
    dot = float(np.clip(np.abs(np.dot(a, b)), -1.0, 1.0))
    return np.degrees(2.0 * np.arccos(dot))

def _ee_reached(target_pos_w, target_q_wxyz, pos_tol=POS_TOL, ang_tol_deg=ANG_TOL_DEG):
    cur_p, cur_q = _ee_pose_world()
    pe = float(np.linalg.norm(cur_p - target_pos_w))
    ae = float(_ang_err_deg(cur_q, target_q_wxyz))
    return (pe <= pos_tol and ae <= ang_tol_deg), pe, ae

def _open_gripper():
    targets = np.array([0.04, 0.04], dtype=np.float32)  # Franka: 0.00(合)~0.04(开)
    action = ArticulationAction(joint_positions=targets, joint_indices=np.array(FRANKA_FINGER_IDXS, dtype=np.int32))
    _arm.apply_action(action)

def _close_gripper():
    targets = np.array([0.00, 0.00], dtype=np.float32)
    action = ArticulationAction(joint_positions=targets, joint_indices=np.array(FRANKA_FINGER_IDXS, dtype=np.int32))
    _arm.apply_action(action)


# === Init ===
def _init_if_ready():
    global _initialized, _arm, _aks, _lks, _after_play_frames
    if _initialized:
        return True

    tl = omni.timeline.get_timeline_interface()
    if not tl.is_playing():
        return False

    _after_play_frames += 1
    if _after_play_frames < 2:
        return False

    if not _prim_is_valid(ARM_PRIM_PATH):
        carb.log_error(f"[LulaIK] prim not found: {ARM_PRIM_PATH}")
        return False

    try:
        # 1) 机器人实例
        _arm = Articulation(prim_path=ARM_PRIM_PATH)
        _arm.initialize()

        # 2) 加载内置 Lula 配置 & 创建求解器
        cfg = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
        _lks = LulaKinematicsSolver(**cfg)

        # 3) 将 Articulation 与 Lula 绑定到 AKS（便于 IK/FK）
        _aks = ArticulationKinematicsSolver(_arm, _lks, EE_FRAME_NAME)

        carb.log_info("[LulaIK] IK/FK solvers initialized.")
        _initialized = True
        return True
    except Exception as e:
        carb.log_error(f"[LulaIK] init failed (will retry): {e}")
        return False


# === Per-frame update ===
def _on_update(dt: float):
    global _state, close_time

    if not _init_if_ready():
        return

    goal = _get_grasp_from_graph(OG_GOAL_ATTR)
    if goal is None:
        return

    # 解析目标抓取位姿（世界/基座取决于你 Action Graph 出来的语义，下面假定世界系）
    tgt_pos = np.asarray(goal[:3], dtype=np.float64)
    tgt_q   = np.asarray(goal[3:], dtype=np.float64)  # wxyz

    # 同步基座位姿给 Lula（5.0 官方建议做法）
    base_p_w, base_q_w = _arm.get_world_pose()
    _lks.set_robot_base_pose(base_p_w, base_q_w)

    # FSM
    if _state == STATE_OPEN:
        _open_gripper()
        _state = STATE_MOVE_TO_GRASP

    elif _state == STATE_MOVE_TO_GRASP:
        try:
            ik_action, success = _aks.compute_inverse_kinematics(tgt_pos, tgt_q)
        except Exception as e:
            carb.log_error(f"[LulaIK] IK compute failed: {e}")
            ik_action, success = None, False

        if success and ik_action is not None:
            _arm.apply_action(ik_action)

            reached, pe, ae = _ee_reached(tgt_pos, tgt_q)
            if reached:
                close_time = time.time()
                _state = STATE_CLOSE

    elif _state == STATE_CLOSE:
        _close_gripper()

        if close_time is not None and (time.time() - close_time) > 1.0:
            _state = STATE_MOVE_TO_PLACE

    elif _state == STATE_MOVE_TO_PLACE:
        place_pos = np.asarray(PLACE_POS_W, dtype=np.float64)
        place_q   = np.asarray(PLACE_QUAT_WXYZ, dtype=np.float64)
        try:
            ik_action, success = _aks.compute_inverse_kinematics(place_pos, place_q)
        except Exception as e:
            carb.log_error(f"[LulaIK] IK compute (place) failed: {e}")
            ik_action, success = None, False

        if success and ik_action is not None:
            _arm.apply_action(ik_action)

            reached, pe, ae = _ee_reached(place_pos, place_q)
            if reached:
                _state = STATE_OPEN_AT_PLACE

    elif _state == STATE_OPEN_AT_PLACE:
        _open_gripper()
        _state = STATE_DONE

    elif _state == STATE_DONE:
        pass


# 注册 per-frame 回调 → 先 Run，再点 ▶Play
app = omni.kit.app.get_app()
_subscription = app.get_update_event_stream().create_subscription_to_pop(_on_update)
carb.log_info("[LulaIK] Registered per-frame callback. Now press ▶ Play.")
