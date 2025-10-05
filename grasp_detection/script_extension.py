# === User config ===
ARM_PRIM_PATH = "/World/Franka/panda_link0"
EE_FRAME_NAME = "right_gripper"
OG_GOAL_ATTR  = "/ActionGraph/PickPose.outputs:grasp_pose_base"

# === Imports ===
import numpy as np
import carb, omni
import omni.graph.core as og
from pxr import Usd
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.controllers import ArticulationController
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver, ArticulationKinematicsSolver
from isaacsim.robot_motion.motion_generation import interface_config_loader

_initialized = False
_after_play_frames = 0
_arm = _aks = _ctrl = None
_lks = None  # ★ NEW: 把 Lula solver 存成全局，方便 set_robot_base_pose

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

def _init_if_ready():
    """等 Play 后第 2 帧再初始化（避免 physics view 尚未建好）"""
    global _initialized, _arm, _aks, _ctrl, _after_play_frames, _lks  # ★ NEW: _lks
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
        # 1) Articulation
        _arm = Articulation(prim_path=ARM_PRIM_PATH)
        _arm.initialize()

        # 2) 加载内置 Lula 配置
        cfg = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
        _lks = LulaKinematicsSolver(**cfg)  # ★ NEW: 存到全局 _lks

        carb.log_info(f"[LulaIK] Frames: {_lks.get_all_frame_names()}")

        # 3) AKS
        _aks = ArticulationKinematicsSolver(_arm, _lks, EE_FRAME_NAME)

        _initialized = True
        carb.log_info("[LulaIK] IK & Controller initialized (with URDF+YAML).")
        return True
    except Exception as e:
        carb.log_error(f"[LulaIK] init failed (will retry next frame): {e}")
        return False

def _on_update(dt: float):
    if not _init_if_ready():
        return
    goal = _get_grasp_from_graph(OG_GOAL_ATTR)
    if goal is None:
        return

    # 直接给一个世界系的“保守可达”目标
    pos = np.asarray(goal[:3], dtype=np.float64).reshape(3)
    quat_wxyz = np.asarray(goal[3:], dtype=np.float64).reshape(4)

    # ★ NEW: 每帧把基座的世界位姿同步给 LULA（即便是恒等，也要同步）
    base_p_w, base_q_w = _arm.get_world_pose()
    _lks.set_robot_base_pose(base_p_w, base_q_w)  # 关键一步（官方范式）:contentReference[oaicite:1]{index=1}

    try:
        action, success = _aks.compute_inverse_kinematics(pos, quat_wxyz)
        carb.log_error(f"[LulaIK] IK success={success}, pos={pos.tolist()}, quat={quat_wxyz.tolist()}")
        if success and action is not None:
            _arm.apply_action(action)
    except Exception as e:
        carb.log_error(f"[LulaIK] IK compute failed: {e}")
        return

    if success and action is not None:
        try:
            _arm.apply_action(action)
        except Exception as e:
            carb.log_error(f"[LulaIK] apply_action failed: {e}")

# 注册 per-frame 回调 → 先 Run，再点 ▶Play
app = omni.kit.app.get_app()
_subscription = app.get_update_event_stream().create_subscription_to_pop(_on_update)
carb.log_info("[LulaIK] Registered per-frame callback. Now press ▶ Play.")
