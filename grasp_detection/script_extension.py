# === User config ===
ARM_PRIM_PATH = "/World/Franka/panda_link0"                                   # 改成真正的 Articulation Root **link** 路径（非 Xform / 非 joint）
EE_FRAME_NAME = "right_gripper"                                   # 如果你想用 tool_center，先确认它在 URDF frame 列表里
OG_GOAL_ATTR  = "/ActionGraph/PickPose.outputs:grasp_pose_base"   # 7元数组 [x,y,z,qx,qy,qz,qw]

# === Imports ===
import numpy as np
import carb, omni
import omni.graph.core as og
from pxr import Usd
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.controllers import ArticulationController
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver, ArticulationKinematicsSolver
from isaacsim.robot_motion.motion_generation import interface_config_loader  # ★ 关键：加载内置 Lula 配置

_initialized = False
_after_play_frames = 0
_arm = _aks = _ctrl = None

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
    global _initialized, _arm, _aks, _ctrl, _after_play_frames
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

        # 2) 加载内置的 Lula Kinematics 配置（★ 解决 urdf_path 必填）
        #    对内置支持的机器人，可直接按名称加载，例如 "Franka"
        cfg = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
        # cfg 等价于 {"robot_description_path": ".../franka/rmpflow/robot_descriptor.yaml",
        #            "urdf_path": ".../franka/lula_franka_gen.urdf"}（文档示例同理）
        # 也可以手动拼路径：见 5.0 教程中 41-47 行
        # https://docs.isaacsim.omniverse.nvidia.com/5.0.0/manipulators/manipulators_lula_kinematics.html
        lks = LulaKinematicsSolver(**cfg)

        # （可选）打印 URDF 里所有可用的 frame 名，确认 EE 是否存在
        carb.log_info(f"[LulaIK] Frames: {lks.get_all_frame_names()}")

        # 3) AKS + 控制器
        _aks = ArticulationKinematicsSolver(_arm, lks, EE_FRAME_NAME)

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
    pos = goal[:3]
    quat_xyzw = goal[3:]

    try:
        action, success = _aks.compute_inverse_kinematics(pos, quat_xyzw)  # AKS 5.0 教程里也是这么用
        if success and action is not None:
            _arm.apply_action(action)
    except Exception as e:
        carb.log_error(f"[LulaIK] IK compute failed: {e}")
        return

    if success and action is not None:
        try:
            _arm.apply_action(action)              # 直接下发 ArticulationAction
        except Exception as e:
            carb.log_error(f"[LulaIK] apply_action failed: {e}")
    # else: IK 不可达，静默跳过

# 注册 per-frame 回调 → 先 Run，再点 ▶Play
app = omni.kit.app.get_app()
_subscription = app.get_update_event_stream().create_subscription_to_pop(_on_update)
carb.log_info("[LulaIK] Registered per-frame callback. Now press ▶ Play.")
