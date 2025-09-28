import json, os, shutil, subprocess, tempfile 
import numpy as np 
import rclpy 

from rclpy.node import Node 
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy 
from sensor_msgs.msg import Image, CameraInfo 
from geometry_msgs.msg import Pose, PoseArray 
from std_msgs.msg import Header 
from cv_bridge import CvBridge 
from image_geometry import PinholeCameraModel
from builtin_interfaces.msg import Time as TimeMsg

def _to_float_sec(t: TimeMsg) -> float:
    return float(t.sec) + float(t.nanosec) * 1e-9

class RGBDAnyGraspNode(Node): 
    def __init__(self): 
        super().__init__('rgbd_anygrasp_node') 
        self.bridge = CvBridge()

        # 参数
        self.declare_parameter('color_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')

        self.declare_parameter('anygrasp_env', 'fuck_anygrasp')  # conda 环境名
        self.declare_parameter('wrapper_script', '/home/jizexian/github/anygrasp_sdk/grasp_detection/anygrasp_wrapper.py')
        self.declare_parameter('stride', 1)
        self.declare_parameter('publish_topic', '/anygrasp/grasps')

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        color_topic = self.get_parameter('color_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        info_topic  = self.get_parameter('camera_info_topic').value

        # 最新缓存
        self.latest_color = None   # (msg, t_sec)
        self.latest_info  = None   # (msg, t_sec)

        # 单独订阅三路；以“深度帧到达”作为触发
        self.create_subscription(Image, color_topic, self._on_color, qos)
        self.create_subscription(CameraInfo, info_topic, self._on_info, qos)
        self.create_subscription(Image, depth_topic, self._on_depth_trigger, qos)

        self.model = PinholeCameraModel()
        self.pub   = self.create_publisher(PoseArray, self.get_parameter('publish_topic').value, 10)
        self.get_logger().info(f'RGBDAnyGraspNode subscribing {color_topic}, {depth_topic}, {info_topic}')

    def _on_color(self, msg: Image):
        self.latest_color = (msg, _to_float_sec(msg.header.stamp))

    def _on_info(self, msg: CameraInfo):
        self.latest_info = (msg, _to_float_sec(msg.header.stamp))

    def _on_depth_trigger(self, depth_msg: Image):
        if self.latest_color is None or self.latest_info is None:
            return
        t_depth = _to_float_sec(depth_msg.header.stamp)
        color_msg, t_color = self.latest_color
        info_msg, t_info = self.latest_info

        self.get_logger().info(f"Depth timestamp: {t_depth}, Color timestamp: {t_color}, Info timestamp: {t_info}")

        # 允许 0.5 秒内的时间差
        if abs(t_depth - t_color) > 0.5 or abs(t_depth - t_info) > 0.05:
            return
        try:
            self.on_rgbd(color_msg, depth_msg, info_msg)
        except Exception as e:
            self.get_logger().error(f'on_rgbd error: {e}')
    
    def on_rgbd(self, color_msg: Image, depth_msg: Image, info_msg: CameraInfo):
        self.get_logger().info("on_rgbd called!")
        self.get_logger().info(
            f"on_rgbd: ts=({color_msg.header.stamp.sec}.{color_msg.header.stamp.nanosec}, "
            f"{depth_msg.header.stamp.sec}.{depth_msg.header.stamp.nanosec}, "
            f"{info_msg.header.stamp.sec}.{info_msg.header.stamp.nanosec})"
        )

        # 转 numpy
        color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')  # HxWx3 BGR
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')  # 32FC1(m) or 16UC1(mm)
        # 相机内参
        self.model.fromCameraInfo(info_msg)
        fx, fy = float(self.model.fx()), float(self.model.fy())
        cx, cy = float(self.model.cx()), float(self.model.cy())
        frame_id = info_msg.header.frame_id or 'camera_link'

        # 深度统一为 mm 的 16UC1，记录 scale
        if depth.dtype == np.float32:
            self.get_logger().info("Depth image is float32, converting to uint16 in mm")
            depth_mm = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
            depth_scale = 0.001
        elif depth.dtype == np.uint16:
            depth_mm = depth
            depth_scale = 0.001
        else:
            self.get_logger().warn(f'Unsupported depth dtype: {depth.dtype}')
            return

        # 下采样（同步缩放内参）
        s = max(1, int(self.get_parameter('stride').value))
        if s > 1:
            color = color[::s, ::s, :]
            depth_mm = depth_mm[::s, ::s]
            fx, fy = fx / s, fy / s
            cx, cy = cx / s, cy / s

        # 写临时目录
        tmpdir = tempfile.mkdtemp(prefix='anygrasp_')
        try:
            # 保存图片
            from PIL import Image as PILImage
            color_rgb = color[:, :, ::-1]  # BGR->RGB
            PILImage.fromarray(color_rgb).save(os.path.join(tmpdir, 'color_1.png'))
            PILImage.fromarray(depth_mm).save(os.path.join(tmpdir, 'depth_1.png'))

            # 保存内参
            meta = {
                'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
                'depth_scale': depth_scale,
                'frame_id': frame_id}
            with open(os.path.join(tmpdir, 'intrinsics.json'), 'w') as f:
                json.dump(meta, f)

            # 调用包装脚本（conda 环境）
            env_name = self.get_parameter('anygrasp_env').value
            wrapper = self.get_parameter('wrapper_script').value
            cmd = ['conda', 'run', '-n', env_name, 'python', wrapper, '--data_dir', tmpdir]
            self.get_logger().info(f'Running AnyGrasp: {" ".join(cmd)}')
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if out.returncode != 0:
                self.get_logger().error(f'AnyGrasp failed: {out.stderr}')
            else:
                self.get_logger().info(f'AnyGrasp output: {out.stdout}')

            # 读取 grasps.json
            grasp_json = os.path.join(tmpdir, 'grasps.json')
            if not os.path.exists(grasp_json):
                self.get_logger().warn('grasps.json not found')
            else:
                self.get_logger().info(f'Found grasps.json at {grasp_json}')


            with open(grasp_json, 'r') as f:
                data = json.load(f)

            # 发布 PoseArray
            pa = PoseArray()
            pa.header = Header()
            pa.header.stamp = depth_msg.header.stamp
            pa.header.frame_id = frame_id

            for g in data.get('grasps', []):
                pose = Pose()
                t = g.get('translation', [0, 0, 0])
                q = g.get('quaternion', [0, 0, 0, 1])
                pose.position.x, pose.position.y, pose.position.z = map(float, t)
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = map(float, q)
                pa.poses.append(pose)

            self.pub.publish(pa)

        except Exception as e:
            self.get_logger().error(f'AnyGrasp call error: {e}')
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

def main(): 
    rclpy.init() 
    node = RGBDAnyGraspNode() 
    rclpy.spin(node) 
    node.destroy_node() 
    rclpy.shutdown()
