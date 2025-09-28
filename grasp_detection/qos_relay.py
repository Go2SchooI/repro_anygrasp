#!/usr/bin/env python3
# 订阅 /camera/color/image_raw、/camera/depth/image_raw、/camera/camera_info（RELIABLE）
# 再以 BEST_EFFORT 重发到同名话题，解决 QoS 不兼容问题。

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo

class QosRelay(Node):
    def __init__(self):
        super().__init__('qos_relay')

        # 订阅端（与 Isaac 匹配）：RELIABLE
        sub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        # 发布端（与 anygrasp 节点匹配）：BEST_EFFORT
        pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        # COLOR
        self.pub_color = self.create_publisher(Image, '/camera/color/image_raw', pub_qos)
        self.sub_color = self.create_subscription(Image, '/camera/color/image_raw', self.cb_color, sub_qos)

        # DEPTH
        self.pub_depth = self.create_publisher(Image, '/camera/depth/image_raw', pub_qos)
        self.sub_depth = self.create_subscription(Image, '/camera/depth/image_raw', self.cb_depth, sub_qos)

        # CAMERA INFO
        self.pub_info = self.create_publisher(CameraInfo, '/camera/camera_info', pub_qos)
        self.sub_info = self.create_subscription(CameraInfo, '/camera/camera_info', self.cb_info, sub_qos)

        self.get_logger().info('QoS relay running: RELIABLE -> BEST_EFFORT for /camera/*')

    def cb_color(self, msg: Image):
        self.pub_color.publish(msg)

    def cb_depth(self, msg: Image):
        self.pub_depth.publish(msg)

    def cb_info(self, msg: CameraInfo):
        self.pub_info.publish(msg)

def main():
    rclpy.init()
    node = QosRelay()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
