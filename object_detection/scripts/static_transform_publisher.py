#!/usr/bin/env python3

import math
import rclpy
from geometry_msgs.msg import TransformStamped
import tf2_ros

def quaternion_from_euler(roll: float, pitch: float, yaw: float):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    q = [0.0, 0.0, 0.0, 1.0]
    q[0] = sr * cp * cy - cr * sp * sy  # x
    q[1] = cr * sp * cy + sr * cp * sy  # y
    q[2] = cr * cp * sy - sr * sp * cy  # z
    q[3] = cr * cp * cy + sr * sp * sy  # w
    return q


class StaticTransformPublisher:
    def __init__(self) -> None:
        self.node = rclpy.create_node('static_transform_publisher_node')
        self.broadcaster = tf2_ros.StaticTransformBroadcaster(self.node)

    def publish_static_transform(self) -> None:
        static_transform_stamped = TransformStamped()
        static_transform_stamped.header.stamp = self.node.get_clock().now().to_msg()
        static_transform_stamped.header.frame_id = 'base_link'
        static_transform_stamped.child_frame_id = 'wrist_rgbd_camera_depth_optical_frame'

        # Translation from urdf
        static_transform_stamped.transform.translation.x = 0.35
        static_transform_stamped.transform.translation.y = 0.45
        static_transform_stamped.transform.translation.z = 0.1

        # r=0, p=pi/6, y=-pi/2 from urdf converted to Quaternion
        qx, qy, qz, qw = quaternion_from_euler(0.0, math.pi / 6.0, -math.pi / 2.0)

        static_transform_stamped.transform.rotation.x = qx
        static_transform_stamped.transform.rotation.y = qy
        static_transform_stamped.transform.rotation.z = qz
        static_transform_stamped.transform.rotation.w = qw

        self.broadcaster.sendTransform(static_transform_stamped)

    def spin(self) -> None:
        rclpy.spin(self.node)
        self.node.destroy_node()
        rclpy.shutdown()


def main(args=None) -> None:
    rclpy.init(args=args)
    static_transform_publisher = StaticTransformPublisher()
    static_transform_publisher.publish_static_transform()
    static_transform_publisher.spin()

if __name__ == '__main__':
    main()