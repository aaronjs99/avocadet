#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math
from geometry_msgs.msg import PointStamped


class WaypointController(Node):
    def __init__(self):
        super().__init__("bunker_nav")

        self.k_v = 0.5
        self.k_w = 1.0

        self.max_v = 0.2
        self.max_w = 0.4

        self.aruco_sub = self.create_subscription(
            PointStamped, "/aruco/marker_0/position", self.aruco_callback, 10
        )
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

    def aruco_callback(self, msg):
        # Robot position
        x = msg.point.x
        z = msg.point.z

        # Errors
        distance = math.sqrt(x * x + z * z)

        target_angle = (
            math.atan2(x, z) * -1
        )  # change the sign because when marker left --> robot rotate left --> ang.z: (+)
        self.get_logger().info(
            f"Distance: {distance:.2f} m, Target Angle: {math.degrees(target_angle):.2f}°"
        )

        cmd = Twist()

        if distance > 0.5:
            cmd.linear.x = min(self.k_v * distance, self.max_v)
            cmd.angular.z = max(min(self.k_w * target_angle, self.max_w), -self.max_w)
            self.get_logger().info(
                f"cmd.linear.x: {cmd.linear.x:.2f} m/s, cmd.angular.z: {cmd.angular.z:.2f} rad/s"
            )
        else:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.get_logger().info("Goal reached!")

        self.cmd_pub.publish(cmd)

    @staticmethod
    def normalize_angle(a):
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a


def main():
    rclpy.init()
    node = WaypointController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
