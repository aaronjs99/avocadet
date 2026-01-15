#!/usr/bin/env python3
# Aruco Marker Generator: https://chev.me/arucogen/ (using 6x6)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import math
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import PointStamped


class ImageSubscriber(Node):
    def __init__(self):
        super().__init__("marker_detection")
        # IMPORTANT: use the depth topic that is ALIGNED to color. Typical RealSense:
        # '/camera/aligned_depth_to_color/image_raw'  or '/camera/depth/image_rect_raw' if already aligned.
        self.color_topic = "/camera/camera/color/image_raw"
        self.depth_topic = "/camera/camera/aligned_depth_to_color/image_raw"
        self.camera_info_topic = (
            "/camera/camera/color/camera_info"  # color intrinsics (cx,cy,fx,fy)
        )

        self.bridge = CvBridge()
        qos = QoSProfile(depth=5)
        qos.reliability = QoSReliabilityPolicy.BEST_EFFORT
        qos.history = QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST

        self.color_sub = self.create_subscription(
            Image, self.color_topic, self.color_callback, qos
        )
        self.depth_sub = self.create_subscription(
            Image, self.depth_topic, self.depth_callback, qos
        )
        self.info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.camera_info_callback, qos
        )

        self.get_logger().info(
            f"Subscribed color:{self.color_topic} depth:{self.depth_topic} info:{self.camera_info_topic}"
        )

        self.pos_pubs = {}  # id -> Publisher(PointStamped)

        # storage for latest depth & intrinsics
        self.latest_depth_img = None
        self.depth_encoding = None
        self.K = None  # camera intrinsics 3x3
        self.fx = self.fy = self.cx = self.cy = None

        # aruco setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        try:
            self.parameters = cv2.aruco.DetectorParameters_create()
        except AttributeError:
            self.parameters = cv2.aruco.DetectorParameters()
        try:
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        except AttributeError:
            self.detector = None

        # self.window_name = 'camera'
        # cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        # cv2.resizeWindow(self.window_name, 1280, 720)

    def camera_info_callback(self, msg: CameraInfo):
        # take first CameraInfo and extract intrinsics
        if self.K is None:
            self.K = np.array(msg.k).reshape((3, 3))
            self.fx = float(self.K[0, 0])
            self.fy = float(self.K[1, 1])
            self.cx = float(self.K[0, 2])
            self.cy = float(self.K[1, 2])
            if msg.header.frame_id:
                self.camera_frame = msg.header.frame_id
            self.get_logger().info(
                f"Camera intrinsics fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}"
            )

    def depth_callback(self, msg: Image):
        # store the latest depth image (do not convert to color!)
        self.depth_encoding = msg.encoding
        try:
            # do not request desired_encoding — keep raw
            depth_cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding=msg.encoding)
        except CvBridgeError as e:
            self.get_logger().error(f"Depth CvBridge error: {e}")
            return
        self.latest_depth_img = depth_cv

    def color_callback(self, msg: Image):
        try:
            color_cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"Color CvBridge error: {e}")
            return

        gray = cv2.cvtColor(color_cv, cv2.COLOR_BGR2GRAY)

        # detect markers
        if self.detector is not None:
            corners, ids, rejected = self.detector.detectMarkers(gray)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.parameters
            )

        if ids is not None:
            try:
                marker_ids = ids.ravel().astype(int).tolist()
            except Exception:
                marker_ids = np.array(ids).flatten().tolist()
            self.get_logger().info(f"Detected markers: {marker_ids}")

            # compute center pixel of each detected marker and get 3D point
            for i, c in enumerate(corners):
                # corners[i] is shape (1,4,2) or (4,2) depending on OpenCV; normalize:
                pts = np.array(c).reshape((-1, 2))
                u = float(np.mean(pts[:, 0]))  # x pixel
                v = float(np.mean(pts[:, 1]))  # y pixel

                z = self.get_depth_meter(u, v)
                if z is None:
                    self.get_logger().warning(
                        f"No valid depth for marker {marker_ids[i]} at pixel ({u:.1f},{v:.1f})"
                    )
                    continue

                X = (u - self.cx) * z / self.fx
                Y = (v - self.cy) * z / self.fy
                Z = z

                # 3D point in camera frame
                self.get_logger().info(
                    f"Marker {marker_ids[i]} 3D (camera frame): X={X:.3f} m, Y={Y:.3f} m, Z={Z:.3f} m (pixel {u:.1f},{v:.1f})"
                )

                # publish pose: ensure publisher exists
                point_msg = PointStamped()
                point_msg.header.frame_id = self.camera_frame
                point_msg.header.stamp = self.get_clock().now().to_msg()

                point_msg.point.x = float(X)
                point_msg.point.y = float(Y)
                point_msg.point.z = float(Z)

                mid = int(marker_ids[i])
                topic = f"/aruco/marker_{mid}/position"
                if mid not in self.pos_pubs:
                    self.pos_pubs[mid] = self.create_publisher(PointStamped, topic, 10)
                    self.get_logger().info(
                        f"Created publisher for marker {mid} on {topic}"
                    )
                self.pos_pubs[mid].publish(point_msg)

                # (Optional) draw the ID and coordinates on the image
                text = f"id:{marker_ids[i]} ({X:.2f},{Y:.2f},{Z:.2f}m)"
                cv2.putText(
                    color_cv,
                    text,
                    (int(u) - 50, int(v) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    2,
                )

        # # draw markers and show
        # if corners is not None and ids is not None:
        #     cv2.aruco.drawDetectedMarkers(color_cv, corners, ids)

        # cv2.imshow(self.window_name, color_cv)
        # key = cv2.waitKey(1) & 0xFF
        # if key == ord('q'):
        #     self.get_logger().info("'q' pressed — shutting down.")
        #     rclpy.shutdown()

    def get_depth_meter(self, u, v, search_radius=3):
        """
        Return depth in meters at pixel (u,v).
        If the exact pixel is invalid (0 or NaN), search a small square neighborhood (radius).
        Handles common encodings: '32FC1' (meters), '16UC1' (integer depth).
        For 16UC1 we assume value is in millimeters (common) -> convert to meters by /1000.0.
        If your depth unit differs, adjust accordingly.
        """
        if self.latest_depth_img is None:
            return None
        if self.fx is None:
            # no intrinsics yet
            return None

        h, w = self.latest_depth_img.shape[:2]
        iu = int(round(u))
        iv = int(round(v))
        if iu < 0 or iu >= w or iv < 0 or iv >= h:
            return None

        # neighborhood sampling
        vals = []
        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                x = iu + dx
                y = iv + dy
                if x < 0 or x >= w or y < 0 or y >= h:
                    continue
                raw = self.latest_depth_img[y, x]
                # handle different dtypes / encodings
                if self.depth_encoding == "32FC1" or self.depth_encoding == "32FC3":
                    # float32 in meters
                    if math.isfinite(float(raw)) and float(raw) > 0.0:
                        vals.append(float(raw))
                else:
                    # assume 16UC1 or similar: integer. common RealSense: depth in millimeters or depth units
                    # convert to meters
                    try:
                        rval = int(raw)
                    except Exception:
                        continue
                    if rval == 0:
                        continue
                    # assume millimeters -> meters
                    vals.append(float(rval) / 1000.0)

        if len(vals) == 0:
            return None
        # robust median
        return float(np.median(vals))


def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt, shutting down node.")
    finally:
        cv2.destroyAllWindows()
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
