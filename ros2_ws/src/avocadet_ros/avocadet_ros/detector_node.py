#!/usr/bin/env python3
"""
Avocadet ROS2 Detector Node

Subscribes to camera images and publishes avocado detections.
"""

import sys
import os

# Add avocadet to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', 'src'))

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

from avocadet_msgs.msg import AvocadoDetection, AvocadoDetectionArray
from avocadet import AvocadoDetector, ColorAnalyzer, SizeEstimator


class AvocadetDetectorNode(Node):
    """ROS2 node for avocado detection."""

    def __init__(self):
        super().__init__('avocadet_detector')
        
        # Declare parameters
        self.declare_parameter('model_path', '')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('mode', 'hybrid')
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('publish_annotated', True)
        
        # Get parameters
        model_path = self.get_parameter('model_path').value
        confidence = self.get_parameter('confidence_threshold').value
        mode = self.get_parameter('mode').value
        image_topic = self.get_parameter('image_topic').value
        self.publish_annotated = self.get_parameter('publish_annotated').value
        
        # Initialize detector
        self.detector = AvocadoDetector(
            model_path=model_path if model_path else None,
            confidence_threshold=confidence,
            mode=mode
        )
        self.analyzer = ColorAnalyzer()
        self.size_estimator = SizeEstimator()
        self.bridge = CvBridge()
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10
        )
        
        # Publishers
        self.detection_pub = self.create_publisher(
            AvocadoDetectionArray,
            '/avocadet/detections',
            10
        )
        
        if self.publish_annotated:
            self.annotated_pub = self.create_publisher(
                Image,
                '/avocadet/annotated_image',
                10
            )
        
        self.get_logger().info(f'Avocadet detector initialized (mode: {mode})')
        self.get_logger().info(f'Subscribing to: {image_topic}')

    def image_callback(self, msg: Image):
        """Process incoming image and publish detections."""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # Update size estimator
            h, w = cv_image.shape[:2]
            self.size_estimator.update_frame_size(w, h)
            
            # Detect avocados
            detections = self.detector.detect(cv_image)
            
            # Create detection array message
            detection_array = AvocadoDetectionArray()
            detection_array.header = msg.header
            
            for det in detections:
                # Analyze each detection
                color, color_name, ripeness = self.analyzer.analyze(cv_image, det.bbox)
                size_cat, rel_size = self.size_estimator.estimate(det.bbox)
                
                # Create detection message
                detection_msg = AvocadoDetection()
                detection_msg.bbox.x = float(det.bbox[0])
                detection_msg.bbox.y = float(det.bbox[1])
                detection_msg.bbox.width = float(det.bbox[2] - det.bbox[0])
                detection_msg.bbox.height = float(det.bbox[3] - det.bbox[1])
                detection_msg.confidence = det.confidence
                detection_msg.ripeness = ripeness.value
                detection_msg.size_category = size_cat.value
                detection_msg.relative_size = rel_size
                detection_msg.dominant_color.r = int(color[2])
                detection_msg.dominant_color.g = int(color[1])
                detection_msg.dominant_color.b = int(color[0])
                
                detection_array.detections.append(detection_msg)
            
            detection_array.count = len(detections)
            self.detection_pub.publish(detection_array)
            
            # Publish annotated image if enabled
            if self.publish_annotated and detections:
                annotated = self._draw_detections(cv_image, detections)
                annotated_msg = self.bridge.cv2_to_imgmsg(annotated, 'bgr8')
                annotated_msg.header = msg.header
                self.annotated_pub.publish(annotated_msg)
                
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def _draw_detections(self, frame, detections):
        """Draw detection boxes on frame."""
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'avocado {det.confidence:.2f}'
            cv2.putText(annotated, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return annotated


def main(args=None):
    rclpy.init(args=args)
    node = AvocadetDetectorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
