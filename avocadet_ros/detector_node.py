#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Avocadet ROS2 Detector Node

This module implements a ROS2 node for real-time avocado detection from
camera image streams. It combines deep learning-based object detection
with color analysis for ripeness classification and size estimation.

Author: Aaron JS
License: MIT
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String


from avocadet_lib import AvocadoDetector, ColorAnalyzer, SizeEstimator
from avocadet_lib.detector import Detection


class AvocadetDetectorNode(Node):
    """
    ROS2 node for real-time avocado detection and analysis.
    
    This node subscribes to camera image topics and publishes detection
    results including bounding boxes, ripeness classification, size
    estimation, and confidence scores.
    
    Subscribed Topics:
        - /camera/image_raw (sensor_msgs/Image): Input camera stream
        
    Published Topics:
        - /avocadet/detections (std_msgs/String): JSON-formatted detection results
        - /avocadet/annotated_image (sensor_msgs/Image): Visualized output
        
    Parameters:
        - model_path (str): Path to custom YOLO model weights
        - confidence_threshold (float): Minimum detection confidence [0.0-1.0]
        - mode (str): Detection mode ['yolo', 'segment', 'hybrid']
        - image_topic (str): Camera topic to subscribe to
        - publish_annotated (bool): Whether to publish annotated images
    """
    
    # Color scheme for ripeness visualization (BGR format)
    RIPENESS_COLORS = {
        'unripe': (0, 255, 0),        # Green
        'nearly_ripe': (0, 255, 255),  # Yellow
        'ripe': (0, 165, 255),         # Orange
        'overripe': (0, 0, 255)        # Red
    }
    
    def __init__(self) -> None:
        """Initialize the detector node with parameters and publishers."""
        super().__init__('avocadet_detector')
        
        # Declare ROS2 parameters with descriptors
        self._declare_parameters()
        
        # Retrieve parameter values
        self._model_path = self.get_parameter('model_path').value
        self._confidence_threshold = self.get_parameter('confidence_threshold').value
        self._mode = self.get_parameter('mode').value
        self._image_topic = self.get_parameter('image_topic').value
        self._publish_annotated = self.get_parameter('publish_annotated').value
        
        # Initialize detection components
        self._initialize_detector()
        
        # Initialize ROS2 communication
        self._cv_bridge = CvBridge()
        self._setup_publishers_and_subscribers()
        
        # Frame dimensions (updated on first image)
        self._frame_width: Optional[int] = None
        self._frame_height: Optional[int] = None
        
        self.get_logger().info(
            f'Avocadet detector initialized | '
            f'mode={self._mode}, confidence={self._confidence_threshold}'
        )
    
    def _declare_parameters(self) -> None:
        """Declare ROS2 parameters with default values."""
        self.declare_parameter('model_path', '')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('mode', 'hybrid')
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('publish_annotated', True)
    
    def _initialize_detector(self) -> None:
        """Initialize the avocado detection pipeline components."""
        model_path = self._model_path if self._model_path else None
        
        self._detector = AvocadoDetector(
            model_path=model_path,
            confidence_threshold=self._confidence_threshold,
            mode=self._mode
        )
        self._color_analyzer = ColorAnalyzer()
        self._size_estimator = SizeEstimator()
        
        self.get_logger().debug('Detection pipeline initialized')
    
    def _setup_publishers_and_subscribers(self) -> None:
        """Configure ROS2 publishers and subscribers."""
        # QoS profile for sensor data
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Subscribe to camera images
        self._image_subscription = self.create_subscription(
            Image,
            self._image_topic,
            self._image_callback,
            sensor_qos
        )
        
        # Publisher for detection results (JSON format)
        self._detection_publisher = self.create_publisher(
            String,
            '/avocadet/detections',
            10
        )
        
        # Publisher for annotated images (optional)
        if self._publish_annotated:
            self._annotated_publisher = self.create_publisher(
                Image,
                '/avocadet/annotated_image',
                10
            )
        
        self.get_logger().info(f'Subscribed to: {self._image_topic}')
    
    def _image_callback(self, msg: Image) -> None:
        """
        Process incoming camera images and publish detection results.
        
        Args:
            msg: ROS2 Image message from camera
        """
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Update frame dimensions
            self._update_frame_dimensions(cv_image)
            
            # Run detection
            detections = self._detector.detect(cv_image)
            
            # Analyze each detection
            analyzed_results = self._analyze_detections(cv_image, detections)
            
            # Publish detection results
            self._publish_detections(analyzed_results, msg.header)
            
            # Publish annotated image if enabled
            if self._publish_annotated and detections:
                self._publish_annotated_image(cv_image, analyzed_results, msg.header)
                
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}', throttle_duration_sec=1.0)
    
    def _update_frame_dimensions(self, frame: np.ndarray) -> None:
        """Update frame dimensions for size estimation."""
        height, width = frame.shape[:2]
        
        if self._frame_width != width or self._frame_height != height:
            self._frame_width = width
            self._frame_height = height
            self._size_estimator.update_frame_size(width, height)
    
    def _analyze_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection]
    ) -> List[Dict[str, Any]]:
        """
        Analyze each detection for ripeness and size.
        
        Args:
            frame: Input image frame
            detections: List of Detection objects
            
        Returns:
            List of analysis results as dictionaries
        """
        results = []
        
        for detection in detections:
            # Analyze color and ripeness
            color, color_name, ripeness = self._color_analyzer.analyze(
                frame, detection.bbox
            )
            
            # Estimate relative size
            size_category, relative_size = self._size_estimator.estimate(
                detection.bbox
            )
            
            results.append({
                'bbox': {
                    'x': int(detection.bbox[0]),
                    'y': int(detection.bbox[1]),
                    'width': int(detection.bbox[2] - detection.bbox[0]),
                    'height': int(detection.bbox[3] - detection.bbox[1])
                },
                'confidence': round(float(detection.confidence), 3),
                'ripeness': ripeness.value,
                'size_category': size_category.value,
                'relative_size': round(float(relative_size), 4),
                'color': {
                    'r': int(color[2]),
                    'g': int(color[1]),
                    'b': int(color[0])
                }
            })
        
        return results
    
    def _publish_detections(
        self,
        results: List[Dict[str, Any]],
        header: Any
    ) -> None:
        """Publish detection results as JSON."""
        message = String()
        message.data = json.dumps({
            'header': {
                'stamp': {
                    'sec': header.stamp.sec,
                    'nanosec': header.stamp.nanosec
                },
                'frame_id': header.frame_id
            },
            'count': len(results),
            'detections': results
        }, separators=(',', ':'))
        
        self._detection_publisher.publish(message)
    
    def _publish_annotated_image(
        self,
        frame: np.ndarray,
        results: List[Dict[str, Any]],
        header: Any
    ) -> None:
        """Draw detections on frame and publish."""
        annotated = frame.copy()
        
        for result in results:
            bbox = result['bbox']
            x1, y1 = bbox['x'], bbox['y']
            x2, y2 = x1 + bbox['width'], y1 + bbox['height']
            
            # Get color based on ripeness
            ripeness = result['ripeness']
            color = self.RIPENESS_COLORS.get(ripeness, (0, 255, 0))
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label = f"{ripeness} ({result['confidence']:.0%})"
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                annotated,
                (x1, y1 - label_h - baseline - 5),
                (x1 + label_w + 5, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated, label, (x1 + 2, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
        
        # Convert and publish
        annotated_msg = self._cv_bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        annotated_msg.header = header
        self._annotated_publisher.publish(annotated_msg)


def main(args: Optional[List[str]] = None) -> None:
    """
    Entry point for the avocadet detector node.
    
    Args:
        args: Command-line arguments (passed to rclpy.init)
    """
    rclpy.init(args=args)
    
    node = AvocadetDetectorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
