#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Avocadet ROS Detector Node

This module implements a ROS node for real-time avocado detection from
camera image streams. It combines deep learning-based object detection
with color analysis for ripeness classification and size estimation.

Author: Aaron JS
License: MIT
"""

from __future__ import print_function

import json
import sys
import os

# Add package source to path
PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PKG_ROOT, 'src'))

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import String

from avocadet import AvocadoDetector, ColorAnalyzer, SizeEstimator


class AvocadetDetectorNode(object):
    """
    ROS node for real-time avocado detection and analysis.
    
    This node subscribes to camera image topics and publishes detection
    results including bounding boxes, ripeness classification, size
    estimation, and confidence scores.
    
    Subscribed Topics:
        - /camera/image_raw (sensor_msgs/Image): Input camera stream
        
    Published Topics:
        - /avocadet/detections (std_msgs/String): JSON-formatted detection results
        - /avocadet/annotated_image (sensor_msgs/Image): Visualized output
        
    Parameters:
        - ~model_path (str): Path to custom YOLO model weights
        - ~confidence_threshold (float): Minimum detection confidence [0.0-1.0]
        - ~mode (str): Detection mode ['yolo', 'segment', 'hybrid']
        - ~image_topic (str): Camera topic to subscribe to
        - ~publish_annotated (bool): Whether to publish annotated images
    """
    
    # Color scheme for ripeness visualization (BGR format)
    RIPENESS_COLORS = {
        'unripe': (0, 255, 0),        # Green
        'nearly_ripe': (0, 255, 255),  # Yellow
        'ripe': (0, 165, 255),         # Orange
        'overripe': (0, 0, 255)        # Red
    }
    
    def __init__(self):
        """Initialize the detector node with parameters and publishers."""
        rospy.init_node('avocadet_detector', anonymous=False)
        
        # Get ROS parameters
        self._model_path = rospy.get_param('~model_path', '')
        self._confidence_threshold = rospy.get_param('~confidence_threshold', 0.5)
        self._mode = rospy.get_param('~mode', 'hybrid')
        self._image_topic = rospy.get_param('~image_topic', '/camera/image_raw')
        self._publish_annotated = rospy.get_param('~publish_annotated', True)
        
        # Initialize detection components
        self._initialize_detector()
        
        # Initialize ROS communication
        self._cv_bridge = CvBridge()
        self._setup_publishers_and_subscribers()
        
        # Frame dimensions
        self._frame_width = None
        self._frame_height = None
        
        rospy.loginfo(
            'Avocadet detector initialized | mode=%s, confidence=%.2f',
            self._mode, self._confidence_threshold
        )
    
    def _initialize_detector(self):
        """Initialize the avocado detection pipeline components."""
        model_path = self._model_path if self._model_path else None
        
        self._detector = AvocadoDetector(
            model_path=model_path,
            confidence_threshold=self._confidence_threshold,
            mode=self._mode
        )
        self._color_analyzer = ColorAnalyzer()
        self._size_estimator = SizeEstimator()
        
        rospy.logdebug('Detection pipeline initialized')
    
    def _setup_publishers_and_subscribers(self):
        """Configure ROS publishers and subscribers."""
        # Subscribe to camera images
        self._image_subscriber = rospy.Subscriber(
            self._image_topic,
            Image,
            self._image_callback,
            queue_size=1,
            buff_size=2**24
        )
        
        # Publisher for detection results (JSON format)
        self._detection_publisher = rospy.Publisher(
            '/avocadet/detections',
            String,
            queue_size=10
        )
        
        # Publisher for annotated images (optional)
        if self._publish_annotated:
            self._annotated_publisher = rospy.Publisher(
                '/avocadet/annotated_image',
                Image,
                queue_size=1
            )
        
        rospy.loginfo('Subscribed to: %s', self._image_topic)
    
    def _image_callback(self, msg):
        """
        Process incoming camera images and publish detection results.
        
        Args:
            msg: ROS Image message from camera
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
                
        except CvBridgeError as e:
            rospy.logerr('CV Bridge error: %s', str(e))
        except Exception as e:
            rospy.logerr('Error processing image: %s', str(e))
    
    def _update_frame_dimensions(self, frame):
        """Update frame dimensions for size estimation."""
        height, width = frame.shape[:2]
        
        if self._frame_width != width or self._frame_height != height:
            self._frame_width = width
            self._frame_height = height
            self._size_estimator.update_frame_size(width, height)
    
    def _analyze_detections(self, frame, detections):
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
    
    def _publish_detections(self, results, header):
        """Publish detection results as JSON."""
        message = String()
        message.data = json.dumps({
            'header': {
                'stamp': {
                    'secs': header.stamp.secs,
                    'nsecs': header.stamp.nsecs
                },
                'frame_id': header.frame_id
            },
            'count': len(results),
            'detections': results
        })
        
        self._detection_publisher.publish(message)
    
    def _publish_annotated_image(self, frame, results, header):
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
            label = '{} ({:.0%})'.format(ripeness, result['confidence'])
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
        try:
            annotated_msg = self._cv_bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            annotated_msg.header = header
            self._annotated_publisher.publish(annotated_msg)
        except CvBridgeError as e:
            rospy.logerr('Error publishing annotated image: %s', str(e))
    
    def run(self):
        """Run the node until shutdown."""
        rospy.spin()


def main():
    """Entry point for the avocadet detector node."""
    try:
        node = AvocadetDetectorNode()
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
