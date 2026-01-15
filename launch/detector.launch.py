#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Avocadet ROS2 Launch Configuration

This module provides launch configurations for the Avocadet detection system.
Supports various camera sources and detection modes.

Usage:
    ros2 launch avocadet detector.launch.py
    ros2 launch avocadet detector.launch.py image_topic:=/camera/color/image_raw

Author: Aaron JS
License: MIT
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """
    Generate the launch description for avocadet detector.

    Returns:
        LaunchDescription with all configured nodes and parameters.
    """

    # Declare launch arguments
    declared_arguments = [
        DeclareLaunchArgument(
            "image_topic",
            default_value="/camera/image_raw",
            description="Input camera image topic",
        ),
        DeclareLaunchArgument(
            "model_path",
            default_value="",
            description="Path to custom YOLO model weights (empty for default)",
        ),
        DeclareLaunchArgument(
            "confidence",
            default_value="0.5",
            description="Detection confidence threshold (0.0-1.0)",
        ),
        DeclareLaunchArgument(
            "mode",
            default_value="hybrid",
            description="Detection mode: yolo, segment, or hybrid",
        ),
        DeclareLaunchArgument(
            "publish_annotated",
            default_value="true",
            description="Publish annotated images with bounding boxes",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Use simulation time from Gazebo",
        ),
    ]

    # Detector node configuration
    detector_node = Node(
        package="avocadet",
        executable="detector_node.py",
        name="avocadet_detector",
        parameters=[
            {
                "image_topic": LaunchConfiguration("image_topic"),
                "model_path": LaunchConfiguration("model_path"),
                "confidence_threshold": LaunchConfiguration("confidence"),
                "mode": LaunchConfiguration("mode"),
                "publish_annotated": LaunchConfiguration("publish_annotated"),
                "use_sim_time": LaunchConfiguration("use_sim_time"),
            }
        ],
        output="screen",
        emulate_tty=True,
    )

    # Log launch configuration
    log_info = LogInfo(
        msg=[
            "Launching Avocadet detector on topic: ",
            LaunchConfiguration("image_topic"),
        ]
    )

    return LaunchDescription(declared_arguments + [log_info, detector_node])
