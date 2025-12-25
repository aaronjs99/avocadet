"""
Avocadet ROS2 Launch File

Launches the avocado detection node.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # Arguments
        DeclareLaunchArgument(
            'image_topic',
            default_value='/camera/image_raw',
            description='Input camera image topic'
        ),
        DeclareLaunchArgument(
            'model_path',
            default_value='',
            description='Path to custom YOLO model'
        ),
        DeclareLaunchArgument(
            'confidence',
            default_value='0.5',
            description='Detection confidence threshold'
        ),
        DeclareLaunchArgument(
            'mode',
            default_value='hybrid',
            description='Detection mode: yolo, segment, or hybrid'
        ),

        # Detector node
        Node(
            package='avocadet_ros',
            executable='detector_node.py',
            name='avocadet_detector',
            parameters=[{
                'image_topic': LaunchConfiguration('image_topic'),
                'model_path': LaunchConfiguration('model_path'),
                'confidence_threshold': LaunchConfiguration('confidence'),
                'mode': LaunchConfiguration('mode'),
                'publish_annotated': True,
            }],
            output='screen'
        ),
    ])
