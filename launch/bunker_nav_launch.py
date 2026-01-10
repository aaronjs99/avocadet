from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution

def generate_launch_description():

    aruco_node = Node(
        package='marker_detection',
        executable='aruco_detector_3d.py',
        name='aruco_detector_node',
        output='screen'
    )

    aruco_nav_node = Node(
        package='bunker_nav',
        executable='aruco_tracking.py',
        name='aruco_tracking_node',
        output='screen'
    )

    return LaunchDescription([
        aruco_node,
        aruco_nav_node
    ])
