from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution

def generate_launch_description():

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('realsense2_camera'),
                'launch',
                'rs_launch.py'
            ])
        ),
        launch_arguments={
            'initial_reset': 'true',
            'enable_color': 'true',
            'enable_depth': 'true',
            'enable_sync': 'true',
            'align_depth.enable': 'true',
            'rgb_camera.color_profile': '1280x720x30',
            'depth_module.depth_profile': '640x360x30',
            'pointcloud.enable': 'true',
            'pointcloud.stream_filter': '0',
            'clip_distance': '4.0',
            'decimation_filter.enable': 'true',
            'decimation_filter.filter_magnitude': '4',
            'spatial_filter.enable': 'true',
            'temporal_filter.enable': 'true',
            'enable_gyro': 'true',
            'enable_accel': 'true',
            'unite_imu_method': '2',
            'motion_module.gyro_sensitivity': '4',
        }.items()
    )   

    aruco_node = Node(
        package='marker_detection',
        executable='aruco_detector_3d.py',
        name='aruco_detector_node',
        output='screen'
    )

    return LaunchDescription([
        realsense_launch,
        aruco_node
    ])
