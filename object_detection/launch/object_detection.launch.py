import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_object_detection = get_package_share_directory('object_detection')

    # Path to the RViz configuration file
    rviz_config_file = os.path.join(
        pkg_object_detection,
        'rviz',
        'object_detection.rviz'
    )

    # 1. Static Transform Publisher node (base_link -> wrist_rgbd_camera_depth_optical_frame)
    static_tf_node = Node(
        package='object_detection',
        executable='static_transform_publisher.py',
        name='static_transform_publisher_node',
        output='screen'
    )

    # 2. Perception / Object Detection node
    object_detection_node = Node(
        package='object_detection',
        executable='object_detection.py',
        name='object_detection_node',
        output='screen'
    )

    # 3. RViz2 node loaded with custom configuration
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        output='screen'
    )

    return LaunchDescription([
        static_tf_node,
        object_detection_node,
        rviz_node
    ])