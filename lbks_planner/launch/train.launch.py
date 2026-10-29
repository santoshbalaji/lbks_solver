from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='lbks_planner',
            executable='train',
            output='screen',
            prefix='/home/santosh/purdue_ws/venv/bin/python',
        ),
    ])
