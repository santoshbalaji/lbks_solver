"""Launch the integrated RViz studio and its isolated inference backend."""

import os
from pathlib import Path
import shlex
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from lbks_gui.robot import load_robot


def start(context):
    value = lambda name: LaunchConfiguration(name).perform(context)
    urdf, root = load_robot(value('urdf_path'))
    backend = Node(
        package='lbks_gui', executable='ik_preview', output='screen',
        prefix=shlex.quote(value('python_executable')),
        parameters=[{'urdf_path': value('urdf_path'),
                     'models_config': value('models_config'), 'tip_link': value('tip_link')}])
    publisher = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        name='lbks_gui_robot_state', output='screen',
        parameters=[{'robot_description': urdf}],
        remappings=[('joint_states', '/lbks_gui/joint_states'),
                    ('robot_description', '/lbks_gui/robot_description'),
                    ('/tf', '/lbks_gui/tf'), ('/tf_static', '/lbks_gui/tf_static')])
    rviz = Node(
        package='rviz2', executable='rviz2', name='lbks_gui_view', output='screen',
        arguments=['-d', value('rviz_config'), '-f', root],
        remappings=[('/tf', '/lbks_gui/tf'), ('/tf_static', '/lbks_gui/tf_static')])
    shutdown = RegisterEventHandler(OnProcessExit(
        target_action=rviz,
        on_exit=[EmitEvent(event=Shutdown(reason='Visualization window closed'))]))
    return [shutdown, publisher, backend, rviz]


def generate_launch_description():
    share = Path(get_package_share_directory('lbks_gui'))
    description = Path(get_package_share_directory('lbks_description'))
    python = str(Path(os.environ['VIRTUAL_ENV']) / 'bin/python') if os.environ.get(
        'VIRTUAL_ENV') else sys.executable
    return LaunchDescription([
        DeclareLaunchArgument('urdf_path', default_value=str(description / 'urdf/ur/ur.urdf')),
        DeclareLaunchArgument('tip_link', default_value='tool0'),
        DeclareLaunchArgument('models_config', default_value=str(share / 'config/models.yaml')),
        DeclareLaunchArgument('rviz_config', default_value=str(share / 'config/studio.rviz')),
        DeclareLaunchArgument('python_executable', default_value=python,
                              description='Python with torch and pytorch_kinematics installed'),
        OpaqueFunction(function=start),
    ])
