"""Start the local web GUI using the training environment's Python."""

import os
from pathlib import Path
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration


def start(context):
    def value(name):
        return LaunchConfiguration(name).perform(context)

    command = [value('python_executable'), '-m', 'lbks_gui.server',
               '--port', value('port'), '--urdf-path', value('urdf_path'),
               '--models-config', value('models_config'), '--tip-link', value('tip_link'),
               '--assets', get_package_share_directory('lbks_description')]
    if value('open_browser').lower() == 'true':
        command.append('--open-browser')
    return [ExecuteProcess(cmd=command, output='screen')]


def generate_launch_description():
    share = Path(get_package_share_directory('lbks_gui'))
    description = Path(get_package_share_directory('lbks_description'))
    python = str(Path(os.environ['VIRTUAL_ENV']) / 'bin/python') if os.environ.get(
        'VIRTUAL_ENV') else sys.executable
    return LaunchDescription([
        DeclareLaunchArgument('python_executable', default_value=python),
        DeclareLaunchArgument('port', default_value='8080'),
        DeclareLaunchArgument('open_browser', default_value='true'),
        DeclareLaunchArgument('urdf_path', default_value=str(description / 'urdf/ur/ur.urdf')),
        DeclareLaunchArgument('models_config', default_value=str(share / 'config/models.yaml')),
        DeclareLaunchArgument('tip_link', default_value='tool0'),
        OpaqueFunction(function=start),
    ])
