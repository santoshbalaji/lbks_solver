from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution, FindExecutable
from launch_ros.actions import Node
from launch.conditions import IfCondition
from launch_ros.substitutions import FindPackageShare
from launch.actions import SetEnvironmentVariable
 
 
def generate_launch_description():
    # Define filenames
    urdf_package = 'lbks_description'
    urdf_filename = 'iiwa7_r800.xacro'
    rviz_config_filename = 'robot_visualizer.rviz'
 
    # Set paths to important files
    pkg_share_description = FindPackageShare(urdf_package)
    default_urdf_model_path = PathJoinSubstitution(
        [pkg_share_description, 'urdf', 'kuka', urdf_filename])
    default_rviz_config_path = PathJoinSubstitution(
        [pkg_share_description, 'rviz', rviz_config_filename])
 
    # Launch configuration variables
    rviz_config_file = LaunchConfiguration('rviz_config_file')
    urdf_model = LaunchConfiguration('urdf_model')
    use_sim_time = LaunchConfiguration('use_sim_time')

    # Declare the launch arguments 
    declare_rviz_config_file_cmd = DeclareLaunchArgument(
        name='rviz_config_file',
        default_value=default_rviz_config_path,
        description='Full path to the RVIZ config file to use')

    declare_urdf_model_path_cmd = DeclareLaunchArgument(
        name='urdf_model',
        default_value=default_urdf_model_path,
        description='Absolute path to robot urdf file')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        name='use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true')

    declare_use_jtp_cmd = DeclareLaunchArgument(
        'use_jtp',
        default_value='true',
        description='Whether to run my_node or not')


    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            urdf_model,
            " ",
            "sim_gazebo:=false",
            " ",
            "sim_ignition:=false"
        ])

    env_rcutils_console_output_format = SetEnvironmentVariable(
        name='RCUTILS_CONSOLE_OUTPUT_FORMAT',
        value='[{severity}] {message}')

    env_rcutils_colorized_output = SetEnvironmentVariable(
        name='RCUTILS_COLORIZED_OUTPUT',
        value='1')

    # Subscribe to the joint states of the robot, and publish the 3D pose of each link.
    start_robot_state_publisher_cmd = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': robot_description_content}])

    # Depending on gui parameter, either launch joint_state_publisher or joint_state_publisher_gui
    start_joint_state_publisher_gui_cmd = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(LaunchConfiguration('use_jtp')))

    # Launch RViz
    start_rviz_cmd = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file],
        parameters=[{'use_sim_time': use_sim_time,
                     'robot_description': robot_description_content,
                     'robot_description_semantic': robot_description_content}])

    # Create the launch description and populate
    ld = LaunchDescription()

    # Declare the launch options
    ld.add_action(declare_rviz_config_file_cmd)
    ld.add_action(declare_urdf_model_path_cmd)
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_use_jtp_cmd)

    # Add any actions
    ld.add_action(env_rcutils_console_output_format)
    ld.add_action(env_rcutils_colorized_output)
    ld.add_action(start_joint_state_publisher_gui_cmd)
    ld.add_action(start_robot_state_publisher_cmd)
    ld.add_action(start_rviz_cmd)

    return ld