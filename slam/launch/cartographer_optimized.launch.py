import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config'))


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('configuration_directory', default_value=CONFIG_DIR),
        DeclareLaunchArgument('configuration_basename', default_value='rplidar_2d_optimized.lua'),

        Node(
            package='cartographer_ros',
            executable='cartographer_node',
            name='cartographer_node',
            output='screen',
            parameters=[{'use_sim_time': False}],
            arguments=[
                '-configuration_directory', LaunchConfiguration('configuration_directory'),
                '-configuration_basename', LaunchConfiguration('configuration_basename')
            ],
            remappings=[
                ('scan', '/scan'),
                ('odom', '/odom')
            ]
        ),

        Node(
            package='cartographer_ros',
            executable='cartographer_occupancy_grid_node',
            name='cartographer_occupancy_grid_node',
            output='screen',
            parameters=[{'use_sim_time': False}],
            arguments=['-resolution', '0.05']
        ),
    ])
