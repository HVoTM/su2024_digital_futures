# robot_description.launch.py

import os
from ament_index_python.packages import get_package_share_directory
import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, ThisLaunchFileDir, Command
from launch_ros.actions import Node


def generate_launch_description():
    xacro_file = "tortoisebot_hook.xacro"
    package_description = "tortoisebot_description"
    robot_description_path = os.path.join(
        get_package_share_directory(package_description), "models", "urdf", xacro_file
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "xacro_file", description="Absolute path to the Xacro file"
            ),
            # Convert Xacro to URDF
            ExecuteProcess(
                cmd=["xacro", LaunchConfiguration("xacro_file")],
                output="screen",
                shell=True,
            ),
            # Launch robot_state_publisher
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                output="screen",
                namespace="my_robot",
                parameters=[
                    {
                        "use_sim_time": False,
                        "robot_description": Command(
                            [
                                "xacro ",
                                robot_description_path,
                                " robot_name:=tortoisebo_hook",
                            ]
                        ),
                    }
                ],  # Set to True if using simulation time
                arguments=[LaunchConfiguration("xacro_file")],
            ),
            # Launch RViz
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                namespace="my_robot",
                parameters=[
                    {"use_sim_time": False}
                ],  # Set to True if using simulation time
                # condition=IfCondition(LaunchConfiguration("gui")),
            ),
        ]
    )
