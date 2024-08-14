# In this documentation, we will learn about how ROS2 works in the programmer side, specifically how to work with the client libraries

## colcon
==colcon== is an iteration on the ROS build tools *catkin_make*, *catkin_make_isolated*, *catkin_tools*, and *ament_tools*.

- Installing colcon: `sudo apt install python3-colcon-common-extensions`
- <div style="color:yellow">src, build, install, log</div> will be the core directories of a ROS packages.

### Create a workspace
1. `mkdir -p ~/ros2_ws/src` to create a workspace with the src directory
2. `cd ~/ros2_ws/`
3. start to write some code, add source, bla bla.

## Underlay - overlay
__*Source an underlay*__:
Basically, underlay will be the main development environment for a ROS2 package to run, most of the time will be the existing ROS2 Installation that we had first installed. If you do not have an underlay for every time you run a ROS package, it will not be able to run.

We will need to __source__ the setup script provided via your installation in order to have an underlay.

## Creating a ROS2 Workspace
1. Source ROS2 environment
`source /opt/ros/foxy/setup.bash`

2. Create a new directory for every new workspace
`mkdir -p ~/<workspace_name>/src`

3. Create your own packages, or clone an existing packages on git repo
- Clone using `git clone`

4. Resolve dependencies
- From the root of your workspace (remember to *cd ..* out of the current source folder)
- Using **rosdep** commands

5. Build the workspace with colcon
- From the root of the workspace, build the package using command `colcon build`
> **NOTE**
Other useful arguments for colcon build:

<div style="color: orange">--packages-up-to</div> builds the package you want, plus all its dependencies, but not the whole workspace (saves time)

<div style="color: orange">--symlink-install</div> saves you from having to rebuild every time you tweak python scripts

<div style="color: orange">--event-handlers console_direct+</div> shows console output while building (can otherwise be found in the log directory)

6. Optional: sourcing the overlay
> ADd new packages without interfering with the existing ROS 2 workspace - underlay
**NOTE**:
- Open a new terminal separating from the one used for building the workspace, or such
- In the new terminal, source the main ROS 2 environment as the "underlay"
- Go into the root of the workspace
- SOurce the overlay (e.g. `source install/local_setup.bash`)

## Creating a ROS 2 package
1. Make sure to be in the <div style="color:magenta">src</div> folder before running the package creation command. e.g. `cd ~/ros2_ws/src`
2. For Python, the syntax to create a new package: `ros2 pkg create --build-type ament_python <package_name>`
    - **--node-name**: optional argument
3. Build a package: `colcon build --packages-select <package_name>` to build only the packages you want

4. Source the setup files: underlay and overlay, if needed
5. USe the package
`ros2 run <package_name> <node_name>

6. Examine package contents and customize package setup files.
Look up [here](https://docs.ros.org/en/foxy/Tutorials/Beginner-Client-Libraries/Creating-Your-First-ROS2-Package.html#customize-package-xml)
