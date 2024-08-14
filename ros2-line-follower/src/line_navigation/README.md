# Running the Line Following
## On Gazebo
- Bringup the simulation world:
`ros2 launch line_navigation new_track.launch.py`
- Run the line following algorithm:
`ros2 run line_navigation follower_w_visual`
- Send ROS service action to start following the line
`ros2 service call /start_follower std_srvs/srv/Empty`
- Pause following
`ros2 service_call /stop_follower std_srvs/srv/Empty`

## On Turtlebot
- Bring up the necessary robot bring up on SSH terminal (REMEMBER: **linefollowing algorithm from remote PC to turtlebot(raspberry pi) does not work at the moment, copy the package to the SD card and run it on the raspi for now**)
`ros2 launch turtlebot3_bringup robot.launch.py`
`ros2 run usb_cam usb_cam_node_exe`
- Start the line following algorithm
`ros2 run line_navigation follower_node`
- Send ROS Service action to start following the line
`ros2 service call /start_follower std_srvs/srv/Empty`
- Pause following
`ros2 service_call /stop_follower std_srvs/srv/Empty`

# HSV Color range:
- Blue:
    - lower = [100. 50. 50]
    - upper = [130, 255, 255]
- Orange: 
    - lower = [0, 100, 100]
    - upper = [30, 255, 255]
- Red:
    - lower = [0, 100, 100]
    - upper = [10, 255, 255]
- Red (additional range due to red being spread out in the HSV wheel):
    - lower = [170, 100, 100]
    - upper = [180, 255, 255]

- Black, Green, White
# Line following
- [] Gaussian blur for image processing in the the line navigation
- [x] Start/stop service for ROS action for line following
- [x] Define the better HSV Color range for: ORange, white, green, red
    +  Interfaces or ROS Action to call for differrent color choosing
- [] Include launch file for usb-cam in turtlebot3_bringup robot.launch.py
- [x] Extra code script to use for gazebo simulations and real testing with the TurtleBot3 -> add required ROS nodes
- [] Add a randomly wandering and choose on any color to follow node
- [] Obstacle avoidance node using LiDAR
## Improve for the camera image to be in line with the turtlebot center
- PID controller parameters and constants to be changed (https://en.wikipedia.org/wiki/Proportional%E2%80%93integral%E2%80%93derivative_controller)
- Offset on the camera with the center of the differential drive since the positions are quite different
    + *NOTE*: Currently solved by angling the camera looking downwards, about 30-45 degrees from the vertical axis
- Time delay to have the camera frame to be in conjunction with the position of the turtlebot
    + *NOTE*: possibly not in need of that, but a great idea to re-track if needed

## Data streaming and communication protocols
- DDS communication 
- ROS bag

## Upgrade with other features
- Obstacle avoidance with Vector Field Histogram in LiDAR
- Wandering algorithm
- multiple (ordered, possibly) color detection for course change
- Road lane detection (+ extrapolation for prediction of the line) and center the motion of travel of the 