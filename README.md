# Concepts and areas of Robotics & Autonomous Vehicle
## Autonomous Vehicle
> The goal of this project is to develop an autonomous wheelchair that can

- goes around a defined course (maybe using lines of tape) (using a webcam)
- be able to detect people with specific classifications (old person, disabled, ...) to help
- also detect to avoid obstacles and visualize the environment around (Lidar, Camera, Depth camera sensor)

We will be working on the algorithms to detect line and actuate line following on the ROS 2 Foxy model, specifically the TurtleBot3. Some concepts to understand: SLAM, Navigation, Object detection & Avoidance, Line Following, Sensors (LiDAR, Cameras, ...)
<br>
Some topics we should get into are OpenCV, Line Detection algorithm and Line Follower Implementation, ROS 2: working with ROS2 workspace, development, and continuous development, ...

For further resources, go to <p style="color: orange;  background-color:white; padding:10px">**reference.txt**</p>

## Autonomous driving with lane-detection
in `/computervision/road_lane_detection`
- Traditional method: Canny Edge Detection with Hough Transform
- Advanced on traditional method: camera calibration with distortion and stuff
- CNN-based approach

## Line Follower
- Using Contour Moment to follow an HSV color range and using `/cmd_vel` to move the robot around

## PID Controller for smoother mobility

## Hand Gestures Recognition (todo)

## SLAM/Localization (todo)

## LiDAR obstacle avoidance (and also other sensors as well) (todo)

## Pose Estimation (todo)

## Path Planning (todo)

## Kalman Filter (todo)

# ROS 2 (https://docs.ros.org/en/foxy/Tutorials.html)
- [x] Linux commands
    - <div style="color:orange">/linux_intro</div> 
- [x] working with ROS 2 using CLI tools (Linux, rqt, colcon, 
    - <div style="color:orange">/ros2_linux</div>
- [x] Gazenbo simulator: world building and simulations on robots
- [x] Working with Client libraries (/src, nodes, rclpy,...)
- [x] Intermediate ROS: dependencies, action, launch, testing, etc
- [x] Advanced ROS: topic statistics, simulators


# Computer Vision & Image Processing
- OpenCV crashcourse
- Textbook: __*Computer Vision: Algorithms and Applications*__ by Szeliski

## Image processing
- Computer Graphics concepts
- Gaussian Blur/filter for noise reduction
- Contour moments and selection
- Canny Edge Detection
- Hough Transform
- HSL, HSV Color masking

## Computer vision: Convolutional Neural Network Approach

# Hardware and Electronics
## Raspberry Pi
Everything with Raspberry Pi documentation
https://www.raspberrypi.com/documentation/

## Jetson Nano (getting started: https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit)

# Data Analysis
- A great textbook is __*Introduction to Statistical Learning in Python*__
- Supported Vector Machine: classification, regression, outlier detection
    - Further statistical models that we can get into later on
- Record data using a service or topic in ROS for data acquisition