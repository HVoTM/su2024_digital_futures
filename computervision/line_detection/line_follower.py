#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import cv_bridge

# create a bridge between ROS and OpenCV
bridge = cv_bridge.CvBridge()


class LineFollower:
    def __init__(self):
        rclpy.init_node('line_follower')
        rclpy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.cmd_vel_pub = rclpy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.twist = Twist()

    def scan_callback(self, scan_msg):
        # Process sensor data and determine robot's movement
        # Adjust self.twist.linear.x and self.twist.angular.z accordingly
        # Example: Follow a line using proportional control
        error = scan_msg# Calculate error based on sensor data
        self.twist.linear.x = 0.2  # Forward speed
        self.twist.angular.z = 0.5 * error  # Proportional control

    def run(self):
        rate = rclpy.Rate(10)  # 10 Hz
        while not rclpy.is_shutdown():
            self.cmd_vel_pub.publish(self.twist)
            rate.sleep()

    def check_shift_turn(angle, shift):
        turn_state = 0 # meaning it's going straight

    def get_turn(turn_state, shift_state):
        pass

    def self_driving():
        pass

if __name__ == '__main__':
    try:
        follower = LineFollower()
        follower.run()
    except rclpy.ROSInterruptException:
        pass
