#!/usr/bin/env python3
"""
A ROS2 Node used to control a differential drive robot with a camera attached to the front
It will trace and follow the line

May change the parameters based on likings and requirements
"""
import rclpy
from rclpy.node import Node
# import QoS(quality of service) for what subscriber QoS Settings are compatible with publishing QoS
# create the subscriber with the following QoS profile
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from rclpy.qos import QoSProfile

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
import numpy as np
import cv2 as cv
from cv_bridge import CvBridge
import random

from processing import crop_size, get_contour_data

# User-defined parameters, customizable
# Minimum size for a contour to be considered anything
MIN_AREA = 500

# Minimum size for a contour to be considered part of the track
# but not too small so that tracks far away (smaller contour) to be considered
MIN_AREA_TRACK = 5000

# Robot's linear speed (NOTE: should be calibrated along with KP)
LINEAR_SPEED = 0.1 # current test range 0.01 - 0.2

# IMPROVE: add linear speed if not turning, less speed when turning

# Proportional constant to be applied on speed when turning
KP = 5/1000 # current test range: 1/1000 - 5/1000

# If the line is completely lost, the error value will be compensated by
# multiplying with this loss factor to return on track
LOSS_FACTOR = 2.5

# Send messages every $TIMER_PERIOD seconds
TIMER_PERIOD = 0.06

# HSV Range of the color of the Line - BLUE
LOWER_HSV_RANGE = np.array([100, 50, 50])
UPPER_HSV_RANGE = np.array([130, 255, 255])

# QoS profile to match compatibility with the existing image publisher endpoint
img_qos = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
)

class LineFollower(Node):
    def __init__(self):
        "Initiate ROS node to run line following feature"
        super().__init__('line_follower') # init a node with a title, which will be called for spinning in the main function
        self.publisher = self.create_publisher(Twist, '/cmd_vel', rclpy.qos.qos_profile_system_default)
        self.timer = self.create_timer(TIMER_PERIOD, self.timer_callback)
        self.subscription = self.create_subscription(
            Image,
            '/image_raw', # gazebo sim topic: /camera/image_raw
            self.image_callback, # NOTE: how did my function receive the data when not specified the name or input arguments yet, lol
            img_qos)
        self.subscription  # prevent unused variable warning
        # create a bridge between ROS and OpenCV
        self.bridge = CvBridge()

        self.start_service = self.create_service(Empty, 'start_follower', self.start_follower_callback)
        self.stop_service = self.create_service(Empty, 'stop_follower', self.stop_follower_callback)
        
        # misc. elements for status checking 
        self.frame_input = 0
        self.error = 0 
        self.just_seen_line = False
        self.should_move = False

    def image_callback(self, data):
        """
        Function to receive image input and turn it into openCV format
        """
        self.frame_input = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        # method to troubleshoot if image callback function has any problem
        self.get_logger().info('nanosec: %d' %(data.header.stamp.nanosec))
        """
        cv.imshow("Original", self.frame_input)
        cv.waitKey(1)
        """

    # TODO: add service call (ROS Action) to start and stop the robot
    def start_follower_callback(self, request, response):
        self.should_move = True
        return response

    def stop_follower_callback(self, request, response):
        self.should_move= False
        return response

    def timer_callback(self):
        """
        Function to be alled when timer ticks, basically will check time as well as 
        determine the speed of the robot so it can follow the contour
        """
        if type(self.frame_input) != np.ndarray:
            return
        
        height, width, _ = self.frame_input.shape

        # Use a copy of the frame/image since findContours() alters the image
        frame = self.frame_input.copy()

        h_start, h_stop, w_start, w_stop = crop_size(height=height, width=width)

        # Get the bottom part of the image (matrix slicing)
        crop = frame[h_start: h_stop, w_start: w_stop]

        # convert from BGR to HSV
        hsv = cv.cvtColor(crop, cv.COLOR_BGR2HSV)

        # Create a binary mask, where non-zero values represent the line
        # we try to filter and isolate just the color of the line(contour) is seen
        mask = cv.inRange(hsv, LOWER_HSV_RANGE, UPPER_HSV_RANGE)

        # get the centroid of the biggest contour in the picture
        # plot its detail on the cropped part of the output image
        output = frame
        line = get_contour_data(mask, output[h_start: h_stop, w_start : w_stop], w_start=w_start)
          
        message = Twist()
        contour_data = 0

        # Check if there is a line                                                                                                                                                                          a line in the image
        if line:
            x = line['x']
            contour_data = line['contour']
            # error:= the difference between the center of the image
            # and the center of the line
            self.error = x - width //2

            message.linear.x = LINEAR_SPEED
            self.just_seen_line = True
            
            # plot the centroid in the image
            cv.circle(output, (line['x'], h_start + line['y']), 5, (0, 255, 0), 7)
        else:
            # if there is no line in the image frame
            # Turn on the spot to find it again
            if self.just_seen_line:
                self.just_seen_line = False
                self.error = self.error * LOSS_FACTOR
            message.linear.x = 0.0

        # Determine the speed to turn and get the line in the center of the camera.
        message.angular.z = float(self.error) * -KP
        print("Error: {} | Angular Z: {:.6f} | Moment Area: {}".format(self.error, message.angular.z, contour_data))

        # plot the boundaries where the image was cropped
        cv.rectangle(output, (w_start, h_start), (w_stop, h_stop), (0, 0, 255), 2)
        """
        # this will have the image window fixed at a location, so you can't drag it anywhere
        cv.namedWindow("Processed", cv.WINDOW_NORMAL)
        cv.moveWindow("Processed", current_processed.shape[1] + 100, 0)
        """
        cv.imshow("Processed", output)
        cv.waitKey(1)

        # Publish the message to `/cmd_vel`
        if self.should_move:
            self.publisher.publish(message)
        else:
            empty_message = Twist()
            self.publisher.publish(empty_message)

    def wander(self):
        msg = Twist()
        # Set random angular velocity (turning motion)
        msg.angular.z = random.uniform(-1.0, 1.0)
        
        # TODO: add the random wandering to the robot message
        return 
    
def main(args=None):
    rclpy.init(args=args)
    line_follower = LineFollower()
    rclpy.spin(line_follower)

    # Destory the node explicityly
    # (optional - otherwise it will be done automatically)
    # when the garbage collector destroys the node object
    line_follower.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()