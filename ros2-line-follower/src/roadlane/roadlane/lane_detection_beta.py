# Import necessary libraries
# Computer vision scripts
from advanced import LaneDetectionbyFrame, UnpackPickle
from hough import LaneDetectionPolyFit

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

"Hyperparameters and Constants"
# Robot's linear speed (NOTE: should be calibrated along with KP)
LINEAR_SPEED = 0.1 # current test range 0.01 - 0.2

# Proportional constant to be applied on speed when turning
KP = 1.5/1000 # current test range: 1/1000 - 5/1000

# If the line is completely lost, the error value will be compensated by
# multiplying with this loss factor to return on track
LOSS_FACTOR = 2.5

# Send messages every $TIMER_PERIOD seconds
TIMER_PERIOD = 0.06
# Rows and columns for region of interest
"""
bottom_left = [cols, rows]
top_left = [cols, rows * 0.5]
top_right = [cols, rows * 0.5]
bottom_right = [cols, rows]
"""

# QoS profile to match compatibility with the existing image publisher endpoint
img_qos = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
)

class LaneDetector(Node):
    def __init__(self):
        "Initiate ROS2 node to run road lane detection feature"
        super().__init__('lane_detector') 
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )   
        self.subscription # prevent unused variable warning
        self.image = 0 # initialize image frame attribute
        self.publisher = self.create_publisher(Twist, '/cmd_vel', rclpy.qos.qos_profile_system_default) # TODO: add publisher node
        self.timer = self.create_timer(timer_period_sec=TIMER_PERIOD, callback=self.timer_callback_hough)
        self.bridge = CvBridge()
        # self.mtx, self.dist = UnpackPickle()
        self.error = 0

    def image_callback(self, data):
        """
        Function to receive image input and turn it into openCV format
        """
        self.image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        # method to troubleshoot if image callback function has any problem
        self.get_logger().info('nanosec: %d' %(data.header.stamp.nanosec))
        """
        cv.imshow("Original", self.frame_input)
        cv.waitKey(1)
        """

    def timer_callback_hough(self):
        """
        Timer callback function used to process and decide the lane using basic Hough methods
        """
        if (type(self.image)) != np.ndarray:
            return
        
        processed, lane_center = LaneDetectionPolyFit(self.image)
        print("Lane Center x-coordinate:", lane_center)

        # TODO: handle if lane_center = 0 with LOss factor or such
        height, width, _ = self.image.shape

        message = Twist()
        
        self.error = lane_center - width//2
        message.linear.x = LINEAR_SPEED
        message.angular.z = float(self.error) * -KP
        print("Error: {} | Angular Z: {:.6f}".format(self.error, message.angular.z))
        
        self.publisher.publish(message)

        cv.imshow("Processed image", processed)
        cv.waitKey(1)    

    def timer_callback_advanced(self):
        """
        Timer function to process and actuate the decisions of road lane detection
        """
        if (type(self.image)) != np.ndarray:
            return
        
        image_copy = self.image.copy()

        processed, left_x, right_x = LaneDetectionbyFrame(image_copy, self.mtx, self.dist)
        print(left_x, right_x)
        height, width, _ = self.image.shape

        # publishing the differential drive wheel actuator
        # (rewrite the comment for readability later)
        message = Twist()
        if left_x > width // 2:
            # if both lines are skewed to the right
            # have the motor turn left a lil bit to recapture and redefine the correct line
            pass
        elif right_x < width // 2:
            # vice versa
            pass
        else:
            # this is the normal case when both lines are well separated on 2 sides
            self.error = (left_x + right_x) // 2 - width//2
        message.linear.x = LINEAR_SPEED
        message.angular.z = float(self.error) * -KP
        print("Error: {} | Angular Z: {:.6f}".format(self.error, message.angular.z))
        
        self.publisher.publish(message)

        cv.imshow("Processed image", processed)
        cv.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    laner = LaneDetector()
    rclpy.spin(laner)
    # Destory the node explicityly
    # (optional - otherwise it will be done automatically)
    # when the garbage collector destroys the node object
    laner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
