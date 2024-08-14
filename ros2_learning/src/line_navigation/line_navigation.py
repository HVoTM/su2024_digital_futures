import rclpy
import rclpy.exceptions
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2 as cv
# Create a bridge between ROS And OpenCV
from cv_bridge import CvBridge
import numpy as np

"CONSTANTS"
# Threshold to view a contour to be part of the track
MIN_AREA_TRACK = 500
# Robot Speed when following the line
LINEAR_SPEED = 0.2
# Proportional constant to be applied on speed when turning
# (multiplied by the error value)
KP = 1.5/100


class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw', # NOTE: find the correct ROS topic
            self.listener_callback,
            10
        )
        self.subscription # prevent unused variable warning
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

    def listener_callback(self, data):
        # Convert ROS Image message to OpenCV image
        current_frame = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        
        # BUG CHECK: use this to test the image input first then go to preprocessing
        # cv.imshow('Camera Feed', current_frame)
        # cv.waitkey(1)

        # convert BGR to HSV
        hsv_image = cv.cvtColor(current_frame, cv.COLOR_BGR2HSV)

        # Define the range of blue color in HSV
        # NOTE: change to the specific color range you like
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])

        # Create a binary mask
        blue_mask = cv.inRange(hsv_image, lower_blue, upper_blue)

        # Apply the mask to the original image
        blue_segmented_image = cv.bitwise_and(current_frame, current_frame, mask=blue_mask)

        # Detect line and get its centroid
        line = self.get_contour_data(blue_mask)

        # COMMAND TO RUN THE WHEELS: Move depending on detection
        cmd = Twist()
        _, width, _ = blue_segmented_image.shape

        """
        Code snippet to work around with controlling the actuators
        """
        # Display the segmented image with line centroid
        if line:
            x = line['x']

            error = x - width // 2

            cmd.linear.x = LINEAR_SPEED
            cv.circle(blue_segmented_image, (line['x'], line['y']), 5, (0, 0, 255), 7)
        
        # Determine the speed to turn and get the line in the center of the camera.
        cmd.angular.z = float(error) * -KP
        print("Error: {} | Angular Z: {}, ".format(error, cmd.angular.z))

        # Send the command to execute
        self.publisher.publish(cmd)       

        # NOTE: Display the segmented image
        cv.imshow("Blue Segmented Image", blue_segmented_image)
        cv.waitKey(1)

    # Algorithm to calculate contour data
    def get_contour_data(self, mask):
        """
        Return the centroid of the largest contour in the binary image 'mask'
        """
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        line = {}

        for contour in contours:
            M = cv.moments(contour)

            if (M['m00'] > MIN_AREA_TRACK):
                # contour is part of the track
                line['x'] = int(M["m10"] / M["m00"])
                line['y'] = int(M["m01"] / M["m00"])
        return (line)
    
    def crop_size(height, width):
        """
        Get measures to crop the image
        output:
        (height_upper, height_lower, width_left, width_right)
        """
        # TODO: get to cropping image for perimeter control for image processing
        return (1*height//3, height, width//4, 3*width//4)
    

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    try: 
        main()
    except(KeyboardInterrupt, rclpy.exceptions.ROSInterruptException):
        empty_message = Twist()
        # TODO: we will get to this later for exception handling
