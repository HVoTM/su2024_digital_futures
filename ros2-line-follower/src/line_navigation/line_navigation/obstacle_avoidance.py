import rclpy
from rclpy import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import math
import numpy as np

# Parameters
# Distance threshold
DISTANCE_THRESHOLD = 0.7

"Define the ROS 2 obstacle avoidance Node for Turtlebot"
class ObstacleAvoidance(Node):
    def __init__(self):
        super().__init__('obstacle avoider?') # initialize the base ROS node with obstacle avoidance title
        self.subscription = self.create_subscription(
            LaserScan, 
            '/scan', 
            self.laser_callback,
            rclpy.qos.qos_profile_system_default,
            queue_size=10,)
        self.subscription # prevent unused variable warning
        self.publisher = self.create_publisher(Twist, '/cmd_vel', rclpy.qos.qos_profile_system_default)
        
        # Minimum distance to obstacle
        self.min_dist = float('inf')

    def laser_callback(self, scan_data):
        "Callback function to process laser scan data and detect obstacles"
       # Convert laser scan to numpy array for easier manipulation
        ranges = np.array(scan_data.ranges)
        
        # Extract the closest obstacle within a defined range
        range_min_index = int(len(ranges) / 3)
        range_max_index = len(ranges) - range_min_index
        
        self.min_dist = np.min(ranges[range_min_index:range_max_index])

    def avoid_obstacle(self):
        # Initialize a Twist message
        message = Twist()

        # Set linear velocity (forward motion)
        message.linear.x = 0.2  # Constant forward velocity

        # Adjust angular velocity (turning) based on the distance to the nearest obstacle
        if self.min_dist < DISTANCE_THRESHOLD:
            # Calculate the angle to turn based on the distance to the obstacle
            angle = math.atan2(1.0, DISTANCE_THRESHOLD)
            
            # Set angular velocity (turning) to avoid the obstacle
            message.angular.z = np.clip(angle, -1.0, 1.0)
        else:
            # No obstacle detected within threshold, continue moving forward
            message.angular.z = 0.0

        # Publish the Twist message to control the TurtleBot3
        self.publisher.publish(message)

def main(args=None):
    rclpy.init(args=args)
    obstacler = ObstacleAvoidance()
    rclpy.spin(obstacler)

    # Destory the node explicityly
    # (optional - otherwise it will be done automatically)
    # when the garbage collector destroys the node object
    obstacler.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()