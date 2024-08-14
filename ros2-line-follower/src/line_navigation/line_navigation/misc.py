#!/user/bin/env python
import rclpy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import random
import math
import numpy as np

"A wandering algorithm to have the robot running aimlessly if idling?"
def wander():
    msg = Twist()
    # preset a linear velocity
    msg.linear.x = 1.4

    # Set randomized angular velocity (turning motion)
    msg.angular.z = random.uniform(-1.0, 1.0)

    return msg

def avoid_obstacle():
    pass

