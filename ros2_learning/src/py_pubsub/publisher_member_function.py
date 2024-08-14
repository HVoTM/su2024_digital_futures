	# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node

from std_msgs.msg import String


class MinimalPublisher(Node):

    def __init__(self):
        # super().__init__ to call Node class's constructor
        super().__init__('minimal_publisher')

        # attribute publisher
        # call with another method of Node to declare that the node publishes messages of type String
        self.publisher_ = self.create_publisher(String, 'topic', 10)

        # attribute timer with a callback
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback) 
        
        # a counter used in the call back
        self.i = 0

    # timer_callback creates a message with the counter value appended
    # and publishes it to the console with get_logger().info 
    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

def main(args=None):
    # 1. first the rclpy is initialized
    rclpy.init(args=args)

    # 2. A node is created
    minimal_publisher = MinimalPublisher()

    # It "spins" the node so its callbacks are called
    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()