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
from rclpy.node import Node # node to run the ROS 2 system

from std_msgs.msg import String # this is the built-in message type that the node uses to structure 
                                # data that it passes on the topic


# Quite similar to the publisher node, the subscriber node's code implements a similar constructor
class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10) # name and message type used by publisher-subscriber must match
        self.subscription  # prevent unused variable warning

    # subscriber does not need a timer definition, because its callback gets
    # called as soon as it receives a message
    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)
        # simply print an info message into the console

# The main function is most definitely the same
# just replace with the class minimalsubscriber
def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()