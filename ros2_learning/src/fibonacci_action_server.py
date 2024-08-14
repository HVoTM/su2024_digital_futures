import time
import rclpy

from rclpy.action import ActionServer # import action API from rclpy
from rclpy.node import Node

from action_tutorials_interfaces.action import Fibonacci


class FibonacciActionServer(Node):

    def __init__(self):
        super().__init__('fibonacci_action_server')
        """
        Action server requires 4 arguments:
        1. ROS 2 node to add to the action client to self
        2. The type of action (defined by another interface package)
        3. Action name
        4. callback function for executing accepted goals, **must** return a result message for the action type
        """
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            self.execute_callback)
    """
    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        # Execute the process to return goal result
        sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            sequence.append(sequence[i] + sequence[i-1])

        # need to set the goal_handle state in order to execute goal request
        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = sequence
        return result
    """
    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        # update message as feedback
        feedback_msg = Fibonacci.Feedback()
        feedback_msg.partial_sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            feedback_msg.partial_sequence.append(
                feedback_msg.partial_sequence[i] + feedback_msg.partial_sequence[i-1])
            self.get_logger().info('Feedback: {0}'.format(feedback_msg.partial_sequence))
            goal_handle.publish_feedback(feedback_msg)
            time.sleep(1)

        goal_handle.succeed()

        result = Fibonacci.Result()
        result.sequence = feedback_msg.partial_sequence
        return result

def main(args=None):
    rclpy.init(args=args)

    fibonacci_action_server = FibonacciActionServer()

    rclpy.spin(fibonacci_action_server)


if __name__ == '__main__':
    main()