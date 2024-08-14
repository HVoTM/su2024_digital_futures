# Import dependencies 
from example_interfaces.srv import AddTwoInts # service type
# NOTE: Look into example_interfaces
import rclpy
from rclpy.node import Node


class MinimalService(Node):

    def __init__(self):
        # same with other publisher-subscriber, refer to parent class's constructor
        super().__init__('minimal_service')
        # Create the service with the built-in node methods
        # defines type, name, and callback function
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)
    
    def add_two_ints_callback(self, request, response):
        "Receives the request data, sums it, and returns the sum as a response"
        response.sum = request.a + request.b
        self.get_logger().info('Incoming request\na: %d b: %d' % (request.a, request.b))

        return response


def main(args=None):
    "The main class initializes the ROS2 python client library, like pygame"
    rclpy.init(args=args)
    # instantiates MinimalService class to create the service node
    minimal_service = MinimalService()

    rclpy.spin(minimal_service)

    rclpy.shutdown()


if __name__ == '__main__':
    main()