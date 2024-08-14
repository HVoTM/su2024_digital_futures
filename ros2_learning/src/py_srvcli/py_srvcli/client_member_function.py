# client node uses sys.argv to get access to command line input arguments for the request
import sys 

from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node


class MinimalClientAsync(Node):

    def __init__(self):
        # Similarly, use parent class's constructor for referential init.
        super().__init__('minimal_client_async')
        # Create a client with the same type and name as the service node
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        
        # The loop checks if a service matching the type and name of a client
        # is available for a second that we set
        # Checks the 'future' to see if there is a response
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        "Request definition followed by main"
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        # NOTE: check the rclpy library
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()


def main(args=None):
    rclpy.init(args=args)

    minimal_client = MinimalClientAsync()
    response = minimal_client.send_request(int(sys.argv[1]), int(sys.argv[2]))
    minimal_client.get_logger().info(
        'Result of add_two_ints: for %d + %d = %d' %
        (int(sys.argv[1]), int(sys.argv[2]), response.sum))

    minimal_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()