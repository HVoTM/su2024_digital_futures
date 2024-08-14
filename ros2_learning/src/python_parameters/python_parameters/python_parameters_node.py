import rclpy
import rclpy.node

class MinimalParam(rclpy.node.Node):
    def __init__(self):
        super().__init__('minimal_param_node')

        # create a parameter with the name 'my_parameter' and a default value of 'world'
        self.declare_parameter('my_parameter', 'world')

        # the timer is initialized with a period of 1
        self.timer = self.create_timer(1, self.timer_callback)

    def timer_callback(self):
        # get the parameter from the node and stores it in my_param
        my_param = self.get_parameter('my_parameter').get_parameter_value().string_value

        # get_logger() ensures the event is logged
        self.get_logger().info('Hello %s!' % my_param)

        my_new_param = rclpy.parameter.Parameter(
            'my_parameter',
            rclpy.Parameter.Type.STRING,
            'world'
        )
        all_new_parameters = [my_new_param]
        # set_parameters sets my_parameter back to the default string value 'world'
        self.set_parameters(all_new_parameters)

def main():
    rclpy.init()
    node = MinimalParam()
    rclpy.spin(node)

if __name__ == '__main__':
    main()