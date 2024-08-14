# ROS2 - Foxy distribution 
## 1. Beginner: CLI tools
### 1.1. COnfiguring environments
- `source /opt/ros/foxy/setup,bash`: source the setup files
- `echo "source /opt/ros/foxy/setup,bash"`: add sourcing to the shell startup script
    - which also works with other bash command line or help set environment variables in new terminal

- `printenv | grep <name>` : to check environment variable with that name
- ROS_DOMAIN_ID
- ROS_LOCALHOST_ONLY

### 1.2. Using turtlesim, ros2, rqt
- `sudo apt update`: to check the updates on the available packages
- `sudo apt install <package_name>`: install package name
- `ros2 pkg executables <package_name>`
- `ros2 run <package_name> <executable_name>`: to run a node
- `ros2 node list` : using list subcommand to see the associated types of nodes
- `ros2 topic list`
- `ros2 service list`
- `ros2 action list`
- `rqt`: is a graphical user interface (GUI) tool for ROS 2. Everything done in rqt can be done on the command line, but rqt provides a more user-friendly way to manipulate ROS 2 elements.
    + plugins, services, topic monitors, bla bla
    + Use a service, update expressions for request and use "call" to run the service
- remapping: a command to reassign default node properties to custom values

### 1.3. Understanding nodes
- ROS graphs, ROS nodes, services, publishers, subscribers, actions, topics, parameters
- `ros2 node info \<insert_node_name>` : to return a list of subscribers, publishers, services, actions.

### 1.4. Understanding topics
> method of communication based on publisher-subscribers
> may be one-to-one, one-to-many, many-to-many, many-to-one
> publish data and continually updating info on the topics
- `rqt_graph`: use this to understand more about the topics being run in the overall ROS graph
- `ros2 topic list`
- `ros2 topic list -t`
- TO see data being published on a topic, use: `ros2 topic echo <topic_name>`
- To look into the type of communication (one-to-one, many-to-one, etc.): `ros2 topic info <topci_name>`
- `ros2 interface show <msg type>` to show the structure of the data of from topic echo above.
- `ros2 topic pub <topic_name> <msg_type> '<args>'` to publish data onto a topic directly from the command line.
    + --once: publish one time then exit
    + --rate 1: option to publish a command at a steady stream of 1Hz (yes, we can change the parameter of the speed)
- `ros2 topic hz <topic_name>`: to view the rate at which the data is published

### 1.4. Understanding services
> a method of communication based on a call-and-response model versus the publisher-subscriber of topics
- `ros2 service type <service_name>`
    + descibe the request and response data of a service
- `ros2 service list -t`(--show-types abbreviated as *-t* ): to display the active services at the same time
- To find all the services of a specific type: `ros2 service find <type_name>`, compared to service type
- `ros2 interface show <type_name>.srv` - is used to know the structure of the input arguments for the service call.
- `ros service call <service_name> <service_type> <arguments>`: used to call a service, (arguments is optional)

### 1.5. Understanding parameters
> Configuration value of a *ROS node* - like node settings - store as integers, floats, booleans, strings, and lists.
- `ros2 param list`
- `ros2 param get <node_name> <parameter_name>`: display the type and current value of a parameter.
- `ros2 param set <node_name> <parameter_name> <value>`: to change a parameter at runtime
- `ros2 param dump <node_name>`: to save all of a node's current parameter values into a file
- `ros2 param load <node_name> <parameter_file>`: to load parameters from a file to a currently running node
- `ros2 run <package_name> <executable_name> --ros-args --params-file <file_name>`: load parameter file on node startup
    + e.g.: <div style="color: magenta">ros2 run turtlesim turtlesim_node --ros-args --params-file ./turtlesim.yaml</div>

### 1.6. Understanding actions
> One of the communication types in ROS 2 and are intended for long running tasks. 
- `ros2 node info <node_name>`: provided the name of the node executed, it will return a list of the node's subscribers, publishers, services, action servers and action clients.

- `ros2 action list`: idenfify all the actions in the ROS graph
    + `ros2 action list -t`: return the actions with the action types.
    + the action type will be needed for an execution from the command line or from code

- `ros2 action info <action_name>`: introspect the action to see which node is the action client and which node is the action server

- `ros2 interface show <action_type>`: given the action type, use it to get information about the structure of the action type.
    + **NOTE**: you can refer `ros2 action list -t` to get the type of the action

- `ros2 action send_goal <action_name> <action_type> <values>`: send an actioin goal from the command line
    + *<values>* needs to be in YAML format
    + add **--feedback** to the end of the command to see the feedback of this goal.

### 1.7. Using rqt_console to view logs
> rqt_console can be very helpful if you need to closely examine the log messages from your system.
- `ros2 run rqt_console rqt_console`: start in a new terminal to run a *rqt_console*..
- Logger levels:
    + Fatal
    + Error
    + Warn
    + Info
    + Debug

- `ros2 run <package_name> <executable_name> --ros-args --log-level <logger_level>`: set the default logger level for the node you want 

### 1.8. Launching nodes
> Launch files allow you to start up and configure a number of executables containing ROS 2 nodes simultaneously.
- More complex systems will mean more and more nodes running simultaneously, so opening terminals and reentering configuration details becomes tedious.

- `ros2 launch <package_name> <launch_file>`    

### 1.9. Recording and playing back data
- "ros2 bag": command line tool for recording data published on topics in your system.
- ros2 bag can only record data from published messages in topics. (use `ros2 topic list` to see the list of topics.)

1. To see the data that a topic is publishing, run command: `ros2 topic echo <topic_name>`
2. Use command syntax to record the data `ros2 bag record <topic_name>`
3. CTRL+C to stop recording

- To record multiple topics: `ros2 bag record -o subset <service_1> <service_2> <so_on>`
    + -o: allows you to choose a unique name for your bag file; in this case, it's "subset"

- `ros2 bag info <bag_file_name>`: see details about the recording
- `ros2 bag play <bag_file_name>`: to replay the bag file

