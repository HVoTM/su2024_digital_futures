# 0. Methodology and practices for testing robots
1. Define your test objectives
    - What you want to achieve, measure, and evaluate
    - Aligned with robot's specifications, requirements, and functionalities

2. Choose the right test methods
    - Unit testing: identify and fix bugs, errors, and defects at an early stage
    - Integration testing: check comparability, interoperability, communication of the robots' parts
    - System testing: Evaluate robot's functionality, performance, and compliance with the specs & requirements
    - Simulation testing: reduce the risks, costs, and time involved in testing with the physical world
    - Field testing: Validate the robot's functionality, performance, and usability. Collect feedback, data, and insights from the end-users and stakeholders

3. Follow the test plan
> A document that outlines the scope, approach, resources, schedule, and deliverables of the testing process
- Help organize and mangage testing activities -> consistency, comprehensiveness, traceablity
- A test plan should include: objectives, methods, cases, data, environment, team, schedule, and deliverables
    - objectives: goals, criteria, metrics
    - methods: types, techniques, and tools
    - cases: scenarios, inputs, outputs, expected results of the testing process
    - data: information, variables, parameters to be used
    - environment: hardware, software, network settings
    - team: roles, responsibilities, skills of the people involved
    - schedule: timeline, milestones, deadlines
    - deliverables: reports, documents, artifacts that you will produce and share in the testing process

4. Document and report the test results
> CLEAR, CONCISE, AND ACCURATE
- Results: quantitative, qualitative
    - e.g.: test cases, pass/fail rates, coverage, defects, performance, usability, interpretation, and evaluation with the obectives and expectations
    - suggestions for improvement

5. Review and improve the testing process
- Continuous and iterative process - like software developement
- Monitor and update the process based on feedback, data, and insights

# 1.Testing for Robot's Programs' Functionality
Model: A Wheelchair (initial phase) robot that can perform the following functions:

1. Line following
2. Object Detection
3. Obstacle Avoidance
4. Hand + Face Tracking (hand gesture)
5. Position Estimation
6. Depth Perception
7. Following a person

## Integration and Performance

- Setup: Combine path following, object detection, and obstacle avoidance scenarios in a more complex environment.
- Procedure:
    - Create a course with a mix of predefined paths, dynamic obstacles, and objects to be detected.
    - Evaluate how well the robot integrates its capabilities to navigate the course autonomously.
    - Assess overall performance metrics, including completion time, accuracy in path following, and successful interaction with detected objects and obstacles.

## Environmental Variability Testing

- Setup: Test the robot under different environmental conditions to assess robustness.
- Procedure:
    - Vary lighting conditions (bright light, low light, shadows).
    - Introduce environmental factors such as changes in surface texture (e.g., from carpet to tile), which may affect wheel traction.
    - Evaluate how the robot adapts its behavior and maintains performance across these conditions.


# 3. Testing for Robot Mechanical Capabilities
- Slope Climb
- Tip Test
- Drawbar Pull
- Optional:
    + Crossing ramp test to go over rough terrain
    + Slalom
    + Pan-tilt-zoom 

# 4. Ideas for other test methods we can use to evaluate our ODOT robot 
- Obstacle Negotiation
- Turning Radius and Maneuverability -> in conjunction with obstacle avoidabce, line following, traversing in general
- Battery Life and Endurance
- Safety Features Testing
- Durability and Reliability
- User interface and Accessibility Testing
- Emergency Situations Simulation
- Communication and connectivity testing

# Additional Considerations:

- Data Logging: Record sensor data and robot actions during tests to analyze performance and identify areas for improvement.
- Human Interaction: Test the robot’s responsiveness to human interaction (e.g., stopping on command or avoiding humans in its path).
- Error Handling: Evaluate how the robot handles unexpected situations (sensor failures, communication issues) and whether it can recover autonomously.