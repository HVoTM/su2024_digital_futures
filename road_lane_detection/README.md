# Approach 1:
## A pattern of comment overall steps
- Image Pre-processing
- Feature Extraction
- Feature Validation
- Trajectory Calculation

## 1a. handcrafted, traditional Houghline method
1. Read and decode video file into frames
2. Grayscale conversion of image
3. Reduce noise by applying filter
4. Detecting edges using Canny Edge detection (https://en.wikipedia.org/wiki/Canny_edge_detector)
5. Mask the canny image
6. Region of interest by using _**fillPoly()**_ to define and constrain to the necessary environment to execute
7. HoughLine Transform and extraction
8. Find coordinates of road lanes and define weighted slopes and integer
9. Draw lines on the image or video for representation
10. Define the center point and use that to have the robot to stay in the center of the lane (like how line following works)

## 1b. Further advanced traditional method
- Based on the 1a method, we now will work on performing more complicated methods such as:
    1. Thresholds for different color spaces and gradients
    2. Sliding window techniques
    3. Warped perspective transforms:
        - Geometric Image Transform: https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html
    4. Polynomial fits 

Which I hope that can be used to tackle the problem of getting distortion on sharp turns and curves, which are the usual characteristics of lane and road formation for transportation.

## To-dos
- [] Design an algorithm flowcharts
- [] Adding HSL Image conversion for necessary color detection -> maybe also use that as a boundary for staying out
- [] Add center of the 2 lanes after we perform Left right differentiation -> new approach 
- [] **IMPORTANT**: Test and tweak for necessary hyperparameters for houghline, canny edge, and region of interest in Turtlebot3 camera or other products
    + "guess" work, hardcoding
    + some other optimization methods
- [] Transfer over to ROS2 Package to run with publisher and such
- *look at improvement to see what comes next...*
- Afterwards, organize and improve for maintainability

# Approach 2: Convolutional Neural Network - CNN-based methods
- Spatial CNN
- LaneNET
- ...

# Concepts
<div style="color:magenta; background-color:tomato">Overall table of applicable traditional methods to work on self-driving, especially lane_detection </div>

1. **Image Pre-processing**: Images taken directly from the camera often need to undergo certain preparations and transformations to make information extraction easier

| Algorithms/ Techniques | Details|
|-|-|
| Region of Interest| the image is cut to exclude unwanted information (_e.g._ visible car, sky, etc) or to focus on a particular area
| Greyscale Transform | Convert coloured pixels into shades of grey|
| Binary Transform | Convert coloured pixels into either black or white given an intensity threshold (shade of grey). |
| Blur Filter (*Gaussian*) | Used to reduce noise and image detail. |
| Inverse Perspective Mapping | Transforms image’s perspective into bird’s eye view. This view makes the width of the road equal size at any distance (i.e. in one point perspective view, the width of the road diminishes the further you look)|
| Fisheye Dewarp, or Extrinsic/Instrinsic Calibration | Cameras equipped with fisheye lenses need to be transformed back to normal perspective to simplify distance calculations |

2. **Feature Extraction**: Information Extraction Phase. Typically, the intent is to extract features the resemble lane markings

| Algorithms/Technique | Details|
| - | -|
| Canny Edge Detection     | The algorithm aims to detect all the edges in an image, may require threshold tuning to achieve the desired range of edges |
| Hough Transform | Detecting arbitrary shapes, mainly used to fine dominant lines |

3. **Feature Validation**: feature validation or fitting is the approximation of extracted features into a smooth path using geometric models. The generated path is later used for various decision making processes typically for trajectory or heading calculations. Aim: fit a curve as accurately as possible with given features

| Algorithms/Techniques | Details |
| -| -| 
| RANSAC (Random sample consesus) | A method that requires as much data as available to function as intended, it aims to remove the invalid data by fitting a desired shape to the detected features, i.e. a curve, a straight line, a circle, and then later applies smoothing. RANSAC paradigm is based on three criteria: 1) error tolerance, 2) determine the compatibility with a model and 3) apply the threshold assuming the correct model has been found. |
| Kalman Filter | Filter out noise from given noise data. It is based on mathematical equations that provide a recursive mean to estimate a process and minimizes the mean of squared error, predicting a future state from the previous one |
| Polynomial Fitting | Curve fitting is a mathematical technique that is widely used in engineering applications. It consists of fitting a set of points to a curve using *Lagrange Interpolation Polynomial*. The principal of this method is that given a set of points, the aim is to fit them in a smooth curve that passes through the aforementioned points and the order of the function depends on the number of points. It can be used to approximate complicated curves. However, a high-degree interpolation may also result in a poor prediction of the function between points. |

4. **Trajectory Calculation**: Finally, the vehicle receives the coordinates for desired heading. The trajectory is derived on a 2D plane, it needs to translated to correspond to real world coordinates - 3D plane


## Camera calibration
- OpenCV's camera calibration procedure
https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

- Intrinsic and extrinsic parameters
https://towardsdatascience.com/what-are-intrinsic-and-extrinsic-camera-parameters-in-computer-vision-7071b72fb8ec

- Zhang's calibration
+https://www.ipb.uni-bonn.de/html/teaching/photo12-2021/2021-pho1-22-Zhang-calibration.pptx.pdf

## Distortion matrix

## HSL/HSV color space

## Bird Eye's View - Warped Perspective
- Geometric Image Transformation
https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html
- In simple terms, we will want a specific region, usually a skewed, tilted area due to camera perspective, which will cause parallel lines, for example, to be converging in the horizon in this lane detection case. We will use getPerspectiveTransform() to get the transformation matrix and lay the desired region out to with `warpPerspective()`. Henceforth, the lane can be seen as parallel, similar to how a bird from above looks down to the road from the sky (Bird Eye's View) 

## Lane Pixels - Finding and fitting to the boundary
NumPy's convolve
https://numpy.org/doc/stable/reference/generated/numpy.convolve.html

## [Vanishing Points](https://en.wikipedia.org/wiki/Vanishing_point)

# Controls & Smoothing technique
To address the issue of sudden oscillations in lane center detection caused by distracting environments, you can implement several techniques to stabilize your lane center estimation and improve the robustness of your lane detection algorithm. Here are some approaches to consider:

### 1. **Smoothing Techniques**

- **Moving Average Filter**: Apply a moving average filter to smooth the lane center position over time. This technique averages the current position with a fixed number of previous positions to reduce rapid fluctuations.
  ```python
  def moving_average(data, window_size):
      return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
  ```

- **Exponential Smoothing**: Use exponential smoothing to give more weight to recent data while still considering past values. This can be useful to adapt quickly to genuine changes but still filter out noise.
  ```python
  alpha = 0.1  # Smoothing factor
  smoothed_value = alpha * current_value + (1 - alpha) * previous_smoothed_value
  ```

### 2. **Kalman Filter**

- **Implement a Kalman Filter**: The Kalman Filter is a powerful tool for estimating the state of a system over time. It combines predictions from a model with noisy measurements to provide a more stable estimate. It can be particularly effective for filtering out noise and reducing oscillations in lane center detection.
  ```python
  import numpy as np

  # Initialize Kalman Filter parameters
  def kalman_filter(prev_state, prev_covariance, measurement, measurement_noise, process_noise):
      # Prediction step
      predicted_state = prev_state
      predicted_covariance = prev_covariance + process_noise

      # Update step
      innovation = measurement - predicted_state
      innovation_covariance = predicted_covariance + measurement_noise
      kalman_gain = predicted_covariance / innovation_covariance
      updated_state = predicted_state + kalman_gain * innovation
      updated_covariance = (1 - kalman_gain) * predicted_covariance

      return updated_state, updated_covariance
  ```

### 3. **Lane Center Filtering**

- **Filter Lane Center**: Apply a filter specifically to the lane center position. This could be a simple median filter that replaces the current lane center with the median of recent positions, reducing the impact of outliers.

### 4. **Robust Lane Detection**

- **Improve Lane Detection Robustness**: Enhance the robustness of your lane detection algorithm to minimize the impact of distracting elements:
  - **Preprocessing**: Use techniques such as adaptive thresholding or region-based segmentation to improve the detection of lanes under varying conditions.
  - **Lane Persistence**: Incorporate lane persistence where the algorithm expects the lanes to follow a smooth path. This can help prevent sudden changes in lane center estimation.

### 5. **Constraint-Based Filtering**

- **Apply Constraints**: Set constraints based on the expected range or behavior of lane centers. For example, if the lane center changes too rapidly, you can limit the maximum allowable change per frame.

### 6. **Feedback Control**

- **Implement Feedback Control**: Use a feedback control system, such as a PID controller, to manage the lane center position. This approach can help smooth out abrupt changes and provide more stable control responses.

```python
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0

    def update(self, setpoint, measured_value, dt):
        error = setpoint - measured_value
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        self.previous_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative
```

### 7. **Post-Processing**

- **Post-Process Detected Lanes**: After detecting lanes and calculating the center, apply additional checks or filters to validate the lane center's consistency before using it for robot control.

# Improvement
- *Lane Departure* problem: Assumptions that the vehicle is **always** inside its lane drives the whole algorithms, no efforts made on trying to return the car back to the lane in case of erroneous prediction -> track memory??
- Interface to give choices not to swerve over to the right or the left, the default would be staying between the lanes
- HSL Color masking choices for different colors
- CNN for better segmentation 
- Some associated topics:
  - *Object detection*
  - *Traffic signs* and *traffic lights* detection
  - *Intersection handling*

# References
Test image and concepts were referred from Udacity's Autonomous Driving Nanodegree

Repo of Lane Detection Reference <br>
https://github.com/amusi/awesome-lane-detection?tab=readme-ov-file#Code

Papers with Code - Lane Detection <br>
https://paperswithcode.com/task/lane-detection

Tutorial: Build A Lane Detector with 2 approaches <br>
https://towardsdatascience.com/tutorial-build-a-lane-detector-679fd8953132

Simple Lane Detection with OpenCV <br>
https://medium.com/@mrhwick/simple-lane-detection-with-opencv-bfeb6ae54ec0

Advanced Lane Line Detection <br>
https://github.com/nachiket273/Self_Driving_Car/tree/master/CarND-Advanced-Lane-Lines

Hough Transform <br>
https://en.wikipedia.org/wiki/Hough_transform

## Research, Paper, Thesis
Applications of Computer Vision in Autonomous Vehicles: Methods, Challenges and Future Directions <br>
https://arxiv.org/pdf/2311.09093

Lane Detection and Following Approach in Self-Driving Miniature Vehicles <br>
https://blog.glugmvit.com/assets/images/self_driving/car11.pdf

Spatial CNN <br>
https://arxiv.org/pdf/1712.06080.pdf

RNN(ReNet) or Instance Segmentation <br>
https://arxiv.org/pdf/1802.05591.pdf

Robust Lane Detection from Continuous Driving Scenes using Deep Neural Network <br>
https://arxiv.org/pdf/1903.02193


## Code, Framework, Documentation
**matplotlib**'s Pyplot styling and references <br>
https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html



