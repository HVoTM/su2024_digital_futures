import numpy as np
import pandas as pd
import cv2 as cv
from typing import Optional, Union

"Hyperparameters of Hough Transform, or HoughLinesP() method"
# Distance resolution of the accumulator in pixels.
RHO = 3     
# Angle resolution of the accumulator in radians.
THETA = np.pi/180   
# Only lines that are greater than threshold will be returned.
THRESHOLD = 50      
# Line segments shorter than that are rejected.
MIN_LINE_LENGTH = 5
# Maximum allowed gap between points on the same line to link them
MAX_LINE_GAP = 100

def RegionOfInterest(input_frame:np.ndarray)->np.ndarray:
    """
    Utility filter function to determine and crop out areas which will never contain information about the lane markers
    and retains the region of interest in the input image
    """
    # Create a blank mask/matrix that matches the height/width
    mask = np.zeros_like(input_frame)
    
    # Define a the number of color-channel of image frame
    if len(input_frame.shape) > 2:
        channel_count = input_frame.shape[2] 
        ignore_mask_color = (255,) * channel_count # mask color: white
    # if there is only one channel
    else:
        # color of the mask polygon is white
        ignore_mask_color = 255

    # Create a polygon to focus only on the road in the picture
    # NOTE: change the values in accordance to the camera and the positioning of the road
    rows, cols = input_frame.shape[:2]
    # TEST: print("rowcol\n",rows, cols, "\n")

    # NOTE: in computer graphics, the point (0, 0) or the origin is in the upper left corner of the image
    bottom_left = [0, rows]
    top_left = [cols * 0.3, rows * 0.6]
    top_right = [cols * 0.7, rows * 0.6]
    bottom_right = [cols, rows]

    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)

    # Fill the polygon with white color and generating the final mask
    # Reference: https://www.geeksforgeeks.org/draw-a-filled-polygon-using-the-opencv-function-fillpoly/
    cv.fillPoly(mask, vertices, ignore_mask_color)

    # Test code:
    cv.imshow("Region of Interest Polygon",mask)
    # performing bitwise-and on the input image frame and mask to get only the edges of the road
    masked = cv.bitwise_and(input_frame, mask)
    return masked

def average_slope_intercept(lines):
    """
    Find the slope and intercept of the left and right lanes of each image.
    Parameters:
        lines: output from Hough Transform
    """
    left_lines    = [] #(slope, intercept)
    left_weights  = [] #(length,)
    right_lines   = [] #(slope, intercept)
    right_weights = [] #(length,)
     
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            # calculating slope of a line
            slope = (y2 - y1) / (x2 - x1)
            # calculating intercept of a line
            intercept = y1 - (slope * x1)
            # calculating length of a line
            length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
            # slope of left lane is negative and for right lane slope is positive
            if slope < -0.5 and slope >  -0.984:
                left_lines.append((slope, intercept))
                left_weights.append((length))
            elif slope > 0.5 and slope < 0.984:
                right_lines.append((slope, intercept))
                right_weights.append((length))

    # Calculate the weighted average of slopes and intercepts for the corresponding left and right lanes
    left_lane  = np.dot(left_weights,  left_lines) / np.sum(left_weights)  if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
    return left_lane, right_lane
   
def pixel_points(y1, y2, line):
    """
    Converts the slope and intercept of each line into pixel points.
        Parameters:
            y1: y-value of the line's starting point.
            y2: y-value of the line's end point.
            line: The slope and intercept of the line.
    """
    if line is None:
        return None
    slope, intercept = line
    
    # TEST CODE:
    # print('\ny1: ', y1, '\ny2: ', y2,'\nSlope: ', slope, '\nIntercept: ', intercept)
    if slope <= 0 and slope > -0.01:
        slope = -0.01
    elif slope >= 0 and slope < 0.01:
        slope = 0.01
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)
    return ((x1, y1), (x2, y2))
   
def lane_lines(image, lines):
    """
    Create full length lines from pixel points.
        Parameters:
            image: The input test image.
            lines: The output lines from Hough Line Transform.
    """
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0]
    y2 = y1 * 0.6
    if left_lane is not None: 
        # For each side, we will return the two weighted pixel endpoints and the colors for differentiation
        left_line  = (*pixel_points(y1, y2, left_lane), (0, 0, 255))
    else:
        left_line = pixel_points(y1, y2, left_lane)
    if right_lane is not None:
        right_line = (*pixel_points(y1, y2, right_lane), (0, 255, 0))
    else:
        right_line = pixel_points(y1, y2, right_lane)
    return left_line, right_line
  
def DrawLines(image: np.ndarray, lines, thickness=5) -> np.ndarray:
    """
    Draw lines onto the input image.
        Parameters:
            image: The input test image (video frame in our case).
            lines: The output lines from Hough Transform with defined color
            thickness (Default = 12): Line thickness. 
    """
    if lines is None:
        return image
    
    # Make a copy of the original image
    img = np.copy(image)

    # Create a blank image that matches the original in size
    line_image = np.zeros_like(image)

    # Loops over all lines and draw them on the blank image
    for line in lines:
        if line is not None:
            cv.line(line_image, *line, thickness) # *line just to unpack the iterable tuple
    
    return cv.addWeighted(img, 1.0, line_image, 1.0, 0.0)

# TODO: add HSL, or HSV, masking for certain colored lines (WHITE AND YELLOW)
def HSLMasking():
    pass

def Fit_Polynomial(lines, image_width):
    """
    To better handle slopes and orientations, fit lines using polynomial regression
    """
    if lines is None:
        return None
    center_line_x = image_width//2
    left_lines = []
    right_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        """
        # calculating slope of a line
            slope = (y2 - y1) / (x2 - x1)
            # calculating intercept of a line
            intercept = y1 - (slope * x1)
            # calculating length of a line
            length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
            # slope of left lane is negative and for right lane slope is positive
            if slope <= 0:
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))
        """
        # Calculate the slope of the line
        slope = (y2 -y1) / (x2 - x1)
        # Categorize left and right lanes based on slope in the corresponding slope range
        # to account for acute or overly obtuse angle on each one
        # Current range: 30 - 80 degree 
        # ======================================================
        if slope < -0.5 and slope > -0.984 and x1 < center_line_x:
            left_lines.append((x1, y1, x2, y2))
        elif slope > 0.5 and slope < 0.984 and x1 > center_line_x:
            right_lines.append((x1, y1, x2, y2))

        """
        # TODO: account for position of the lines as well?
        if x1 < image_width // 2:
            left_lines.append((x1, y1, x2, y2))
        else:
            right_lines.append((x1, y1, x2, y2))
        """
    def fit_line(lines):
        if len(lines) == 0:
            return None
        xs = []
        ys = []
        # this below is the same as for line in lines, then x1, y1, x2, y2 = line
        for x1, y1, x2, y2 in lines:
            xs.extend([x1, x2])
            ys.extend([y1, y2])
        coefficients = np.polyfit(xs, ys, 1)  # Fit a curved line withpolynomial degree of 2
        # DEBUG
        print(coefficients)

        return coefficients
    
    left_fit = fit_line(left_lines)
    right_fit = fit_line(right_lines)
    return left_fit, right_fit

def CalculateLaneCenter(left_fit, right_fit, image_size):
    # Base case to check for missing left / right lanes
    # if no left lanes to be seen, have the lane_center to be to the left a lil bit
    if left_fit is None:
        return image_size[1] * 0.2
    # same goes for right lanes
    if right_fit is None:
        return image_size[1] * 0.8
    
    left_slope, left_intercept = left_fit
    right_slope, right_intercept = right_fit
    
    # Check case for slopes reaching 0, which will make the theoretically derived, 
    # or virtual coordinates of the left/right lanes
    if left_slope > -0.001: # REMIND: left_slope < 0, and right_slope > 0 always since we already checked that in 
                            # Fit_Polynomial()
        left_slope = 0.001
    if right_slope < 0.001:
        right_slope = 0.001

    y1 = image_size[0]
    y2 = image_size[0] // 2
    
    left_x1 = (y1 - left_intercept) / left_slope
    left_x2 = (y2 - left_intercept) / left_slope
    
    right_x1 = (y1 - right_intercept) / right_slope
    right_x2 = (y2 - right_intercept) / right_slope
    
    # TODO: check for case when there are too many distracting environments
    # 
    lane_center = (left_x1 + right_x1) / 2
    return lane_center

def FrameProcessing(frame:np.ndarray) -> np.ndarray:
    """
    Process the input frame to detect lane lines
    Input argument:
        frame: image/frame of a road where one wants to detect lane lines
    """

    # convert image color scale (BGR) to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # applying Gaussian Blur to remove noise from the image and focus on region of interest
    # kernel size = 5 (normally distributed numbers to run across the entire image)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    
    """
    Parameters:
    lower_threshold = 50
    higher_threshold = 150
    apertureSize = [3, 7], the larger the aperture, the more edges will be defined, 3 would be the best
    """
    edges = cv.Canny(blurred, 50, 75, apertureSize=3)

    # Limiting the region of interest to reduce confusion and constrain to the necessary
    # edges for lane detection
    region = RegionOfInterest(edges)
    """
    # DEBUG: IMAGE OUTPUT TEST
    cv.imshow('Grayscale image', gray)
    cv.imshow("Gaussian Blurred", blurred)
    cv.imshow("Canny Edge Detection", edges)
    cv.imshow("Region of Interest shown", region)
    """
    cv.imshow("Region of Interest shown", region)
    # Deduce Hough Lines with suitable hyperparameters
    lines = cv.HoughLinesP(region, rho=RHO, theta=THETA, threshold=THRESHOLD, minLineLength=MIN_LINE_LENGTH, maxLineGap=MAX_LINE_GAP)
    # DEBUG 
    # print(lines)
    return lines

def LaneDetection(image: np.ndarray) -> np.ndarray:
    """The main function to be called to Perform image processing 
    and detect road lanes on input image"""
    # call for image processing and return the lines 
    lines = FrameProcessing(image)

    # Rendering Hough lines detected as an overlay for a sense of the real features in the scene
    if lines is not None:
        for line in lines:
            # print(line[0])
            # Extracted points nested in the list
            x1, y1, x2, y2 = line[0]

            # Draw the lines joining the points on the original image
            cv.line(image, (x1, y1), (x2, y2), (0, 255, 0), 5)

    return image

def LeftRightLaneDetection(image: np.ndarray) -> np.ndarray:
    "Perform the similar processing techniques for the image input, but use slope to identify the two lanes"
    lines = FrameProcessing(image)
    if lines is not None:
        # lastly, draw the lines on the resulting frame and return it as output
        result = DrawLines(image, lane_lines(image, lines))
        return result
    return image

def LaneDetectionPolyFit(image:np.ndarray) -> np.ndarray:
    "Lane Detection Processing Pipeline for image input"
    lines = FrameProcessing(frame=image)

    if lines is not None:
        left_fit, right_fit = Fit_Polynomial(lines=lines, image_width=image.shape[1])
        lane_center = CalculateLaneCenter(left_fit=left_fit, right_fit=right_fit, image_size=image.shape)
        
        # Display detected lines found by Hough Transform
        for line in lines:
            # print(line[0])
            # Extracted points nested in the list
            x1, y1, x2, y2 = line[0]

            # Draw the lines joining the points on the original image
            cv.line(image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    else:
        lane_center = image.shape[1] // 2
    # DEBUG
    print(lane_center)
    # =======================================================================
    # TODO: adding offset for the camera position and orientation, may be a list with delay value to control the differential drive
    # since the central point of the camera and the central position of the motor are not very aligned, so it will skip
    # =======================================================================
    # Calculate offset due to camera position and orientation
    offset = (image.shape[1] // 2) - lane_center
    if offset > 0:
        print(f"Turn right by {offset} units")
    elif offset < 0:
        print(f"Turn left by {-offset} units")
    else:
        print("Move straight")
        
    # Draw the center of the lane, with an offset of 10 pixels upwards for visualization
    # cv.circle(image, center (x, y), radius, color, thickness)
    cv.circle(image, (int(lane_center), image.shape[0] - 10), 5, (0, 50, 255), 5)

    return image, lane_center


def ShowProcessedImg(filename: str) -> None:
    "Driver function to test for read an image file and perform lane detection"
    img = cv.imread(filename)
  
    cv.imshow('Original', img)
    # either call Lane Detection or LeftRight Lane Detection
    processed = LaneDetectionPolyFit(img)
    cv.imshow('Road lane detected', processed)
    # Pauses execution until a key is pressed. '0' means waiting indefinitely
    cv.waitKey(0)
    cv.destroyAllWindows() # closes all openCV windows when script exits

def ReturnProcessedImage(input_image: np.ndarray) -> np.ndarray:
    return LeftRightLaneDetection(input_image)

def ShowProcessedVideo(test_video:str, output_video: str):
    pass

def ShowProcessedWebcam(source:Union[int, str]= 0) -> None:
    cap = cv.VideoCapture(source)
    
    while True:
        # Capture frame-by-frame
        # ret = boolean return value from getting the frame, frame = current frame being projected in the video
        ret, frame = cap.read()
        
        # Check if frame is successfully captured
        if not ret:
            print("Failed to capture frame")
            break
        
        # Process the frame
        processed_frame = LaneDetectionPolyFit(frame)
        
        # Display processed frame
        cv.imshow('Processed', processed_frame)
        
        # set waitKey to 1 for video playback or real-time image processing
        # it will allow the program to update displayed image
        # Break the loop when 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the capture
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    #process_img('roadline.jpg')
    #ShowProcessedImg("roadline.jpg")
    ShowProcessedWebcam()