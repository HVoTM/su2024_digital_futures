# https://www.geeksforgeeks.org/line-detection-python-opencv-houghline-method/
# Python program to illustrate HoughLine for line detection

# Many of these implementations are learnt from
# https://github.com/kenshiro-o/CarND-LaneLines-P1/blob/master/Lane_Detection_Term_1.ipynb
import cv2 as cv
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def detect_line(filename: str):
	# Reading the image
	# Make sure that the image has to be in the same directory as the algorithm
	img = cv.imread(filename)
	"""
	# Next TODO: convert to HSL color space and isolate with the white lines
	# if there is a different colored lines, we can implement similarly to isolate_white_hsl()
	# convert to HSL colorspace to distinguish the white line
	hsl = cv.cvtColor(image, cv.COLOR_RGB2HLS)
	
	# Isolate white lines using isolate_white_hsl()
	"""
	# Convert the img to grayscale
	# cv2.cvtColor() to convert an image from one color space to another
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

	# apply Gaussian blur to reduce noise and improve line detection
	blurred = cv.GaussianBlur(gray, (9, 9), 0)

	# Apply Canny edge detection method on the image
	# the 2 middle parameters are low-high threshold to include edges
	# higher than high threshold: keep. in the threshold range, keep if near the edges beyond the high threshold
	# below the low threshold: discarded.
	# Recommended ratio: 1:3, 1:2
	edges = cv.Canny(blurred, 50, 150, apertureSize=3)

	# This returns an array of (r, theta) values. r is measured in pixels, theta is measure in radians
	lines = cv.HoughLines(edges, 1, np.pi/180, 200)

	# The below for loop runs till r and theta values
	# are in the range of the 2d array
	for r_theta in lines:
		arr = np.array(r_theta[0], dtype=np.float64)
		r, theta = arr
		# Stores the value of cos(theta) in a
		a = np.cos(theta)
		# Stores the value of sin(theta) in b
		b = np.sin(theta)

		# x0 stores the value rcos(theta)
		x0 = a*r
		# y0 stores the value rsin(theta)
		y0 = b*r

		# x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
		x1 = int(x0 + 1000*(-b))
		# y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
		y1 = int(y0 + 1000*(a))
		# x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
		x2 = int(x0 - 1000*(-b))
		# y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
		y2 = int(y0 - 1000*(a))

		# cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
		# (0,0,255) denotes the colour of the line to be drawn. This case is red
		cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

	# All the changes made in the input image are finally
	# written on a new image houghlines.jpg
	cv.imwrite('detected/houghline_1.jpg', img)

"Another line detection algorithmg with simpler methoQd"
def detect_line_simpler(filename: str):
	# Read image
	image = cv.imread(filename)

	"""
	# Next TODO: convert to HSL color space and isolate with the white lines
	# if there is a different colored lines, we can implement similarly to isolate_white_hsl()
	# convert to HSL colorspace to distinguish the white line
	hsl = cv.cvtColor(image, cv.COLOR_RGB2HLS)
	
	# Isolate white lines using isolate_white_hsl()
	"""
	# Convert image to grayscale
	gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	
	# Gaussian blur
	blurred = cv.GaussianBlur(gray, (9, 9), 0)

	# Use canny edge detection
	edges = cv.Canny(blurred, 50, 150, apertureSize=3)

	# Region of interest on the image to focus on the line
	# NOTE: to add or not?

	# Apply HoughLinesP method to directly obtain line end points
	# https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#ga46b4e588934f6c8dfd509cc6e0e4545a
	lines_list =[]
	lines = cv.HoughLinesP(
				edges, # Input preprocessed images
				rho=1, # Distance resolution in pixels
				theta=np.pi/180, # Angle resolution in radians
				threshold=100, # accumulator threshold: Min number of votes for valid line
				minLineLength=5, # Min allowed length of line
				maxLineGap=25 # Max allowed gap between line for joining them
				)
	
	# Iterate over points and draw marker on the detected line
	for line in lines:
		print(line[0])
		# Extracted points nested in the list
		x1,y1,x2,y2 = line[0]

		# Draw the lines joing the points on the original image
		cv.line(image,(x1,y1),(x2,y2),(0,255,0),2)
		# Maintain a simples lookup list for points
		lines_list.append([(x1,y1),(x2,y2)])
	
	# drawn = trace_lane_line(image, lines, make_copy=False)

	# Save the result image
	cv.imwrite('detected/houghline_2.jpg', image)

# NOTE: Do JPEG and PNG make any difference with efficiency and memory size?
# with different compression processes, JPEGs and JPG contain less data than PNGs, makeing it smaller in size -> use the former to work on it?

def preprocessing(image):
	"""
	# Next TODO: convert to HSL color space and isolate with the white lines
	# if there is a different colored lines, we can implement similarly to isolate_white_hsl()
	# convert to HSL colorspace to distinguish the white line
	hsl = cv.cvtColor(image, cv.COLOR_RGB2HLS)
	
	# Isolate white lines using isolate_white_hsl()
	"""
	# Convert image to grayscale
	gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	
	# Gaussian blur
	blurred = cv.GaussianBlur(gray, (9, 9), 0)

	# Use canny edge detection
	edges = cv.Canny(blurred, 50, 150, apertureSize=3)

	# Region of interest on the image to focus on the line
	segmented_image = region_of_interest(edges)

	return segmented_image

# Image should have already been converted to HSL color space
def isolate_white_hsl(img):
    # Caution - OpenCV encodes the data in ***HSL*** format
    # Lower value equivalent pure HSL is (30, 45, 15)
    low_threshold = np.array([0, 200, 0], dtype=np.uint8)
    # Higher value equivalent pure HSL is (360, 100, 100)
    high_threshold = np.array([180, 255, 255], dtype=np.uint8)  
    
    white_mask = cv.inRange(img, low_threshold, high_threshold)
    
    return white_mask

# REGION OF INTEREST
# "guess" what the region may be by following the contours of the line the vehicle is in and define a polygon
# - which will act as a region of interest
def get_vertices_for_img(img):
    imshape = img.shape
    height = imshape[0]
    width = imshape[1]

    vert = None
    
    if (width, height) == (960, 540):
        region_bottom_left = (130 ,imshape[0] - 1)
        region_top_left = (410, 330)
        region_top_right = (650, 350)
        region_bottom_right = (imshape[1] - 30,imshape[0] - 1)
        vert = np.array([[region_bottom_left , region_top_left, region_top_right, region_bottom_right]], dtype=np.int32)
    else:
        region_bottom_left = (200 , 680)
        region_top_left = (600, 450)
        region_top_right = (750, 450)
        region_bottom_right = (1100, 650)
        vert = np.array([[region_bottom_left , region_top_left, region_top_right, region_bottom_right]], dtype=np.int32)

    return vert

def region_of_interest(img):
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
        
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    vert = get_vertices_for_img(img)    
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv.fillPoly(mask, vert, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv.bitwise_and(img, mask)
    return masked_image

# LANE EXTRAPOLATION
def draw_lines(img, lines, color = [255, 0, 0], thickness=10, make_copy=True):
	# Copy the img
	img_copy = np.copy(img) if make_copy else img

	for line in lines:
		for x1,y1,x2,y2 in line:
			cv.line(img_copy, (x1, y1), (x2, y2), color, thickness)
    
	return img_copy

def find_lane_lines_formula(lines):
	xs = []
	ys = []

	for line in lines:
		for x1, y1, x2, y2 in line:
			xs.append(x1)
			xs.append(x2)
			ys.append(y1)
			ys.append(y2)

	# with coordinate of the two endpoints, we use scipy stats.linregress to predict the next line with slope and y-intercept
	slope, intercept, r_value, p_value, std_err = stats.linregress(xs, ys)
	# test code
	print("The linear regression output: ", (slope, intercept))

	# straight line: y = ax + b 
	return (slope, intercept)

# define a function to trace a line on the lane
def trace_lane_line(img, lines, make_copy=True):
	A, b = find_lane_lines_formula(lines)
	vert = get_vertices_for_img(img)
	
	region_top_left = vert[0][1]
	# test code: print(region_top_left)
	img_shape = img.shape
	bottom_y = img_shape[0] -1
	# y = Ax + b, therefore x = (y - b) / A
	x_to_bottom_y = (bottom_y - b) / A
    
	top_x_to_y = (region_top_left[1] - b) / A 
    
	new_lines = [[[int(x_to_bottom_y), int(bottom_y), int(top_x_to_y), int(region_top_left[1])]]]
	return draw_lines(img, new_lines, make_copy=make_copy)

# LANE DETECTOR WITH MEMORY
from collections import deque

def create_lane_line_coefficients_list(length=10):
	return deque(maxlen=length)

def trace_lane_line_with_coefficients(img, line_coefficients, top_y, make_copy=True):
	A = line_coefficients[0]
	b = line_coefficients[1]

	img_shape = img.shape
	bottom_y = img_shape[0] - 1

	# y = Ax + b, therefore x = (y - b)/A
	x_to_bottom_y = (bottom_y - b) /A

	top_x_to_y = (top_y - b) / A
	
	new_lines = [[int(x_to_bottom_y), int(bottom_y), int(top_x_to_y), int(top_y)]]

	return draw_lines(img, new_lines, make_copy=make_copy)

def trace_both_lane_lines_with_lines_coefficients(img, left_line_coefficients, right_line_coefficients):
    vert = get_vertices_for_img(img)
    region_top_left = vert[0][1]
    
    full_left_lane_img = trace_lane_line_with_coefficients(img, left_line_coefficients, region_top_left[1], make_copy=True)
    full_left_right_lanes_img = trace_lane_line_with_coefficients(full_left_lane_img, right_line_coefficients, region_top_left[1], make_copy=False)
    
    # image1 * α + image2 * β + λ
    # image1 and image2 must be the same shape.
    img_with_lane_weight =  cv.addWeighted(img, 0.7, full_left_right_lanes_img, 0.3, 0.0)
    
    return img_with_lane_weight

import math

MAXIMUM_SLOPE_DIFF = 0.1
MAXIMUM_INTERCEPT_DIFF = 50.0

class LaneDetectorWithMemory:
    def __init__(self):
        self.left_lane_coefficients  = create_lane_line_coefficients_list()
        self.right_lane_coefficients = create_lane_line_coefficients_list()
        
        self.previous_left_lane_coefficients = None
        self.previous_right_lane_coefficients = None
        
    
    def mean_coefficients(self, coefficients_queue, axis=0):        
        return [0, 0] if len(coefficients_queue) == 0 else np.mean(coefficients_queue, axis=axis)
    
    def determine_line_coefficients(self, stored_coefficients, current_coefficients):
        if len(stored_coefficients) == 0:
            stored_coefficients.append(current_coefficients) 
            return current_coefficients
        
        mean = self.mean_coefficients(stored_coefficients)
        abs_slope_diff = abs(current_coefficients[0] - mean[0])
        abs_intercept_diff = abs(current_coefficients[1] - mean[1])
        
        if abs_slope_diff > MAXIMUM_SLOPE_DIFF or abs_intercept_diff > MAXIMUM_INTERCEPT_DIFF:
            #print("Identified big difference in slope (", current_coefficients[0], " vs ", mean[0],
             #    ") or intercept (", current_coefficients[1], " vs ", mean[1], ")")
            
            # In this case use the mean
            return mean
        else:
            # Save our coefficients and returned a smoothened one
            stored_coefficients.append(current_coefficients)
            return self.mean_coefficients(stored_coefficients)
        
"""
    def lane_detection_pipeline(self, img):
        combined_hsl_img = filter_img_hsl(img)
        grayscale_img = grayscale(combined_hsl_img)
        gaussian_smoothed_img = gaussian_blur(grayscale_img, kernel_size=5)
        canny_img = canny_edge_detector(gaussian_smoothed_img, 50, 150)
        segmented_img = region_of_interest(canny_img)
        hough_lines = hough_transform(segmented_img, rho, theta, threshold, min_line_length, max_line_gap)

        try:
            left_lane_lines, right_lane_lines = separate_lines(hough_lines, img)
            left_lane_slope, left_intercept = find_lane_lines_formula(left_lane_lines)
            right_lane_slope, right_intercept = find_lane_lines_formula(right_lane_lines)
            smoothed_left_lane_coefficients = self.determine_line_coefficients(self.left_lane_coefficients, [left_lane_slope, left_intercept])
            smoothed_right_lane_coefficients = self.determine_line_coefficients(self.right_lane_coefficients, [right_lane_slope, right_intercept])
            img_with_lane_lines = trace_both_lane_lines_with_lines_coefficients(img, smoothed_left_lane_coefficients, smoothed_right_lane_coefficients)
        
            return img_with_lane_lines

        except Exception as e:
            print("*** Error - will use saved coefficients ", e)
            smoothed_left_lane_coefficients = self.determine_line_coefficients(self.left_lane_coefficients, [0.0, 0.0])
            smoothed_right_lane_coefficients = self.determine_line_coefficients(self.right_lane_coefficients, [0.0, 0.0])
            img_with_lane_lines = trace_both_lane_lines_with_lines_coefficients(img, smoothed_left_lane_coefficients, smoothed_right_lane_coefficients)
        
            return img_with_lane_lines

"""
# Drive code to test stuff
if __name__== "__main__":

	filename = "frame_input/parking_lot.jpg"
	# detect_line(filename)
	detect_line_simpler(filename)