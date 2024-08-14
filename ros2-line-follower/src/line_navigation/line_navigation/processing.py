import cv2 as cv
import numpy as np
from typing import Union, Tuple

ArrayTuple = Tuple[np.ndarray, np.ndarray]
MIN_AREA_TRACK = 5000

def crop_size(height, width):
        """
        Get measures to crop the image output:
        (height_upper, height_lower, width_left, width_right)
        Width : from left to right: 1/6 w -> 5/6 w
        Height: top down: 1/3 h -> h
        """
        # TODO: optimal cropping size for perimeter control for image processing
        return (1*height//3, height, width//6, 5*width//6)

def get_contour_data(mask, processed_img, w_start=0):
        """
        Return the centroid of the largest contour in the binary image 'mask'
        """
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        line = {}

        for contour in contours:
            M = cv.moments(contour)

            if (M['m00'] > MIN_AREA_TRACK):
                # contour is part of the track
                line['x'] = w_start + int(M["m10"] / M["m00"]) # TODO: check if need to add crop_w_start
                line['y'] = int(M["m01"] / M["m00"])
                # adding contour information for testing on the physical robot
                line['contour'] = M['m00']

                # plot the area (countour) in pink
                cv.drawContours(processed_img, contour, -1, (255, 0, 255), 1)
                # putText(image, text, org (tuple of x-, y-coordinate), font, fontScale, color, thickness)
                cv.putText(processed_img, str(M['m00']), (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])),
                    cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
                
                # NOTE: track mark?
                """
                else:
                # Contour is a track mark
                if (not mark) or (mark['y'] > int(M["m01"]/M["m00"])):
                    # if there are more than one mark, consider only 
                    # the one closest to the robot 
                    mark['y'] = int(M["m01"]/M["m00"])
                    mark['x'] = w_start + int(M["m10"]/M["m00"])

                    # plot the area in pink
                    cv2.drawContours(processed_img, contour, -1, (255,0,255), 1) 
                    cv2.putText(processed_img, str(M['m00']), (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)
                """
        """
        if mark and line:
        # if both contours exist
            if mark['x'] > line['x']:
                mark_side = "right"
            else:
                mark_side = "left"
        else:
            mark_side = None
        """
        return (line)

# Run a video stream for image processing and line navigation
def line_detection_video(source:Union[int, str]=6):
    # check the webcam index to use 
    # Here, I am using webcam index number 4, or else, video files such as mp4
    # would also work 
    cap = cv.VideoCapture(source)
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # Check if frame is successfully captured
        if not ret:
            print("Failed to capture frame")
            break
        
        # Create two windows and set positions for them to not overlap
        cv.namedWindow("Original", cv.WINDOW_NORMAL)
        cv.moveWindow("Original", 0, 0)

        # Display original frame
        cv.imshow('Original', frame)
        
        # Process the frame
        processed_frame = line_detection(frame)
        
        cv.namedWindow("Processed", cv.WINDOW_NORMAL)
        cv.moveWindow("Processed", processed_frame.shape[1] + 100, 0)

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


def line_detection_img(filename: str) -> None:
    # NOTE: use this to test the image input first then go to preprocessing
    # cv.imshow('Camera Feed', current_frame)
    # cv.waitkey(1)
    # Read image

    img = cv.imread(filename)

    # convert BGR to HSV
    hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Define the range of white color in HSV
    # NOTE: change to the specific color range you like
    lower_hsv = np.array([0, 0, 200])
    upper_hsv = np.array([255, 50, 255])


    # Create a binary mask
    color_mask = cv.inRange(hsv_image, lower_hsv, upper_hsv)

    # Apply the mask to the original image
    color_segmented_image = cv.bitwise_and(img, img, mask=color_mask)
    
    # Detect line and get its centroid
    line = get_contour_data(color_mask)

    # Display the segmented image with line centroid
    if line:
        cv.circle(color_segmented_image, (line['x'], line['y']), 5, (0, 0, 255), 7)

    # print(type(color_segmented_image))
    cv.imwrite('detected/contour_output.jpg', color_segmented_image)

def line_detection(img):
    # Defining region of interest for image processing
    focused = roi(img)

    # apply Gaussian blur to reduce noise and improve line detection
    blurred = cv.GaussianBlur(focused, (9, 9), 0)

    # convert BGR to HSV
    hsv_image = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)

    # Define the range of specified color in HSV
    # NOTE: this is currently BLUE
    lower_hsv = np.array([100, 100, 100])
    upper_hsv = np.array([140, 255, 255])


    # Create a binary mask
    color_mask = cv.inRange(hsv_image, lower_hsv, upper_hsv)
    
    # NOTE: this partially works, decided to keep to see if needed further    
    # Apply Canny edge detection technique on the image
    # edges = cv.Canny(color_mask, 50, 150, apertureSize=3)

    # Apply the mask to the original image
    color_segmented_image = cv.bitwise_and(img, img, mask=color_mask)
        
    "TEST: FURTHER FEATURE EXTRACTION"
    # apply grayscale conversion to further filter the color
    # gray = cv.cvtColor(color_segmented_image, cv.COLOR_BGR2GRAY)

    # apply Gaussian blur to reduce noise and improve line detection
    # blurred = cv.GaussianBlur(color_segmented_image, (9, 9), 0)

    # apply Canny edge detection method on the image
    # edges = cv.Canny(color_segmented_image, 50, 150, apertureSize=3)

    "END OF TEST"
    # Detect line and get its centroid
    line = get_contour_data(color_mask)
 
    # Display the segmented image with line centroid
    if line:
        cv.circle(color_segmented_image, (line['x'], line['y']), 5, (0, 0, 255), 7)

    return color_segmented_image

# Algorithm to calculate contour data
def get_contour_data(mask):
    """
    Return the centroid of the largest contour in the binary image 'mask'
    """
    # NOTE: parameter to affect
    MIN_AREA_TRACK = 500

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    line = {}

    for contour in contours:
        M = cv.moments(contour)

        if (M['m00'] > MIN_AREA_TRACK):
            # contour is part of the track
            line['x'] = int(M["m10"] / M["m00"])
            line['y'] = int(M["m01"] / M["m00"])
    return (line)

def roi(img):
    """
    Function to get the Region of Interest (R.O.I.), so that the image processing
    module would only focus on the certain point
    while keeping the original image uncropped.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    
    # For the sake of simplifying this line navigation algorithm
    # the image values are hardcoded to reduce computational cost
    vert = np.array([[(0,480), (0, 480//2), (640, 480//2), (640, 480)]], dtype=np.int32)   
        
    # filling pixels inside the polygon defined by "vertices" with the fill color    
    cv.fillPoly(mask, vert, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv.bitwise_and(img, mask)
    
    return masked_image

def get_vertices_for_img(img) -> np.ndarray:
    """
    For this method, we will just keep it simple to have the module focus on the
    lower half of the frame
    """
    img_shape = img.shape
    height = img_shape[0]
    width = img_shape[1]

    # Get only the lower half of the frame
    region_bottom_left = (0, 0)
    region_top_left = (0, height//2)
    region_top_right = (width, height//2)
    region_bottom_right = (width, 0)

    vert = np.array([[region_bottom_left, region_top_left, region_top_right, region_bottom_right]], dtype=np.int32)
        
    return vert 


# Function to calculate the HSV Range
def get_hsv_range() -> ArrayTuple:
    # Convert color to lowercase
    color = int(input('Specify color of the line: \n0 - red\n1 - green\n2 - blue\n3 - white\n4 - black\n'))
    
    # Define dictionary of color ranges in BGR format
    color_ranges = {
        0: ([0, 0, 100], [10, 255, 255]),
        1: ([40, 40, 40], [70, 255, 255]),
        2: ([90, 50, 50], [130, 255, 255]),
        3: ([0, 0, 200], [255, 50, 255]),
        4: ([0, 0, 0], [180, 255, 30])
    }
    
    # Convert BGR color ranges to HSV
    lower_bgr = np.array(color_ranges[color][0], dtype=np.uint8)
    upper_bgr = np.array(color_ranges[color][1], dtype=np.uint8)
    lower_hsv = cv.cvtColor(np.array([[lower_bgr]]), cv.COLOR_BGR2HSV)[0][0]
    upper_hsv = cv.cvtColor(np.array([[upper_bgr]]), cv.COLOR_BGR2HSV)[0][0]
    
    return lower_hsv, upper_hsv

if __name__ == "__main__":

    # filename = "frame_input/single_line.jpg"
    # line_detection_img(filename=filename)

    # FInd hsv range
    # print(get_hsv_range())

    line_detection_video()