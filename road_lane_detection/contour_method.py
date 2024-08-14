import cv2 as cv
from typing import Union
import numpy as np


def ROI(image:np.ndarray) -> np.ndarray:
    # blank mask that matches the input image dimensions
    mask = np.zeros_like(image)
    
    # Define a the number of color-channel of image frame
    if len(image.shape) > 2:
        channel_count = image.shape[2] 
        ignore_mask_color = (255,) * channel_count # mask color: white
    # if there is only one channel
    else:
        # color of the mask polygon is white
        ignore_mask_color = 255

    # Create a polygon to focus only on the road in the picture
    # NOTE: change the values in accordance to the camera and the positioning of the road
    rows, cols = image.shape[:2]
    # TEST: print("rowcol\n",rows, cols, "\n")

    # NOTE: in computer graphics, the point (0, 0) or the origin is in the upper left corner of the image
    bottom_left = [0, rows]
    top_left = [0, rows * 0.5]
    top_right = [cols, rows * 0.5]
    bottom_right = [cols, rows]

    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)

    # Fill the polygon with white color and generating the final mask
    # Reference: https://www.geeksforgeeks.org/draw-a-filled-polygon-using-the-opencv-function-fillpoly/
    cv.fillPoly(mask, vertices, ignore_mask_color)

    # Test code:
    # cv.imshow("Region of Interest Polygon",mask)
    # performing bitwise-and on the input image frame and mask to get only the edges of the road
    masked = cv.bitwise_and(image, mask)
    return masked

def BinaryThresholding(image: np.ndarray) -> np.ndarray:
    """
    Applying binary thresholding for further differentiation between lanes and road
    """
    # The binary threshold can be dynamically defined from the room's ambient light
    # measured by the light sensor
    # NOTE for IMPROVEMENT: add light sensor
    threshold = 110
    max_value = 255

    _, binary_image = cv.threshold(image, threshold, max_value, cv.THRESH_BINARY)
    """
    ===================================
    BINARY THRESHOLDING:
    
    Parameteres:
    threshold: threshold intensity value
    max_value: maximum value to use for the output binary image
    type: cv.THRESH_BINARY, any pixel with intensity value lower than the threshold will be set to 0, and the higher onees will be set to max_value

    >> Other types of threshold <<
    cv.THRESH_BINARY_INV: Inverse binary thresholding.
    cv.THRESH_TRUNC: Thresholding that truncates pixel values above the threshold.
    cv.THRESH_TOZERO: Sets pixel values to zero if they are below the threshold.
    cv.THRESH_TOZERO_INV: Inverse of cv2.THRESH_TOZERO.
    ====================================
    """
    return binary_image

def ContourExtraction(image:np.ndarray) -> np.ndarray:
    """
    Using OpenCV to get the contours of all the shapes in the frame, which are usually separated by black spaces
    Best work in binary/grayscale image
    """
    contours, hierarchy = cv.findContours(image=image, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)
    """
    findContours()

    parameters
    - image: input image, preferably binary input
    - mode: contour-retrieval mode
        + RETR_EXTERNAL: the extreme outer contours
        + RETR_TREE: retrieves all of the contours and reconstructs a full hierarchy of nested contours
        + for further mode: https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga819779b9857cc2f8601e6526a3a5bc71
    - method: defines the contour-approximation method (which points within a contour is stored)
        + CHAIN_APPROX_NONE: no approximation
        + CHAIN_APPROX_SIMPLE: simple approximation to 4 points
        + further: https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga4303f45752694956374734a03c54d5ff
    """
    print(contours)
    return contours

def GetRectangles():
    pass

def LaneDetection(image):
    """
    Overall pipeline for lane detection and following
    """

    image_copy = image.copy()

    grey = cv.cvtColor(image_copy, cv.COLOR_BGR2GRAY)
    binary_img = BinaryThresholding(grey)
    blurred = cv.GaussianBlur(binary_img, (5, 5), 0)

    region = ROI(blurred)

    contour_detected = ContourExtraction(region)

    # draw contours on the original image
    cv.drawContours(image, contours=contour_detected, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA )
    cv.imshow('Contours detected', image)

    return image

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
        
        # Display original for comparison
        cv.imshow('Original', frame)   

        # Process the frame
        processed_frame = LaneDetection(frame)

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