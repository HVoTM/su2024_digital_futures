import cv2 
import numpy as np
# https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html

colors_bgr = {"black":[[[0,0,0]]], "white":[[[255, 255, 255]]], "blue":[[[255, 0, 0]]], "red": [[[0, 0,255]]],
               "green":[[[0, 255, 0]]], "orange": [[[0, 165, 255]]]}

def displayHSVRange():
    for color_name, color_range in colors_bgr.items():
        color_range = np.uint8(color_range)
        hsv_range = cv2.cvtColor(color_range, cv2.COLOR_BGR2HSV)
        upper =  hsv_range[0][0][0] +10, 255, 255
        lower = hsv_range[0][0][0]- 10, 100, 100
        print("Color: {}\nBGR range:{} \nLower: {}\nUpper: {}".format(color_name, color_range, lower, upper))

def webcamColorDetection(webcam_index=0):
    """
    color = str.lower(input("Choose a color by typing in the list down below:\n\
                  Black, White, Blue, Red, Green, Orange"))
    bgr_range = np.uint8(colors_bgr[color])
    hsv_range = cv2.cvtColor(bgr_range, cv2.COLOR_BGR2HSV)
    """
    upper =  np.array([29, 255, 255])
    lower = np.array([9, 100, 100])

    # Start video and start searching for the color
    cap = cv2.VideoCapture(webcam_index)
    while True:
        # Capture frame by frame
        ret, frame = cap.read()

        if not ret:
            KeyError("Failed to capture FRame!")
            break
        
        # Display original frame, cam check
        # cv2.imshow('Output', frame)
        processed = line_detection(frame, lower=lower, upper=upper)

        cv2.imshow('Processed image', processed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # release the capture
    cap.release()
    cv2.destroyAllWindows()


def line_detection(img, lower: np.ndarray, upper:np.ndarray):
    # convert BGR to HSV
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    color_mask = cv2.inRange(hsv_image, lower, upper)

    # apply  the mask to the original image
    color_segmented_image = cv2.bitwise_and(img, img, mask=color_mask)

    line = get_contour_data(color_mask)
    # Display the segmented image with line centroid
    if line:
        cv2.circle(color_segmented_image, (line['x'], line['y']), 5, (0, 0, 255), 7)
    return color_segmented_image

# Algorithm to calculate contour data
def get_contour_data(mask):
    """
    Return the centroid of the largest contour in the binary image 'mask'
    """
    # NOTE: parameter to affect
    MIN_AREA_TRACK = 500

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    line = {}

    for contour in contours:
        M = cv2.moments(contour)

        if (M['m00'] > MIN_AREA_TRACK):
            # contour is part of the track
            line['x'] = int(M["m10"] / M["m00"])
            line['y'] = int(M["m01"] / M["m00"])
    return (line)


if __name__ == "__main__":
    webcamColorDetection()
    pass