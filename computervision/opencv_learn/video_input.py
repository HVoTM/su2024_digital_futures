import cv2 as cv
import numpy as np

def video(webcam_index: int = 10):
    # Open webcam
    cap = cv.VideoCapture(webcam_index)

    while True:
        # Capture frame by frame
        ret, frame = cap.read()

        # Check if frame is successfully captured

        if not ret:
            print("Failed to capture frame!")
            break

        # Display original frame
        cv.imshow('Output', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # release the capture
    cap.release()
    cv.destroyAllWindows()

def check_webcam_indices():
    max_num_cameras = 10 # Assume there are 10 cameras
    for i in range(max_num_cameras):
        cap = cv.VideoCapture(i)
        if not cap.isOpened():
            print(f"No camera found at index {i}")
        else:
            print(f"Camera found at index {i}")
            cap.release()

# Function to calculate the HSV Range
def get_hsv_range(color):
    # Convert color to lowercase
    color = color.lower()
    
    # Define dictionary of color ranges in BGR format
    color_ranges = {
        'red': ([0, 0, 100], [10, 255, 255]),
        'green': ([40, 40, 40], [70, 255, 255]),
        'blue': ([90, 50, 50], [130, 255, 255]),
        'white': ([0, 0, 200], [255, 50, 255]),
        'black': ([0, 0, 0], [180, 255, 30])
    }
    
    # Convert BGR color ranges to HSV
    lower_bgr = np.array(color_ranges[color][0], dtype=np.uint8)
    upper_bgr = np.array(color_ranges[color][1], dtype=np.uint8)
    lower_hsv = cv.cvtColor(np.array([[lower_bgr]]), cv.COLOR_BGR2HSV)[0][0]
    upper_hsv = cv.cvtColor(np.array([[upper_bgr]]), cv.COLOR_BGR2HSV)[0][0]
    
    return lower_hsv, upper_hsv
# Driver code
if __name__=="__main__":
    video(0)
    # check_webcam_indices()
