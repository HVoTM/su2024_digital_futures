import numpy as np
import cv2 as cv
import pickle
import glob
from tracker import Tracker
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec # https://matplotlib.org/stable/api/_as_gen/matplotlib.gridspec.GridSpec.html
from typing import Tuple, Union

"""HYPERPARAMETERS"""
# TODO: Hyperparameters to be put in the header for configuration and calibration for scalability

# TODO: set BirdEyeView's parameter for geometric warp
# default: BOT_W, TOP_W, TOP_H, BOT_H = 0.76, 0.08, 0.62, 0.935; OFFSET=0.25
# simulation default: BOT_W, TOP_W, TOP_H, BOT_H = 0.92, 0.20, 0.40, 0.87
BOT_W, TOP_W, TOP_H, BOT_H = 0.92, 0.34, 0.6, 0.87
OFFSET = 0.25

# Size of sliding windows in Warping Perspective
WINDOW_WIDTH = 50
WINDOW_HEIGHT = 80

def ShowChessboardCorners():
    """
    On a personal camera input with multiple corners and distance of a printed 
    A4 Chessboard, find and show the given corners with matplotlib 
    """
    # number of inside corners in x & y directions
    nx, ny = 9, 6

    # prepare object points
    objp = np.zeros((6*9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane

    # Make a list of calibration images
    images = glob.glob('./camera_cal/calibration*.jpg')

    plt.figure(figsize = (18,12))
    grid = gridspec.GridSpec(5,4)
    # set the spacing between axes.
    grid.update(wspace=0.05, hspace=0.15)  

    for idx, fname in enumerate(images):
        img = cv.imread(fname)
        # Convert to grayscale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chessboard corners using openCV's findChessboardCorners()
        ret, corners = cv.findChessboardCorners(gray, (nx, ny), None)

        # If found, add to object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            img = cv.drawChessboardCorners(img, (nx, ny), corners, ret)
            write_name = 'corners_found'+str(idx)+'.jpg'
            #cv2.imwrite(write_name,img)
            img_plt = plt.subplot(grid[idx])
            plt.axis('on')
            img_plt.set_xticklabels([])
            img_plt.set_yticklabels([])
            #img_plt.set_aspect('equal')
            plt.imshow(img)
            plt.title(write_name)
            plt.axis('off')
            
    plt.show()
    # plt.axis('off')

# NOTE: assign as a distinct ROS node to run initial camera calibration before first launch of the road lane detection
def CameraCalibration(): # reference: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    """
    Given your own camera input, use a printed A4 chessboard and have samples of different angles
    and corners taken of the chessboard image. We will take those datasets to configure the 
    appropriate camera calibration and distortion coefficients
    """
    # number of inside corners in x & y directions
    nx, ny = 9, 6

    # prepare object points, like (0, 0, 0), (1, 0, 0), (2, 0, 0), ...
    objp = np.zeros((6*9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images
    objpoints = [] # 3-D points in real world space
    imgpoints = [] # 2-D points in image plane

    # Make a list of calibration images
    images = glob.glob('./camera_cal/calibration*.jpg')

    for idx, fname in enumerate(images):
        img = cv.imread(fname)
        # Convert to grayscale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chessboard corners using openCV's findChessboardCorners()
        ret, corners = cv.findChessboardCorners(gray, (nx, ny), None)

        # If found, add to object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # DEBUG: Draw and display the corners
            # cv.drawChessboardCorners(img, (nx, ny), corners, ret)
            # write_name = 'corners_found'+str(idx)+'.jpg'
            # cv.imwrite(write_name,img)
            
    # DEBUG (or to be decided for deprecation)
    # Load image for reference
    img = cv.imread('./calibration1.jpg')
    img_size = (img.shape[1], img.shape[0])

    # Compute camera calibration matrix (mtx) & distortion coefficients (dist) with the given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    # rvecs: rotation vectors
    # tvecs: translation vector

    # Save thee camera calibration results for later use
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open("./calibration_pickle.p", "wb"))

# TODO: if unpacking pickle file on other specified file name, prepare case handling for that
def UnpackPickle() -> Tuple:
    with open('./calibration_pickle.p', 'rb') as f:
        data = pickle.load(f)
    assert data is not None, "Missing camera calibration matrix and distortion coefficients"
    
    mtx = data['mtx']
    dist = data['dist']
    # print("mtx:\n", mtx, "\ndist:\n", dist)
    return mtx, dist

def VisualizeDistortion(image_filename: str, mtx, dist) -> None:
    """
    Visualize the before/after distortion on chessboard images using cv2.undistort
    """
    # Convert image file into OpenCV image format
    image = cv.imread(image_filename)
    
    undist = cv.undistort(image, mtx, dist, None, mtx)
    plt.figure(figsize= (18, 12))
    grid = gridspec.GridSpec(1, 2)
    # set the spacing between axes.
    grid.update(wspace=0.1, hspace=0.1)  

    img_plt = plt.subplot(grid[0])
    plt.imshow(image)
    plt.title('Original Image')

    img_plt = plt.subplot(grid[1])
    plt.imshow(undist)
    plt.title('Undistorted Image')
    plt.show()

def VisualizeTestImageDistortion(image_filename: str, mtx, dist) -> None:
    """
    Choose from the test images to demonstrate the before/after applying undistortion 
    """
    testImg = cv.imread(image_filename)
    testImg = cv.cvtColor(testImg, cv.COLOR_BGR2RGB)

    undistTest = cv.undistort(testImg, mtx, dist, None, mtx)

    #Visualize the before/after distortion on test images
    plt.figure(figsize = (18,12))
    grid = gridspec.GridSpec(1,2)
    # set the spacing between axes.
    grid.update(wspace=0.1, hspace=0.1)  

    img_plt = plt.subplot(grid[0])
    plt.imshow(testImg)
    plt.title('Original test Image')

    img_plt = plt.subplot(grid[1])
    plt.imshow(undistTest)
    plt.title('Undistorted test Image')
    plt.show()

"USEFUL FUNCTIONS FOR EXPERIMENTING WITH DIFFERENT COLOR THRESHOLDS AND GRADIENTS"

def AbsSobelThreshold(img, orient='x', thresh_min=25, thresh_max=255):
    """
    Define a function that takes an image, gradient orientation, and threshold min/max value
    """
    # Convert to grayscale
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hls = cv.cvtColor(img, cv.COLOR_RGB2HLS).astype(np.float32)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv.Sobel(l_channel, cv.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv.Sobel(l_channel, cv.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output

def MagnitudeThreshold(img, sobel_kernel=3, mag_thresh=(0, 255)):
    """
    Define a function to return the magnitude of the gradient for a given sobel kernel size and threshold values
    """
    # Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

def DirectThreshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    """
    Define a function to threshold an image for a given range and Sobel kernel
    """
    # Grayscale
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def ColorThreshold(image, sthresh=(0,255), vthresh=(0,255)):
    hls = cv.cvtColor(image, cv.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel > sthresh[0]) & (s_channel <= sthresh[1])] = 1

    hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV)
    v_channel = hsv[:,:,2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel > vthresh[0]) & (v_channel <= vthresh[1])] = 1

    output = np.zeros_like(s_channel)
    output[(s_binary == 1) & (v_binary) == 1] = 1

    # Return the combined s_channel & v_channel binary image
    return output

def S_ChannelThreshold(image: np.ndarray, sthresh=(0,255)):
    hls = cv.cvtColor(image, cv.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]  # use S channel

    # create a copy and apply the threshold
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel >= sthresh[0]) & (s_channel <= sthresh[1])] = 1
    return binary_output

def WindowMasking(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height), max(0,int(center-width)):min(int(center+width),img_ref.shape[1])] = 1
    return output

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
    cv.imshow("Region of Interest Polygon",mask)
    masked = cv.bitwise_and(input_frame, mask)
    return masked

def BirdEyeView(image: np.ndarray) -> np.ndarray:
    """
    Performing Warp Perspective to get the top-down, or Bird Eye's View to account for the 2D display of two lanes 
    converging at a distance to the horizon so that we can see that the lanes are always parallel to each other
    -- For further grasp on the concept: https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html
    """
    # ! -- NOTE on OPTIMIZE: think of how to refer/hardcode the image size since this step is 
    # performed on multiple functions, which causes redundancy and overusage
    img_size = (image.shape[1], image.shape[0]) # width, height

    bot_width = BOT_W # percentage of bottom trapezoidal width according to the image width, centered at the middle
    upper_width = TOP_W # percentage of upper trapezoidal height
    upper_height = TOP_H# percentage of the upper trapezoidal height according the image height coordinate
    bottom_height= BOT_H # percentage from top to bottom avoiding the hood of the car
    # Source image input array (region of interest for warping)
    src = np.float32([[image.shape[1]*(0.5-upper_width/2), image.shape[0]*upper_height],
                      [image.shape[1]*(0.5+upper_width/2),image.shape[0]*upper_height],
                      [image.shape[1]*(0.5+bot_width/2), image.shape[0]*bottom_height],
                      [image.shape[1]*(0.5-bot_width/2), image.shape[0]*bottom_height]])
    
    # Offset for destination image, anything not in the warped region will be transformed without consideration
    offset = img_size[0]*OFFSET
    # Destination image output array
    dst = np.float32([[offset,0],
                      [img_size[0]-offset,0],
                      [img_size[0]-offset,img_size[1]],
                      [offset,img_size[1]]])
    
    # Perform the warp perspective transform to get the transformation matrix M
    # from the source image with a size of src to destination image with a size of dst
    M = cv.getPerspectiveTransform(src, dst)
    Minv = cv.getPerspectiveTransform(dst, src) # inverse mapping
    # cv.warpPerspective(input image, transformation matrix, size, interpolation methods)
    warped = cv.warpPerspective(image, M, img_size, flags=cv.INTER_LINEAR)
    return warped, dst, src, Minv

# TODO + OPTIMIZE: COnfigure for better accuracy
def FindLanePixels(warped):
    """
    Applying convolution under the sliding window method to maximize the number of "hot" pitxels in each window
    """
    # Set up the overall class to do the lane line tracking
    curve_centers = Tracker(window_width=WINDOW_WIDTH, window_height=WINDOW_HEIGHT, margin = 25, ym = 10/720, xm = 4/384, smooth_factor=15)
    
    window_centroids = curve_centers.find_window_centroids(warped)
    
    # Points used to draw all the left and right windows
    l_points = np.zeros_like(warped)
    r_points = np.zeros_like(warped)
        
    # points used to find the right & left lanes
    rightx = []
    leftx = []

    # Go through each level and draw the windows 
    for level in range(0,len(window_centroids)):
        # Window_mask is a function to draw window areas
        # Add center value found in frame to the list of lane points per left, right
        leftx.append(window_centroids[level][0])
        rightx.append(window_centroids[level][1])

        l_mask = WindowMasking(WINDOW_WIDTH, WINDOW_HEIGHT, warped, window_centroids[level][0],level)
        r_mask = WindowMasking(WINDOW_WIDTH, WINDOW_HEIGHT, warped, window_centroids[level][1],level)
        # Add graphic points from window mask here to total pixels found 
        l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
        r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

    # Draw the results
    template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
    zero_channel = np.zeros_like(template) # create a zero color channel
    template = np.array(cv.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
    warpage = np.array(cv.merge((warped,warped,warped)),np.uint8) # making the original road pixels 3 color channels
    result = cv.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the original road image with window results

    return result, rightx, leftx, curve_centers

def LaneFitting(image: np.ndarray, warped: np.ndarray, Minv, rightx, leftx, curve_centers):
    img_size = (image.shape[1], image.shape[0]) # width, height

    #fit the lane boundaries to the left, right center positions found
    yvals = range(0,warped.shape[0])
    
    res_yvals = np.arange(warped.shape[0]-(WINDOW_HEIGHT/2),0,-WINDOW_HEIGHT)
    left_fit = np.polyfit(res_yvals, leftx, 2)
    left_fitx = left_fit[0]*yvals*yvals + left_fit[1]*yvals + left_fit[2]
    left_fitx = np.array(left_fitx,np.int32)
    
    right_fit = np.polyfit(res_yvals, rightx, 2)
    right_fitx = right_fit[0]*yvals*yvals + right_fit[1]*yvals + right_fit[2]
    right_fitx = np.array(right_fitx,np.int32)
    
    left_lane = np.array(list(zip(np.concatenate((left_fitx-WINDOW_WIDTH/2, left_fitx[::-1]+WINDOW_WIDTH/2),axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)
    right_lane = np.array(list(zip(np.concatenate((right_fitx-WINDOW_WIDTH/2, right_fitx[::-1]+WINDOW_WIDTH/2),axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)

    road = np.zeros_like(image)
    road_bkg = np.zeros_like(image)

    cv.fillPoly(road,[left_lane],color=[255,0,0])
    cv.fillPoly(road,[right_lane],color=[0,0,255])
    cv.fillPoly(road_bkg,[left_lane],color=[255,255,255])
    cv.fillPoly(road_bkg,[right_lane],color=[255,255,255])

    road_warped = cv.warpPerspective(road, Minv, img_size, flags=cv.INTER_LINEAR)
    road_warped_bkg= cv.warpPerspective(road_bkg, Minv, img_size, flags=cv.INTER_LINEAR)
    
    base = cv.addWeighted(image, 1.0, road_warped, -1.0, 0.0)
    lanes_overlap = cv.addWeighted(base, 1.0, road_warped, 1.0, 0.0)
    overall_result = CalculateLaneStatistics(warped=warped, lanes_overlap=lanes_overlap, curve_centers=curve_centers,
                                             res_yvals=res_yvals, leftx=leftx, rightx=rightx, yvals=yvals,
                                             left_fitx=left_fitx, right_fitx=right_fitx)
    return road, lanes_overlap, overall_result

def CalculateLaneStatistics(warped, lanes_overlap, curve_centers, res_yvals, leftx, rightx, yvals, left_fitx, right_fitx):
    """
    This function encapsulates the core functionality of calculating lane statistics 
    such as radius of curvature and lane position.
    """
    ym_per_pix = curve_centers.ym_per_pix # meters per pixel in y dimension
    xm_per_pix = curve_centers.xm_per_pix # meters per pixel in x dimension

    curve_fit_cr = np.polyfit(np.array(res_yvals,np.float32)*ym_per_pix,np.array(leftx,np.float32)*xm_per_pix,2)
    curverad = ((1 + (2*curve_fit_cr[0]*yvals[-1]*ym_per_pix + curve_fit_cr[1])**2)**1.5) /np.absolute(2*curve_fit_cr[0])
    
    # Calculate the offset of the car on the road
    camera_center = (left_fitx[-1] + right_fitx[-1])/2
    center_diff = (camera_center-warped.shape[1]/2)*xm_per_pix
    side_pos = 'left'
    if center_diff <= 0:
        side_pos = 'right'

    # Draw the text showing curvature, offset & speed
    cv.putText(lanes_overlap, 'Radius of Curvature='+str(round(curverad,3))+'m ',(50,50),cv.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv.putText(lanes_overlap, 'Vehicle is '+str(abs(round(center_diff,3)))+'m '+side_pos+' of center',(50,100), cv.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    return lanes_overlap

def PreprocessingPipeline(image_filename: str, mtx, dist):
    """
    The main preprocessing pipeline for the image, which includes the combination of binary thresholded image
    from the Sobel threshold in the x & y directions along with the color thresholds in the H & V channels, to get
    clear lane lines in all the test images
    """
    # STEP 0: Initial configuration for camera calibration matrix and distortion coefficients
    image = cv.imread(image_filename)
    """
    TODO: refine the camera matrix based on a free scaling parameter using cv.getOptimalNewCameraMatrix()
    h,  w = image.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    # then replace the second mtx argument in undistort with newcameramtx
    """
    # STEP 1: undistort the image
    undistorted = cv.undistort(image, mtx, dist, None, mtx)
    
    # NOTE + TODO: Configure the appropriate hyperparameters 
    # STEP 2: Pass through the appropriate image processing and thresholding procedures
    # blank mask
    preprocessImage = np.zeros_like(undistorted[:, :,0])
    # go through binary thresholded image from the Sobel threshold in the x & y directions
    # and colors thresholds in the H & V channel and filter the preprocessed pixels
    gradx = AbsSobelThreshold(undistorted, orient="x", thresh_min=12, thresh_max=255)
    grady = AbsSobelThreshold(undistorted, orient="y", thresh_min=25, thresh_max=255)
    c_binary = ColorThreshold(undistorted, sthresh=(100, 255), vthresh=(50, 255))
    preprocessImage[((gradx == 1) & (grady ==1) | (c_binary == 1))] = 255

    # STEP 3: Perform Warp Perspective for the Bird Eye's of View 
    warped, dst, src, Minv = BirdEyeView(preprocessImage) # get Minv(inverse matrix mapping to warp the boudaries back onto the original image)

    # STEP 4: Apply convolution and sliding window to detect lane pixels and fit the boundaries
    window_fitted, rightx, leftx, curve_centers = FindLanePixels(warped)

    # STEP 5 + 6: 
    # 5. Apply a polynomial fit to the indentified lane lines on the left and the right
    # 6. Calculate the radius of curvature using polynomial fit functions, and the position of the vehicle's center 
    # from the left or the right lane. Also, display these results along with the fitted lane lines 
    road, lanes_overlap, overall_results = LaneFitting(image, warped, Minv, rightx, leftx, curve_centers)

    # !-------------------------------------------------------!
    # TODO: work on the next steps of image processing
    # !-------------------------------------------------------!

    # DEBUG: checking image processing results
    plt.figure(figsize = (24,24))
    grid = gridspec.GridSpec(3, 3)
    # set the spacing between axes.
    grid.update(wspace=0.5, hspace=0.5)  

    plt.subplot(grid[0])
    plt.imshow(image, cmap="gray")
    plt.title('Original test Image')

    plt.subplot(grid[1])
    plt.imshow(undistorted, cmap="gray")
    for i in range(4):
        plt.plot(src[i][0],src[i][1],'rs')
        """
        NOTE - REMINDER:
        'rs' as a string argument to specify the style of markers in `plt.plot()`
        - 'r': color - red; other choices: 'b' (blue), 'g' (green)
        - 's': shape of the marker - square; other choices: 'o' (circle), '+' (plus), 'x': cross
        """
    plt.title('Undistorted Image')

    plt.subplot(grid[2])
    plt.imshow(preprocessImage, cmap="gray")
    plt.title('After Preprocessing Pipeline')

    plt.subplot(grid[3])
    plt.imshow(warped, cmap="gray")
    for i in range(4):
        plt.plot(dst[i][0],dst[i][1],'ro')
    plt.title("Bird's Eye view")

    plt.subplot(grid[4])
    plt.imshow(window_fitted, cmap='gray')
    plt.title('Window Fitting Results')
    
    # Visualize the results of identified lane lines and overlapping them on to the original undistorted image
    plt.subplot(grid[5])
    plt.imshow(road, cmap='gray')
    plt.title('Identified Lane Lines')

    plt.subplot(grid[6])
    plt.imshow(lanes_overlap, cmap='gray')
    plt.title('Lane lines overlapped on original image')

    plt.subplot(grid[7])
    plt.imshow(overall_results, cmap='gray')
    plt.title('Final Image Results')

    plt.show() 
    # ----------- DEBUG END --------------------------------!

def ProcessingPipeline2(image: np.ndarray, mtx, dist) -> np.ndarray:
    """ 
    A testing processing technique which is modified for the use of a TurtleBot3 comprises of
    some initial steps from the traditional Hough finding for more defined lines
    - Some issues when working with:
        + lanes being closer -> different orientation and positioning
    """
    # STEP 0: Initial configuration for camera calibration matrix and distortion coefficients
    assert image is not None, "file could not be read, check with os.path.exists()"
    
    undistorted = cv.undistort(image, mtx, dist, None, mtx)

    # convert image color scale (BGR) to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # applying Gaussian Blur to remove noise from the image and focus on region of interest
    # kernel size = 5 (normally distributed numbers to run across the entire image)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    
    """
    thresholds for hysteresis procedure in Canny Edge Detection
    lower_threshold = 10
    higher_threshold = 65
    """
    edges = cv.Canny(blurred, 10, 65)
    # Limiting the region of interest to reduce confusion and constrain to the necessary
    # edges for lane detection
    region = RegionOfInterest(edges)

    # STEP 3: Perform Warp Perspective for the Bird Eye's of View 
    warped, dst, src, Minv = BirdEyeView(region) # get Minv(inverse matrix mapping to warp the boudaries back onto the original image)

    # STEP 4: Apply convolution and sliding window to detect lane pixels and fit the boundaries
    window_fitted, rightx, leftx, curve_centers = FindLanePixels(warped)

    # STEP 5 + 6: 
    # 5. Apply a polynomial fit to the indentified lane lines on the left and the right
    # 6. Calculate the radius of curvature using polynomial fit functions, and the position of the vehicle's center 
    # from the left or the right lane. Also, display these results along with the fitted lane lines 
    road, lanes_overlap, overall_results = LaneFitting(image, warped, Minv, rightx, leftx, curve_centers)
    """
    opencv_display_1 = np.concatenate((image, edges, region), axis=1)
    opencv_display_2 = np.concatenate((warped, road, overall_results), axis =1)
    cv.imshow("Lane Detection Pipeline shown", np.concatenate(opencv_display_1, opencv_display_2), axis=0)
    """
    """
    # DEBUG: checking image processing results
    plt.figure(figsize = (24,24))
    grid = gridspec.GridSpec(3, 3)
    # set the spacing between axes.
    grid.update(wspace=0.5, hspace=0.5)  

    plt.subplot(grid[0])
    plt.imshow(image, cmap="gray")
    plt.title('Original test Image')

    plt.subplot(grid[1])
    plt.imshow(undistorted, cmap="gray")
    for i in range(4):
        plt.plot(src[i][0],src[i][1],'rs')
    plt.title('Undistorted Image')

    plt.subplot(grid[2])
    plt.imshow(region, cmap="gray")
    plt.title("Region of Interest applied")

    plt.subplot(grid[3])
    plt.imshow(warped, cmap="gray")
    for i in range(4):
        plt.plot(dst[i][0],dst[i][1],'ro')
    plt.title("Bird's Eye view")

    plt.subplot(grid[4])
    plt.imshow(window_fitted, cmap='gray')
    plt.title('Window Fitting Results')
    
    # Visualize the results of identified lane lines and overlapping them on to the original undistorted image
    plt.subplot(grid[5])
    plt.imshow(road, cmap='gray')
    plt.title('Identified Lane Lines')

    plt.subplot(grid[6])
    plt.imshow(lanes_overlap, cmap='gray')
    plt.title('Lane lines overlapped on original image')

    plt.subplot(grid[7])
    plt.imshow(overall_results, cmap='gray')
    plt.title('Final Image Results')
    
    plt.subplot(grid[8])
    plt.imshow(edges, cmap="gray")
    plt.title("Canny Edge detection method applied")
    plt.show()
    """
    return overall_results, leftx[-1], rightx[-1]

def WebcamLaneDetection(source, mtx, dist):
    """
    Lane Detection on webcam's "cv.VideoCapture" 
    """
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
        processed_frame = ProcessingPipeline2(frame, mtx, dist)
        
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

def LaneDetectionbyFrame(image: np.ndarray, mtx, dist):
    """
    Perform lane detection on frame by frame
    """
    assert mtx is not None and dist is not None

    processed, left_x, right_x = ProcessingPipeline2(image=image, mtx=mtx, dist=dist)
    return processed, left_x, right_x

# Driver code for testing functions and progression of the overall algorithm of deeper methods on lane detection
if __name__ == "__main__":
    # coefficients and values for distortion
    # NOTE: see if we might need to hardcode it, removing the intial step of refering to the zipped pickle file
    # like having something to save as you run the ROS node, and use that value for the rest of the running time of ROS node
    mtx, dist = UnpackPickle()
    # VisualizeTestImageDistortion('test_images/test5.jpg', mtx, dist)
    # PreprocessingPipeline("test_images/test5.jpg", mtx, dist)

    # Preprocess2('test_images/real_track_0.png', mtx, dist)

    WebcamLaneDetection(0, mtx, dist)