import cv2
import numpy as np

# Read the input image
def crop_img(filename: str) -> None:
    image = cv2.imread(filename)

    # Define the coordinates of the ROI (top-left and bottom-right corners)
    x1, y1 = 100, 100  # Top-left corner
    x2, y2 = 300, 300  # Bottom-right corner

    # Extract the ROI using numpy array slicing
    roi = image[y1:y2, x1:x2]

    # Display the original image and the extracted ROI
    cv2.imshow('Original Image', image)
    cv2.imshow('ROI', roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_vertices_for_img(img) -> np.ndarray:
    imshape = img.shape
    height = imshape[0]
    width = imshape[1]

    vert=None

    # FIXME: have yet to understand why he put the default values
    # maybe he's guessing
    if (width, height) == (960, 540):
        region_bottom_left = (130, height - 1)
        region_top_left = (410, 330)
        region_top_right = (650, 350)
        region_bottom_right = (width - 30, height - 1)
        vert = np.array([[region_bottom_left, region_top_left, region_top_right, region_bottom_right]], dtype=np.int32)
    
    else:
        region_bottom_left = (200, 680)
        region_top_left = (600, 450)
        region_top_right = (750, 450)
        region_bottom_right = (1100, 650)
        vert = np.array([[region_bottom_left, region_top_left, region_top_right, region_bottom_right]], dtype=np.int32)
    
    return vert

def roi(filename: str):     
    img = cv2.imread(filename)  
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    vert = get_vertices_for_img(img)    
        
    # filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vert, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    cv2.imshow('Original Image', img)
    cv2.imshow('ROI', masked_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Driver code
if __name__ == "__main__":
    filename = "roadline.jpg"
    # crop_img(filename)
    roi(filename=filename)