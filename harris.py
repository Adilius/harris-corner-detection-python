import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# Reads image from system
def read_image(image_name: str):
    return cv2.imread(image_name)

# Converts image to grayscale
def convert_grayscale(image):
    return cv2.cvtColor(image, cv2.cv2.COLOR_BGR2GRAY)

# Shows image
def show_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.xticks([]), plt.yticks([])
    plt.show()

# Implementation of harris corner detection
def harris(image_name: str, scale_factor: float, window_size: int, k: float, threshold: str):

    # Read and scale image
    image = read_image(image_name)
    image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
    height, width, _ = image.shape

    # Convert to grayscale for faster computational speeds
    optimized_image = convert_grayscale(image)

    # Blur
    #optimized_image = cv2.GaussianBlur(src=optimized_image, ksize=(3,3), sigmaX=0, sigmaY=0, borderType=cv2.BORDER_DEFAULT)

    # Get horizontal and vertical sobel
    dx = cv2.Sobel(src=optimized_image, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    dy = cv2.Sobel(src=optimized_image, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)

    dxdy = dx * dy
    dx = np.square(dx)
    dy = np.square(dy)

    # Get window offset
    offset = int(window_size/2)
    y_range = height - offset
    x_range = width - offset

    # Go through image with window
    for y in range(offset, y_range):
        for x in range(offset, x_range):

            sum_x2 = np.sum(dx  [y-offset: y+offset+1, x-offset: x+offset+1])
            sum_y2 = np.sum(dy  [y-offset: y+offset+1, x-offset: x+offset+1])
            sum_xy = np.sum(dxdy[y-offset: y+offset+1, x-offset: x+offset+1])

            # Create H-matrix
            H = np.array([[sum_x2, sum_xy], [sum_xy, sum_y2]])

            # Calculate determinant and trace of matrix
            #det = (sum_x2 * sum_y2) - (sum_xy**2)     
            #trace = sum_x2 - sum_y2
            det = np.linalg.det(H)
            trace = np.matrix.trace(H)

            # Calculate r for Harris Corner equation
            r = det - k * (trace**2)

            if r < threshold:
                image[y,x] = (0,0,255)

    show_image(image)

def ORB(image_name_1: str, image_name_2: str):
    img1 = cv2.imread(image_name_1,cv2.IMREAD_GRAYSCALE)          # queryImage
    img2 = cv2.imread(image_name_2,cv2.IMREAD_GRAYSCALE) # trainImage
    # Initiate ORB detector
    orb = cv2.ORB_create()
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3),plt.show()

if __name__ == '__main__':
    #image_name = 'right_feature.png'
    #harris(image_name=image_name, scale_factor=0.5, window_size=5, k=0.04, threshold=10000.00)
    ORB('left_feature.png', 'right_feature.png')