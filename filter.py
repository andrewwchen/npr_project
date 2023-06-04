import numpy as np
from scipy.ndimage import convolve
import cv2

# luminance: luminance channel of an image
def sobel_x(luminance):
    sobel_x_kernel = np.array([
        [1,0,-1],
        [2,0,-2],
        [1,0,-1],
    ])
    return convolve(luminance, sobel_x_kernel)

# luminance: luminance channel of an image
def sobel_y(luminance):
    sobel_y_kernel = np.array([
        [1,2,1],
        [0,0,0],
        [-1,-2,-1],
    ])
    return convolve(luminance, sobel_y_kernel)

def bilateral(input, distance=15, sigmaColor=100, sigmaSpace=100, useCV2=True):
    if useCV2:
        return cv2.bilateralFilter(input, distance, sigmaColor, sigmaSpace)