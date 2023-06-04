import numpy as np
from scipy.ndimage import gaussian_filter
from filter import sobel_x, sobel_y
import cv2
import skimage

# luminance: luminance channel of an image
def gaussian_edge_detection(luminance, sigma=5.0, sharpness=5.0):
    K = np.sqrt(1.6)
    D1 = gaussian_filter(luminance, sigma=sigma)
    D2 = gaussian_filter(luminance, sigma=K*sigma)
    D = D1-D2
    E = np.copy(D)
    E[E > 0] = 1
    E[E < 0] = 0
    E = -E + 1
    E *= np.tanh(D*sharpness)
    E += 1
    return E

def canny_edge_detection(luminance, minVal, maxVal, useCV2=True):
    if useCV2: # use faster cv2 implementation
        edges = skimage.img_as_float(cv2.Canny(np.uint8(luminance), minVal, maxVal))
        edges -= 0.5
        edges *= -1
        edges += 0.5
        return edges
    
    # gaussian filter to reduce noise
    filtered = gaussian_filter(luminance, sigma=1.0)

    # sobel filter to get gradient and direction
    G_x = sobel_x(filtered)
    G_y = sobel_y(filtered)
    G = np.nan_to_num(np.sqrt(G_x + G_y))
    angle = np.arctan(G_y/G_x) # [-pi/2,pi/2]
    angle /= np.pi/2 # [-1,1]
    angle = np.round(angle)
    angle = np.abs(angle) # [0, 1]
    # 0 = horizontal direction
    # 1 = vertical direction

    # non-maximum suppression
    M, N = filtered.shape
    left_shift = np.zeros((M, N))
    left_shift[:M-1,:] = filtered[1:,:]

    right_shift = np.zeros((M, N))
    right_shift[1:,:] = filtered[:M-1,:]

    up_shift = np.zeros((M, N))
    up_shift[:,:N-1] = filtered[:,1:]

    down_shift = np.zeros((M, N))
    down_shift[:,1:] = filtered[:,:N-1]

    local_max_horizontal = (filtered == np.maximum(np.maximum(right_shift, left_shift), filtered)).astype(int)
    
    local_max_vertical = (filtered == np.maximum(np.maximum(up_shift, down_shift), filtered)).astype(int)

    suppressed = angle * local_max_horizontal + (angle*-1+1) * local_max_vertical

    # Hysteresis thresholding

    G *= suppressed
    strong = 1
    weak = 0.75
    minVal = 0.1
    maxVal = 0.5
    G[G<=minVal] = 0
    G[G>=maxVal] = strong
    
    G[np.logical_and(np.less(G,maxVal), np.greater(G,minVal))] = weak

    img = G
    M, N = img.shape  
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] >= strong) or (img[i+1, j] >= strong) or (img[i+1, j+1] >= strong)
                        or (img[i, j-1] >= strong) or (img[i, j+1] >= strong)
                        or (img[i-1, j-1] >= strong) or (img[i-1, j] >= strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass

    edges = img - 1
    edges *= -1
    return edges

