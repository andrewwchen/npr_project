#!/usr/bin/python
import os
import numpy as np
import skimage
from skimage import io, color
from quantization import linear_quantization
from edge_detection import gaussian_edge_detection, canny_edge_detection
from combination import combine_edge_as_albedo, combine_edge_as_cutoff
from filter import bilateral

### 1. Bilateral Filtering Cartoonization (BLF)
if not os.path.isdir('./results'):
    os.mkdir('./results')

## Step 1: rgb to lab
img_name = "walk"
filetype = "jpg"
filename = './data/'+img_name+"."+filetype
rgb = skimage.img_as_float32(io.imread(filename))
lab = skimage.img_as_float32(color.rgb2lab(rgb))

## Step 2: bilateral filter
distance = 15
sigmaColor = 100
sigmaSpace = 100
bil_lab = bilateral(lab, distance, sigmaColor, sigmaSpace)

## Step 3: luminance quantization
bin_size = 10
sharpnessQ = 5.0
Q_L = linear_quantization(bil_lab[:,:,0], bin_size, sharpnessQ)
Q = np.copy(bil_lab)
Q[:,:,0] = Q_L
Q_rgb = color.lab2rgb(Q)

## Step 4: edge detection
sigmaEdge = 3.0
sharpnessEdge = 3.0
minVal = 40
maxVal = 50
#edges = gaussian_edge_detection(bil_lab[:,:,0], sigmaEdge, sharpnessEdge)
edges = canny_edge_detection(lab[:,:,0], minVal, maxVal)


## Step 5: combine images
threshold = 0.5
final_alb = combine_edge_as_albedo(Q_rgb, edges, threshold)
final_cut = combine_edge_as_cutoff(Q_rgb, edges, threshold)


# uncomment to save og_lab
"""
io.imsave("results/"+img_name+"_og_lab.jpg", lab)
"""

# uncomment to save bil_lab
"""
io.imsave("results/"+img_name+"_bil_lab.jpg", bil_lab)
"""
# uncomment to save bil_L
"""
io.imsave("results/"+img_name+"_bil_L.jpg", bil_lab[:,:,0])
"""

# uncomment to save Q_lab
"""
io.imsave("results/"+img_name+"_Q_lab.jpg", Q)
"""
# uncomment to save Q_L
"""
io.imsave("results/"+img_name+"_Q_L.jpg", Q_L)
"""
# uncomment to save Q_rgb
#"""
io.imsave("results/"+img_name+"_Q_rgb.jpg", Q_rgb)
#"""
# uncomment to save E
#"""
io.imsave("results/"+img_name+"_edges.jpg", edges)
#"""
# uncomment to save final_alb
#"""
io.imsave("results/"+img_name+"_final_alb.jpg", final_alb)
#"""
# uncomment to save final_cut
#"""
io.imsave("results/"+img_name+"_final_cut.jpg", final_cut)
#"""