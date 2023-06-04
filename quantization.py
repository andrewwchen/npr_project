import numpy as np

# luminance: luminance channel of an image
def linear_quantization(luminance, bin_size=10, sharpness=5.0):
    min_L = int(np.min(luminance))
    max_L = int(np.max(luminance))
    bins = np.array(range(min_L, max_L, bin_size))
    bin_centers = (bins[1:]+bins[:-1])/2
    q_nearest = bins[np.digitize(luminance, bin_centers)] # nearest bin
    return q_nearest + (bin_size/2*np.tanh((luminance-q_nearest)*sharpness))