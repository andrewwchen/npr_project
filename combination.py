from skimage.color import gray2rgb

# fill_img: RGB image
# edge_img: Grayscale image [0,1]
def combine_edge_as_albedo(fill_img, edge_img, threshold=0.5):
    albedo = gray2rgb(edge_img)
    albedo[albedo >= threshold] = 1
    return fill_img * albedo

def combine_edge_as_cutoff(fill_img, edge_img, threshold=0.5):
    cut = gray2rgb(edge_img)
    cut[cut >= threshold] = 1
    cut[cut < threshold] = 0
    return fill_img * cut