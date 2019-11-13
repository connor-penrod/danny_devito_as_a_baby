import numpy as np

def hist_match(source, template):
    """
    Author: ali_m @ StackOverflow
    
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)    
    
def normalize(source, template):
    match_r = hist_match(source[:,:,0], template[:,:,0])
    match_g = hist_match(source[:,:,1], template[:,:,1])
    match_b = hist_match(source[:,:,2], template[:,:,2])

    match = np.dstack((match_r,match_g,match_b))
    match = match.astype(np.uint8)
    return match
    
if "__name__" == "__main__":
    from io import BytesIO
    from PIL import Image, ImageDraw
    import matplotlib.pyplot as plt
    image1 = Image.open("devito1.png")
    image2 = Image.open("family1.jpg")

    image1 = np.array(image1)
    image2 = np.array(image2)

    match = normalize(image1,image2)

    plt.figure()
    plt.imshow(image1)

    plt.figure()
    plt.imshow(image2)

    print(match)
    plt.figure()
    plt.imshow(match)

plt.show()