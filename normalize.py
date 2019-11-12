from io import BytesIO
from PIL import Image, ImageDraw
import numpy as np

def normalization(im1,im2):
    #im1.show()
    im1 = np.array(im1)
    im1_min = np.min(im1, axis=(0,1))
    im1_max = np.max(im1, axis=(0,1))

    im2 = np.array(im2)
    im2_min = np.min(im2, axis=(0, 1))
    im2_max = np.max(im2, axis=(0, 1))

    comb_min = (im1_min + im2_min)/2
    comb_max = (im1_max + im2_max)/2
    comb_avg = (comb_min+comb_max)/2
    print(comb_avg)
    
    im1 = im1/comb_avg * 50
    #print(im1)
    im2 = im2/comb_avg * 50
    
    im1_pil = Image.fromarray(im1.astype('uint8'), 'RGB')
    im2_pil = Image.fromarray(im2.astype('uint8'), 'RGB')    
    im1_pil.show()
    im2_pil.show()
    
    
    
image1 = Image.open("devito1.png")
image2 = Image.open("family1.jpg")

normalization(image1,image2)