import numpy as np
import skimage
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import os
import scipy.misc as sm

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def load_data(dir_name):    

    img = mpimg.imread(dir_name)
    img = rgb2gray(img)
    return img


def visualize(img, format=None, gray=False):
    plt.figure(figsize=(20, 40))
    if img.shape == 3:
        img = img.transpose(1,2,0)
    plt.imshow(img, format)
    plt.show()