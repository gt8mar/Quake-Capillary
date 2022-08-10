"""
Filename: standard_dev.py
------------------------------------------------------
Compare standard deviations of each pixel.

By: Marcus Forst
sort_nicely credit: Ned B (https://nedbatchelder.com/blog/200712/human_sorting.html)
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import re
import time
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable


FILEFOLDER = 'C:\\Users\\Luke\\Documents\\Marcus\\Data\\220513\\pointer2_slice_stable'
BIN_FACTOR = 8
BORDER = 50

ticks = time.time()


# Sort images first
def tryint(s):
    try:
        return int(s)
    except:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)


images = [img for img in os.listdir(FILEFOLDER) if img.endswith(".tif") or img.endswith(
    ".tiff")]  # if this came out of moco the file suffix is .tif otherwise it's tiff
sort_nicely(images)

"""
------------------------------------------------------------------------------------------------------------------
"""


def bin_image_by_2_space(image):
    return (image[:, ::2, ::2] + image[:, 1::2, ::2]
            + image[:, ::2, 1::2] + image[:, 1::2, 1::2]) // 4


def bin_image_by_2_time(image):
    return (image[:-1:2, :, :] + image[1::2, :, :]) // 2


# Initialize array for images
z_time = len(images)
z_time_extended = len(images) + 1
image_example = cv2.imread(os.path.join(FILEFOLDER, images[0]))
rows, cols, layers = image_example.shape
image_array = np.zeros((z_time, rows, cols), dtype='uint16')

# loop to populate array
for i in range(z_time):
    image = cv2.imread(os.path.join(FILEFOLDER, images[i]))
    image_2D = np.mean(image, axis=2)
    # im = Image.fromarray(image_2D)
    # bin = im.resize((cols//BIN_FACTOR, rows//BIN_FACTOR), Image.BILINEAR)
    # plt.imshow(bin)
    # plt.show()
    image_array[i] = image_2D

background = np.mean(image_array, axis=0)

# Get rid of edges
image_array_slice = image_array[:, BORDER:-BORDER, BORDER:-BORDER]
standard_dev = np.std(image_array_slice, axis=0)

# plot image background
ax = plt.subplot()
im = ax.imshow(standard_dev)
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
plt.show()
