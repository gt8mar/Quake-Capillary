"""
Filename: correlation.py
------------------------------------------------------
TBD
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

FILEFOLDER = 'C:\\Users\\Luke\\Documents\\Marcus\\Data\\220513\\pointer2_slice_stable'
BIN_FACTOR = 8

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
            + image[:, ::2, 1::2] + image[:, 1::2, 1::2])//4

def bin_image_by_2_time(image):
    return (image[:-1:2,:,:] + image[1::2, :, :])//2



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

"""Bin images to conserve memory and improve resolution"""
image_array_binned_space = bin_image_by_2_space(image_array)
image_array_binned = bin_image_by_2_time(image_array_binned_space)

max = np.max(image_array_binned)

# Initialize correllation matrix
up_left = image_array_binned[:-1,1:-1,1:-1]*image_array_binned[1:,:-2,:-2]
up_mid = image_array_binned[:-1,1:-1,1:-1]*image_array_binned[1:,:-2,1:-1]
up_right =image_array_binned[:-1,1:-1,1:-1]*image_array_binned[1:,:-2,2:]
center_right =image_array_binned[:-1,1:-1,1:-1]*image_array_binned[1:,1:-1,2:]
center_left =image_array_binned[:-1,1:-1,1:-1]*image_array_binned[1:,1:-1,:-2]
down_right =image_array_binned[:-1,1:-1,1:-1]*image_array_binned[1:,2:,2:]
down_left =image_array_binned[:-1,1:-1,1:-1]*image_array_binned[1:,2:,:-2]
down_mid =image_array_binned[:-1,1:-1,1:-1]*image_array_binned[1:,2:,1:-1]

"""-----------------------------------------------------------------------------------------------------------------"""

# total correlations
corr_total_right = 707 * (up_right + down_right)//1000 + center_right
corr_total_left = 707 * (up_left + down_left)//1000 + center_left
corr_total_up = 707 * (up_left + up_right)//1000 + up_mid
corr_total_down = 707 * (down_left + down_right)//1000 + down_mid

print(corr_total_right.shape)

# Now we need to make vectors for the correllations:
corr_total_right_2D = np.mean(corr_total_right, axis=0)
corr_total_left_2D = np.mean(corr_total_left, axis=0)
corr_total_up_2D = np.mean(corr_total_up, axis=0)
corr_total_down_2D = np.mean(corr_total_down, axis=0)
corr_x = corr_total_right_2D - corr_total_left_2D
# corr_x /= 5000
corr_y = corr_total_up_2D - corr_total_down_2D
# corr_y /= 5000
print(np.max(corr_x))
print(np.max(corr_y))

"""
TODO:
Bin by 4?
"""
corr_x_slice = corr_x[10:-10:BIN_FACTOR, 10:-10:BIN_FACTOR]
corr_y_slice = corr_y[10:-10:BIN_FACTOR, 10:-10:BIN_FACTOR]
print(corr_y_slice.shape)

# y, x = np.meshgrid(np.arange(0, rows // (BIN_FACTOR*2), 1), np.arange(0, cols // (BIN_FACTOR*2), 1))
# plt.quiver(x, y, corr_x_slice, corr_y_slice)
plt.quiver(corr_y_slice, corr_x_slice)
plt.show()

# # plot image background
# ax = plt.subplot()
# im = ax.imshow(background)
# # create an axes on the right side of ax. The width of cax will be 5%
# # of ax and the padding between cax and ax will be fixed at 0.05 inch.
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(im, cax=cax)
# plt.show()


print("--------------------")
print("--------------------")

print(time.time() - ticks)
ticks = time.time()
