"""
Filename: correlation.py
------------------------------------------------------
TBD
By: Marcus Forst
sort_nicely credit: Ned B (https://nedbatchelder.com/blog/200712/human_sorting.html)
"""

import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.registration import phase_cross_correlation
from skimage.registration._phase_cross_correlation import _upsampled_dft
from scipy.ndimage import fourier_shift
from scipy.ndimage import shift as regular_shift
import matplotlib.pyplot as plt

import cv2
import os
import re
import time
from PIL import Image

FILEFOLDER = "C:\\Users\\gt8ma\\OneDrive\\Documents\\Quake\\Bluud\\Data\\pol_series_min_illum"
BIN_FACTOR = 16

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

# Initialize array for images
z_time = len(images)
z_time_extended = len(images) + 1
image_example = cv2.imread(os.path.join(FILEFOLDER, images[0]))
rows, cols, layers = image_example.shape
image_array = np.zeros((z_time, rows, cols))

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

# Initialize correllation matrix
corr_x = np.zeros((z_time, rows, cols))
corr_y = np.zeros((z_time, rows, cols))

pocket_frame = np.zeros((1, rows, cols))
image_array_extended = np.concatenate((image_array, pocket_frame))
next_frame = np.concatenate((pocket_frame, image_array))
"""-----------------------------------------------------------------------------------------------------------------"""

zero_col = np.zeros((z_time_extended, rows, 1))
zero_row = np.zeros((z_time_extended, 1, cols))
next_frame_shifted_right = np.dstack((zero_col, next_frame[:, :, 1:]))
next_frame_shifted_left = np.dstack((next_frame[:, :, 0:-1], zero_col))
corr_right = image_array_extended * next_frame_shifted_right
corr_left = image_array_extended * next_frame_shifted_left
# print((corr_left, corr_right))

# Up and down:
next_frame_shifted_up = np.hstack((next_frame[:, 1:, :], zero_row))
next_frame_shifted_down = np.hstack((zero_row, next_frame[:, 0:-1, :]))
corr_up = image_array_extended * next_frame_shifted_up
corr_down = image_array_extended * next_frame_shifted_down
# Diagonals
zero_row_small = np.zeros((z_time_extended, 1, cols - 1))
next_frame_shifted_up_right = np.hstack((next_frame[:, 1:, 1:], zero_row_small))
next_frame_shifted_up_right = np.dstack((zero_col, next_frame_shifted_up_right))
next_frame_shifted_up_left = np.hstack((next_frame[:, 1:, 0:-1], zero_row_small))
next_frame_shifted_up_left = np.dstack((next_frame_shifted_up_left, zero_col))

next_frame_shifted_down_right = np.hstack((next_frame[:, 0:-1, 1:], zero_row_small))
next_frame_shifted_down_right = np.dstack((zero_col, next_frame_shifted_down_right))
next_frame_shifted_down_left = np.hstack((next_frame[:, 0:-1, 0:-1], zero_row_small))
next_frame_shifted_down_left = np.dstack((next_frame_shifted_down_left, zero_col))

# correlations
corr_up_right = image_array_extended * next_frame_shifted_up_right
corr_up_left = image_array_extended * next_frame_shifted_up_left
corr_down_right = image_array_extended * next_frame_shifted_down_right
corr_down_left = image_array_extended * next_frame_shifted_down_left

# total correlations
corr_total_right = np.sqrt(2) * 0.5 * (corr_up_right + corr_down_right) + corr_right
corr_total_left = np.sqrt(2) * 0.5 * (corr_up_left + corr_down_left) + corr_left
corr_total_up = np.sqrt(2) * 0.5 * (corr_up_left + corr_up_right) + corr_up
corr_total_down = np.sqrt(2) * 0.5 * (corr_down_left + corr_down_right) + corr_down

print(corr_total_right.shape)

# Now we need to make vectors for the correllations:
corr_total_right_2D = np.mean(corr_total_right, axis=0)
corr_total_left_2D = np.mean(corr_total_left, axis=0)
corr_total_up_2D = np.mean(corr_total_up, axis=0)
corr_total_down_2D = np.mean(corr_total_down, axis=0)
corr_x = corr_total_right_2D - corr_total_left_2D
corr_x /= 5000
corr_y = corr_total_up_2D - corr_total_down_2D
corr_y /= 5000
print(np.max(corr_x))
print(np.max(corr_y))

"""
TODO:
Bin by 4?
"""
corr_x_slice = corr_x[::BIN_FACTOR, ::BIN_FACTOR]
corr_y_slice = corr_y[::BIN_FACTOR, ::BIN_FACTOR]
print(corr_y_slice.shape)

y, x = np.meshgrid(np.arange(0, rows // BIN_FACTOR, 1), np.arange(0, cols // BIN_FACTOR, 1))
plt.quiver(x, y, corr_x_slice, corr_y_slice)
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
