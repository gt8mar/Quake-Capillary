"""
Filename: write_background_file.py
------------------------------------------------------
This file takes a series of images, creates a background file, and creates a folder with
background subtracted files.
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


image_folder = 'C:\\Users\\Luke\\Documents\\Marcus\\Data\\220427_tyler\\220427_vid2_moco'
images = [img for img in os.listdir(image_folder) if img.endswith(".tif")]                          # if this came out of moco the file suffix is .tif otherwise it's tiff
sort_nicely(images)


# Here we make a list of image files
image_files = []
for i in range(len(images)):
    picture = np.array(cv2.imread(os.path.join(image_folder, images[i])))
    new_picture = np.mean(picture, axis=2)
    # # This chops the image into smaller pieces (important if there has been motion correction)
    new_new_picture = new_picture[25:1055, 25:1425]
    # new_new_picture[new_new_picture > 5] = 5
    image_files.append(new_new_picture)

ROWS, COLS = image_files[0].shape


image_files = np.array(image_files)
background = np.mean(image_files, axis=0)

ax = plt.subplot()
im = ax.imshow(background)

# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
plt.show()


"""
Extra functions
-----------------------------------------------------------------------------------------
"""

# Enhance contrast
image_files = image_files - background
print(np.max(image_files))
print(np.min(image_files))
MIN = 0 - np.min(image_files)
image_files = image_files + MIN
MULT = 255 / np.max(image_files)
image_files = image_files * round(MULT, 1)
print('the following should never be less than 0')
print(np.min(image_files))

# Plot with newly enhanced contrast
ax = plt.subplot()
im = ax.imshow(image_files[10])
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
plt.show()


# write new folder of reduced images:
cwd = os.getcwd()
folder = image_folder + "_background"
path = os.path.join(cwd, folder)
if folder not in os.listdir(cwd):
    os.mkdir(path)

for i in range(len(image_files)):
    file = image_files[i]
    filename = images[i]
    # Make 3D again
    # indexing with np.newaxis inserts a new 3rd dimension, which we then repeat the
    # array along, (you can achieve the same effect by indexing with None, see below)
    file_3D = np.repeat(file[:, :, np.newaxis], 3, axis=2)
    file_3D = np.uint8(file_3D)
    # write to new folder
    cv2.imwrite(os.path.join(path, filename), file)

# Add background file
bkgd_name = str(images[0].strip("."))
bkgd_name += "_background"
bkgd_name += ".tiff"
cv2.imwrite(os.path.join(path, bkgd_name), background)


print("--------------------")
print("--------------------")


print(time.time() - ticks)
ticks = time.time()

