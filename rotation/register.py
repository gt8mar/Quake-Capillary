"""
Filename: no_wobble.py
------------------------------------------------------------------------
This file takes a series of images and reduces the wobble of the resulting
video.

By: Marcus Forst

"""
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import os
import glob
import skimage
from skimage import data
from skimage.feature import register_translation
from skimage.feature.register_translation import _upsampled_dft
from scipy.ndimage import fourier_shift
import re
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc
from itertools import product


shift, error, diffphase = register_translation(my_images[999], my_images[1000], 100)
print(f"Detected subpixel offset (y, x): {shift}")

print(shift[0], shift[1])




shift, error, diffphase = register_translation(file1, file2, 100)

file4 = skimage.external.tifffile.imread(file_list[3])
file4_shift = (-30.92, -45.76)

file4_offset = fourier_shift(np.fft.fftn(file4), file4_shift)
file4_offset = np.fft.ifftn(file4_offset)

shift, error, diffphase = register_translation(file2, file4_offset)
print(f"Detected subpixel offset (y, x): {shift}")

shift, error, diffphase = register_translation(image, offset_image)



print(f"Detected pixel offset (y, x): {shift}")

# subpixel precision
shift, error, diffphase = register_translation(image, offset_image, 100)


# Calculate the upsampled DFT, again to show what the algorithm is doing
# behind the scenes.  Constants correspond to calculated values in routine.
# See source code for details.
cc_image = _upsampled_dft(image_product, 150, 100, (shift*100)+75).conj()
ax3.imshow(cc_image.real)
ax3.set_axis_off()
ax3.set_title("Supersampled XC sub-area")


plt.show()

print(f"Detected subpixel offset (y, x): {shift}")

# Make a new dictionary for the stabilized video files

my_images_new = {}

# Make a list of tuples that match each subsequent file with the first file


key_combos = list(product([sorted_myimages[999]], sorted_myimages[:999]))
# print(key_combos)

# Iterate through each file and correct it to the base file

for i, j in key_combos:
    shift, error, diffphase = register_translation(my_images[i], my_images[j], 100)
    if shift[0] > 100:
        print(f"Discontinuity between {my_images[i]} and {my_images[j]} in y")
        break
    if shift[1] > 100:
        print(f"Discontinuity between {my_images[i]} and {my_images[j]} in x")
        break
    else:
        print(i, j)
        print(f"Detected subpixel offset (y, x): {shift}")

        # Take the fourier transform of the file, shift it, and then transform it back.

        file_shift = shift
        file_offset = fourier_shift(np.fft.fftn(my_images[j]), file_shift)
        file_offset = np.fft.ifftn(file_offset)

        # Create new entries in the stabilized video library
        my_images_new[i] = my_images[i]
        my_images_new[j] = file_offset.real.astype(int)

        shift, error, diffphase = register_translation(my_images_new[i], my_images_new[j], 100)
        print(f"Detected subpixel offset after adjustment (y, x): {shift}")
        print()
        print('------------------')
        print()
