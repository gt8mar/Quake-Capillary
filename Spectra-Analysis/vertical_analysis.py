"""
Filename: vertical_analysis.py thingy
-------------------------------------------------
This file averages the wavelength information from the spectrum and plots position versus time.

By: Marcus Forst

sort_nicely credit: Ned B (https://nedbatchelder.com/blog/200712/human_sorting.html)
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import re
from mpl_toolkits.axes_grid1 import make_axes_locatable


FILEFOLDER = os.path.join("C:\\", 'Users', 'Luke', 'Documents', 'Marcus', 'Data', "220427_tyler",
                          '220427_vid2_moco')
"""
Function definitions:
--------------------------------------------------------------------------------------------------------------
""""""
Sort nicely: 
------------------------------------------------------------------------------------------------------------------
"""

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


images = [img for img in os.listdir(FILEFOLDER) if img.endswith(".tif")]
sort_nicely(images)
"""
---------------------------------------------------------------------------------------------------------------------
"""



"""
Analyze image: 
------------------------------------------------------------------------------------------------------------------
"""

number_of_files = len(images)
vertical_list = []
time_series = np.empty((number_of_files, 500))  # 500 is the difference between the row bounds aka the number
                                                # of rows in the wavelength_slice

# We iterate through every other file because the diffraction files are oddly indexed.
for i in range(len(images)):
    # Read in image from the filefolder
    picture = np.array(cv2.imread(os.path.join(FILEFOLDER, images[i])))
    # Collapse from 3D (color) image array to 2D array
    new_picture = np.mean(picture, axis=2)
    # Select the bottom left of the wavelength image. The Wavelengths on the left are more
    # different between white and red. the bottom is where the capillaries are.
    wavelength_slice = new_picture[400:900, 200:800]
    vertical_picture = np.mean(wavelength_slice, axis = 1)
    vertical_list.append(vertical_picture)
    time_series[i] = vertical_picture
    # add the spectrum to the plot
    # plt.plot(vertical_picture)
# plt.show()

time_series_T = np.transpose(time_series)
ax = plt.subplot()
im = ax.imshow(time_series_T)



# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
plt.show()



# # look at the difference in spectra
# diff_rbc = spectrum_list[1]-spectrum_list[0]
# plt.plot(diff_rbc)
# diff_wbc = spectrum_list[2]-spectrum_list[0]
# plt.plot(diff_wbc)
# plt.show()
