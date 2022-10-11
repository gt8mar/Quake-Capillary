"""
Filename: center_spectra.py thingy
-------------------------------------------------
This file selects a few rows centered around the center of the image and averages them to create a
spectrum. It does this for a few files in a folder and then plots them for comparison.

By: Marcus Forst

sort_nicely credit: Ned B (https://nedbatchelder.com/blog/200712/human_sorting.html)
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import re

FILEFOLDER = os.path.join("C:\\", 'Users', 'Luke', 'Documents', 'Marcus', 'Data',
                          '220427_tylerhelps')
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


images = [img for img in os.listdir(FILEFOLDER) if img.endswith(".tiff")]
sort_nicely(images)
"""
---------------------------------------------------------------------------------------------------------------------
"""



"""
Analyze image: 
------------------------------------------------------------------------------------------------------------------
"""


spectrum_list = []
# We iterate through every other file because the diffraction files are oddly indexed.
for i in range(1, len(images), 2):
    # Read in image from the filefolder
    picture = np.array(cv2.imread(os.path.join(FILEFOLDER, images[i])))
    # Collapse from 3D (color) image array to 2D array
    new_picture = np.mean(picture, axis=2)
    # Slice/select the middle 5 rows (blood cells should be 3-5 pixels in diameter )(check this)
    middle_rows = new_picture[538:543,:]
    print(middle_rows.shape)
    # Average the middle rows together to get a spectrum.
    spectrum = np.mean(middle_rows, axis = 0)
    spectrum_list.append(spectrum)
    print(i)
    print(images[i])
    print(spectrum.shape)
    # add the spectrum to the plot
    plt.plot(spectrum)
plt.show()

# look at the difference in spectra
diff_rbc = spectrum_list[1]-spectrum_list[0]
plt.plot(diff_rbc)
diff_wbc = spectrum_list[2]-spectrum_list[0]
plt.plot(diff_wbc)
plt.show()
