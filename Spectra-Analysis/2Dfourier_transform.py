"""
Filename: 2Dfourier_transform.py
------------------------------------------------------
This file calculates the 2D fourier transform of an image.

By: Marcus Forst
sort_nicely credit: Ned B (https://nedbatchelder.com/blog/200712/human_sorting.html)
average_in_circle credit: Nicolas Gervais (https://stackoverflow.com/questions/49330080/numpy-2d-array-selecting-indices-in-a-circle)
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import re
import time
import pandas as pd
import statsmodels.api as sm

FILEFOLDER = 'C:\\Users\\Luke\\Documents\\Marcus\\Data\\220513\\pointer2small'
FLOW_FILE = 'centerline_array_7_long.csv'



def main():
    # Import images
    flow_image = np.genfromtxt(FLOW_FILE, delimiter=',', dtype=int)
    print(flow_image.shape)
    flow_image_fft = np.fft.rfft2(flow_image)
    flow_image_fft_log = np.log(flow_image_fft)
    # # ignore low freq noise:
    # flow_image_fft[0] = 0
    # flow_image_fft[:][0] = 0
    # flow_image_fft[-1] = 0
    # flow_image_fft[:][-1] = 0
    print(np.abs(flow_image_fft))

    # np.savetxt('centerline_array_7.csv', centerline_array, delimiter=',')
    plt.imshow(flow_image)
    plt.show()

    # Plot pixels vs time:
    plt.imshow(np.abs(flow_image_fft_log))
    plt.title('centerline pixel values per time')
    plt.xlabel('frame')
    plt.ylabel('centerline pixel')
    plt.show()

    # TODO: calculate flow rate
    return 0



# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    main()
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))

