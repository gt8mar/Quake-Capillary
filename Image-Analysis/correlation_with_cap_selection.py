"""
Filename: correlation_with_cap_selection.py
------------------------------------------------------
This program selects out capillary vectors from the flow vector field.
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
from skimage.measure import block_reduce

FILEFOLDER = 'C:\\Users\\gt8mar\\Desktop\\data\\221010\\vid4_moco'
FILEFOLDER_SEGMENT = 'C:\\Users\\gt8mar\\Desktop\\data\\221010'
BIN_FACTOR = 4
SCALE_FACTOR = 1

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
def get_images(FILEFOLDER):
    """
    this function grabs image names, sorts them, and puts them in a list.
    :param FILEFOLDER: string
    :return: images: list of images
    """
    images = [img for img in os.listdir(FILEFOLDER) if img.endswith(".tif") or img.endswith(
        ".tiff")]  # if this came out of moco the file suffix is .tif otherwise it's tiff
    sort_nicely(images)
    return images
def load_image_array(image_list):
    """
    This function loads images into a numpy array.
    :param image_list: List of images
    :return: image_array: 3D numpy array
    """
    # Initialize array for images
    z_time = len(image_list)
    image_example = cv2.imread(os.path.join(FILEFOLDER, image_list[0]))
    rows, cols, layers = image_example.shape
    image_array = np.zeros((z_time, rows, cols), dtype='uint16')
    # loop to populate array
    for i in range(z_time):
        image_array[i] = cv2.imread(os.path.join(FILEFOLDER, image_list[i]), cv2.IMREAD_GRAYSCALE)
    return image_array
def make_correlation_matrix(image_array_binned):
    """
    This function calculates the correlation of a pixel with its nearest pixels in the next frame
    :param image_array_binned: 2D numpy array: (n row, m col)
    :return: 2D numpy array: (n-2 row, m-2 col)
    """
    # Initialize correlation matrix
    up_left = image_array_binned[:-1, 1:-1, 1:-1] * image_array_binned[1:, :-2, :-2]
    up_mid = image_array_binned[:-1, 1:-1, 1:-1] * image_array_binned[1:, :-2, 1:-1]
    up_right = image_array_binned[:-1, 1:-1, 1:-1] * image_array_binned[1:, :-2, 2:]
    center_right = image_array_binned[:-1, 1:-1, 1:-1] * image_array_binned[1:, 1:-1, 2:]
    center_left = image_array_binned[:-1, 1:-1, 1:-1] * image_array_binned[1:, 1:-1, :-2]
    down_right = image_array_binned[:-1, 1:-1, 1:-1] * image_array_binned[1:, 2:, 2:]
    down_left = image_array_binned[:-1, 1:-1, 1:-1] * image_array_binned[1:, 2:, :-2]
    down_mid = image_array_binned[:-1, 1:-1, 1:-1] * image_array_binned[1:, 2:, 1:-1]

    """-----------------------------------------------------------------------------------------------------------------"""

    # total correlations
    corr_total_right = 707 * (up_right + down_right) // 1000 + center_right
    corr_total_left = 707 * (up_left + down_left) // 1000 + center_left
    corr_total_up = 707 * (up_left + up_right) // 1000 + up_mid
    corr_total_down = 707 * (down_left + down_right) // 1000 + down_mid

    # Now we need to make vectors for the correllations:
    corr_total_right_2D = np.mean(corr_total_right, axis=0)
    corr_total_left_2D = np.mean(corr_total_left, axis=0)
    corr_total_up_2D = np.mean(corr_total_up, axis=0)
    corr_total_down_2D = np.mean(corr_total_down, axis=0)
    corr_x = corr_total_right_2D - corr_total_left_2D
    # corr_x /= 5000
    corr_y = corr_total_up_2D - corr_total_down_2D
    # corr_y /= 5000
    return corr_x, corr_y
def calc_avg_flow(corr_x, corr_y, segmented_array_binned):
    """
    This function selects capillary vector fields and sums their magnitudes.
    :param corr_x:
    :param corr_y:
    :param segmented_array_binned:
    :return:
    """
    number_nonzero = len(np.transpose(np.nonzero(segmented_array_binned)))
    avg_flow_rate = np.sqrt(
        (corr_x * corr_x * segmented_array_binned) + (corr_y * corr_y * segmented_array_binned))  # / number_nonzero
    print("The average flow magnitude is: ")
    print(np.sum(avg_flow_rate) / number_nonzero)
    print("I have no idea what these units are.")
    return avg_flow_rate / number_nonzero

def main():
    images = get_images(FILEFOLDER)
    image_array = load_image_array(images)
    background = np.mean(image_array, axis=0)

    # Even out the shapes of the images. The segmented image is chopped by 25 on all sides but the image array is already chopped by 10 rows.
    background = background[15:-25, 25:-25]
    image_array = image_array[:, 15:-25, 25:-25]
    segmented = cv2.imread(os.path.join(FILEFOLDER_SEGMENT, "vid40000segmented.png"), cv2.IMREAD_GRAYSCALE)   # this comes out as shape [row, col, 3] so in the next frame we make it even and take the mean

    """Bin images to conserve memory and improve resolution"""
    segmented_array_binned = block_reduce(segmented, (2, 2), func=np.mean)
    image_array_binned = block_reduce(image_array, (2, 2, 2), func=np.mean)
    segmented_array_binned[segmented_array_binned > 0] = 1

    corr_x, corr_y = make_correlation_matrix(image_array_binned)
    segmented_array_binned = segmented_array_binned[1:-1, 1:-1]         # This is the adjustment for the correlation matrix shrinkage

    corr_x_slice = corr_x[10:-10:BIN_FACTOR, 10:-10:BIN_FACTOR]
    corr_y_slice = corr_y[10:-10:BIN_FACTOR, 10:-10:BIN_FACTOR]
    seg_slice = segmented_array_binned[10:-10:BIN_FACTOR, 10:-10:BIN_FACTOR]

    rows2, cols2 = corr_y_slice.shape

    # plt.quiver(corr_y_slice, corr_x_slice, angles = 'xy') #, scale = 10000
    # plt.gca().invert_yaxis()
    # plt.show()

    # plt.quiver(corr_y_slice*seg_slice / SCALE_FACTOR, corr_x_slice*seg_slice / SCALE_FACTOR, angles = 'xy', scale = 10000)  #scale = 5000
    # plt.gca().invert_yaxis()
    # plt.show()

    # corr_x_round = np.around(corr_x, decimals=2)
    # corr_y_round = np.around(corr_y, decimals=2)
    #
    # # np.savetxt("corr_x_total.txt", corr_x)
    # # np.savetxt("corr_y_total.txt", corr_y)
    # # np.savetxt("corr_x_selected.txt", corr_x * segmented_array_binned)
    # # np.savetxt("corr_y_selected.txt", corr_y * segmented_array_binned)

    avg_flow = calc_avg_flow(corr_x, corr_y, segmented_array_binned)
    return 0

"""
-----------------------------------------------------------------------------
"""
# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    main()
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))
