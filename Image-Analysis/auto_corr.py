"""
Filename: auto_corr.py
------------------------------------------------------
This file shows how a pixel varies with time.

By: Marcus Forst
sort_nicely credit: Ned B (https://nedbatchelder.com/blog/200712/human_sorting.html)
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
CAPILLARY_ROW = 584
CAPILLARY_COL = 950
BKGD_COL = 1024
BKGD_ROW = 584
FRAME = 50

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
        image = cv2.imread(os.path.join(FILEFOLDER, image_list[i]))
        image_2D = np.mean(image, axis=2)
        image_array[i] = image_2D
    return image_array
def generate_operator(vector):
    """
    Use a vector to generate an operator of a series of row vectors stacked next to each other.
    This operator, when multiplied by a matrix with diagonals equal to the original vector, (an eigenvector)
    can be used to find diagonals and off-diagonals.
    :param vector: 1D numpy array
    :return: operator: 2D numpy array
    """
    operator = np.array(vector)
    A = np.array(vector)
    for n in range(len(vector) - 1):
        operator = np.vstack((operator, A))
    return np.transpose(operator)
def diagonalize(operator, vector):
    """
    This multiplies the operator and the diagonalized eigenvector to get the diagonalized matrix.
    :param operator: 2D array. This operator is a series of the same vector, vertically stacked.
    :param vector: 1D array. This is the eigenvector
    :return: diag_matrix: 2D array
    """
    eigenvector = np.diag(vector)
    return np.matmul(operator, eigenvector)
def vector_to_diag(vector):
    """
    this takes a vector and turns it into a diagonal similarity matrix.
    :param vector: 1D numpy array
    :return: diag_matrix: 2D numpy array
    """
    vector = np.array(vector)
    operator = generate_operator(vector)
    return diagonalize(operator, vector)
def cycle_rows(array):
    """
    Cycle arrays by taking the top row and putting it on the bottom.
    :param array: 2D numpy array
    :return: cycled: 2D array
    """
    return np.vstack((array[1:], array[0]))
def test():
    A = np.array([1, 2, 3, 4])
    B = generate_operator(A)
    print(B)
    C = diagonalize(B, A)
    print(C)
    D = cycle_rows(C)
    print(D)
    E = cycle_rows(D)
    print(E)
def average_array(array):
    """
    This returns an averaged array with half length of the input 1d array
    :param array: 1d numpy array length 2n
    :return: 1d array length n
    """
    if np.mod(len(array), 2) == 0:
        return (array[::2] + array[1::2]) // 2
    else:
        return (array[:-1:2] + array[1::2]) // 2

def main():
    # Import images
    images = get_images(FILEFOLDER)
    image_array = load_image_array(images)
    # write background
    background = np.mean(image_array, axis=0)
    max = np.max(image_array)
    # Select points to do auto correlation, on and off the capillary. This gives us 1D arrays of frames for each pixel
    pix_cap_vector = image_array[:, CAPILLARY_ROW, CAPILLARY_COL]
    pix_bkgd_vector = image_array[:, BKGD_ROW, BKGD_COL]

    auto_corr_fn = sm.tsa.acf(pix_cap_vector, nlags= pix_cap_vector.shape[0], fft = False)
    auto_corr_fn_fft = sm.tsa.acf(pix_cap_vector, nlags=pix_cap_vector.shape[0], fft = True)
    acf_bkgd = sm.tsa.acf(pix_bkgd_vector, nlags=pix_bkgd_vector.shape[0], fft = False)
    print(len(auto_corr_fn))

    """
    Plot log of the exponential function
    """
    log_acf_sqr = -1 * np.log(auto_corr_fn * auto_corr_fn)
    # find line of best fit
    a, b = np.polyfit(range(log_acf_sqr.shape[0]), log_acf_sqr, 1)
    # add points to plot
    plt.scatter(range(log_acf_sqr.shape[0]), log_acf_sqr)
    # add line of best fit to plot
    plt.plot(range(log_acf_sqr.shape[0]), a * range(log_acf_sqr.shape[0]) + b)
    plt.title("Log of autocorrelation")
    plt.show()
    print(a)
    plt.plot(auto_corr_fn * auto_corr_fn)
    plt.plot(acf_bkgd * acf_bkgd)
    plt.title("Autocorrelation for capillary and background")
    plt.show()

# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    main()
    print("--------------------")
    print(time.time() - ticks)

