"""
Filename: find_centerline_v3.py
-------------------------------------------------
This file segments an image using ____ technique from scikit image

By: Marcus Forst

png to polygon credit: Stephan Hügel (https://gist.github.com/urschrei/a391f6e18a551f8cbfec377903920eca)
find skeletons: (https://scikit-image.org/docs/dev/auto_examples/edges/plot_skeleton.html#sphx-glr-auto-examples-edges-plot-skeleton-py)
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import re
from skimage import measure
from skimage.morphology import medial_axis, skeletonize
from fil_finder import FilFinder2D
import astropy.units as u


FILEFOLDER = os.path.join("C:\\", 'Users', 'Luke', 'Documents', 'Marcus', 'Data', '220513')
FILENAME = "background_segmented.png"
BRANCH_THRESH = 40

"""
Function definitions:
--------------------------------------------------------------------------------------------------------------
"""

def test():
    a = np.arange(6).reshape((2, 3))
    b = a.transpose()
    print(a)
    print(b)
    return 0
def enumerate_capillaries(image):
    """
    This function finds the number of capillaries and returns an array of images with one
    capillary per image.
    :param image: 2D numpy array
    :return: 3D numpy array: [capillary index, row, col]
    """
    row, col = image.shape
    print(row, col)
    contours = measure.find_contours(image, 0.8)
    print("The number of capillaries is " + str(len(contours)))
    contour_array = np.zeros((len(contours), row, col))
    for i in range(len(contours)):
        grid = np.array(measure.grid_points_in_poly((row, col), contours[i]))
        contour_array[i] = grid
    return contour_array
def make_skeletons(image):
    """
    This function uses the FilFinder package to find and prune skeletons of images.
    :param image: 2D numpy array or list of points that make up polygon mask
    :return: 2D numpy array with skeletons
    """
    # Load in skeleton class for skeleton pruning
    fil = FilFinder2D(image, mask=image)
    # Use separate method to get distances
    skeleton, distance = medial_axis(image, return_distance=True)
    # This is a necessary step for the fil object. It does nothing.
    fil.preprocess_image(skip_flatten=True)
    # This makes the skeleton
    fil.medskel()
    # This prunes the skeleton
    fil.analyze_skeletons(branch_thresh=BRANCH_THRESH * u.pix, prune_criteria='length',
                          skel_thresh=BRANCH_THRESH * u.pix)
    # Multiply the distances by the skeleton, selects out the distances we care about.
    distance_on_skeleton = distance * fil.skeleton
    distances = add_radii_value(distance_on_skeleton)           # adds the distance values into a list
    plt.hist(distances)
    plt.show()
    plt.imshow(distance_on_skeleton, cmap='magma')
    plt.show()
    return fil.skeleton, distances
def add_radii_value(distance_array):
    """
    This function creates a list of distances for the skeleton of an image
    :param distance_array: array of skeleton distance values
    :return: list of distances
    """
    skeleton_coordinates = np.transpose(np.nonzero(distance_array))
    distances = []
    for i in range(len(skeleton_coordinates)):
        row = skeleton_coordinates[i][0]
        col = skeleton_coordinates[i][1]
        distances.append(distance_array[row][col])
    return distances

"""
---------------------------------------------------------------------------------------------------------------------
"""
def main():
    # Read in the mask
    segmented = cv2.imread(os.path.join(FILEFOLDER, FILENAME))
    # Make mask 2D
    segmented_2D = np.mean(segmented, axis=2)
    # Make mask either 1 or 0
    segmented_2D[segmented_2D != 0] = 1
    # Make a numpy array of images with isolated capillaries. The mean/sum of this is segmented_2D.
    contours = enumerate_capillaries(segmented_2D)
    skeletons = []
    capillary_distances = []
    for contour in contours:
        skeleton, distances = make_skeletons(contour)
        skeletons.append(skeleton)
        capillary_distances.append(distances)
        

    return 0







"""
-----------------------------------------------------------------------------
"""
# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    main()
