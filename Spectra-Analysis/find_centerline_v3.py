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
from skimage import measure
from skimage.morphology import medial_axis
from fil_finder import FilFinder2D
import astropy.units as u
import time


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
        # plt.plot(contours[i][:, 1], contours[i][:, 0], linewidth=2)   # this shows all of the enumerated capillaries
    # plt.show()
    return contour_array
def make_skeletons(image, verbose = True):
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
    # This plots the histogram of the capillary and the capillary with distance values.
    if verbose:
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
def unbend_capillaries_1D(array):
    """
        This returns an array with every other value taken out and sent to the back
        :param array: 1d numpy array length n
        :return: 1d array length n but every other value is removed and sent to the end
        """
    if np.mod(len(array), 2) == 0:
        return np.hstack((array[::2], np.flip(array[1::2])))
    else:
        return np.hstack((array[:-1:2], np.flip(array[1::2])))
def unbend_capillaries_2D(array_2D):
    """
        This returns an array with every other value taken out and sent to the back
        :param array: 1d numpy array length n
        :return: 1d array length n but every other value is removed and sent to the end
        """
    x = unbend_capillaries_1D(array_2D[0])
    y = unbend_capillaries_1D(array_2D[1])
    new_array = np.vstack((x, y))
    return np.transpose(new_array)


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
    capillary_distances_unwound = []
    flattened_distances = []
    for i in range(contours.shape[0]):
        skeleton, distances = make_skeletons(contours[i], verbose=False)
        skeletons.append(skeleton)
        capillary_distances.append(distances)
        flattened_distances += list(distances)
        capillary_distances_unwound.append(unbend_capillaries_1D(np.array(distances)))
    # plt.plot(capillary_distances_unwound[0])
    # # plt.plot(np.array(capillary_distances[0]))                            # This is interesting, it gives radii as found from the top
    # # plt.plot(average_array(np.array(capillary_distances[0])))             # This does the same as above but averages.
    # plt.title('Capillary radii')
    # plt.show()

    # save csv file to import into blood_flow_linear.py
    # skeleton_coords = unbend_capillaries_2D(np.nonzero(skeleton))
    # np.savetxt('test.csv', skeleton_coords, delimiter=',')

    # Make overall histogram
    # plt.hist(flattened_distances)
    # plt.show()

    # TODO: Write program to register radii maps with each other :)
    # TODO: Abnormal capillaries, how do.

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
