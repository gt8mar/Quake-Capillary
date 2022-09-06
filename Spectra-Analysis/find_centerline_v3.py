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
    print(len(contours))
    contour_array = np.zeros((len(contours), row, col))
    for i in range(len(contours)):
        # plt.plot(contours[i][:, 1], contours[i][:, 0], linewidth=2)
        grid = np.array(measure.grid_points_in_poly((row, col), contours[i]))
        contour_array[i] = grid
        # plt.imshow(grid)
    # plt.show()
    print(contour_array.shape)
    return contour_array

def make_skeletons(image):
    """
    This function uses the FilFinder package to find and prune skeletons of images.
    :param image: 2D numpy array or list of points that make up polygon mask
    :return: 2D numpy array with skeletons
    """
    fil = FilFinder2D(image, mask=image)
    fil.preprocess_image(skip_flatten=True)
    fil.medskel(verbose=True)
    fil.analyze_skeletons(branch_thresh=BRANCH_THRESH * u.pix, prune_criteria='length',
                          skel_thresh=BRANCH_THRESH * u.pix)
    plt.imshow(fil.skeleton, origin='lower')
    plt.show()
    return fil.skeleton

def add_radii_value(distance_array):
    """
    This function creates a list of distances for the skeleton of an image
    :param distance_array: array of skeleton distance values
    :return: list of distances
    """
    skeleton_coordinates = np.transpose(np.nonzero(distance_array))
    print(len(skeleton_coordinates))
    distances = []
    for i in range(len(skeleton_coordinates)):
        row = skeleton_coordinates[i][0]
        col = skeleton_coordinates[i][1]
        distances.append(distance_array[row][col])
    return distances

"""
Import and select polygons:
---------------------------------------------------------------------------------------------------------------------
"""
segmented = cv2.imread(os.path.join(FILEFOLDER, FILENAME))
segmented_2D = np.mean(segmented, axis=2)
segmented_2D[segmented_2D != 0] = 1
# TODO: clean this up
contours = enumerate_capillaries(segmented_2D)
print(contours[0])
skeleton2 = make_skeletons(contours[0])
plt.imshow(skeleton2)
plt.show()
# fil = FilFinder2D(segmented_2D, mask=segmented_2D)
# fil.preprocess_image(skip_flatten=True)
# fil.medskel(verbose=True)
# fil.analyze_skeletons(branch_thresh = BRANCH_THRESH * u.pix, prune_criteria='length',
#                       skel_thresh=BRANCH_THRESH * u.pix)
# plt.imshow(fil.skeleton, origin= 'lower')
# plt.show()
#
# skeleton, distance = medial_axis(segmented_2D, return_distance=True)
# skeleton = skeleton.astype(np.uint8)
# skel = skeletonize(segmented_2D)
# skel_lee = skeletonize(segmented_2D, method='lee')
# dist_on_skel = distance*skeleton
# print(distance)
# print(np.transpose(np.nonzero(distance)))
# distances = add_radii_value(dist_on_skel)           # This adds the distance values into a list
# print(len(distances))
#
# plt.hist(distances)
# plt.show()
#
#
# fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
# ax = axes.ravel()
#
# ax[0].imshow(segmented_2D, cmap=plt.cm.gray)
# ax[0].set_title('original')
# ax[0].axis('off')
#
# ax[1].imshow(dist_on_skel, cmap='magma')
# ax[1].contour(segmented_2D, [0.5], colors='w')
# ax[1].set_title('medial_axis')
# ax[1].axis('off')
#
# ax[2].imshow(skel, cmap=plt.cm.gray)
# ax[2].set_title('skeletonize')
# ax[2].axis('off')
#
# ax[3].imshow(skel_lee, cmap=plt.cm.gray)
# ax[3].set_title("skeletonize (Lee 94)")
# ax[3].axis('off')
#
# fig.tight_layout()
# plt.show()
#
# plt.imshow(dist_on_skel, cmap='magma')
# plt.contour(segmented_2D, [0.5], colors = 'w')
# plt.title('Centerlines with distance values')
# plt.show()


"""
-----------------------------------------------------------------------------
"""


# """
# --------------------------------------------------------------------------------------------
# """
#
# # def main():
# #     masks = []
# #     load_masks()
# #     for mask in masks:
# #         mask_centerline = centerline(mask)
#
#
#
#
