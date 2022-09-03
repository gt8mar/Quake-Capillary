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
    contours = measure.find_contours(image, 1.0)



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
fil = FilFinder2D(segmented_2D, mask=segmented_2D)
fil.preprocess_image(skip_flatten=True)
fil.medskel(verbose=True)
fil.analyze_skeletons(branch_thresh = BRANCH_THRESH * u.pix, prune_criteria='length',
                      skel_thresh=BRANCH_THRESH * u.pix)
plt.imshow(fil.skeleton, origin= 'lower')
plt.show()

skeleton, distance = medial_axis(segmented_2D, return_distance=True)
skeleton = skeleton.astype(np.uint8)
skel = skeletonize(segmented_2D)
skel_lee = skeletonize(segmented_2D, method='lee')
dist_on_skel = distance*skeleton
print(distance)
print(np.transpose(np.nonzero(distance)))
distances = add_radii_value(dist_on_skel)           # This adds the distance values into a list
print(len(distances))

plt.hist(distances)
plt.show()


fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(segmented_2D, cmap=plt.cm.gray)
ax[0].set_title('original')
ax[0].axis('off')

ax[1].imshow(dist_on_skel, cmap='magma')
ax[1].contour(segmented_2D, [0.5], colors='w')
ax[1].set_title('medial_axis')
ax[1].axis('off')

ax[2].imshow(skel, cmap=plt.cm.gray)
ax[2].set_title('skeletonize')
ax[2].axis('off')

ax[3].imshow(skel_lee, cmap=plt.cm.gray)
ax[3].set_title("skeletonize (Lee 94)")
ax[3].axis('off')

fig.tight_layout()
plt.show()

plt.imshow(dist_on_skel, cmap='magma')
plt.contour(segmented_2D, [0.5], colors = 'w')
plt.title('Centerlines with distance values')
plt.show()


# find contours
# Not sure why 1.0 works as a level -- maybe experiment with lower values


# # build polygon, and simplify its vertices if need be
# # this assumes a single, contiguous shape
# # if you have e.g. multiple shapes, build a MultiPolygon with a list comp
#
# # RESULTING POLYGONS ARE NOT GUARANTEED TO BE SIMPLE OR VALID
# # check this yourself using e.g. poly.is_valid
# poly = Polygon(contours[0]).simplify(1.0)

# xp, yp = poly.exterior.xy
# plt.plot(xp, yp)
# plt.show()
# # attributes_cap = {"id":1, "name": "polygon", "valid": True}
# # centerline_cap = Centerline(poly, **attributes_cap)
# # print(len(centerline_cap))
# array_x = np.array(xp)
# array_y = np.array(yp)
# points = np.vstack((array_x, array_y)).transpose()  # This gives us the points in the form [(x, y), (x, y),
# print(points)  # ...] which is what Voroni needs.
# vor = Voronoi(points)
#
# plt.plot(points[:, 0], points[:, 1], 'o')
# plt.plot(vor.vertices[:, 0], vor.vertices[:, 1], '*')
# plt.xlim(100, 400);
# plt.ylim(1000, 1200)
# plt.title("first pass voroni")
# plt.show()
#
# voronoi_plot_2d(vor, show_vertices=True, line_colors='orange',
#                 line_width=2, line_alpha=0.6, point_size=2)
# plt.title("voroni regions")
# plt.show()

"""
-----------------------------------------------------------------------------
"""

#
# plt.plot(xp, yp)
# # plt.plot(x4, y4, "o")
# plt.title("Capillary outline")
# plt.show()
#
# """
# Deal with the polygons:
# ---------------------------------------------------------------
# """
#
# polygon = Polygon([[0, 0], [0, 4], [4, 4], [4, 0]])
# attributes = {"id": 1, "name": "polygon", "valid": True}
#
# centerline = Centerline(polygon, **attributes)
# print(centerline.id == 1)
# print(centerline.name)
# print(centerline.geoms)
# print(centerline.verticies)
# print(len(centerline.verticies))
# print(centerline)
# print(centerline.ridges)
#
# x,y = polygon.exterior.xy
# print(x)
# print(y)
# plt.plot(x,y)
# # plt.plot(centerline)
# x2 = []
# y2 = []
# x3 = []
# y3 = []
# for i in range(len(centerline.verticies)):
#     x2.append(centerline.verticies[i][0])
#     y2.append(centerline.verticies[i][1])
# # for i in range(len(centerline.ridges)):
# #     x3.append(centerline.ridges[i][0])
# #     y3.append(centerline.ridges[i][1])
# plt.plot(x2, y2, 'o')
# # plt.plot(x3, y3, 'o')
# plt.show()
#
#
#
#
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
