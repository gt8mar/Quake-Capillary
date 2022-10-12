"""
Filename: segment.py
-------------------------------------------------
This file segments an image using ____ technique from scikit image

By: Marcus Forst

sort_nicely credit: Ned B (https://nedbatchelder.com/blog/200712/human_sorting.html)
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import re
import skimage.segmentation as seg
from centerline.geometry import Centerline
from scipy.interpolate import interp1d
from scipy.spatial import Voronoi
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon
from shapely.ops import unary_union


FILEFOLDER = os.path.join("C:\\", 'Users', 'Luke', 'Documents', 'Marcus', 'Data',
                          '220513')
FILENAME = "background_segmented.png"
"""
Function definitions:
--------------------------------------------------------------------------------------------------------------
""""""
Sort nicely: 
------------------------------------------------------------------------------------------------------------------
"""

# # Sort images first
# def tryint(s):
#     try:
#         return int(s)
#     except:
#         return s
# def alphanum_key(s):
#     """ Turn a string into a list of string and number chunks.
#         "z23a" -> ["z", 23, "a"]
#     """
#     return [tryint(c) for c in re.split('([0-9]+)', s)]
# def sort_nicely(l):
#     """ Sort the given list in the way that humans expect.
#     """
#     l.sort(key=alphanum_key)
#
# images = [img for img in os.listdir(FILEFOLDER) if img.endswith(".tif") or img.endswith(".tiff")]
# sort_nicely(images)



"""
Import and select polygons:
---------------------------------------------------------------------------------------------------------------------
"""
segmented = cv2.imread(os.path.join(FILEFOLDER, FILENAME))
segmented_2D = np.mean(segmented, axis=2)
print(segmented_2D.shape)

from skimage import measure
# find contours
# Not sure why 1.0 works as a level -- maybe experiment with lower values
contours = measure.find_contours(segmented_2D, 1.0)

# build polygon, and simplify its vertices if need be
# this assumes a single, contiguous shape
# if you have e.g. multiple shapes, build a MultiPolygon with a list comp

# RESULTING POLYGONS ARE NOT GUARANTEED TO BE SIMPLE OR VALID
# check this yourself using e.g. poly.is_valid
poly = Polygon(contours[0]).simplify(1.0)
print(poly.exterior.xy)

"""
Deal with the polygons:
---------------------------------------------------------------
"""

polygon = Polygon([[0, 0], [0, 4], [4, 4], [4, 0]])
attributes = {"id": 1, "name": "polygon", "valid": True}

centerline = Centerline(polygon, **attributes)
print(centerline.id == 1)
print(centerline.name)
print(centerline.geoms)
print(centerline.verticies)
print(len(centerline.verticies))
print(centerline)
print(centerline.ridges)

x,y = polygon.exterior.xy
print(x)
print(y)
plt.plot(x,y)
# plt.plot(centerline)
x2 = []
y2 = []
x3 = []
y3 = []
for i in range(len(centerline.verticies)):
    x2.append(centerline.verticies[i][0])
    y2.append(centerline.verticies[i][1])
# for i in range(len(centerline.ridges)):
#     x3.append(centerline.ridges[i][0])
#     y3.append(centerline.ridges[i][1])
plt.plot(x2, y2, 'o')
# plt.plot(x3, y3, 'o')
plt.show()




"""
--------------------------------------------------------------------------------------------
"""

# def main():
#     masks = []
#     load_masks()
#     for mask in masks:
#         mask_centerline = centerline(mask)




