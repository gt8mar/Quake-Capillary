"""
Filename: plot_poly.py
-------------------------------------------------
This file segments an image using ____ technique from scikit image

By: Marcus Forst

sort_nicely credit: Ned B (https://nedbatchelder.com/blog/200712/human_sorting.html)
png to polygon credit: Stephan Hügel (https://gist.github.com/urschrei/a391f6e18a551f8cbfec377903920eca)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from centerline.geometry import Centerline
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon
from skimage import measure
import geopandas as gpd
import shapely.wkt

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
def test():
    a = np.arange(6).reshape((2, 3))
    b = a.transpose()
    print(a)
    print(b)
    return 0

def main():
    """
    Import and select polygons:
    ---------------------------------------------------------------------------------------------------------------------
    """
    segmented = cv2.imread(os.path.join(FILEFOLDER, FILENAME))
    segmented_2D = np.mean(segmented, axis=2)
    print(segmented_2D.shape)

    contours = measure.find_contours(segmented_2D, 1.0) # TODO: for loop to find multiple of these
    poly = Polygon(contours[0]).simplify(1.0)
    xp, yp = poly.exterior.xy

    # Create centerline:
    attributes_cap = {"id":1, "name": "polygon", "valid": True}
    centerline_cap = Centerline(poly, **attributes_cap)

    print(len(centerline_cap.geoms))
    print(centerline_cap.geoms[0][0])
    plt.plot(centerline_cap.geoms[0][0])
    plt.show()

    # make dataframe of linestrings from centerline:
    geo_df = pd.DataFrame(centerline_cap.geoms)
    print(geo_df.head())
    print(geo_df.shape)

    for line_string in centerline_cap.geoms:
        gs_ls = gpd.GeoSeries(pd.Series(line_string).apply(shapely.wkt.loads))
        ax = gs_ls.plot()
        plt.plot()

    # gdf = gpd.GeoDataFrame(geometry=poly)
    # gdf.plot()



# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    main()
