"""
Filename: find_centerline_v3.py
-------------------------------------------------
This file segments an image using ____ technique from scikit image

By: Marcus Forst

png to polygon credit: Stephan Hügel (https://gist.github.com/urschrei/a391f6e18a551f8cbfec377903920eca)
find skeletons: (https://scikit-image.org/docs/dev/auto_examples/edges/plot_skeleton.html#sphx-glr-auto-examples-edges-plot-skeleton-py)
sort_continuous credit: Imanol Luengo (https://stackoverflow.com/questions/37742358/sorting-points-to-form-a-continuous-line)
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
from sklearn.neighbors import NearestNeighbors
import networkx as nx


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
def enumerate_capillaries(image, short = False, verbose = False):
    """
    This function finds the number of capillaries and returns an array of images with one
    capillary per image.
    :param image: 2D numpy array
    :param short: boolian, if you want to test using one capillary. Default is false.
    :return: 3D numpy array: [capillary index, row, col]
    """
    row, col = image.shape
    print(row, col)
    contours = measure.find_contours(image, 0.8)
    print("The number of capillaries is: " + str(len(contours)))
    if short == False:
        contour_array = np.zeros((len(contours), row, col))
        for i in range(len(contours)):
            grid = np.array(measure.grid_points_in_poly((row, col), contours[i]))
            contour_array[i] = grid
            if verbose == True:
                plt.plot(contours[i][:, 1], contours[i][:, 0], linewidth=2, label = "capilary " + str(i)) #plt.imshow(contour_array[i])   # plt.plot(contours[i][:, 1], contours[i][:, 0], linewidth=2) this shows all of the enumerated capillaries
                # plt.show()
            else:
                pass
        plt.gca().invert_yaxis()
        plt.legend()
        plt.show()
        return contour_array
    # This is only used if we don't want to iterate through the whole set of contours.
    # This is for testing.
    else:
        contour_array = np.zeros((1, row, col))
        for i in range(1):
            grid = np.array(measure.grid_points_in_poly((row, col), contours[i]))
            contour_array[i] = grid
            # plt.plot(contours[i][:, 1], contours[i][:, 0], linewidth=2)   # this shows all of the enumerated capillaries
        # plt.show()
        return contour_array
def make_skeletons(image, verbose = True):
    """
    This function uses the FilFinder package to find and prune skeletons of images.
    :param image: 2D numpy array or list of points that make up polygon mask
    :return fil.skeleton: 2D numpy array with skeletons
    :return distances: 1D numpy array that is a list of distances (which correspond to the skeleton coordinates)
    """
    # Load in skeleton class for skeleton pruning
    fil = FilFinder2D(image, beamwidth=0 * u.pix, mask=image)
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
    distances = distance[fil.skeleton.astype(bool)]
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
    # TODO: fix if not equal indexes between x and y
    x = array_2D[0]   #unbend_capillaries_1D(array_2D[0])
    y = unbend_capillaries_1D(array_2D[1])
    new_array = np.vstack((x, y))
    return np.transpose(new_array)
def sort_continuous(array_2D, verbose = False):
    """
    This function takes a 2D array of shape (2, length) and sorts it in order of continuous points
    :param array_2D: 2D numpy array
    :param verbose: bool, shows plots if true.
    :return sorted_array: 2D numpy array
    :return opt_order: something that slices into the correct order when given a 1D array
    """
    if isinstance(array_2D, (list, np.ndarray)):

        points = np.c_[array_2D[0], array_2D[1]]
        neighbors = NearestNeighbors(n_neighbors=2).fit(points)
        graph = neighbors.kneighbors_graph()
        graph_connections = nx.from_scipy_sparse_array(graph)
        paths = [list(nx.dfs_preorder_nodes(graph_connections, i)) for i in range(len(points))]
        min_dist = np.inf
        min_idx = 0

        for i in range(len(points)):
            order = paths[i]  # order of nodes
            ordered = points[order]  # ordered nodes
            # find cost of that order by the sum of euclidean distances between points (i) and (i+1)
            cost = (((ordered[:-1] - ordered[1:]) ** 2).sum(1)).sum()
            if cost < min_dist:
                min_dist = cost
                min_idx = i
        opt_order = paths[min_idx]
        row = array_2D[0][opt_order]
        col = array_2D[1][opt_order]
        sorted_array = np.c_[row, col]
        if verbose == True:
            plt.plot(col, row)
            plt.show()
            print(sorted_array)
            print(opt_order)
        return sorted_array, opt_order
    else:
        raise Exception('wrong type')


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
    contours = enumerate_capillaries(segmented_2D, short=False, verbose=True)
    skeletons = []
    capillary_distances = []
    skeleton_coords = []
    flattened_distances = []
    for i in range(contours.shape[0]):
        skeleton, distances = make_skeletons(contours[i], verbose=True)     # Skeletons come out in the shape
        sorted_skeleton_coords, optimal_order = sort_continuous(np.asarray(np.nonzero(skeleton)), verbose= False)
        ordered_distances = distances[optimal_order]
        capillary_distances.append(ordered_distances)
        flattened_distances += list(distances)
        skeleton_coords.append(sorted_skeleton_coords)
    plt.show()

    # Plot example of capillary
    plt.plot(capillary_distances[0])
    plt.show()
    # Plot all capillaries together
    for i in range(len(capillary_distances)):
        plt.plot(capillary_distances[i])
    plt.title('Capillary radii')
    plt.show()


    # sorted_skeleton_example = skeleton_coords[7]
    # # save csv file to import into blood_flow_linear.py
    # # np.savetxt('test3.csv', skeleton_coords, delimiter=',')
    #
    # np.savetxt('test_skeleton_coords_7.csv', sorted_skeleton_example, delimiter=',')


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
