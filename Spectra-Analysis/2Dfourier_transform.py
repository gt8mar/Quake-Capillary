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
import time
from PIL import Image
from scipy.fft import fft2, fftshift
from skimage.filters import window


FILEFOLDER = 'C:\\Users\\Luke\\Documents\\Marcus\\Data\\220513\\pointer2small'
FLOW_FILE = 'test_skeleton_coords_5.csvcenterline_array_long.csv'
SIGMA = 50

def make_pretty_fft(image, title = 'FFT', saturate = False, filter = False, verbose = True):
    """
    This function makes an fft and plots it unless verbose is false.
    :param image: 2D numpy array
    :param title: string, Title of plot
    :param saturate: bool, Whether to saturate the fft
    :param filter: bool, whether to use hann filter to reduce low freq noise
    :param verbose: bool, whether to plot or not
    :return:
    """
    if filter:
        wimage = image * window('hann', image.shape)
        image_fft = fft2(wimage)

    else:
        image_fft = fft2(image)
    # Plot centerline vs time
    fft = np.log(np.abs(fftshift(image_fft)))

    if verbose:
        if saturate:
            plt.imshow(fft,
                       vmin=4, vmax=12
                       )
        else:
            plt.imshow(fft)
        plt.title(title)
        plt.xlabel('1/time')
        plt.ylabel('1/position')
        plt.colorbar()
        plt.show()
        return fft
    else:
        return fft
def fft_test(image):
    """
    This function displays ffts of different slices of the image
    :param image: 2D image array
    :return: 0
    """
    make_pretty_fft(image[:215], "centerline vs time, top half")
    make_pretty_fft(image[225:], "centerline vs time, bottom half")
    # make_pretty_fft(flow_image[256:315])
    # make_pretty_fft(flow_image[327:411])
    # make_pretty_fft(flow_image[421:])
    return 0
def normalize_image(image):
    """
    This function normalizes images to make it possible to save as a png or tiff file.
    :param image: 2d numpy array
    :return: 2D numpy array, scaled to 255
    """
    image = image - np.min(image)
    image = image / np.max(image)
    image *= 255
    image = np.rint(image)
    return image
def gaussian(x, mu, sig):
    """
    This outputs a gaussian function given x inputs
    :param x: range of input values
    :param mu: mean of gaussian
    :param sig: standard deviation of gaussian
    :return: range of values as the output of gaussian function.
    """
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
def gaussian_test(y_values, mu_list, sig):
    """
    This function tests out the gaussian function
    :param y_values:
    :param mu_list:
    :param sig:
    :return:
    """
    for mu in mu_list:
        plt.plot(y_values, gaussian(y_values, mu, sig))
    plt.show()
def select_flow(image, y_values, mu, sig):
    selector_1D = gaussian(y_values, SIGMA, SIGMA)
    selector_2D = np.transpose(np.tile(selector_1D, (image.shape[1], 1)))
    selected_flow = selector_2D * image
    # make_pretty_fft(selected_flow, saturate=True)
    return selected_flow
def gaussian_test_2(flow_image):
    y_values = np.arange(flow_image.shape[0])
    y_sections = np.arange(SIGMA, flow_image.shape[0] + (SIGMA), 2 * SIGMA)
    print(y_sections)
    # gaussian_test(y_values, y_sections, SIGMA)
    for i in range(len(y_sections)-1):
        image_fft = fft2(flow_image[i * 2 * SIGMA: (i + 1) * 2 * SIGMA])
        image_fft_gauss = fft2(select_flow(flow_image, y_values, y_sections[i], SIGMA))
        fig, axes = plt.subplots(2, 1, figsize=(8, 4))
        ax = axes.ravel()
        ax[0].imshow(np.log(np.abs(fftshift(image_fft))),
                     vmin=4, vmax=12
                     )
        ax[0].set_title("fft " + str(i))
        ax[1].imshow(np.log(np.abs(fftshift(image_fft_gauss))),
                     vmin=4, vmax=12
                     )
        plt.show()
def plot_sections(flow_image, filter = False):
    """
    This function plots ffts of different sections of the capillaries all together
    :param flow_image: 2D numpy array
    :return: 0
    """
    fig, axes = plt.subplots(4, 2, figsize=(12, 6))
    ax = axes.ravel()
    # y_sections = range(0, flow_image.shape[0], 2 * SIGMA)
    for i in range(7):
        image = flow_image[i * 2 * SIGMA: (i + 1) * 2 * SIGMA]
        if filter:
            wimage = image * window('hann', image.shape)
        else:
            wimage = image
        image_fft = fft2(wimage)
        ax[i].imshow(np.log(np.abs(fftshift(image_fft))),
                     vmin=4, vmax=12
                     )
        ax[i].set_title("fft " + str(i))
    plt.show()
    return 0
def plot_fft_contour(flow_image):
    image_fft = fft2(flow_image)
    # create the x and y coordinate arrays (here we just use pixel indices)
    lena = np.array(Image.fromarray(np.log(np.abs(fftshift(image_fft))))
                    .resize(size=(flow_image.shape[0] // 4, flow_image.shape[1] // 4))
                    )
    xx, yy = np.mgrid[0:lena.shape[0], 0:lena.shape[1]]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(xx, yy, lena, rstride=1, cstride=1, linewidth=0)
    plt.show()
def chop_fft(fft, window_radius = 50, center_reduced = False, radius = 7):
    """
    This function prunes the fft to be a square and zeros out the three inner rows and columns
    :param fft: 2D numpy array with zero centered
    :param window_radius: window to create
    :return: 2D numpy array with shape (2 * window_radius, 2 * window_radius)
    """
    rows = fft.shape[0]
    cols = fft.shape[1]
    center_row = rows//2
    center_col = cols//2
    x = np.arange(0, fft.shape[1])
    y = np.arange(0, fft.shape[0])
    mask = (x[np.newaxis, :] - center_col) ** 2 + (y[:, np.newaxis] - center_row) ** 2 < radius ** 2
    if center_reduced:
        fft[mask] = 0
        # fft[center_row - 1:center_row + 2] = 0
        # fft[:, center_col -1: center_col + 2] = 0
    fft_slice = fft[center_row - window_radius:center_row + window_radius,
                    center_col - window_radius:center_col + window_radius]
    return fft_slice
def find_max_rows_cols(fft_slice):
    """
    This function returns the indicies of the maximum values of each row
    :param fft_slice: 2D numpy array
    :return row_max_list: list of coordinates
    """
    row_max_list = []
    col_max_list = []
    transpose_fft = np.transpose(fft_slice)
    fig, axes = plt.subplots(1, 2)
    ax = axes.ravel()
    ax[0].imshow(fft_slice)
    ax[1].imshow(transpose_fft)
    plt.show()
    for i in range(fft_slice.shape[0]):
        row_max = np.argmax(fft_slice[i])
        row_max_list.append(row_max)
    # for j in range(fft_slice.shape[1]):
    #     col_max = np.argmax(fft_slice[:][j])
    #     col_max_list.append(col_max)
    for j in range(transpose_fft.shape[0]):
        col_max = np.argmax(transpose_fft[j])
        col_max_list.append(col_max)
    return row_max_list, col_max_list



def main():
    # Import images
    flow_image = np.genfromtxt(FLOW_FILE, delimiter=',', dtype=int)
    norm_image = normalize_image(flow_image)
    # im = Image.fromarray(norm_image)
    # im.save("centerline_array_7_long.tiff")

    # print(flow_image.shape)
    # flow_image_fft = np.fft.rfft2(flow_image)
    # plt.imshow(flow_image)
    # plt.show()
    fft = make_pretty_fft(flow_image, "full centerline vs time fft", saturate= True, filter=True, verbose=False)
    fft_slice = chop_fft(fft, center_reduced=True)

    plt.imshow(fft_slice)
    plt.show()
    row_max_list, col_max_list = find_max_rows_cols(fft_slice)
    print(col_max_list)
    plt.scatter(range(len(row_max_list)), row_max_list)
    plt.scatter(col_max_list, range(len(col_max_list)))
    plt.show()

    # gaussian_test_2(flow_image)

    # plot_sections(flow_image, filter = True)


    # flow_image_fft_log = np.log(flow_image_fft)
    # row_example = flow_image[290]
    # row_example_fft = np.fft.fft(row_example)
    # plt.plot(row_example)
    # plt.plot(np.abs(row_example_fft[1:]))
    # plt.show()

    # np.savetxt('centerline_array_7.csv', flow_image_fft, delimiter=',')
    # plt.imshow(flow_image)
    # plt.show()
    #



    # TODO: calculate flow rate
    return 0



# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    main()
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))

