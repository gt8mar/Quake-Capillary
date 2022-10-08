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
FLOW_FILE = 'test_skeleton_coords_5v3.csvcenterline_array_long_offset.csv'
LOOP_BUFFER_START = 216 #350
LOOP_BUFFER_STOP = 252 #415
SIGMA = 50
RADIUS = 9
SLOPE_CORRECTION = 0.08

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
def plot_sections_fft(flow_image, sections, filter = False, buffer = False):
    """
    This function plots ffts of different sections of the capillaries all together
    :param flow_image: 2D numpy array
    :return: 0
    """
    rows = flow_image.shape[0]
    slopes = []
    fig, axes = plt.subplots(sections // 2 + 1, 2, figsize=(12, 6))
    ax = axes.ravel()
    if buffer:
        """
        first calculate the fft for the hook part of the capillary
        """
        image = flow_image[LOOP_BUFFER_START:LOOP_BUFFER_STOP]
        flow_image = np.vstack((flow_image[:LOOP_BUFFER_START],
                                flow_image[LOOP_BUFFER_STOP:]))
        if filter:
            wimage = image * window('hann', image.shape)
        else:
            wimage = image
        image_fft = np.log(np.abs(fftshift(fft2(wimage))))
        image_fft = chop_fft(image_fft, window_radius=30)
        ax[sections-1].imshow(image_fft,
                     vmin=4, vmax=12
                     )
        ax[sections-1].set_title("buffer fft " + str(sections-1))
        print(image.shape)
        print(image_fft.shape)
        a, b, a1, b1 = find_slope_max_method(image_fft)
        slopes.append([a1, b1])
        ax[sections-1].plot(a1 * range(15, image.shape[0] - 15) + b1,
                            range(15, image.shape[0] - 15),
                            color = 'white'
                            )
        ax[sections - 1].set_xlim(0, 59)
        sections -= 1
    section_size = rows//sections

    for i in range(sections):
        """
        Now iterate through the sections
        """
        image = flow_image[i * section_size: (i + 1) * section_size]
        if filter:
            wimage = image * window('hann', image.shape)
        else:
            wimage = image
        image_fft = np.log(np.abs(fftshift(fft2(wimage))))
        image_fft = chop_fft(image_fft, window_radius=30)
        ax[i].imshow(image_fft,
                     vmin=4, vmax=12
                     )
        ax[i].set_title("fft " + str(i))
        a, b, a1, b1 = find_slope_max_method(image_fft)
        slopes.append([a1, b1])
        ax[i].plot(a1 * range(15, image_fft.shape[0] - 15) + b1,
                   range(15, image_fft.shape[0] -15),
                   color = 'white'
                   )
        ax[i].set_xlim(0, 59)
    plt.show()
    return slopes
def plot_sections_space(flow_image, sections, filter = False, buffer = False, correction = False):
    """
    This function plots ffts of different sections of the capillaries all together
    :param flow_image: 2D numpy array
    :return: 0
    """
    rows = flow_image.shape[0]
    slopes = []
    fig, axes = plt.subplots(sections // 2 + 1, 2, figsize=(12, 6))
    ax = axes.ravel()
    if buffer:
        """
        first calculate the fft for the hook part of the capillary
        """
        image = flow_image[LOOP_BUFFER_START:LOOP_BUFFER_STOP]
        flow_image = np.vstack((flow_image[:LOOP_BUFFER_START],
                                flow_image[LOOP_BUFFER_STOP:]))
        if filter:
            wimage = image * window('hann', image.shape)
        else:
            wimage = image
        image_fft = np.log(np.abs(fftshift(fft2(wimage))))
        image_fft = chop_fft(image_fft, window_radius=30)
        ax[sections -1].imshow(image)
        ax[sections -1].set_xlim(0, image.shape[1])
        ax[sections -1].set_aspect('equal')
        ax[sections-1].set_title("Loop section " + str(sections-1))
        print(image.shape)
        print(image_fft.shape)
        a, b, a1, b1 = find_slope_max_method(image_fft)
        ax[sections - 1].plot(a1 * range(-(image.shape[0]//3) , image.shape[0] - (image.shape[0]//3)),
                              range(image.shape[0]), color='white')
        perp_slope = -1/a1
        if correction:
            ax[sections - 1].plot((perp_slope+SLOPE_CORRECTION) * range(-(image.shape[0] // 3),
                                   image.shape[0] - (image.shape[0] // 3)) + image.shape[1] // 2,
                                   range(image.shape[0]), color='pink')
        else:
            ax[sections - 1].plot(perp_slope * range(-(image.shape[0] // 3),
                                  image.shape[0] - (image.shape[0] // 3)) + image.shape[1] // 2,
                                  range(image.shape[0]), color='red')

        slopes.append([a1, b1])
        # plot_line(image, [a1, b1])
        sections -= 1
    section_size = rows//sections

    for i in range(sections):
        """
        Now iterate through the sections
        """
        image = flow_image[i * section_size: (i + 1) * section_size]
        if filter:
            wimage = image * window('hann', image.shape)
        else:
            wimage = image
        image_fft = np.log(np.abs(fftshift(fft2(wimage))))
        image_fft = chop_fft(image_fft, window_radius=30)
        ax[i].imshow(image)
        ax[i].set_title("Section " + str(i))
        ax[i].set_xlim(0, image.shape[1])
        ax[i].set_aspect('equal')
        a, b, a1, b1 = find_slope_max_method(image_fft)
        ax[i].plot(a1 * range(-(image.shape[0]//3) , image.shape[0] - (image.shape[0]//3)),
                              range(image.shape[0]), color='white')
        perp_slope = -1/a1
        if correction:
            for j in range(15):
                ax[i].plot((perp_slope + SLOPE_CORRECTION) * range(-(image.shape[0] // 3),
                           image.shape[0] - (image.shape[0] // 3)) + (20 * j),
                           range(image.shape[0]), color='pink')
        else:
            ax[i].plot(perp_slope * range(-(image.shape[0] // 3),
                       image.shape[0] - (image.shape[0] // 3)) + image.shape[1]//2,
                       range(image.shape[0]), color='red')
        slopes.append([a1, b1])
        # plot_line(image, [a1, b1])
        # ax[i].plot(a1 * range(15, image_fft.shape[0] - 15) + b1,
        #            range(15, image_fft.shape[0] -15),
        #            color = 'white'
        #            )
        # ax[i].set_xlim(0, 59)
    plt.show()
    return slopes
def plot_fft_contour(flow_image, filter = False):
    if filter:
        wimage = flow_image * window('hann', flow_image.shape)
        image_fft = fft2(wimage)

    else:
        image_fft = fft2(flow_image)
    # create the x and y coordinate arrays (here we just use pixel indices)
    lena = np.array(Image.fromarray(np.log(np.abs(fftshift(image_fft))))
                    .resize(size=(flow_image.shape[0] // 4, flow_image.shape[1] // 4))
                    )
    plt.imshow(lena)
    plt.show()
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
def find_max_rows_cols(fft_slice, verbose = False):
    """
    This function returns the indicies of the maximum values of each row
    :param fft_slice: 2D numpy array
    :return row_max_list: list of coordinates
    """
    row_max_list = []
    col_max_list = []
    transpose_fft = np.transpose(fft_slice)
    if verbose:
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
def test_fft(flow_image, fft):
    wimage = flow_image * window('hann', flow_image.shape)
    test_fft = fftshift(fft2(wimage))
    test_fft = np.abs(test_fft)
    test_fft = np.log(test_fft)
    rows = test_fft.shape[0]
    cols = test_fft.shape[1]
    center_row = rows // 2
    center_col = cols // 2
    window_radius = 50
    fig, axes = plt.subplots(1, 3)
    ax = axes.ravel()
    ax[0].imshow(test_fft[center_row - window_radius:center_row + window_radius,
                 center_col - window_radius:center_col + window_radius])
    ax[1].imshow(fft)
    ax[2].imshow(flow_image)
    plt.show()
def test_speeds():
    a = [3.94, 4.26, 0,  5.15, 11.58]
    ax = range(0, 700, 700//len(a)+1)
    b = [1.89, 0, 3.56]
    bx = range(0, 700, 700//len(b)+1)
    c = [5.67, 5.26, 7.5, 0,  7.8, 9.73, 10]
    cx = range(0, 700, 700//len(c)+1)

    plt.plot(ax, a, marker = 'o', label = "5 sections")
    plt.plot(bx, b, marker = 'o', label = "3 sections")
    plt.plot(cx, c, marker = 'o', label = "7 sections")
    plt.title("blood speed at each capillary section")
    plt.xlabel("capillary section")
    plt.ylabel("speed (pixels/frame)")
    plt.legend()
    plt.show()
    return 0
def find_slope_max_method(fft_slice, verbose = False):
    """
    This function finds the slope of an fft by finding the maximum of every column and
    using those points to fit a line. The symmetry of an fft forces the line through the
    centerpoint.
    :param fft_slice: 2D numpy array
    :param verbose: bool, to plot or not to plot
    :return: parameters for a line. b will rarely be zero, the line must go through
    the midpoint of the image, not zero.
    """
    row_max_list, col_max_list = find_max_rows_cols(fft_slice)
    a, b = np.polyfit(row_max_list, range(len(row_max_list)), 1)
    a2, b2 = np.polyfit(col_max_list, range(len(col_max_list)), 1)
    # print(a, b)
    print(a2, b2)
    if verbose:
        plt.scatter(range(len(row_max_list)), row_max_list)
        plt.scatter(col_max_list, range(len(col_max_list)))
        row_fit = plt.plot(range(len(row_max_list)), (a * range(len(row_max_list))) + b, label="row fit")
        col_fit = plt.plot(range(len(col_max_list)), (a2 * range(len(col_max_list))) + b2, label='col fit')
        plt.legend()
        plt.title("220513 capillary 5 fft slope")
        plt.show()
    return a, b, a2, b2
def plot_row_fft(flow_image):
    # This plots an example row of the image and the fourier transform.
    row_example = flow_image[290]
    row_example_fft = np.fft.fft(row_example)
    fig, axes = plt.subplots(1, 2)
    ax = axes.ravel()
    ax[0].plot(row_example)
    ax[1].plot(np.abs(row_example_fft[1:]))
    plt.show()
    return 0
def plot_lines(image, slopes):
    slope = slopes[0]
    a = slope[0]
    slopes = slopes[1:]
    sections = len(slopes)
    section_size = image.shape[0]//sections
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_xlim(0, image.shape[1])
    ax.plot(a * range(-LOOP_BUFFER_START, image.shape[0]-LOOP_BUFFER_START),
            range(image.shape[0]), color = 'white')
    # for i in range(len(slopes)):
    #     ax.plot(a * range(0, image.shape[0]-360), range(image.shape[0]), color = 'white')

    plt.show()
    return 0
def plot_line(image, slope):
    a = slope[0]
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_xlim(0, image.shape[1])
    ax.plot(a * range(image.shape[0]),
            range(image.shape[0]), color='white')
    plt.show()
    return 0

def main():
    # Import images
    flow_image = np.genfromtxt(FLOW_FILE, delimiter=',', dtype=int)
    norm_image = normalize_image(flow_image)
    plt.imshow(flow_image)
    plt.show()
    # Save image as png:
    # im = Image.fromarray(norm_image)
    # im.save("centerline_array_7_long.tiff")

    # plot_fft_contour(flow_image, filter = True)

    fft = make_pretty_fft(flow_image, "full centerline vs time fft", saturate= True, filter=True, verbose=False)

    fft_slice = chop_fft(fft, center_reduced=True, radius=RADIUS)

    # test_fft(flow_image, fft)
    # plt.imshow(fft_slice)
    # plt.show()

    # find_slope_max_method(fft_slice)

    # slopes = plot_sections_fft(flow_image, sections=5, filter = True, buffer = True)
    slopes = plot_sections_space(flow_image, sections=5, filter = True, buffer = True, correction=True)
    # plot_lines(flow_image, slopes)

    # plot_row_image

    # test_speeds()


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

