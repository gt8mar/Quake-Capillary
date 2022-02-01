"""
File: signal_to_noise.py
---------------------
This program inputs tiff files and looks at their signal to noise ratios.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import re
import time

FILEFOLDER = os.path.join("C:\\", 'Users', 'gt8ma', 'OneDrive', 'Documents', 'Quake', 'Bluud', 'Data',
                          'pol_series_min_illum')

SELECTED_ROW = 706
OFFSET = 20
RANGE_CUTOFF = 1001

"""
Sort nicely: 
------------------------------------------------------------------------------------------------------------------
"""
ticks = time.time()


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


images = [img for img in os.listdir(FILEFOLDER) if img.endswith(".tiff")]
sort_nicely(images)
"""
---------------------------------------------------------------------------------------------------------------------
"""
print(images)

"""
------------------------------------------------------------------------------------------------------------------------
Signal to noise function:
"""
def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)


"""
Begin signal processing:
"""
# import files:
signals = []
signals_fft = []
snr_list = []
mean_pixel_list = []
for i in range(len(images)):
    picture = np.array(cv2.imread(os.path.join(FILEFOLDER, images[i])))
    new_picture = np.mean(picture, axis=2)  # This makes it 2D instead of 3D
    # # This chops the image into smaller pieces
    row = new_picture[SELECTED_ROW, :]
    mean_pixel = np.mean(row)/10
    mean_pixel_list.append(mean_pixel)
    # Calculate signal to noise ratio using mean and standard deviation.
    snr = signaltonoise(row, axis = 0, ddof = 0)
    snr_list.append(snr)
    signals.append(row[0:RANGE_CUTOFF])
    # plt.plot(row)
    # plt.show()

    row_fft = np.fft.fft(row[0:RANGE_CUTOFF])
    row_fft[0] = 0
    row_fft[-1] = 0
    signals_fft.append(row_fft)
    xvals = range(0,50)
    # plt.plot(xvals, np.abs(np.real(row_fft))[0:50])
    # ax = plt.gca()
    # ax.set_xlim([0, 50])
    # ax.set_ylim([0, 1000])
    # print("\nsignaltonoise ratio for " + str(i) + ": ", signaltonoise(row, axis=0, ddof=0))
    # plt.show()

plt.plot(snr_list)
plt.plot(mean_pixel_list)
plt.show()

while True:
    fig, (ax1, ax2) = plt.subplots(2)
    # fig.suptitle('Vertically stacked subplots')


    for i in range(5):
        ax1.plot(signals[i] + (OFFSET * i), linewidth=1)
        ax2.plot(np.abs(np.real(signals_fft[i]))+ (OFFSET * i *5), linewidth=1)

    plt.show()

    for i in range(5, 10):
        plt.plot(signals[i] + (OFFSET * i), linewidth=1)
    plt.show()

    break

print(signals[0].shape)
