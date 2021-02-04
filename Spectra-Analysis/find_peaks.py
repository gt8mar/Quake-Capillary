"""
File: find_peaks.py
---------------------
This program inputs a tiff file of horizontally diffracted light and outputs a line spectrum
for that light with the peaks identified.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from scipy import signal
import pandas as pd


# Insert file here
FILENAME_SAMPLE = "Basler_acA1300-200um__23253950__20201007_191129578_21.tiff"
FILENAME_BKGD = "Basler_acA1300-200um__23253950__20201007_191619058_75.tiff"
FILENAME_HGAR = "Basler_acA1300-200um__23253950__20201007_191244451_4.tiff"
FILENAME_HGAR2 = "Basler_acA1300-200um__23253950__20201007_191323400_2.tiff"
FILENAME_LSR = "Basler_acA1300-200um__23253950__20201007_190716443_20.tiff"

DATA_FOLDER = os.path.join('E:\\', 'Quake', '2020-10-07_Selected_pics')

# Matplotlib Parameters:
plt.rcParams['figure.figsize'] = (6, 4)
plt.rcParams['figure.dpi'] = 150

"""
This program opens a file and displays it to the user. 
"""
def main():
    # Read in images of sample, background, and calibration wavelengths
    im_sample = cv2.imread(os.path.join(DATA_FOLDER, FILENAME_SAMPLE), 0)
    im_background = cv2.imread(os.path.join(DATA_FOLDER, FILENAME_BKGD), 0)
    im_hgar = cv2.imread(os.path.join(DATA_FOLDER, FILENAME_HGAR), 0)
    im_hgar2 = cv2.imread(os.path.join(DATA_FOLDER, FILENAME_HGAR2), 0)
    im_laser = cv2.imread(os.path.join(DATA_FOLDER, FILENAME_LSR), 0)


    # Check to ensure that the dimensions of the files make sense.
    height, width = im_sample.shape
    print(height, width)

    # Collapse the image into a 1D array:
    mean_sample_spectra = np.mean(im_sample, axis = 0)
    mean_background_spectra = 1.65*np.mean(im_background, axis = 0) +20
    # mean_absorbtion_spectra = mean_background_spectra - mean_sample_spectra
    # mean_hgar_spectra = np.mean(im_hgar, axis = 0)
    mean_hgar2_spectra = np.mean(im_hgar2, axis = 0)
    mean_laser_spectra = np.mean(im_laser, axis = 0)




    plt.ylabel('intensity')

    peaks, _ = signal.find_peaks(mean_hgar2_spectra, prominence = 1)
    peaks2, _2 = signal.find_peaks(mean_laser_spectra, prominence = 1)
    print(peaks)
    print(peaks2)

    """
    Use pandas to import selected peaks, then use a linear regression to find conversion from pixels to nm
    """
    point_array = np.array([[435.8, 1160],
                                [546.1, 600],
                                [577, 434],
                                [650, 73],
                                [532, 671]])
    df = pd.DataFrame(point_array,
                      columns = ["nm","pixels"]
                      )
    #df.insert(2,"nm")
    # nm_line = 663.7 -0.1966(pixels)


    def pixel_to_nm(spectrum):
        df = pd.DataFrame(data = spectrum, columns=["intensity"])
        df["nm"] = 663.7136 - 0.1966 * df.index
        return df

    df_hg2 = pixel_to_nm(mean_hgar2_spectra)
    plt.plot(df_hg2["nm"], df_hg2["intensity"])
    plt.show()

    # Plot the original curves:
    plt.plot(mean_sample_spectra)
    plt.plot(mean_background_spectra)
    # plt.plot(mean_hgar_spectra)
    plt.plot(mean_hgar2_spectra)
    plt.plot(mean_laser_spectra)
    # plt.plot(mean_absorbtion_spectra)

    # Plot the peaks
    plt.plot(peaks, mean_hgar2_spectra[peaks], "x")
    plt.plot(peaks2, mean_laser_spectra[peaks2], "x")

    plt.show()


    # cv2.imshow("Sample with white light", im_sample)
    # # Wait for the user to press any key
    # cv2.waitKey(0)
    # # Then close the windows
    # cv2.destroyAllWindows()




if __name__ == '__main__':
    main()
