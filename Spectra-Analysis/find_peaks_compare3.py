"""
File: find_peaks_compare_3.py
---------------------
This program inputs a tiff file of horizontally diffracted light and outputs a line spectrum
for that light with the peaks identified. It is meant to compare 3 different spectral regimes
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from scipy import signal
import pandas as pd
import sklearn


# Insert files here
FILENAME_RBC_HG = "Basler_acA1300-200um__23253950__20201024_201100038_4.tiff"
FILENAME_WBC_HG = "Basler_acA1300-200um__23253950__20201024_201302105_4.tiff"
FILENAME_PAPER_HG = "Basler_acA1300-200um__23253950__20201024_201427690_4.tiff"


FILENAME_HGAR = "Basler_acA1300-200um__23253950__20201007_191244451_4.tiff"
FILENAME_HGAR2 = "Basler_acA1300-200um__23253950__20201007_191323400_2.tiff"
FILENAME_LSR = "Basler_acA1300-200um__23253950__20201007_190716443_20.tiff"

DATA_FOLDER_RBC = os.path.join('E:\\', 'Quake', '2020-10-24', 'RBC')
DATA_FOLDER_WBC = os.path.join('E:\\', 'Quake', '2020-10-24', 'WBC')
DATA_FOLDER_PAPER = os.path.join('E:\\', 'Quake', '2020-10-24', 'PAPER')

# Matplotlib Parameters:
plt.rcParams['figure.figsize'] = (6, 4)
plt.rcParams['figure.dpi'] = 150

"""
This program opens a file and displays it to the user. 
"""
def main():
    # Read in images of sample, background, and calibration wavelengths
    im_rbc_hg = cv2.imread(os.path.join(DATA_FOLDER_RBC, FILENAME_RBC_HG), 0)
    im_wbc_hg = cv2.imread(os.path.join(DATA_FOLDER_WBC, FILENAME_WBC_HG), 0)
    im_paper_hg = cv2.imread(os.path.join(DATA_FOLDER_PAPER, FILENAME_PAPER_HG), 0)


    # Check to ensure that the dimensions of the files make sense.
    height, width = im_rbc_hg.shape
    print(height, width)

    # Collapse the image into a 1D array:
    mean_rbc_hg_spectra = np.mean(im_rbc_hg, axis = 0)
    mean_paper_hg_spectra = np.mean(im_paper_hg, axis = 0)
    mean_wbc_hg_spectra = np.mean(im_wbc_hg, axis = 0)

    # mean_absorbtion_spectra = mean_background_spectra - mean_spectra
    # mean_hgar_spectra = np.mean(im_hgar, axis = 0)


    plt.ylabel('intensity')

    # Find peaks for each respective spectra
    peaks_rbc, properties_rbc = signal.find_peaks(mean_rbc_hg_spectra, prominence = 1, height = 0)
    peaks_wbc, properties_wbc = signal.find_peaks(mean_wbc_hg_spectra, prominence = 1, height = 0)
    peaks_paper, properties_paper = signal.find_peaks(mean_paper_hg_spectra, prominence = 1, height = 0)

    peaks_rbc_intensities = properties_rbc["peak_heights"]
    peaks_wbc_intensities = properties_wbc["peak_heights"]
    peaks_paper_intensities = properties_paper["peak_heights"]

    # """
    # Use pandas to import selected peaks, then use a linear regression to find conversion from pixels to nm
    # """
    # point_array = np.array([[435.8, 1160],
    #                             [546.1, 600],
    #                             [577, 434],
    #                             [650, 73],
    #                             [532, 671]])
    # df = pd.DataFrame(point_array,
    #                   columns = ["nm","pixels"]
    #                   )
    # #df.insert(2,"nm")
    # # nm_line = 663.7 -0.1966(pixels)

    """
    This function converts pixel values to nanometers using a linear regression. 
    """
    def pixel_to_nm(spectrum):
        df = pd.DataFrame(data = spectrum, columns=["intensity"])
        df["nm"] = 664.5629 - 0.1964 * df.index
        return df



    # Plot the original curves:
    plt.plot(mean_rbc_hg_spectra, color = "red")
    plt.plot(mean_wbc_hg_spectra, color = "blue")
    plt.plot(mean_paper_hg_spectra, color = "green")


    # Plot the peaks
    plt.plot(peaks_rbc, mean_rbc_hg_spectra[peaks_rbc], "x", color = "red")
    plt.plot(peaks_wbc, mean_wbc_hg_spectra[peaks_wbc], "x", color = "blue")
    plt.plot(peaks_paper, mean_paper_hg_spectra[peaks_paper], "x", color = "green")
    plt.show()

    # Add nm values to each dataframe
    df_rbc_hg = pixel_to_nm(mean_rbc_hg_spectra)
    df_wbc_hg = pixel_to_nm(mean_wbc_hg_spectra)
    df_paper_hg = pixel_to_nm(mean_paper_hg_spectra)
    # Convert peak arrays to nm
    peaks_rbc_nm = 664.5629 - 0.1964 *peaks_rbc
    peaks_wbc_nm = 664.5629 - 0.1964 *peaks_wbc
    peaks_paper_nm =  664.5629 - 0.1964 *peaks_paper
    print(peaks_rbc_nm)
    print(peaks_wbc_nm)
    print(peaks_paper_nm)


    plt.plot(df_rbc_hg["nm"], df_rbc_hg["intensity"], color="red")
    plt.plot(df_wbc_hg["nm"], df_wbc_hg["intensity"], color="blue")
    plt.plot(df_paper_hg["nm"], df_paper_hg["intensity"], color="green")
    plt.plot(peaks_rbc_nm, peaks_rbc_intensities, "x", color = "red")
    plt.plot(peaks_wbc_nm, peaks_wbc_intensities, "x", color = "blue")
    plt.plot(peaks_paper_nm, peaks_paper_intensities, "x", color = "green")
    plt.show()

    # Plot the absorbtion spectra rbc
    plt.plot(df_rbc_hg["nm"], df_paper_hg["intensity"] - df_rbc_hg["intensity"], color="orange", label = "rbc_abs")
    plt.plot(df_wbc_hg["nm"], df_paper_hg["intensity"] - df_wbc_hg["intensity"], color="purple", label = "wbc_abs")
    plt.show()

    # cv2.imshow("Sample with white light", im)
    # # Wait for the user to press any key
    # cv2.waitKey(0)
    # # Then close the windows
    # cv2.destroyAllWindows()




if __name__ == '__main__':
    main()
