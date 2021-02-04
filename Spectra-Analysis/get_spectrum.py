"""
File: get_spectrum.py
---------------------
This program inputs a tiff file of horizontally diffracted light and outputs a line spectrum
for that light.
"""

import numpy as np
import matplotlib.pyplot as mpl
import cv2
import os

# Insert file here
FILENAME_SAMPLE = "Basler_acA1300-200um__23253950__20201007_191129578_21.tiff"
FILENAME_BKGD = "Basler_acA1300-200um__23253950__20201007_191619058_75.tiff"
FILENAME_HGAR = "Basler_acA1300-200um__23253950__20201007_191244451_4.tiff"
FILENAME_HGAR2 = "Basler_acA1300-200um__23253950__20201007_191323400_2.tiff"
FILENAME_LSR = "Basler_acA1300-200um__23253950__20201007_190716443_20.tiff"

DATA_FOLDER = os.path.join('E:\\', 'Quake', '2020-10-07_Selected_pics')

# Matplotlib Parameters:
mpl.rcParams['figure.figsize'] = (6, 4)
mpl.rcParams['figure.dpi'] = 150

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



    height, width = im_sample.shape
    print(height, width)

    mean_sample_spectra = np.mean(im_sample, axis = 0)
    mean_background_spectra = 1.65*np.mean(im_background, axis = 0) +20
    mean_absorbtion_spectra = mean_background_spectra - mean_sample_spectra
    mean_hgar_spectra = np.mean(im_hgar, axis = 0)
    mean_hgar2_spectra = np.mean(im_hgar2, axis = 0)
    mean_laser_spectra = np.mean(im_laser, axis = 0)


    mpl.plot(mean_sample_spectra)
    mpl.plot(mean_background_spectra)
    # mpl.plot(mean_hgar_spectra)
    # mpl.plot(mean_hgar2_spectra)
    # mpl.plot(mean_laser_spectra)
    mpl.plot(mean_absorbtion_spectra)

    mpl.ylabel('intensity')
    mpl.show()



    cv2.imshow("Sample with white light", im_sample)
    # Wait for the user to press any key
    cv2.waitKey(0)
    # Then close the windows
    cv2.destroyAllWindows()




if __name__ == '__main__':
    main()
