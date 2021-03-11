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
from PIL import Image
import pandas as pd



# Insert file here
# FILENAME_SAMPLE = "Basler_acA1300-200um__23253950__20201007_191129578_21.tiff"
# FILENAME_BKGD = "Basler_acA1300-200um__23253950__20201007_191619058_75.tiff"
# FILENAME_HGAR = "Basler_acA1300-200um__23253950__20201007_191244451_4.tiff"
# FILENAME_HGAR2 = "Basler_acA1300-200um__23253950__20201007_191323400_2.tiff"
# FILENAME_LSR = "Basler_acA1300-200um__23253950__20201007_190716443_20.tiff"

DATA_FOLDER = os.path.join('E:\\', 'Quake', '2020-10-07_Selected_pics')
DATA_FOLDER_RBC = os.path.join('E:\\', 'Quake', '2020-10-24', 'RBC', 'LED')
DATA_FOLDER_WBC = os.path.join('E:\\', 'Quake', '2020-10-24', 'WBC', 'LED')
DATA_FOLDER_PAPER = os.path.join('E:\\', 'Quake', '2020-10-24', 'PAPER')
# DATA_FOLDER_RBC = os.path.join('C:\\', 'Users', 'Luke', 'Documents', 'Marcus', 'Data', '2020-10-24', 'RBC', 'LED')
# DATA_FOLDER_WBC = os.path.join('C:\\', 'Users', 'Luke', 'Documents', 'Marcus', 'Data', '2020-10-24', 'WBC', 'LED')
# DATA_FOLDER_PAPER = os.path.join('C:\\', 'Users', 'Luke', 'Documents', 'Marcus', 'Data', '2020-10-24', 'PAPER')
DATA_FOLDER_IMAGE = os.path.join('C:\\', 'Users', 'gt8ma', 'OneDrive', 'Documents', 'Quake', 'Bluud', 'Data',
                                 '020521', '50xv2')
# DATA_FOLDER_IMAGE = os.path.join('C:\\', 'Users', 'Luke', 'Documents', 'Marcus', 'Data', '020521', '50xv2')
# DATA_FOLDER_CALIBRATION = os.path.join('C:\\', 'Users', 'Luke', 'Documents', 'Marcus', 'Data', 'camera_calibration')
DATA_FOLDER_CALIBRATION = os.path.join('C:\\', 'Users', 'gt8ma', 'OneDrive', 'Documents', 'Quake',
                                       'Bluud', 'Data', 'camera_calibration')

IMAGE_FILENAME = "Image__2021-02-05__18-49-31.tiff"
IMAGE_FILENAME_2 = "6.png"
CALIBRATION_FILENAME = "Image__2021-02-12__19-00-59.tiff"
CALIBRATION_FILENAME_A = "987.png"

BIN_SIZE = 60

# Matplotlib Parameters:
mpl.rcParams['figure.figsize'] = (6, 4)
mpl.rcParams['figure.dpi'] = 150

"""
This program opens a file and displays it to the user. 
"""
def main():
    binned, original = bin_image(DATA_FOLDER_IMAGE, IMAGE_FILENAME)
    mean_binned = np.mean(binned, axis = 0)
    mean_original = np.mean(original, axis=0)
    mpl.plot(mean_binned)
    mpl.plot(mean_original)
    mpl.show()
    compare_rows(binned, original)

    # # Read in images of sample, background, and calibration wavelengths
    # im_sample = cv2.imread(os.path.join(DATA_FOLDER, FILENAME_SAMPLE), 0)
    # im_background = cv2.imread(os.path.join(DATA_FOLDER, FILENAME_BKGD), 0)
    # im_hgar = cv2.imread(os.path.join(DATA_FOLDER, FILENAME_HGAR), 0)
    # im_hgar2 = cv2.imread(os.path.join(DATA_FOLDER, FILENAME_HGAR2), 0)
    # im_laser = cv2.imread(os.path.join(DATA_FOLDER, FILENAME_LSR), 0)
    #
    #
    #
    # height, width = im_sample.shape
    # print(height, width)
    #
    # mean_sample_spectra = np.mean(im_sample, axis = 0)
    # mean_background_spectra = 1.65*np.mean(im_background, axis = 0) +20
    # mean_absorbtion_spectra = mean_background_spectra - mean_sample_spectra
    # mean_hgar_spectra = np.mean(im_hgar, axis = 0)
    # mean_hgar2_spectra = np.mean(im_hgar2, axis = 0)
    # mean_laser_spectra = np.mean(im_laser, axis = 0)
    #
    #
    # mpl.plot(mean_sample_spectra)
    # mpl.plot(mean_background_spectra)
    # # mpl.plot(mean_hgar_spectra)
    # # mpl.plot(mean_hgar2_spectra)
    # # mpl.plot(mean_laser_spectra)
    # mpl.plot(mean_absorbtion_spectra)
    #
    # mpl.ylabel('intensity')
    # mpl.show()
    #
    #
    #
    # cv2.imshow("Sample with white light", im_sample)
    # # Wait for the user to press any key
    # cv2.waitKey(0)
    # # Then close the windows
    # cv2.destroyAllWindows()


def bin_image(folder, filename):
    # First read in file:
    image = cv2.imread(os.path.join(folder,filename), 0)
    image2 = np.array(image)
    df = pd.DataFrame(data=image)
    array = df.to_numpy()
    for row in range(array.shape[0]):
        for col in range(10, array.shape[1], BIN_SIZE):
            binned = array[row][col:col+BIN_SIZE]
            average = np.mean(binned)
            array[row][col:col+BIN_SIZE] = average
    im = Image.fromarray(array)
    im.save("binned_image.png", "PNG")
    im.show()
    return array, image2

def compare_rows(binned, image):
    spectra_list = []
    VERTICAL_BINS = 20
    for row in range(0, binned.shape[0], binned.shape[0]//10):
        vert_slice = binned[row:row+binned.shape[0]//VERTICAL_BINS][:]
        chunk_value = np.mean(vert_slice, axis=0)
        binned[row:row+binned.shape[0]//VERTICAL_BINS] = chunk_value
        spectra_list.append(chunk_value)
        mpl.plot(chunk_value)
    print(len(spectra_list))
    # mpl.plot(spectra_list)
    mpl.show()
    im = Image.fromarray(binned)
    im.save("vertbin.png", "PNG")
    im.show()
    return spectra_list

if __name__ == '__main__':
    main()
