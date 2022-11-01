"""
Filename: pipeline.py
------------------------------------------------------
This program runs a sequence of python programs to analyze capillaries
By: Marcus Forst
"""

import correlation_with_cap_selection
import chop_top
import write_background_file
import time
import os
import auto_corr
import correlation
import pic2vid


UMBRELLA_FOLDER = 'C:\\Users\\gt8mar\\Desktop\\data\\221019'
DATE = "221019"
PARTICIPANT = "Participant4"
# FILEFOLDER = 'C:\\Users\\gt8mar\\Desktop\\data\\221019\\vid4_moco'
FILEFOLDER_SEGMENTED = 'C:\\Users\\gt8mar\\Desktop\\data\\221019'
# UMBRELLA_FOLDER_MOCO = 'C:\\Users\\gt8mar\\Desktop\\data\\221019\\moco'
# CAPILLARY_ROW = 565
# CAPILLARY_COL = 590
# BKGD_COL = 669
# BKGD_ROW = 570


def service_function():
    print('service function')

"""
-----------------------------------------------------------------------------
"""
# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    print("Run full pipeline")
    print("-------------------------------------")
    ticks_first = time.time()
    ticks = time.time()

    # """ Pic2Vid """
    # for folder in os.listdir(UMBRELLA_FOLDER):
    #     print(folder)
    #     path = os.path.join(UMBRELLA_FOLDER, folder)
    #     pic2vid.main(path, folder = folder, date = DATE, participant = PARTICIPANT)
    # print("-------------------------------------")
    # print("pic2vid Runtime: " + str(time.time() - ticks))
    # ticks = time.time()

    # """ Chop Top """
    # chop_top.main(UMBRELLA_FOLDER, DATE, PARTICIPANT)
    # print("-------------------------------------")
    # print("Chop Top Runtime: " + str(time.time() - ticks))
    # ticks = time.time()

    # TODO: imagej stuff, not clear if this is possible
    service_function()

    """ Write Background """
    # for folder in os.listdir(UMBRELLA_FOLDER):
    #     path = os.path.join(UMBRELLA_FOLDER, folder)
    #     write_background_file.main(folder, path, DATE, PARTICIPANT)
    # print("-------------------------------------")
    # print("Background Runtime: " + str(time.time() - ticks))
    # ticks = time.time()


    """ Correlation files """
    # for folder in os.listdir(UMBRELLA_FOLDER_MOCO):
    #     path = os.path.join(UMBRELLA_FOLDER, folder)
    #     pic2vid.main(path, folder, DATE, PARTICIPANT)
    #     segmented_file_name = folder + '0000segmented'
    #     correlation_with_cap_selection.main(path, UMBRELLA_FOLDER_MOCO, segmented_file_name)
    #     auto_corr.main(UMBRELLA_FOLDER_MOCO, CAPILLARY_ROW, CAPILLARY_COL, BKGD_ROW, BKGD_COL)
    #     correlation.main(UMBRELLA_FOLDER_MOCO)

    # print("-------------------------------------")
    # print("Correlation Runtime: " + str(time.time() - ticks))
    # ticks = time.time()

    print("-------------------------------------")
    print("Total Pipeline Runtime: " + str(time.time() - ticks_first))

