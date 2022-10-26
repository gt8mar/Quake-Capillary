"""
Filename: pic2vid.py
-------------------------------------------------------------
This file turns a group of files into a video. It correctly orders misordered files.
by: Marcus Forst
sort_nicely credit: Ned B (https://nedbatchelder.com/blog/200712/human_sorting.html)
"""


import cv2
import os
import glob
import re
import time
import numpy as np


# UMBRELLA_FOLDER = 'C:\\Users\\gt8mar\\Desktop\\data\\221010'
FILEFOLDER_PATH = "C:\\Users\\gt8mar\\Desktop\\data\\221019\\raw\\vid12"
DATE = "221025"
PARTICIPANT = "Participant4"
FOLDER_NAME = 'vid12'
SET = '01'
SAMPLE = '0002'

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
def get_images(FILEFOLDER):
    """
    this function grabs image names, sorts them, and puts them in a list.
    :param FILEFOLDER: string
    :return: images: list of images
    """
    images = [img for img in os.listdir(FILEFOLDER) if img.endswith(".tif") or img.endswith(
        ".tiff")]  # if this came out of moco the file suffix is .tif otherwise it's tiff
    sort_nicely(images)
    return images
def add_set_sample(img):
    # Write some Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (50, 50)
    fontScale = 1
    fontColor = (255, 255, 255)
    thickness = 1
    lineType = 2

    cv2.putText(img, str(SET) + "." + str(SAMPLE) + ":",
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return 0
def add_hardware_software(img):
    # Write some Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (50, 80)
    fontScale = 1
    fontColor = (255, 255, 255)
    thickness = 1
    lineType = 2

    cv2.putText(img, "HW: 01",
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    cv2.putText(img, "SW: 01",
                (50, 110),
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return 0
def add_frame_counter_pressure(img, frame):
    # Write some Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    topLeftCornerOfText = (200, 50)
    fontScale = 1
    fontColor = (255, 255, 255)
    thickness = 1
    lineType = 2

    cv2.putText(img, str(frame).zfill(4),
                topLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    cv2.putText(img, "P: " + '0.2' + ' psi',  # 1.3 will be called from a txt file with frame index.
                (1000, 50),
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return 0
def add_focus_bar(img, focus):
    # Write some Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontColor = (255, 255, 255)
    thickness = 1
    lineType = 2

    cv2.putText(img, 'F' + str(focus),
                (1150, 900),
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return 0
def calculate_focus_measure(image,method='LAPE'):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) # optional
    if method == 'LAPE':
        if image.dtype == np.uint16:
            lap = cv2.Laplacian(image, cv2.CV_32F)
        else:
            lap = cv2.Laplacian(image, cv2.CV_16S)
        focus_measure = np.mean(np.square(lap))
    elif method == 'GLVA':
        focus_measure = np.std(image,axis=None)# GLVA
    else:
        focus_measure = np.std(image,axis=None)# GLVA
    return focus_measure
def pic2vid(image_folder, images, video_name):
    """
    this takes an image folder and a list of image files and makes a movie
    :param image_folder: string
    :param images: list of image filenames (strings)
    :return:
    """
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, 0, 60, (width, height))
    print(frame.shape)
    # for i in range(len(images)):
    #     img = cv2.imread(os.path.join(image_folder, images[i]))
    #     focus_measure = calculate_focus_measure(img)
    #     add_frame_counter_pressure(img, i)
    #     add_set_sample(img)
    #     add_hardware_software(img)
    #     add_focus_bar(img, focus_measure)
    #     video.write(img)
    cv2.destroyAllWindows()
    video.release()
    return 0

def main(filefolder = FILEFOLDER_PATH, folder = FOLDER_NAME, date = DATE, participant = PARTICIPANT):
    # for folder in os.listdir(UMBRELLA_FOLDER):
    #     path = os.path.join(UMBRELLA_FOLDER, folder)
    #     images = get_images(path)
    #     video_name = DATE + "_" + PARTICIPANT + "_"  + folder + ".avi"
    #     pic2vid(path, images, video_name)

    """------------------for only one file --------------------------"""
    images = get_images(filefolder)
    video_name = date + "_" + participant + "_"  + folder + ".avi"
    pic2vid(filefolder, images, video_name)
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
