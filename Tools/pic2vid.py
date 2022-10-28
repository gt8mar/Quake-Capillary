"""
Filename: pic2vid.py
-------------------------------------------------------------
This file turns a group of files into a video. It correctly orders misordered files.
by: Marcus Forst
sort_nicely credit: Ned B (https://nedbatchelder.com/blog/200712/human_sorting.html)
frames_to_timecode credit: Copyright (c) 2016 Shotgun Software Inc.
(https://github.com/shotgunsoftware/tk-core/blob/master/LICENSE)
"""


import cv2
import os
import glob
import re
import time
import numpy as np


# UMBRELLA_FOLDER = 'C:\\Users\\gt8mar\\Desktop\\data\\221010'
FILEFOLDER_PATH = "C:\\Users\\Luke\\Documents\\Marcus\\Data\\220513\\pointer2small"
DATE = "221028"
PARTICIPANT = "Participant0"
FOLDER_NAME = 'pointer2small'
SET = '01'
SAMPLE = '0000'
FRAMERATE = 100

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
def frames_to_timecode(frame_number, frame_rate, drop = False):
    """
    Method that converts frames to SMPTE timecode.
    :param frame_number: Number of frames
    :param frame_rate: frames per second
    :param drop: true if time code should drop frames, false if not
    :returns: SMPTE timecode as string, e.g. '01:02:12:32' or '01:02:12;32'
    """
    fps_int = int(round(frame_rate))

    if drop:
        # drop-frame-mode
        # add two 'fake' frames every minute but not every 10 minutes
        #
        # example at the one minute mark:
        #
        # frame: 1795 non-drop: 00:00:59:25 drop: 00:00:59;25
        # frame: 1796 non-drop: 00:00:59:26 drop: 00:00:59;26
        # frame: 1797 non-drop: 00:00:59:27 drop: 00:00:59;27
        # frame: 1798 non-drop: 00:00:59:28 drop: 00:00:59;28
        # frame: 1799 non-drop: 00:00:59:29 drop: 00:00:59;29
        # frame: 1800 non-drop: 00:01:00:00 drop: 00:01:00;02
        # frame: 1801 non-drop: 00:01:00:01 drop: 00:01:00;03
        # frame: 1802 non-drop: 00:01:00:02 drop: 00:01:00;04
        # frame: 1803 non-drop: 00:01:00:03 drop: 00:01:00;05
        # frame: 1804 non-drop: 00:01:00:04 drop: 00:01:00;06
        # frame: 1805 non-drop: 00:01:00:05 drop: 00:01:00;07
        #
        # example at the ten minute mark:
        #
        # frame: 17977 non-drop: 00:09:59:07 drop: 00:09:59;25
        # frame: 17978 non-drop: 00:09:59:08 drop: 00:09:59;26
        # frame: 17979 non-drop: 00:09:59:09 drop: 00:09:59;27
        # frame: 17980 non-drop: 00:09:59:10 drop: 00:09:59;28
        # frame: 17981 non-drop: 00:09:59:11 drop: 00:09:59;29
        # frame: 17982 non-drop: 00:09:59:12 drop: 00:10:00;00
        # frame: 17983 non-drop: 00:09:59:13 drop: 00:10:00;01
        # frame: 17984 non-drop: 00:09:59:14 drop: 00:10:00;02
        # frame: 17985 non-drop: 00:09:59:15 drop: 00:10:00;03
        # frame: 17986 non-drop: 00:09:59:16 drop: 00:10:00;04
        # frame: 17987 non-drop: 00:09:59:17 drop: 00:10:00;05

        # calculate number of drop frames for a 29.97 std NTSC
        # workflow. Here there are 30*60 = 1800 frames in one
        # minute

        FRAMES_IN_ONE_MINUTE = 1800 - 2

        FRAMES_IN_TEN_MINUTES = (FRAMES_IN_ONE_MINUTE * 10) - 2

        ten_minute_chunks = frame_number / FRAMES_IN_TEN_MINUTES
        one_minute_chunks = frame_number % FRAMES_IN_TEN_MINUTES

        ten_minute_part = 18 * ten_minute_chunks
        one_minute_part = 2 * ((one_minute_chunks - 2) / FRAMES_IN_ONE_MINUTE)

        if one_minute_part < 0:
            one_minute_part = 0

        # add extra frames
        frame_number += ten_minute_part + one_minute_part

        # for 60 fps drop frame calculations, we add twice the number of frames
        if fps_int == 60:
            frame_number = frame_number * 2

        # time codes are on the form 12:12:12;12
        smpte_token = ";"
    else:
        # time codes are on the form 12:12:12:12
        smpte_token = ":"

    # now split our frames into time code
    hours = int(frame_number / (3600 * fps_int) % 24)
    minutes = int((frame_number / (60 * fps_int)) % 60)
    seconds = int((frame_number / fps_int) % 60)
    frames = int(frame_number)
    return "%02d:%02d:%02d%s%02d" % (hours, minutes, seconds, smpte_token, frames)
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
    timecode = frames_to_timecode(frame, 169)
    cv2.putText(img, timecode,                          # str(frame).zfill(4)
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
def calculate_focus_measure(image,method='LAPE', crop = False):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) # optional
    if crop:
        image = image[25:-25, 25:-25]
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
    for i in range(len(images)):
        img = cv2.imread(os.path.join(image_folder, images[i]))
        focus_measure = calculate_focus_measure(img, crop = True)
        add_frame_counter_pressure(img, i)
        add_set_sample(img)
        add_hardware_software(img)
        add_focus_bar(img, focus_measure)
        video.write(img)
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
