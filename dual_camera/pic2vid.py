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

UMBRELLA_FOLDER = 'C:\\Users\\gt8mar\\Desktop\\data\\221010'
DATE = "221010"
PARTICIPANT = "Participant3"

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

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

def main():
    # for folder in os.listdir(UMBRELLA_FOLDER):
    #     path = os.path.join(UMBRELLA_FOLDER, folder)
    #     images = get_images(path)
    #     video_name = DATE + "_" + PARTICIPANT + "_"  + folder + ".avi"
    #     pic2vid(path, images, video_name)

    """
    ------------------for only one file ------------------------------------
    """
    path = os.path.join(UMBRELLA_FOLDER)
    images = get_images(path)
    video_name = DATE + "_" + PARTICIPANT + "_"  + "vid1" + ".avi"
    pic2vid(path, images, video_name)
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
