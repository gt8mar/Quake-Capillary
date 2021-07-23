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


image_folder = 'camA'
# video_name = 'video3_B.avi'

# for name in glob.glob('camB\\cameraB_*.bmp'):
#     print(name)

images = [img for img in os.listdir(image_folder) if img.endswith(".bmp")]
sort_nicely(images)
video_name = str(images[0].strip(".bmp"))
video_name += ".avi"
print(video_name)

for image in images:
    print(image)

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 60, (width, height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
