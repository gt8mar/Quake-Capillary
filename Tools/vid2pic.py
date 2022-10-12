"""
Filename: vid2pic.py
-------------------------------------------------------------
This file turns a video into a group of files.
by: Marcus Forst (not really)

"""

import skvideo.io
import os
import numpy as np
import cv2

VIDEO_NAME = "Basler_acA1440-220um__40131722__20220224_172150255.avi"
FILEFOLDER = os.path.join("C:\\", 'Users', 'Luke', 'Documents', 'Marcus', 'Data',
                          '220224')
PATH = os.path.join("C:\\", 'Users', 'Luke', 'Documents', 'Marcus', 'Data',
                          '220224', VIDEO_NAME)

videodata = skvideo.io.vread(PATH)
print(videodata.shape)                          # (time, rows, cols, colors)
new_picture = np.mean(videodata, axis=3)
print(new_picture.shape)

# write new folder of reduced images:
# cwd = os.getcwd()
# folder = FILEFOLDER
# path = os.path.join(cwd, folder)
# if folder not in os.listdir(cwd):
#     os.mkdir(path)

for i in range(new_picture.shape[0]):
    pic = new_picture[i]
    print(pic.shape)
    name = VIDEO_NAME + "_" + str(i) +".tiff"
    cv2.imwrite(os.path.join(FILEFOLDER, name), pic)
