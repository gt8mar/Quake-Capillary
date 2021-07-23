# ===========================================================================#
#                                                                           #
#  Copyright (C) 2006 - 2018                                                #
#  IDS Imaging Development Systems GmbH                                     #
#  Dimbacher Str. 6-8                                                       #
#  D-74182 Obersulm, Germany                                                #
#                                                                           #
#  The information in this document is subject to change without notice     #
#  and should not be construed as a commitment by IDS Imaging Development   #
#  Systems GmbH. IDS Imaging Development Systems GmbH does not assume any   #
#  responsibility for any errors that may appear in this document.          #
#                                                                           #
#  This document, or source code, is provided solely as an example          #
#  of how to utilize IDS software libraries in a sample application.        #
#  IDS Imaging Development Systems GmbH does not assume any responsibility  #
#  for the use or reliability of any portion of this document or the        #
#  described software.                                                      #
#                                                                           #
#  General permission to copy or modify, but not for profit, is hereby      #
#  granted, provided that the above copyright notice is included and        #
#  reference made to the fact that reproduction privileges were granted     #
#  by IDS Imaging Development Systems GmbH.                                 #
#                                                                           #
#  IDS Imaging Development Systems GmbH cannot assume any responsibility    #
#  for the use or misuse of any portion of this software for other than     #
#  its intended diagnostic purpose in calibrating and testing IDS           #
#  manufactured cameras and software.                                       #
#                                                                           #
# ===========================================================================#

# Developer Note: I tried to let it as simple as possible.
# Therefore there are no functions asking for the newest driver software or freeing memory beforehand, etc.
# The sole purpose of this program is to show one of the simplest ways to interact with an IDS camera via the uEye API.
# (XS cameras are not supported)
# ---------------------------------------------------------------------------------------------------------------------------------------

# Libraries
from pyueye import ueye
from pypylon import pylon
import numpy as np
import cv2
import time
import sys
import datetime

# ---------------------------------------------------------------------------------------------------------------------------------------
# Constants
EXPOSURE_TIME = 46.462 #27.065
FRAME_RATE = 15.2
PXCL = 98
DATE = int(str(datetime.date.today()).replace("-",""))
WAVELENGTH = str(input("Please enter the wavelength.").strip())
SAMPLE = str(input("Please enter the sample name.").strip())
NUMBER = int(input("Please enter the run number."))

print("cameraB_vid_%d_%s_%s_%d.bmp" % (DATE, WAVELENGTH, SAMPLE, 1))
# Variables
hCam = ueye.HIDS(0)  # 0: first available camera;  1-254: The camera with the specified camera ID
sInfo = ueye.SENSORINFO()
cInfo = ueye.CAMINFO()
pcImageMemory = ueye.c_mem_p()
MemID = ueye.int()
rectAOI = ueye.IS_RECT()


pitch = ueye.INT()
nBitsPerPixel = ueye.INT(24)  # 24: bits per pixel for color mode; take 8 bits per pixel for monochrome
channels = 3  # 3: channels for color mode(RGB); take 1 channel for monochrome
m_nColorMode = ueye.INT()  # Y8/RGB16/RGB24/REG32
bytes_per_pixel = int(nBitsPerPixel / 8)



# ---------------------------------------------------------------------------------------------------------------------------------------
# Camera B setup:
# connecting to the first available camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
# Grabbing Continuosely (video) with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()
# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

# tl_factory = pylon.TlFactory.GetInstance()
# img = pylon.PylonImage()
# camera = pylon.InstantCamera()
# camera.Attach(tl_factory.CreateFirstDevice())
# camera.Open()
# camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
# camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
# converter = pylon.ImageFormatConverter()

# -------------------------------------------------------------------------------------------------------------------------------------
print("START")
print()

# Starts the driver and establishes the connection to the camera
nRet = ueye.is_InitCamera(hCam, None)
if nRet != ueye.IS_SUCCESS:
    print("is_InitCamera ERROR")
# Reads out the data hard-coded in the non-volatile camera memory and writes it to the data structure that cInfo points to
nRet = ueye.is_GetCameraInfo(hCam, cInfo)
if nRet != ueye.IS_SUCCESS:
    print("is_GetCameraInfo ERROR")
# You can query additional information about the sensor type used in the camera
nRet = ueye.is_GetSensorInfo(hCam, sInfo)
if nRet != ueye.IS_SUCCESS:
    print("is_GetSensorInfo ERROR")
nRet = ueye.is_ResetToDefault(hCam)
if nRet != ueye.IS_SUCCESS:
    print("is_ResetToDefault ERROR")
# Set display mode to DIB
nRet = ueye.is_SetDisplayMode(hCam, ueye.IS_SET_DM_DIB)
# Set the right color mode
if int.from_bytes(sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_BAYER:
    # setup the color depth to the current windows setting
    ueye.is_GetColorDepth(hCam, nBitsPerPixel, m_nColorMode)
    bytes_per_pixel = int(nBitsPerPixel / 8)
    print("IS_COLORMODE_BAYER: ", )
    print("\tm_nColorMode: \t\t", m_nColorMode)
    print("\tnBitsPerPixel: \t\t", nBitsPerPixel)
    print("\tbytes_per_pixel: \t\t", bytes_per_pixel)
    print()
elif int.from_bytes(sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_CBYCRY:
    # for color camera models use RGB32 mode
    m_nColorMode = ueye.IS_CM_BGRA8_PACKED
    nBitsPerPixel = ueye.INT(32)
    bytes_per_pixel = int(nBitsPerPixel / 8)
    print("IS_COLORMODE_CBYCRY: ", )
    print("\tm_nColorMode: \t\t", m_nColorMode)
    print("\tnBitsPerPixel: \t\t", nBitsPerPixel)
    print("\tbytes_per_pixel: \t\t", bytes_per_pixel)
    print()




# Can be used to set the size and position of an "area of interest"(AOI) within an image
nRet = ueye.is_AOI(hCam, ueye.IS_AOI_IMAGE_GET_AOI, rectAOI, ueye.sizeof(rectAOI))
# rectAOI.s32X = ueye.c_int(464)
# rectAOI.s32Y = ueye.c_int(348)
# rectAOI.s32Width = ueye.c_int(1632)
# rectAOI.s32Height = ueye.c_int(1224)
# nRet = ueye.is_AOI(hCam, ueye.IS_AOI_IMAGE_SET_AOI, rectAOI, ueye.sizeof(rectAOI))

if nRet != ueye.IS_SUCCESS:
    print("is_AOI ERROR")


width = rectAOI.s32Width
height = rectAOI.s32Height

print(height)








# ---------------------------------------------------------------------------------------------------------------------------------------
# Allocates an image memory for an image having its dimensions defined by width and height and its color depth defined by nBitsPerPixel
nRet = ueye.is_AllocImageMem(hCam, width, height, nBitsPerPixel, pcImageMemory, MemID)
if nRet != ueye.IS_SUCCESS:
    print("is_AllocImageMem ERROR")
else:
    # Makes the specified image memory the active memory
    nRet = ueye.is_SetImageMem(hCam, pcImageMemory, MemID)
    if nRet != ueye.IS_SUCCESS:
        print("is_SetImageMem ERROR")
    else:
        # Set the desired color mode
        nRet = ueye.is_SetColorMode(hCam, m_nColorMode)
# Activates the camera's live video mode (free run mode)
nRet = ueye.is_CaptureVideo(hCam, ueye.IS_DONT_WAIT)
if nRet != ueye.IS_SUCCESS:
    print("is_CaptureVideo ERROR")
# Enables the queue mode for existing image memory sequences
nRet = ueye.is_InquireImageMem(hCam, pcImageMemory, MemID, width, height, nBitsPerPixel, pitch)
if nRet != ueye.IS_SUCCESS:
    print("is_InquireImageMem ERROR")
else:
    print("Press q to leave the program")

"""
Please note that for the following code, we follow the C++ process of initializing variables,
passing them into functions that have binary output, and assigning values to the variables
after they are passed through. 

Each function has their own special command (ex: ueye.IS_PIXELCLOCK_CMD_GET_DEFAULT ), takes an input 
initialized variable, and then occasionally requires a size designation. 

"""

# Set pixel clock and exposure time
maxPxClk = ueye.c_int(PXCL)
maxFps = ueye.c_double()
pxDefault = ueye.c_int()
nRet = ueye.is_PixelClock(hCam, ueye.IS_PIXELCLOCK_CMD_GET_DEFAULT, pxDefault, ueye.sizeof(pxDefault))
if nRet == ueye.IS_SUCCESS:
    nRet = ueye.is_PixelClock(hCam, ueye.IS_PIXELCLOCK_CMD_SET, maxPxClk, ueye.sizeof(maxPxClk))
# nRet = ueye.is_SetOptimalCameraTiming(hCam, ueye.IS_BEST_PCLK_RUN_ONCE, 2000, maxPxClk, maxFps)
print(maxPxClk)

# Set framerate
fps = ueye.c_double()
new_fps_set = ueye.c_double(FRAME_RATE)
new_fps = ueye.c_double()
# Check current framerate
nRet = ueye.is_GetFramesPerSecond(hCam, fps)
if nRet != ueye.IS_SUCCESS:
    print("tf is going on")
# Set new framerate
nRet = ueye.is_SetFrameRate(hCam, new_fps_set, new_fps)
if nRet != ueye.IS_SUCCESS:
    print("shit")
dblFPS = ueye.c_double()
nRet = ueye.is_GetFramesPerSecond(hCam, dblFPS)
print(fps)
print(new_fps)
print(dblFPS)


# Set Exposuretime
exposure = ueye.c_double(EXPOSURE_TIME)
nRet = ueye.is_Exposure(hCam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, exposure, 8)
new_exposure = ueye.c_double()
nRet = ueye.is_Exposure(hCam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE, new_exposure, 8)
print(new_exposure)

# ---------------------------------------------------------------------------------------------------------------------------------------
i = 0
# Continuous image display
t0 = time.time()
framert = []
while camera.IsGrabbing():
    while (nRet == ueye.IS_SUCCESS):
        # In order to display the image in an OpenCV window we need to extract the data of our image memory
        # ---------------------------------------------------------------------------------------------------------------------------------------
        # Check how fast stuff is moving within the loop.
        # t1 = time.time() - t0
        # framert.append(t1)
        # t0 = time.time()
        # ---------------------------------------------------------------------------------------------------------------------------------------

        # Camera A setup
        array = ueye.get_data(pcImageMemory, width, height, nBitsPerPixel, pitch, copy=False)           # the shape of this is (19660800,) lol
        bytes_per_pixel = int(nBitsPerPixel / 8)
        # ...reshape it in an numpy array...
        frame = np.reshape(array, (height.value, width.value, bytes_per_pixel))
        # ...resize the image by a half
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # ---------------------------------------------------------------------------------------------------------------------------------------
        # Camera B setup
        result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        image = converter.Convert(result)
        img = image.GetArray()
        # ---------------------------------------------------------------------------------------------------------------------------------------

        # Show live images
        cv2.imshow('Camera B', img)
        cv2.imshow("Camera A", frame)


        # Take image
        if cv2.waitKey(1) & 0xFF == ord('p'):
            # dblFPS = ueye.c_double()
            # nRet = ueye.is_GetFramesPerSecond(hCam, dblFPS)
            # print(dblFPS)
            img2 = pylon.PylonImage()
            img2.AttachGrabResultBuffer(result)
            filename1 = "cameraB_%d_%s_%s_%d_%d.bmp" % (DATE, WAVELENGTH, SAMPLE, NUMBER, i)
            img2.Save(pylon.ImageFileFormat_Bmp, filename1)
            filename2 = "cameraA_%d_%s_%s_%d_%d.bmp" % (DATE, WAVELENGTH, SAMPLE, NUMBER, i)
            cv2.imwrite(filename2, frame)
            print("saved " + filename2)
            print(frame)
            i += 1

        elif cv2.waitKey(1) & 0xFF == ord('v'):
            for j in range(100):
                array = ueye.get_data(pcImageMemory, width, height, nBitsPerPixel, pitch,
                                      copy=False)  # the shape of this is (19660800,) lol
                bytes_per_pixel = int(nBitsPerPixel / 8)
                # ...reshape it in an numpy array...
                frame = np.reshape(array, (height.value, width.value, bytes_per_pixel))
                # ...resize the image by a half
                frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                #-------------------------------------------------------------------------
                result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                image = converter.Convert(result)
                img = image.GetArray()
                img2 = pylon.PylonImage()
                img2.AttachGrabResultBuffer(result)
                filename1 = "cameraB_vid_%d_%s_%s_%d_%d.bmp" % (DATE, WAVELENGTH, SAMPLE, NUMBER, j)
                img2.Save(pylon.ImageFileFormat_Bmp, filename1)
                filename2 = "cameraA_vid_%d_%s_%s_%d_%d.bmp" % (DATE, WAVELENGTH, SAMPLE, NUMBER, j)
                cv2.imwrite(filename2, frame)
            print("video saved")

        # Press q if you want to end the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    break
print(framert)
# ---------------------------------------------------------------------------------------------------------------------------------------

# Releases an image memory that was allocated using is_AllocImageMem() and removes it from the driver management
ueye.is_FreeImageMem(hCam, pcImageMemory, MemID)

# Disables the hCam camera handle and releases the data structures and memory areas taken up by the uEye camera
ueye.is_ExitCamera(hCam)

# Destroys the OpenCv windows
cv2.destroyAllWindows()

print()
print("END")
