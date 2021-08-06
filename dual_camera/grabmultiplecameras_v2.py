# Grab_MultipleCameras_v2.cpp
# ============================================================================
# This sample illustrates how to grab and process images from multiple cameras
# using the CInstantCameraArray class. The CInstantCameraArray class represents
# an array of instant camera objects. It provides almost the same interface
# as the instant camera for grabbing.
# The main purpose of the CInstantCameraArray is to simplify waiting for images and
# camera events of multiple cameras in one thread. This is done by providing a single
# RetrieveResult method for all cameras in the array.
# Alternatively, the grabbing can be started using the internal grab loop threads
# of all cameras in the CInstantCameraArray. The grabbed images can then be processed by one or more
# image event handlers. Please note that this is not shown in this example.
# ============================================================================

import os

os.environ["PYLON_CAMEMU"] = "3"

from pypylon import genicam
from pypylon import pylon
import sys
import time
import datetime
import os

# Constants
DATE = int(str(datetime.date.today()).replace("-",""))
WAVELENGTH = str(input("Please enter the wavelength.").strip())
SAMPLE = str(input("Please enter the sample name.").strip())
NUMBER = int(input("Please enter the run number."))

# Create folders for aquisition
cwd = os.getcwd()
folder1 = "cameraA_%d_%s_%s" % (DATE, WAVELENGTH, SAMPLE)
folder2 = "cameraB_%d_%s_%s" % (DATE, WAVELENGTH, SAMPLE)
path1 = os.path.join(cwd, folder1)
path2 = os.path.join(cwd, folder2)
if folder1 not in os.listdir(cwd):
    os.mkdir(path1)
if folder2 not in os.listdir(cwd):
    os.mkdir(path2)
print("cameraB_vid_%d_%s_%s_%d.bmp" % (DATE, WAVELENGTH, SAMPLE, 1))

# Limits the amount of cameras used for grabbing.
# It is important to manage the available bandwidth when grabbing with multiple cameras.
# This applies, for instance, if two GigE cameras are connected to the same network adapter via a switch.
# To manage the bandwidth, the GevSCPD interpacket delay parameter and the GevSCFTD transmission delay
# parameter can be set for each GigE camera device.
# The "Controlling Packet Transmission Timing with the Interpacket and Frame Transmission Delays on Basler GigE Vision Cameras"
# Application Notes (AW000649xx000)
# provide more information about this topic.
# The bandwidth used by a FireWire camera device can be limited by adjusting the packet size.
maxCamerasToUse = 2
num_img_to_save = 100

# The exit code of the sample application.
exitCode = 0

ALPHABET_LIST = ["A", "B"]

tlFactory = pylon.TlFactory.GetInstance()
devices = tlFactory.EnumerateDevices()

# Create an array of instant cameras for the found devices and avoid exceeding a maximum number of devices.
cameras = pylon.InstantCameraArray(min(len(devices), maxCamerasToUse))
l = cameras.GetSize()

# Create and attach all Pylon Devices.
for i, cam in enumerate(cameras):
    cam.Attach(tlFactory.CreateDevice(devices[i]))
    # Print the model name of the camera.
    print("Using device ", cam.GetDeviceInfo().GetModelName())

"""
Note that this does not grab simultaneously. We can use a hardware trigger. 
"""
cameras.StartGrabbing()
# Grab c_countOfImagesToGrab from the cameras.
for j in range(num_img_to_save):
    if not cameras.IsGrabbing():
        break
    grabResult = cameras.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    # Starts grabbing for all cameras starting with index 0. The grabbing
    # is started for one camera after the other. That's why the images of all
    # cameras are not taken at the same time.
    # However, a hardware trigger setup can be used to cause all cameras to grab images synchronously.
    # According to their default configuration, the cameras are
    # set up for free-running continuous acquisition.

    # When the cameras in the array are created the camera context value
    # is set to the index of the camera in the array.
    # The camera context is a user settable value.
    # This value is attached to each grab result and can be used
    # to determine the camera that produced the grab result.
    camera_index = grabResult.GetCameraContext()

    # Print the index and the model name of the camera.
    print("Camera ", camera_index, ": ", cameras[camera_index].GetDeviceInfo().GetModelName())

    # Now, the image data can be processed.
    print("GrabSucceeded: ", grabResult.GrabSucceeded())
    print("SizeX: ", grabResult.GetWidth())
    print("SizeY: ", grabResult.GetHeight())
    img = grabResult.GetArray()
    # img2 = pylon.PylonImage()
    # img2.AttachGrabResultBuffer(result)
    filename1 = "camera%s_vid_%d_%s_%s_%d_%d.bmp" % (camera_index, DATE, WAVELENGTH, SAMPLE, NUMBER, j)
    img.Save(pylon.ImageFileFormat_Bmp, os.path.join(path1, filename1))
    print("Gray value of first pixel: ", img[0, 0])
    img.Release()

cameras.StopGrabbing()
cameras.Close()

# Comment the following two lines to disable waiting on exit.
sys.exit(exitCode)
