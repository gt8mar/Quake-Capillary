"""
Filename: 2Dradon_transform.py
--------------------------------------------------------------------------------
By: Marcus Forst (not really)

Code credit: skimage docs (https://scikit-image.org/docs/stable/auto_examples/transform/plot_radon_transform.html)

"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, rescale
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable


FILEFOLDER = 'C:\\Users\\Luke\\Documents\\Marcus\\Data\\220513\\pointer2small'
FLOW_FILE = 'centerline_array_7_long.csv'

def main():
    # Import images
    flow_image = np.genfromtxt(FLOW_FILE, delimiter=',', dtype=int)
    flow_image = rescale(flow_image, scale=0.4, mode='reflect', channel_axis=None)
    fig, ax2 = plt.subplots(1, 1, figsize=(8, 4.5))
    theta = np.linspace(0., 180., max(flow_image.shape), endpoint=False)
    sinogram = radon(flow_image, theta=theta)
    ax2.set_title("Radon transform\n(Sinogram)")
    ax2.set_xlabel("Projection angle (deg)")
    ax2.set_ylabel("Projection position (pixels)")
    im = ax2.imshow(np.log(sinogram), cmap = plt.cm.hsv, vmin = -13, vmax = -11.8)
    plt.colorbar(im)
    plt.show()
    return 0

# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    main()
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))
