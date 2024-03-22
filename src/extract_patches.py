"""
Script to extract damaged patches from a given image using a mask,
"""
from PIL import Image
import os
import numpy as np
from matplotlib import pyplot as plt
import sys.argv as arguments

#@TODO -- get interation over files working

#Parse arguments into constants
# Usage: python3 extract_patches.py patch_size input_dir output dir
PATCH_SIZE = int(arguments[1])
INPUT_DIR = arguments[2]
OUTPUT_DIR = arguments[3]

# The "center" of 32x32 is 15 (0 indexing)
HALF = PATCH_SIZE // 2

def calc_patch(pixel_coords):
    x, y = tuple(pixel_coords)
    return (x - HALF + 1, y - HALF + 1, x + HALF + 1, y + HALF +1)


#@TODO -- refactor into functions and generalize to work on many different files


"""
#def extract_patches(

# open images
mask_image = Image.open(MASK_PATH)
rgb_image = Image.open(IMAGE_PATH)

# create proper mask from mask_image
mask_array = np.array(mask_image) >= 2

#create an array from rgb image
rgb_array = np.array(rgb_image)

#@DEBUG --  display overlay
#tensor_mask = np.stack((mask_array,)*3, axis=-1)
#overlay_image = Image.fromarray(rgb_array*tensor_mask)
#overlay_image.show()

# get coordinates of damaged pixels
damaged_pixels = np.argwhere(mask_array == 1)
print(damaged_pixels)

# for now, just try to get pixels that are within the borders- each point is within a border of the image.
# each values should be between 16 and 1008  (1024-16)

# bounded pixels = fdsafds

# show an example of a patch
#example_pixel = damaged_pixels[600]
"""


"""
for coords in damaged_pixels:
    patch_box = calc_patch(coords)
    a, b, c, d = patch_box

    #mask
    mask_patch = mask_image.crop(patch_box)
    filename = f"{a}_{b}_{c}_{d}_mask.png"
    mask_patch.save(os.path.join("extraction_test", "patches", filename))

    #rgb
    rgb_patch = rgb_image.crop(patch_box)
    filename = f"{a}_{b}_{c}_{d}_rgb.png"
    mask_patch.save(os.path.join("extraction_test", "patches", filename))
"""
