from PIL import Image
import os
import numpy as np

IMAGE_PATH = os.path.join("test_images", "test_image.png")
MASK_PATH = os.path.join("test_images", "test_mask.png")

PATCH_SIZE = (32, 32)
# The "center" of 32x32 is 16, 16
CENTER = (16, 16)

with Image.open(MASK_PATH) as im:
    im_array = np.asarray(im)

damaged_pixels = np.argwhere(im_array > 2)

#for now, just try to get pixels that are within the borders- each point is within a border of the image.
# each values should be between 16 and 1008  (1024-16)

#bounded pixels = fdsafds