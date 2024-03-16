from PIL import Image
import os
import numpy as np

IMAGE_PATH = os.path.join("extraction_test", "test_image.png")
MASK_PATH = os.path.join("extraction_test", "test_mask.png")


PATCH_SIZE = 32
# The "center" of 32x32 is 15 (0 indexing)
HALF = PATCH_SIZE // 2

def calc_patch(pixel_coords):
    x, y = tuple(pixel_coords)
    return (x - HALF + 1, y - HALF + 1, x + HALF + 1, y + HALF +1)

# open images
mask_image = Image.open(MASK_PATH)
rgb_image = Image.open(IMAGE_PATH)

#get coordinates
mask_array = np.asarray(mask_image)
damaged_pixels = np.argwhere(mask_array > 2)

#display them
#damage_bitmap = Image.fromarray(im_array > 2)
#damage_bitmap.show()

#example = damaged_pixels[0]



# for now, just try to get pixels that are within the borders- each point is within a border of the image.
# each values should be between 16 and 1008  (1024-16)

#bounded pixels = fdsafds

#print(f"damaged = {example}")
#print(f"patch = {calc_patch(tuple(example))}")



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

