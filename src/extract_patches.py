"""
Script to extract damaged patches from a given image using a mask,
"""

from PIL import Image
import os
import numpy as np
import sys
import glob

# @TODO -- get interation over files working


# Parse arguments into constants
# Usage: python3 extract_patches.py patch_size input_dir
PATCH_SIZE = int(sys.argv[1])
INPUT_DIR = sys.argv[2]
OUTPUT_DIR = f"patch_data_{PATCH_SIZE}"

os.makedirs(f"{OUTPUT_DIR}/images")
os.makedirs(f"{OUTPUT_DIR}/targets")

# Globbing
IMAGES = sorted(glob.glob(f"{INPUT_DIR}/images/*"))
MASKS = sorted(glob.glob(f"{INPUT_DIR}/targets/*"))

# print(IMAGES)
# print(MASKS)


# The "center" of 32x32 is 15 (0 indexing)
HALF = PATCH_SIZE // 2


def extract_patches(image_file_path, mask_file_path):
    """
    Extract patches given file paths
    """

    # Internal function to calculate patch
    def calc_patch(pixel_coords):
        x, y = tuple(pixel_coords)
        return (x - HALF + 1, y - HALF + 1, x + HALF + 1, y + HALF + 1)

    # extract image file names for later
    rgb_name = os.path.basename(image_file_path)[:-4]
    mask_name = os.path.basename(mask_file_path)[:-4]

    # print("opening images...")
    # open up the images for later
    rgb_image = Image.open(image_file_path)
    mask_image = Image.open(mask_file_path)

    # create mask array to use
    mask_array = np.array(mask_image)

    # get coordinates of damaged pixels
    damaged_pixels = np.argwhere(mask_array > 0)

    # @TODO -- only work with pixels inside range and skip pixels
    # for now, just try to get pixels that are within the borders- each point is within a border of the image.
    # each values should be between 16 and 1008  (1024-16)

    # interate over skip
    SKIP = 10
    total_patches = len(damaged_pixels) // SKIP
    # print(f"Generating {num_patches} patches:")
    for i, coords in enumerate(damaged_pixels[::SKIP]):
        patch_box = calc_patch(coords)
        a, b, c, d = patch_box

        # rgb
        rgb_patch = rgb_image.crop(patch_box)
        filename = f"{rgb_name}--patch{i:05d}.png"
        rgb_patch.save(os.path.join(OUTPUT_DIR, "images", filename))

        # mask
        mask_patch = mask_image.crop(patch_box)
        filename = f"{mask_name}--patch{i:05d}.png".replace("_target", "")
        mask_patch.save(os.path.join(OUTPUT_DIR, "targets", filename))


if __name__ == "__main__":
    # @DEBUG
    # print(PATCH_SIZE)
    print("This script no longer removes class data")

    total_images = len(IMAGES)
    count = 1

    for image, mask in zip(IMAGES, MASKS):
        print(f"Unpacking image {count}/{total_images}", end="\r", flush=True)
        extract_patches(image, mask)
        count += 1

    print("finished!")

    # @DEBUG --  display overlay
    # tensor_mask = np.stack((mask_array,)*3, axis=-1)
    # overlay_image = Image.fromarray(rgb_array*tensor_mask)
    # overlay_image.show()
