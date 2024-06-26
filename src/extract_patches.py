"""
Script to extract damaged patches from a given image using a mask,
"""

from PIL import Image
import os
import sys
import glob
import numpy as np
from patchify import patchify

# Parse arguments into constants

if len(sys.argv) < 2 or sys.argv[1] == "help" or sys.argv[1] == "-h":
    print("Usage: python3 extract_patches.py patch_size input_dir")
    exit()
    
PATCH_SZ = int(sys.argv[1])
INPUT_SZ = 1024

INPUT_DIR = sys.argv[2]
OUTPUT_DIR = f"patch_data_{PATCH_SZ}"

os.makedirs(f"{OUTPUT_DIR}/images")
os.makedirs(f"{OUTPUT_DIR}/targets")

# Globbing
IMAGES = sorted(glob.glob(f"{INPUT_DIR}/images/*"))
MASKS = sorted(glob.glob(f"{INPUT_DIR}/targets/*"))


def open_image(fname):
    img = Image.open(fname)
    return np.array(img)

def split_array(x):
    """
    split array into sub-arrays
    """
    return patchify(x, (PATCH_SZ, PATCH_SZ), step=PATCH_SZ)


def extract_patches(image_file_path, mask_file_path):
    """
    Extract patches given file paths
    """

    # extract image file names for later
    rgb_name = os.path.basename(image_file_path)[:-4]
    mask_name = os.path.basename(mask_file_path)[:-4]

    # print("opening images...")
    # open up the images for later
    
    rgb_ten = open_image(image_file_path)
    mask_ten = open_image(mask_file_path)
    
    rgb_sliced = split_tensor(rgb_ten)
    mask_sliced = split_tensor(mask_ten)
    
    for i, (rgb_patch_ten, mask_patch_ten) in enumerate(zip(rgb_sliced, mask_sliced)):
        
        # rgb
        rgb_im = Image.fromarray(rgb_patch_ten)
        filename = f"{rgb_name}--patch{i:05d}.png"
        rgb_im.save(os.path.join(OUTPUT_DIR, "images", filename))

        # mask
        mask_im = Image.fromarray(mask_patch_ten)
        filename = f"{mask_name}--patch{i:05d}.png".replace("_target", "")
        mask_im.save(os.path.join(OUTPUT_DIR, "targets", filename))


if __name__ == "__main__":
    print("This script no longer removes class data")

    total_images = len(IMAGES)
    count = 1

    for image, mask in zip(IMAGES, MASKS):
        print(f"Unpacking image {count}/{total_images}", end="\r", flush=True)
        extract_patches(image, mask)
        count += 1

    print("finished!")
