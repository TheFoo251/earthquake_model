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

if len(sys.argv) < 2 or sys.argv[1] == "help" or sys.argv[1] == "-h":
    print("Usage: python3 extract_patches.py patch_size input_dir")
    exit()
    
PATCH_SIZE = int(sys.argv[1])
INPUT_SIZE = 1024

INPUT_DIR = sys.argv[2]
OUTPUT_DIR = f"patch_data_{PATCH_SIZE}"

os.makedirs(f"{OUTPUT_DIR}/images")
os.makedirs(f"{OUTPUT_DIR}/targets")

# Globbing
IMAGES = sorted(glob.glob(f"{INPUT_DIR}/images/*"))
MASKS = sorted(glob.glob(f"{INPUT_DIR}/targets/*"))

# print(IMAGES)
# print(MASKS)


def split(array):
    """
    Split a matrix into sub-matrices.
    # https://stackoverflow.com/questions/11105375/how-to-split-a-matrix-into-4-blocks-using-numpy
    
    """
    
    nrows, ncols = INPUT_SIZE // PATCH_SIZE, INPUT_SIZE // PATCH_SIZE
    r, h = array.shape
    return (array.reshape(h//nrows, nrows, -1, ncols)
                 .swapaxes(1, 2)
                 .reshape(-1, nrows, ncols))



def extract_patches(image_file_path, mask_file_path):
    """
    Extract patches given file paths
    """

    # extract image file names for later
    rgb_name = os.path.basename(image_file_path)[:-4]
    mask_name = os.path.basename(mask_file_path)[:-4]

    # print("opening images...")
    # open up the images for later
    rgb_image = Image.open(image_file_path)
    mask_image = Image.open(mask_file_path)
    
    rgb_array = np.asarray(rgb_image)
    mask_array = np.asarray(mask_image)
    
    rgb_sliced = split(rgb_array)
    mask_sliced = split(mask_array)
    
    for i, rgb_arr, mask_arr in enumerate(zip(rgb_sliced, mask_sliced)):
        
        # rgb
        rgb_im = Image.fromarray(rgb_arr)
        filename = f"{rgb_name}--patch{i:05d}.png"
        rgb_im.save(os.path.join(OUTPUT_DIR, "images", filename))

        # mask
        mask_im = Image.fromarray(mask_arr)
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
