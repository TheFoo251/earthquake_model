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




def split_tensor(x, grid_length=4):
    """
    https://discuss.pytorch.org/t/split-an-image-into-a-2-by-2-grid/189895
    split images
    """
    sz = x.shape[1] // grid_length
    row_length, col_length = (sz, sz)
    return (x
               .unfold(1,row_length,col_length)
               .unfold(2,row_length,col_length)
               .reshape(3,grid_length**2,row_length,col_length)
               .permute(1,0,2,3))



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
    
    rgb_sliced = split_tensor(rgb_array)
    mask_sliced = split_tensor(mask_array)
    
    for i, rgb_patch_arr, mask_patch_arr in enumerate(zip(rgb_sliced, mask_sliced)):
        
        # rgb
        rgb_im = Image.fromarray(rgb_patch_arr)
        filename = f"{rgb_name}--patch{i:05d}.png"
        rgb_im.save(os.path.join(OUTPUT_DIR, "images", filename))

        # mask
        mask_im = Image.fromarray(mask_patch_arr)
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
