# coding: utf-8
from fastai.data.all import *
from patchify import patchify
from PIL import Image
from tqdm import tqdm

# simpler implementation, more readable if perhaps less dry...
def extract_patches(full_arr, sz):
    patch_list = []
    if len(full_arr.shape) < 3 : # account for no channel dim (masks)
        patch_arr = patchify(full_arr, (sz, sz), sz)
        for i in range(patch_arr.shape[0]):
            for j in range(patch_arr.shape[1]):
                patch_list.append(patch_arr[i,j,:,:])
    
    else:
        channels = full_arr.shape[-1]
        patch_arr = patchify(full_arr, (sz, sz, channels), sz)
        for i in range(patch_arr.shape[0]):
            for j in range(patch_arr.shape[1]):
                patch_list.append(patch_arr[i,j,:,:,:])


    return patch_list


#TODO - screw around calculating class imbalance

def get_all_patches(path, sz):
    """
    Given an input data directory,
    returns a list with tuples of form (img_patch, msk_patch)
    """
    
    def get_arrays(path):
        #This NEEDS to be sorted or everything else will be messed up...
        paths = sorted(path.glob("*"))
        return [np.array(Image.open(path)) for path in paths]

    img_arrs = get_arrays(path/"images")
    msk_arrs = get_arrays(path/"targets")

    img_patches = [extract_patches(img_arr, sz) for img_arr in img_arrs]
    msk_patches = [extract_patches(msk_arr, sz) for msk_arr in msk_arrs]

    img_patches = [patch for patches in img_patches for patch in patches]
    msk_patches = [patch for patches in msk_patches for patch in patches]

    # extra processing-- not efficient, but necessary!
    img_patches = [patch.squeeze() for patch in img_patches]
    msk_patches = [patch.squeeze() for patch in msk_patches]

    all_patches = list(zip(img_patches, msk_patches))

    return all_patches

# don't optimize prematurely! >:3

def save_patches(patches, output_dir):
    """
    given patch arrays and an output dir, save all patches
    """
    total = len(patches) # TODO-- proper progress bar with tqdm
    for i, (img, msk) in enumerate(patches): #TODO -- give proper number padding
        Image.fromarray(img).save(output_dir/"images"/f"{i}.png")
        Image.fromarray(msk).save(output_dir/"targets"/f"{i}.png")
        print(f"Saved tuple {i}/{total}", end="\r", flush=True)
    print(end="\r", flush=True)


def mk_patches(patch_size):
    base_path = Path("../data/")
    data_dir = base_path/"full"/"post-disaster"
    patch_dir = base_path/f"{patch_size}_patches"
    
    
    if not patch_dir.is_dir():
        # make necessary directories
        patch_dir.mkdir()
        (patch_dir/"images").mkdir()
        (patch_dir/"targets").mkdir()
    
        print("extracting patches...")
        patches = get_all_patches(data_dir, patch_size)
        print("saving patches...")
        save_patches(patches, patch_dir)
        print("all patches saved!")
    else:
        print("patches already extracted! skipping.")


for sz in [32, 64, 128, 256]:
    mk_patches(sz)





