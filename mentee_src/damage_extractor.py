import numpy as np
from PIL import Image
from pathlib import Path
import random

random.seed(7)


post_disaster_masks = sorted(Path("data/256_patches/post-disaster/targets").glob("*"))
post_disaster_images = sorted(Path("data/256_patches/post-disaster/images").glob("*"))
pre_disaster_masks = sorted(Path("data/256_patches/pre-disaster/targets").glob("*"))
pre_disaster_images = sorted(Path("data/256_patches/pre-disaster/images").glob("*"))

all_data = list(zip(post_disaster_masks, post_disaster_images, pre_disaster_masks, pre_disaster_images))

# data_path_tuple = all_data[0]
undamaged_data = [x for x in all_data if np.max(np.asarray(Image.open(x[0]))) <= 1]
damaged_data = [x for x in all_data if np.max(np.asarray(Image.open(x[0]))) > 1]

shorter_undamaged_data = random.sample(undamaged_data, k=len(damaged_data))
assert len(damaged_data) == len(shorter_undamaged_data) 
all_data = damaged_data + shorter_undamaged_data
random.shuffle(all_data)

# HOMEWORK (os.mkdir)

# create new directory within data called 256_patches_even

# create post-disaster/images, post-disaster/targets, pre-disaster/images, pre-diaster/targets under 256_patches_even



