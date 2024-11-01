import numpy as np
from PIL import Image
from pathlib import Path
import random
import os
import shutil

random.seed(7)

base_path = "data/256_patches"
post_disaster_masks = sorted((base_path / "post-disaster/targets").glob("*"))
post_disaster_images = sorted((base_path / "post-disaster/images").glob("*"))
pre_disaster_masks = sorted((base_path / "pre-disaster/targets").glob("*"))
pre_disaster_images = sorted((base_path / "pre-disaster/images").glob("*"))

all_data = list(
    zip(
        post_disaster_masks,
        post_disaster_images,
        pre_disaster_masks,
        pre_disaster_images,
    )
)

# data_path_tuple = all_data[0]
undamaged_data = [x for x in all_data if np.max(np.asarray(Image.open(x[0]))) <= 1]
damaged_data = [x for x in all_data if np.max(np.asarray(Image.open(x[0]))) > 1]

shorter_undamaged_data = random.sample(undamaged_data, k=len(damaged_data))
assert len(damaged_data) == len(shorter_undamaged_data)
all_data = damaged_data + shorter_undamaged_data
random.shuffle(all_data)

base_path = Path("data/256_patches_even")
paths = [
    base_path / "post-disaster/targets",
    base_path / "post-disaster/images",
    base_path / "pre-disaster/targets",
    base_path / "pre-disaster/images",
]


for path in paths:
    os.makedirs(path, exist_ok=True)

for data_idx, data in enumerate(all_data):
    for i in range(4):
        shutil.copy(data[i], paths[i] / f"{data_idx:03}.png")
