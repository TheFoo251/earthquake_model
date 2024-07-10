import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path


# the class is responsible for knowing where the data is stored
class SiameseDataset(Dataset):
    def __init__(self, patch_sz, transform=None):
        self.base_path = Path(f"data/{patch_sz}_patches")
        self.transform = transform

        self.pre_images = sorted(
            list((self.base_path / "pre-disaster" / "images").glob("*.png"))
        )
        self.pre_masks = sorted(
            list((self.base_path / "pre-disaster" / "targets").glob("*.png"))
        )
        self.post_images = sorted(
            list((self.base_path / "post-disaster" / "images").glob("*.png"))
        )
        self.post_masks = sorted(
            list((self.base_path / "post-disaster" / "targets").glob("*.png"))
        )

    def __len__(self):
        return len(self.post_images)

    def __getitem__(self, index):
        pre_image = np.array(Image.open(self.pre_images[index]).convert("RGB"))
        pre_mask = np.array(
            Image.open(self.pre_masks[index]).convert("L"), dtype=np.float32
        )
        post_image = np.array(Image.open(self.post_images[index]).convert("RGB"))
        post_mask = np.array(
            Image.open(self.post_masks[index]).convert("L"), dtype=np.float32
        )

        # if self.transform is not None:
        #     augmentations = self.transform(image=image, mask=mask)
        #     image = augmentations["image"]
        #     mask = augmentations["mask"]

        damaged = np.max(post_mask) > 1

        return pre_image, pre_mask, post_image, post_mask, damaged


if __name__ == "__main__":
    ds = SiameseDataset(patch_sz=256)
    print("number of damaged instances: ", len([x for x in ds if x[4] == True]))
