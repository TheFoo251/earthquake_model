import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import torchvision.transforms.v2.functional as TF
import torch
import torch.nn.functional as F


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
        pre_image = TF.pil_to_tensor(
            Image.open(self.pre_images[index]).convert("RGB")
        ).float()
        pre_mask = TF.pil_to_tensor(Image.open(self.pre_masks[index]).convert("L"))
        post_image = TF.pil_to_tensor(
            Image.open(self.post_images[index]).convert("RGB"),
        ).float()  # when should it become a float? Here or the transforms?
        post_mask = TF.pil_to_tensor(Image.open(self.post_masks[index]).convert("L"))

        # if self.transform is not None:
        #     augmentations = self.transform(image=image, mask=mask)
        #     image = augmentations["image"]
        #     mask = augmentations["mask"]
        # what should be transforms, and what should be part of the Dataset?

        # don't be too clever for your own good. This works fine.
        label = (
            torch.max(post_mask) > 1
        ).long()  # best practice is to have it be the label number??
        # needs to be long to work with any pytorch stuff for some reason

        return pre_image, pre_mask, post_image, post_mask, label


if __name__ == "__main__":
    ds = SiameseDataset(patch_sz=256)
    damaged_data = [x for x in ds if x[4] == 1]
    undamaged_data = [x for x in ds if x[4] == 0]
    print("number of damaged instances: ", len(damaged_data))
    print("number of undamaged instances: ", len(undamaged_data))

    from torch_utils import imshow

    ex = ds[0]
    labels = ["No Damage", "Damaged"]
    print(ex[4])
    # print(ds[0][:])
    imshow(list(ex[0:4]), title=labels[ex[4].item()])
