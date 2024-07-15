"""
This file contains all the code relating to the dataset, augmentations, and dataloaders.
"""

import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import torchvision.transforms.v2.functional as TF
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random


# the class is responsible for knowing where the data is stored
class SiameseDataset(Dataset):
    """
    Expects a dictionary with "image_transforms" and "mask_transforms".
    The dataset returns PIL images and expects ALL transforms to take place externally.
    """

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

        if self.transform is not None:
            pre_image = self.transform(pre_image)
            post_image = self.transform(post_image)  # NOT CONSISTENT
        # what should be transforms, and what should be part of the Dataset?

        # don't be too clever for your own good. This works fine.
        label = (
            torch.max(post_mask) > 1
        ).long()  # best practice is to have it be the label number??
        # needs to be long to work with any pytorch stuff for some reason

        return pre_image, pre_mask, post_image, post_mask, label


def get_loaders(
    patch_sz,
    batch_size,
    num_workers=4,
    pin_memory=True,
    transforms=None,
    split=0.9,
):
    full_ds = SiameseDataset(patch_sz=patch_sz, transform=transforms)
    num_train = int(len(full_ds) * split)
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds,
        [num_train, len(full_ds) - num_train],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,  # TODO look up 'workers'
        pin_memory=pin_memory,  # TODO look up pin memory
        shuffle=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        drop_last=True,
    )

    return {"train": train_loader, "val": val_loader}


# keep separate
def get_even_loaders(
    patch_sz,
    batch_size,
    num_workers=4,
    pin_memory=True,
    transforms=None,
    split=0.9,
):
    full_ds = SiameseDataset(patch_sz=patch_sz, transform=transforms)
    damaged_ds = [x for x in full_ds if x[4] == 1]
    undamaged_ds = [x for x in full_ds if x[4] == 0]
    undamaged_ds = torch.utils.data.Subset(undamaged_ds, range(len(damaged_ds)))
    full_ds = torch.utils.data.ConcatDataset([damaged_ds, undamaged_ds])
    num_train = int(len(full_ds) * split)
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds,
        [num_train, len(full_ds) - num_train],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,  # TODO look up 'workers'
        pin_memory=pin_memory,  # TODO look up pin memory
        shuffle=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        drop_last=True,
    )

    return {"train": train_loader, "val": val_loader}


if __name__ == "__main__":
    ds = SiameseDataset(patch_sz=256)
    damaged_data = [x for x in ds if x[4] == 1]
    undamaged_data = [x for x in ds if x[4] == 0]
    print("number of damaged instances: ", len(damaged_data))  # 147
    print("number of undamaged instances: ", len(undamaged_data))  # 2941

    from torch_utils import imshow

    ex = ds[0]
    labels = ["No Damage", "Damaged"]
    print(ex[4])
    # print(ds[0][:])
    imshow(list(ex[0:4]), title=labels[ex[4].item()])
