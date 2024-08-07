"""
This file is in charge of knowing where the data is kept and how to create a usable dataset out of it.
Returns images as HWC. Masks are NOT one-hot encoded.
"""

import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from pathlib import Path
import torchvision.transforms.v2 as T
import torch.nn.functional as F


class EarthquakeDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))  # NEVER FORGET TO SORT!
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):  # TODO -- make the transform system better
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.masks[index])
        # requires PILToTensor rather than ToTensor, or masks get scaled to be between 0 and 1
        # this also does the required format permutation for us
        image = T.PILToTensor()(Image.open(img_path).convert("RGB")).float()
        mask = T.PILToTensor()(
            Image.open(mask_path).convert("L")
        ).long()  # must be a long so it can be one-hotted later
        # needs .long to fix one-hot issue

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask


def get_loaders(patch_sz, batch_size, num_workers=4, pin_memory=True, transforms=None):
    base = Path(f"/home/dawson/Desktop/repos/earthquake_model/data/{patch_sz}_patches")

    full_ds = EarthquakeDataset(
        image_dir=base / "images", mask_dir=base / "targets", transform=transforms
    )
    num_train = int(len(full_ds) * 0.9)
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
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def show_image_and_mask(image, mask):
    """
    Given an image and mask, plot both.
    """
    plt.subplot(1, 2, 1)
    plt.imshow(
        image.permute(1, 2, 0)
    )  # for visualization we have to transpose back to HWC
    plt.subplot(1, 2, 2)
    plt.imshow(
        mask.permute(1, 2, 0)
    )  # need argmax for viewing # SOMETHING HERE IS WRONG!!
    plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    train_loader, val_loader = get_loaders(128, 32)
    sample = train_loader.dataset[0]

    print(sample[0].shape)
    print(sample[1].shape)

    print(sample[1])

    # show_image_and_mask(sample[0], sample[1])
