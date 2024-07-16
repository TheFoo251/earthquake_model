"""
This file contains all the code relating to the dataset, augmentations, and dataloaders.
"""

from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from torchvision.transforms import v2
import torch
from torch.utils.data import DataLoader
import random


# the class is responsible for knowing where the data is stored
class SiameseDataset(Dataset):
    """
    Expects a dictionary with "image_transforms" and "mask_transforms".
    The dataset returns PIL images and expects ALL transforms to take place externally.
    Labels are returned as long ints.
    """

    def __init__(self, patch_sz, transforms=None, even=True):
        self.base_path = Path(f"data/{patch_sz}_patches")
        self.transforms = transforms

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

        self.labels = [
            (np.max(np.asarray(Image.open(mask).convert("L"))) > 1).item()
            for mask in self.post_masks
        ]

        if even:
            self.all_data = list(
                zip(
                    self.pre_images,
                    self.pre_masks,
                    self.post_images,
                    self.post_masks,
                    self.labels,
                )
            )
            damaged_data = [x for x in self.all_data if x[4]]
            undamaged_data = [x for x in self.all_data if not x[4]]

            undersampled_undamaged_data = random.sample(
                undamaged_data, k=len(damaged_data)
            )
            self.all_data = damaged_data + undersampled_undamaged_data

            (
                self.pre_images,
                self.pre_masks,
                self.post_images,
                self.post_masks,
                self.labels,
            ) = zip(*self.all_data)
            # unzip

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        pre_image = Image.open(self.pre_images[index]).convert("RGB")
        pre_mask = Image.open(self.pre_masks[index]).convert("L")
        post_image = Image.open(self.post_images[index]).convert("RGB")
        post_mask = Image.open(self.post_masks[index]).convert("L")
        label = torch.tensor(self.labels[index], dtype=torch.long)

        if self.transforms is not None:
            if self.transforms["image"] is not None:
                pre_image = self.transforms["image"](pre_image)
                post_image = self.transforms["image"](post_image)

            if self.transforms["mask"] is not None:
                pre_mask = self.transforms["mask"](pre_mask)
                post_mask = self.transforms["mask"](post_mask)

        return pre_image, pre_mask, post_image, post_mask, label


class DamagedOnlyDataset(Dataset):
    """
    Returns damaged image and mask pairs from post-disaster.
    This dataset does not process the masks lazily, they're already all loaded when
    the dataset is instantiated. It does process the images lazily.
    """

    def __init__(self, patch_sz, transforms=None):
        self.base_path = Path(f"data/{patch_sz}_patches/post-disaster")
        self.transforms = transforms

        self.pairs = zip(
            sorted(list((self.base_path / "images").glob("*.png"))),
            sorted(list((self.base_path / "targets").glob("*.png"))),
        )

        self.images, self.masks = zip(
            *[  # basically just unzip
                x
                for x in self.pairs
                if (np.max(np.asarray(Image.open(x[1]).convert("L"))) > 1)
            ]
        )

        # get only damage
        self.masks = [np.asarray(Image.open(x)) for x in self.masks]  # open arrays
        self.masks = [x > 1 for x in self.masks]  # make masks
        self.masks = [
            Image.fromarray(x, mode="L") for x in self.masks
        ]  # make images again

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        mask = self.masks[index]  # already an image...

        if self.transforms is not None:
            if self.transforms["image"] is not None:
                image = self.transforms["image"](image)
            if self.transforms["mask"] is not None:
                mask = self.transforms["mask"](mask)

        return image, mask


def get_loaders(
    batch_size,
    num_workers=4,
    pin_memory=True,
    split=0.9,
    full_ds=None,
):
    """
    Requires you pass in an already instantiated dataset with its own transforms
    """
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
    uneven_ds = SiameseDataset(patch_sz=256, even=False)
    damaged_data = [x for x in uneven_ds if x[4] == 1]
    undamaged_data = [x for x in uneven_ds if x[4] == 0]
    even_ds = SiameseDataset(patch_sz=256, even=True)
    assert len(damaged_data) * 2 == len(even_ds)

    from torch_utils import imshow

    ex = damaged_data[0]
    labels = ["No Damage", "Damaged"]
    print(ex[4])
    # print(ds[0][:])
    imshow(list(ex[:4]), title=labels[ex[4].item()])

    ex = even_ds[0]
    print(ex[4])
    # print(ds[0][:])
    imshow(list(ex[:4]), title=labels[ex[4].item()])

    damage_only_ds = DamagedOnlyDataset(patch_sz=256)
    ex = damage_only_ds[4]
    imshow(list(ex))
