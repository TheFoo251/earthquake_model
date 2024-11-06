import torch
import torchvision

import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF
import matplotlib.pyplot as plt
import numpy as np
import random


def save_checkpoint(state, filename):
    print("=> Saving checkpoint")
    torch.save(state, filename)


# TODO -- research tuning models from a checkpoint
def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def dice_score(loader, model, device="cuda"):
    """
    simplistic accuracy checker
    """

    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.3f}")
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()


def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            torchvision.utils.save_image(preds, f"{folder}/{idx}_pred.png")
            torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/{idx}_real.png")

    model.train()


def imshow(imgs, title=None, show_colorbar=False):
    """Plot tensors or PIL images as images. Can mix and match types."""
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        if torch.is_tensor(img):
            img = img.detach()
            img = TF.to_pil_image(img)
        im = axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        if title is not None:
            plt.title(title)
        if show_colorbar:
            cbar = fig.colorbar(im, ax=axs[0, i])
    plt.ioff()
    plt.show()