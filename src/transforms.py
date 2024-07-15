from torchvision.transforms import v2
import torch
import torchvision
from torchvision import models

# matches https://pytorch.org/vision/main/models/generated/torchvision.models.convnext_base.html#torchvision.models.ConvNeXt_Base_Weights
CONVNEXT = {
    "image": models.ConvNeXt_Base_Weights.DEFAULT.transforms(),
    "mask": None,  # don't need any
}

# currently, this can't have any randomness, or the image/mask don't match anymore
DETECTOR = {
    "image": v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(224),
            v2.ToDtype(torch.float32, scale=True),  # scale images to [0.0, 1.0]
            v2.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # make sure to normalize to match ConvNext
        ]
    ),
    "mask": v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(224),
            v2.ToDtype(
                torch.float32, scale=False
            ),  # make sure to turn off scaling for mask
        ]
    ),
}

if __name__ == "__main__":
    print(CONVNEXT["image"])
