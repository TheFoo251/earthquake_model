from data import SiameseDataset, get_loaders
from torchvision import models
import torch.nn as nn
from torchvision.models.convnext import LayerNorm2d
from torchvision.transforms import v2
import torch


"""
All Model Systems are expected to return a constructed model, dataloaders, and a loss function
"""


class ConvNextSystem:
    def __init__(self, patch_size, batch_size):
        # currently, this can't have any randomness, or the images don't line up anymore...
        # matches https://pytorch.org/vision/main/models/generated/torchvision.models.convnext_base.html#torchvision.models.ConvNeXt_Base_Weights
        transforms = {
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
        model_weights = models.ConvNeXt_Base_Weights.DEFAULT
        dataset = SiameseDataset(patch_sz=patch_size, transforms=transforms, even=True)
        self.dataloaders = get_loaders(
            full_ds=dataset, batch_size=batch_size, split=0.8
        )
        model = models.convnext_base(weights=model_weights)

        # this section copied from https://medium.com/exemplifyml-ai/image-classification-with-resnet-convnext-using-pytorch-f051d0d7e098
        n_inputs = None
        for name, child in model.named_children():
            if name == "classifier":
                for sub_name, sub_child in child.named_children():
                    if sub_name == "2":
                        n_inputs = sub_child.in_features
        n_outputs = 2

        sequential_layers = nn.Sequential(
            LayerNorm2d((1024,), eps=1e-06, elementwise_affine=True),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(n_inputs, 2048, bias=True),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, n_outputs),
        )
        model.classifier = sequential_layers

        self.model = model

        self.loss_fn = nn.CrossEntropyLoss()
