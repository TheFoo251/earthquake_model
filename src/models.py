from full_dataset import SiameseDataset, get_loaders
from torchvision import models, transforms
import torch.nn as nn
from torchvision.models.convnext import LayerNorm2d

"""
All Model Systems are expected to return a constructed model, dataloaders, and a loss function
"""

class ConvNextSystem:
    def __init__(self, patch_size, batch_size):
        model_weights = models.ConvNeXt_Base_Weights.DEFAULT
        dataset = SiameseDataset(patch_sz=patch_size, transforms=transforms.CONVNEXT, even=True)
        self.dataloaders = get_loaders(full_ds=dataset, batch_size=batch_size, split=0.8)
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
