"""
A baseline model created using lighnting and segmodels.
"""

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import os
import torch
import matplotlib.pyplot as plt
import lightning as L
import segmentation_models_pytorch as smp
import torch.nn.functional as F

from pprint import pprint
from torch.utils.data import DataLoader

# my imports
from eq_dataset import get_loaders


# model
class MyModel(L.LightningModule):
    loss_functions = {
        "dice": smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
    }

    def __init__(
        self,
        arch,
        encoder_name,
        in_channels,
        out_classes,
        loss_function="dice",
        **kwargs
    ):
        super().__init__()
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = MyModel.loss_functions[loss_function]

    def forward(self, image):
        mask = self.model(image)
        return mask

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)  # y_pred, y_true
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        y_hat = self.model(x)
        val_loss = self.loss_fn(y_hat, y)  # y_pred, y_true
        self.log("val_loss", val_loss)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    train_loader, val_loader = get_loaders(128, 32)

    model = MyModel("FPN", "resnet34", in_channels=3, out_classes=5)

    trainer = L.Trainer(
        max_epochs=5,
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # run validation dataset
    valid_metrics = trainer.validate(model, dataloaders=val_loader, verbose=False)
    pprint(valid_metrics)

    # visualization
    batch = next(iter(val_loader))
    with torch.no_grad():
        model.eval()
        logits = model(batch[0])
    pr_masks = logits.sigmoid()

    for image, gt_mask, pr_mask in zip(batch[0], batch[1], pr_masks):
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(image.numpy().transpose(1, 2, 0))  # convert CHW -> HWC
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(gt_mask.numpy().argmax(0))
        plt.title("Ground truth")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pr_mask.numpy().argmax(0))
        plt.title("Prediction")
        plt.axis("off")

        plt.show()

# something is way off. ground truth never looks anything like the preds this time around.
# this is mostly a loss problem, everything worked fine-ish with mse.
