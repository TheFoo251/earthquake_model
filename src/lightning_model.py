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

# get data
train_loader, val_loader = get_loaders(128, 32)


# # MODEL
# class MyModel(L.LightningModule):


#     def forward(self, image):
#         # normalize image here
#         image = (image - self.mean) / self.std
#         mask = self.model(image)
#         return mask

#     def shared_step(self, batch, stage):

#         image = batch[0]

#         # Shape of the image should be (batch_size, num_channels, height, width)
#         # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
#         assert image.ndim == 4

#         # Check that image dimensions are divisible by 32,
#         # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
#         # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have
#         # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
#         # and we will get an error trying to concat these features
#         h, w = image.shape[2:]
#         assert h % 32 == 0 and w % 32 == 0

#         mask = batch[1]

#         # Shape of the mask should be [batch_size, num_classes, height, width]
#         # for binary segmentation num_classes = 1
#         assert mask.ndim == 4

#         # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
#         assert mask.max() <= 1.0 and mask.min() >= 0

#         logits_mask = self.forward(image)

#         # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
#         loss = self.loss_fn(logits_mask, mask)

#         # Lets compute metrics for some threshold
#         # first convert mask values to probabilities, then
#         # apply thresholding
#         prob_mask = logits_mask.sigmoid()
#         pred_mask = (prob_mask > 0.5).float()

#         # We will compute IoU metric by two ways
#         #   1. dataset-wise
#         #   2. image-wise
#         # but for now we just compute true positive, false positive, false negative and
#         # true negative 'pixels' for each image and class
#         # these values will be aggregated in the end of an epoch
#         tp, fp, fn, tn = smp.metrics.get_stats(
#             pred_mask.long(), mask.long(), mode="binary"
#         )

#         return {
#             "loss": loss,
#             "tp": tp,
#             "fp": fp,
#             "fn": fn,
#             "tn": tn,
#         }

#     def shared_epoch_end(self, outputs, stage):
#         # aggregate step metics
#         tp = torch.cat([x["tp"] for x in outputs])
#         fp = torch.cat([x["fp"] for x in outputs])
#         fn = torch.cat([x["fn"] for x in outputs])
#         tn = torch.cat([x["tn"] for x in outputs])

#         # per image IoU means that we first calculate IoU score for each image
#         # and then compute mean over these scores
#         per_image_iou = smp.metrics.iou_score(
#             tp, fp, fn, tn, reduction="micro-imagewise"
#         )

#         # dataset IoU means that we aggregate intersection and union over whole dataset
#         # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
#         # in this particular case will not be much, however for dataset
#         # with "empty" images (images without target class) a large gap could be observed.
#         # Empty images influence a lot on per_image_iou and much less on dataset_iou.
#         dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

#         metrics = {
#             f"{stage}_per_image_iou": per_image_iou,
#             f"{stage}_dataset_iou": dataset_iou,
#         }

#         self.log_dict(metrics, prog_bar=True)

#     def training_step(self, batch, batch_idx):
#         return self.shared_step(batch, "train")

#     def validation_step(self, batch, batch_idx):
#         return self.shared_step(batch, "valid")

#     def test_step(self, batch, batch_idx):
#         return self.shared_step(batch, "test")

#     def test_epoch_end(self, outputs):
#         return self.shared_epoch_end(outputs, "test")

#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=0.0001)


class MyModel(L.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
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
        self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)

    def forward(self, image):
        mask = self.model(image)
        return mask

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y, y_hat)
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        y_hat = self.model(x)
        val_loss = F.mse_loss(y_hat, y)
        self.log("val_loss", val_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    # training

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
        plt.imshow(
            gt_mask.numpy().squeeze()
        )  # just squeeze classes dim, because we have only one class
        plt.title("Ground truth")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(
            pr_mask.numpy().squeeze()
        )  # just squeeze classes dim, because we have only one class
        plt.title("Prediction")
        plt.axis("off")

        plt.show()
