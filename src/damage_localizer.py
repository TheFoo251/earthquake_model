import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_params
from segmentation_models_pytorch import losses
import torchmetrics.classification
import torchmetrics.segmentation
from torchvision.transforms import v2
import torch
from tqdm import tqdm
import lightning as L
import torchmetrics
from full_dataset import DamagedOnlyDataset, get_loaders
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import logging
import sys
import torch_utils

OPTIMIZE = False
ENCODER = "resnet50"
PRETRAINING = "imagenet"
EPOCHS = 5
NUM_TRIALS = 20


params = get_preprocessing_params(encoder_name=ENCODER, pretrained=PRETRAINING)


transforms = {
    "image": v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(224),
            v2.ToDtype(torch.float32, scale=True),  # scale images to [0.0, 1.0]
            v2.Normalize(
                mean=params["mean"], std=params["std"]
            ),  # make sure to normalize to match ConvNext
        ]
    ),
    "mask": v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(224),
            v2.ToDtype(
                torch.long, scale=False
            ),  # make sure to turn off scaling for mask
        ]
    ),
}

# same for ery-body
DATASET = DamagedOnlyDataset(patch_sz=256, transforms=transforms)
DATALOADERS = get_loaders(16, split=0.7, full_ds=DATASET)


class MyModel(L.LightningModule):
    def __init__(self, lr, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs
        )
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.train_dice = torchmetrics.segmentation.GeneralizedDiceScore(num_classes=1)
        self.val_dice = torchmetrics.segmentation.GeneralizedDiceScore(num_classes=1)
        self.lr = lr

    def training_step(self, batch, batch_idx):
        image, mask = batch
        logits_pred_mask = self.model(image)
        loss = self.loss_fn(logits_pred_mask, mask)

        # mask from logits
        pred_mask = (logits_pred_mask.sigmoid() > 0.5).float()

        self.train_dice(pred_mask, mask)
        self.log(
            "train_dice", self.train_dice, on_step=True, on_epoch=False, prog_bar=True
        )
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def validation_step(self, batch, batch_idx):
        image, mask = batch
        logits_pred_mask = self.model(image)
        val_loss = self.loss_fn(logits_pred_mask, mask)

        # mask from logits
        pred_mask = (logits_pred_mask.sigmoid() > 0.5).float()

        self.val_dice(pred_mask, mask)

        self.log("val_dice", self.val_dice, on_step=True, on_epoch=True)
        self.log("val_loss", val_loss)

    def predict_step(self, batch, batch_idx):
        image, _ = batch
        logits_pred_mask = self.model(image)
        pred_mask = (logits_pred_mask.sigmoid() > 0.5).float()
        return pred_mask


def objective(trial: optuna.trial.Trial) -> float:

    lr = trial.suggest_float("learning_rate", 1e-6, 1e-2)

    model = MyModel(
        lr=lr,
        arch="unet",
        encoder_name=ENCODER,
        in_channels=3,
        out_classes=1,
        decoder_attention_type="scse",
    )

    trainer = L.Trainer(
        logger=True,
        enable_checkpointing=False,
        max_epochs=EPOCHS,
        accelerator="auto",
        callbacks=[
            PyTorchLightningPruningCallback(trial, monitor="val_dice"),
        ],
    )

    hyperparameters = dict(learning_rate=lr)
    trainer.logger.log_hyperparams(hyperparameters)

    trainer.fit(
        model=model,
        train_dataloaders=DATALOADERS["train"],
        val_dataloaders=DATALOADERS["val"],
    )

    return trainer.callback_metrics["val_dice"].item()


if __name__ == "__main__":

    if OPTIMIZE:
        optuna.logging.get_logger("optuna").addHandler(
            logging.StreamHandler(sys.stdout)
        )
        study_name = "damage-localizer-study-1"  # Unique identifier of the study.
        storage_name = "sqlite:///{}.db".format(study_name)
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            direction="maximize",
            load_if_exists=True,
        )
        study.optimize(objective, n_trials=NUM_TRIALS)

    else:
        model = MyModel(
            lr=1e-3,
            arch="unet",
            encoder_name=ENCODER,
            in_channels=3,
            out_classes=1,
            decoder_attention_type="scse",
        )

        trainer = L.Trainer(max_epochs=EPOCHS, log_every_n_steps=1)
        # trainer.fit(
        #     model=model,
        #     train_dataloaders=DATALOADERS["train"],
        #     val_dataloaders=DATALOADERS["val"],
        # )

        predictions = trainer.predict(
            dataloaders=DATALOADERS["val"], model=model
        )  # list of batches
        print(torch.max(predictions[0][0]))
        print(predictions[0][0].shape)
        predictions = torch.cat(
            predictions
        ).float()  # make it one giant tensor, and convert type
        predictions = torch.split(
            predictions, split_size_or_sections=1
        )  # split into samples
        predictions = list(
            map(torch.squeeze, predictions)
        )  # split into squeezed samples

        # print(predictions[0])
        torch_utils.imshow(imgs=predictions[:4])  # just take 4
