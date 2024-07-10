import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn


def get_model():
    return smp.create_model(
        arch="unet",  # name of the architecture, e.g. 'Unet'/ 'FPN' / etc. Case INsensitive!
        encoder_name="mit_b0",
        encoder_weights="imagenet",
        in_channels=3,
        classes=5,
    )


# preprocess_input = get_preprocessing_fn("resnet18", pretrained="imagenet")


# # train....
# for images, gt_masks in dataloader:

#     predicted_mask = model(image)
#     loss = loss_fn(predicted_mask, gt_masks)

#     loss.backward()
#     optimizer.step()
