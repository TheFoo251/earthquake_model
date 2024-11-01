# https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch_unet import UNET

from torch_utils import (
    save_checkpoint,
    load_checkpoint,
    check_accuracy,
    save_predictions_as_imgs,
)

from full_dataset import get_loaders

# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else exit()
BATCH_SIZE = 16
NUM_EPOCHS = 15
NUM_WORKERS = 2
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False
IMG_DIR = "data/256_patches/images"
MASK_DIR = "data/256_patches/targets/"
#CHECKPOINT_PATH = f"model_checkpoints/checkpoint.{IMAGE_WIDTH}.tar"





def train(loader, model, optimizer, loss_fn, scaler):

    def train_one_epoch():
        loop = tqdm(loader) # progress bar

        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=DEVICE)
            # important for binary crossentropy to cast as float??
            targets = targets.float().unsqueeze(1).to(device=DEVICE)

            # forward
            # has to do with fp16 training
            with torch.cuda.amp.autocast():
                predictions = model(data)
                loss = loss_fn(predictions, targets)

            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward() # scaler has to do with FP16
            scaler.step(optimizer)
            scaler.update()

            # update tqdm loop
            loop.set_postfix(loss=loss.item())
        
    

    """
    val_transforms = A.Compose(
        [   A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH), # not necessary
            A.Normalize( # basically just divides by 255
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    """

    # to generalize to more classes, just up out channels
    # and change to regular CE loss.
    #model = UNET(in_channels=3, out_channels=5).to(device=DEVICE) #don't forget to send the model to the device
    model = segmodel.get_model().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()  # no sigmoid on output?
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        IMG_DIR, MASK_DIR, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, transforms
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load(CHECKPOINT_PATH), model)

    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, CHECKPOINT_PATH)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        # save_predictions_as_imgs(
        #    val_loader, model, folder="saved_images/", device=DEVICE
        # )


if __name__ == "__main__":
    pass