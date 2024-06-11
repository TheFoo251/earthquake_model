import torch
import albumentations as A 
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch-unet import UNET

from torch-utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs
)


# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_availible() else exit()
BATCH_SIZE = 8
NUM_EPOCHS = 100
NUM_WORKERS = 2
IMAGE_HEIGHT = 1024
IMAGE_WIDTH = 1024
PIN_MEMORY = True
LOAD_MODEL = False
IMG_DIR = "data/post-disaster/images/"
MASK_DIR = "data/post-disaster/targets/"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader) #TODO -- LOOK THIS UP
    
    for batch_idx, (data, targets) in enumerate(loop):
        data.to(DEVICE)
        # important for binary crossentropy to cast as float??
        targets = targets.float().unsqeeze(1).to(DEVICE)
        
        #forward TODO -- look up what float16 training is
        with torch.cuda.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, target)
            
        #backward
        optimizer.zero_grad()
        scaler.scale(loss).backward() # TODO -- look up what this scaler tomfoolery is
        scaler.step(optimizer)
        scaler.update()
        
        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        
    
def main():
    #TODO -- look ub albumentations or whatever
    transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH), # not necessary
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5)
            A.VerticalFlip(p=0.1)
            A.Normalize( # basically just divides by 255
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
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
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss() # no sigmoid on output
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_loader, val_loader = get_loaders(
        IMG_DIR,
        MASK_DIR,
        BATCH_SIZE,
        NUM_WORKERS,
        PIN_MEMORY,
        transforms
    )
    
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        
        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(val_loader)
        
        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )
    
if __name__ == "__main__:
    main()

