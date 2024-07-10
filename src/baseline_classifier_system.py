import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
from tqdm import tqdm

cudnn.benchmark = True

NUM_EPOCHS = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = device

from torch_utils import get_loaders

dataloaders = get_loaders(256, 16)
dataset_sizes = {x: len(dataloaders[x]) for x in ["train", "val"]}


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)  # progress bar

    for batch_idx, (_, _, data, _, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        # important for binary crossentropy to cast as float??
        targets = targets.float().to(device=DEVICE) # whoops

        # forward
        # has to do with fp16 training
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()  # scaler has to do with FP16
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


# here's where the transfer magic happens...

model_ft = models.resnet18(weights="IMAGENET1K_V1")
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

loss_fn = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

scaler = torch.cuda.amp.GradScaler()

# train it!
for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    train_fn(dataloaders["train"], model_ft, optimizer_ft, loss_fn, scaler)
