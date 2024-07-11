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
import optuna
import torcheval.metrics.functional as FM


# other files
from torch_utils import get_loaders

cudnn.benchmark = True

NUM_EPOCHS = 3
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PATCH_SZ, BATCH_SZ = 256, 16

NUM_TRIALS = 2

# get data
dataloaders = get_loaders(PATCH_SZ, BATCH_SZ)
dataset_sizes = {x: len(dataloaders[x]) for x in ["train", "val"]}


def train_one_epoch(loader, model, optimizer, loss_fn, scaler, scheduler):
    loop = tqdm(loader)  # progress bar wrapped over dataloader
    model.train(True)

    running_loss = 0.0

    for batch_idx, (_, _, data, _, targets) in enumerate(loop):
        data, targets = data.to(device=DEVICE), targets.to(device=DEVICE)

        optimizer.zero_grad()  # supposed to be first in this loop

        # forward
        with torch.cuda.amp.autocast():  # automatically casts as fp16
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # metrics -- these go outside of autograd because they are not computationally intensive
        running_loss += loss.item()

        # backward (backward -> optimizer step -> scheduler step)
        scaler.scale(loss).backward()  # backward pass using scaler
        scaler.step(optimizer)  # optimizer step using scaler
        scaler.update()  # updates for next iteration
        scheduler.step()  # has to be after the optimizer step

        # update tqdm loop
        loop.set_postfix(loss=running_loss)


def val_one_epoch():
    pass


def check_accuracy(loader, model):
    correct = 0
    model.eval()
    with torch.no_grad():
        for _, _, data, _, targets in loader:
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            preds = torch.sigmoid(model(data))
            true_positives += torch.sum(targets == torch.argmax(preds, dim=1)).item()
        accuracy = true_positives / torch.sum(targets == 1)

    model.train()
    return accuracy


def check_f1_score(loader, model):
    running_f1_score = 0
    model.eval()
    with torch.no_grad():
        for _, _, data, _, targets in loader:
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            preds = model(data)
            running_f1_score += FM.multiclass_f1_score(preds, targets, num_classes=2)
        avg_f1_score = running_f1_score / len(loader)

    model.train()
    return avg_f1_score


def check_recall(loader, model):  # What I'm most interested in....
    running_f1_score = 0
    model.eval()
    with torch.no_grad():
        for _, _, data, _, targets in loader:
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            preds = model(data)
            running_f1_score += FM.multiclass_recall(preds, targets, num_classes=2)
        avg_f1_score = running_f1_score / len(loader)

    model.train()
    return avg_f1_score


def check_recall_manually():
    pass


def objective(trial):

    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2)
    # mom = trial.suggest_float("momentum", 0.0, 1.0) # not necessary for adam
    step_size = trial.suggest_int("step_size", 1, 10)
    gamma = trial.suggest_float("gamma", 0.0, 1.0)

    # here's where the transfer magic happens...
    model_ft = models.resnet18(weights="IMAGENET1K_V1")
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)  # 2 classes

    model_ft = model_ft.to(DEVICE)

    loss_fn = (
        nn.CrossEntropyLoss()
    )  # get a better loss function to deal with class imbalance...

    # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr, momentum=mom)
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=lr)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_ft, step_size=step_size, gamma=gamma
    )

    scaler = torch.cuda.amp.GradScaler()

    # train it!
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        train_one_epoch(
            dataloaders["train"],
            model_ft,
            optimizer_ft,
            loss_fn,
            scaler,
            scheduler=exp_lr_scheduler,
        )

    recall = check_recall(model=model_ft, loader=dataloaders["val"])

    return recall


if __name__ == "__main__":
    study = optuna.create_study()
    study.optimize(objective, n_trials=NUM_TRIALS)
