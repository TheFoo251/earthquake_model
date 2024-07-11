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
import torch.nn.functional as F
from torchvision.models.convnext import LayerNorm2d
import math
import logging
import sys

# other files
from torch_utils import get_loaders

cudnn.benchmark = True

NUM_EPOCHS = 2
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PATCH_SZ, BATCH_SZ = 256, 16

NUM_TRIALS = 20


# losses


# copied from https://github.com/ashawkey/FocalLoss.pytorch/blob/master/Explaination.md
class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss
    """

    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        input: [N, C], float32
        target: [N, ], int64
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight)
        return loss


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
        loop.set_postfix(loss=running_loss / (batch_idx + 1))


def val_one_epoch(loader, model, loss_fn):
    loop = tqdm(loader)  # progress bar wrapped over dataloader
    model.eval()

    running_loss = 0.0

    with torch.no_grad():
        for batch_idx, (_, _, data, _, targets) in enumerate(loop):
            data, targets = data.to(device=DEVICE), targets.to(device=DEVICE)
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            running_loss += loss.item()

            # update tqdm loop
            loop.set_postfix(val_loss=running_loss / (batch_idx + 1))


# --- METRICS ---

# def check_accuracy(loader, model):
#     correct = 0
#     model.eval()
#     with torch.no_grad():
#         for _, _, data, _, targets in loader:
#             data, targets = data.to(DEVICE), targets.to(DEVICE)
#             preds = torch.sigmoid(model(data))
#             true_positives += torch.sum(targets == torch.argmax(preds, dim=1)).item()
#         accuracy = true_positives / torch.sum(targets == 1)

#     model.train()
#     return accuracy


# def check_f1_score(loader, model):
#     running_f1_score = 0
#     model.eval()
#     with torch.no_grad():
#         for _, _, data, _, targets in loader:
#             data, targets = data.to(DEVICE), targets.to(DEVICE)
#             preds = model(data)
#             running_f1_score += FM.multiclass_f1_score(
#                 preds, targets, num_classes=2
#             ).item()
#         avg_f1_score = running_f1_score / len(loader)

#     model.train()
#     return avg_f1_score


def check_recall(loader, model):  # What I'm most interested in....
    running_recall = 0
    model.eval()
    with torch.no_grad():
        for _, _, data, _, targets in loader:
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            preds = model(data)
            running_recall += FM.multiclass_recall(preds, targets, num_classes=2).item()
        avg_recall = running_recall / len(loader)

    model.train()
    return avg_recall


def check_precision(loader, model):  # What I'm most interested in....
    running_precision = 0
    model.eval()
    with torch.no_grad():
        for _, _, data, _, targets in loader:
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            preds = model(data)
            running_precision += FM.multiclass_precision(
                preds, targets, num_classes=2
            ).item()
        avg_precision = running_precision / len(loader)

    model.train()
    return avg_precision


# def check_recall_manually(loader, model):
#     running_target_p = 0
#     running_true_p = 0
#     model.eval()
#     with torch.no_grad():
#         for _, _, data, _, targets in loader:
#             data, targets = data.to(DEVICE), targets.to(DEVICE)
#             preds = model(data)
#             running_true_p += torch.sum(
#                 (targets == torch.argmax(preds, dim=-1)) * targets  # dim NEEDS to be -1
#             ).item()
#             running_target_p += torch.sum(targets).item()
#         recall = running_true_p / running_target_p
#     model.train()
#     return recall


# def count_positive_pred(loader, model):
#     running_target_p = 0
#     running_p_preds = 0
#     model.eval()
#     with torch.no_grad():
#         for _, _, data, _, targets in loader:
#             data, targets = data.to(DEVICE), targets.to(DEVICE)
#             preds = model(data)
#             running_p_preds += torch.sum(torch.argmax(preds, dim=-1)).item()
#             running_target_p += torch.sum(targets).item()
#     print(
#         f"model made {running_p_preds} p preds; there are {running_target_p} p in the dataset"
#     )


def objective(trial):

    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2)
    gamma = trial.suggest_int(
        "gamma", 10, 20
    )  # a higher gamma is good for imbalance --> model freaks out at 20, 10 not enough

    # get data
    dataloaders = get_loaders(PATCH_SZ, BATCH_SZ)

    # here's where the transfer magic happens...
    model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)

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

    model = model.to(DEVICE)

    loss_fn = FocalLoss(gamma=gamma)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=math.floor(len(dataloaders["train"]) / BATCH_SZ)
    )

    scaler = torch.cuda.amp.GradScaler()

    # train it!
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        train_one_epoch(
            dataloaders["train"],
            model,
            optimizer,
            loss_fn,
            scaler,
            scheduler=scheduler,
        )
        val_one_epoch(model=model, loss_fn=loss_fn, loader=dataloaders["val"])
        # print("training:")
        # count_positive_pred(model=model, loader=dataloaders["train"])
        # print("validation")
        # count_positive_pred(model=model, loader=dataloaders["val"])

    # recall = check_recall_manually(model=model, loader=dataloaders["val"])

    recall = check_recall(model=model, loader=dataloaders["val"])
    precision = check_precision(model=model, loader=dataloaders["val"])

    return precision, recall


if __name__ == "__main__":

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "convnext-classifier-study"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        directions=["maximize", "maximize"],
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=NUM_TRIALS)
