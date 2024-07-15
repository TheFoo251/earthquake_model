import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import models, transforms
from tqdm import tqdm
from torchvision.models.convnext import LayerNorm2d
import math
import transforms

# other files
from full_dataset import get_loaders
from torch_utils import plot_loss_curves

cudnn.benchmark = True

NUM_EPOCHS = 30
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PATCH_SZ, BATCH_SZ = 256, 32


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

    return running_loss / len(loader)


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

    return running_loss / len(loader)


# --- METRICS ---


def check_accuracy(loader, model):
    correct = 0
    model.eval()
    with torch.no_grad():
        for _, _, data, _, targets in loader:
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            preds = model(data)
            correct += torch.sum(targets == torch.argmax(preds, dim=1)).item()
        accuracy = correct / len(loader.dataset)

    model.train()
    return accuracy


model_weights = models.ConvNeXt_Base_Weights.DEFAULT
dataloaders = get_loaders(PATCH_SZ, BATCH_SZ, transforms=transforms.CONVNEXT)


lr = 9e-6  # from optimizer study, close to 1e-5

# here's where the transfer magic happens...

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

model = model.to(DEVICE)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)

# Decay LR by a factor of 0.1 every 7 epochs
scheduler = lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=math.floor(len(dataloaders["train"]) / BATCH_SZ)
)

scaler = torch.cuda.amp.GradScaler()

metrics = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": [],
}


# train it!
for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    train_loss = train_one_epoch(
        dataloaders["train"],
        model,
        optimizer,
        loss_fn,
        scaler,
        scheduler=scheduler,
    )
    val_loss = val_one_epoch(model=model, loss_fn=loss_fn, loader=dataloaders["val"])
    train_accuracy = check_accuracy(loader=dataloaders["train"], model=model)
    val_accuracy = check_accuracy(loader=dataloaders["val"], model=model)
    metrics["train_loss"].append(train_loss)
    metrics["val_loss"].append(val_loss)
    metrics["train_acc"].append(train_accuracy)
    metrics["val_acc"].append(val_accuracy)

final_accuracy = check_accuracy(loader=dataloaders["val"], model=model)
print("final accuracy:", final_accuracy)
plot_loss_curves(metrics)
