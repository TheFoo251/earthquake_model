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
from full_dataset import SiameseDataset, get_loaders
from torch_utils import plot_loss_curves


# check for CUDA
if not torch.cuda.is_available():
    print("CUDA isn't working!!")
    exit()


cudnn.benchmark = True

NUM_EPOCHS = 30
DEVICE = torch.device("cuda:0")

PATCH_SZ, BATCH_SZ = 256, 16  # lower batch size?


class Learner:
    def __init__(
        self, loader, model, optimizer, loss_fn, scaler, scheduler, device=DEVICE
    ):
        self.loader = loader
        self.model = model.to(DEVICE)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scaler = scaler
        self.scheduler = scheduler
        self.device = device
        self.metrics = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

    def check_accuracy(self, loader):
        correct = 0
        self.model.eval()
        with torch.no_grad():
            for _, _, data, _, targets in loader:
                data, targets = data.to(DEVICE), targets.to(DEVICE)
                preds = self.model(data)
                correct += torch.sum(targets == torch.argmax(preds, dim=1)).item()
            accuracy = correct / len(loader.dataset)

        self.model.train()
        return accuracy

    def train_one_epoch(self):
        loop = tqdm(self.loader["train"])  # progress bar wrapped over dataloader
        self.model.train(True)

        running_loss = 0.0

        for batch_idx, (_, _, data, _, targets) in enumerate(loop):
            data, targets = data.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()  # supposed to be first in this loop

            # forward
            with torch.cuda.amp.autocast():  # automatically casts as fp16
                predictions = self.model(data)
                loss = self.loss_fn(predictions, targets)

            # metrics -- these go outside of autograd because they are not computationally intensive
            running_loss += loss.item()

            # backward (backward -> optimizer step -> scheduler step)
            self.scaler.scale(loss).backward()  # backward pass using scaler
            self.scaler.step(self.optimizer)  # optimizer step using scaler
            self.scaler.update()  # updates for next iteration

            # update tqdm loop
            loop.set_postfix(loss=running_loss / (batch_idx + 1))

        return running_loss / len(self.loader)

    def val_one_epoch(self):
        loop = tqdm(self.loader["val"])  # progress bar wrapped over dataloader
        self.model.eval()

        running_loss = 0.0

        with torch.no_grad():
            for batch_idx, (_, _, data, _, targets) in enumerate(loop):
                data, targets = data.to(self.device), targets.to(self.device)
                predictions = self.model(data)
                loss = self.loss_fn(predictions, targets)
                running_loss += loss.item()

                # update tqdm loop
                loop.set_postfix(val_loss=running_loss / (batch_idx + 1))

        return running_loss / len(self.loader)

    def train_model(self, num_epochs):
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            train_loss = self.train_one_epoch()
            val_loss = self.val_one_epoch()
            self.scheduler.step(
                val_loss
            )  # this assumes reduceLRonplateu, unfortunately
            train_accuracy = self.check_accuracy(loader=self.loader["train"])
            val_accuracy = self.check_accuracy(loader=self.loader["val"])
            self.metrics["train_loss"].append(train_loss)
            self.metrics["val_loss"].append(val_loss)
            self.metrics["train_acc"].append(train_accuracy)
            self.metrics["val_acc"].append(val_accuracy)


# --- METRICS ---


model_weights = models.ConvNeXt_Base_Weights.DEFAULT
dataset = SiameseDataset(patch_sz=PATCH_SZ, transforms=transforms.CONVNEXT, even=True)
dataloaders = get_loaders(full_ds=dataset, batch_size=BATCH_SZ, split=0.8)


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

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)


# Decay LR by a factor of 0.1 every 7 epochs
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, "min")

scaler = torch.cuda.amp.GradScaler()

learner = Learner(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    scaler=scaler,
    scheduler=scheduler,
    device=DEVICE,
    loader=dataloaders,
)

learner.train_model(num_epochs=NUM_EPOCHS)

plot_loss_curves(learner.metrics)
