import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from torchvision.models.convnext import LayerNorm2d
import math
import transforms
import matplotlib.pyplot as plt 

# other files
from torch_utils import plot_loss_curves

from models import ConvNextSystem


# check for CUDA
if not torch.cuda.is_available():
    print("CUDA isn't working!!")
    exit()


cudnn.benchmark = True

NUM_EPOCHS = 30
DEVICE = torch.device("cuda:0")

PATCH_SZ, BATCH_SZ = 256, 16  # lower batch size?


class Metrics:
    def __init__(self):
        self.data = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

    def append(self, metric, value):
        self.data[metric].append(value)

        # from https://www.learnpytorch.io/04_pytorch_custom_datasets/#78-plot-the-loss-curves-of-model-0
    def plot_loss_curves(self, show=True, save=None):
        """
        Plots training curves from metrics.
        """

        # Get the loss values of the results dictionary (training and test)
        loss = self.data["train_loss"]
        test_loss = self.data["val_loss"]

        # Get the accuracy values of the self.data dictionary (training and test)
        accuracy = self.data["train_acc"]
        test_accuracy = self.data["val_acc"]

        # Figure out how many epochs there were
        epochs = range(len(self.data["train_loss"]))

        # Setup a plot
        plt.figure(figsize=(15, 7))

        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, loss, label="train_loss")
        plt.plot(epochs, test_loss, label="val_loss")
        plt.title("Loss")
        plt.xlabel("Epochs")
        plt.legend()

        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, accuracy, label="train_accuracy")
        plt.plot(epochs, test_accuracy, label="val_accuracy")
        plt.title("Accuracy")
        plt.xlabel("Epochs")
        plt.legend()

        if show:
            plt.show()
        if not save == None:
            plt.savefig(save)


class Learner:
    def __init__(
        self, model_system, optimizer, scaler, scheduler, device=DEVICE
    ):
        self.loader = model_system.loader
        self.model = model_system.model.to(DEVICE)
        self.optimizer = optimizer
        self.loss_fn = model_system.loss_fn
        self.scaler = scaler
        self.scheduler = scheduler
        self.device = device
        self.metrics = Metrics()

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
            self.metrics.append("train_loss", train_loss)
            self.metrics.append("val_loss", val_loss)
            self.metrics.append("train_acc", train_accuracy)
            self.metrics.append("val_acc", val_accuracy)


# --- METRICS ---


if __name__ == "__main__":

    model_system = ConvNextSystem()


lr = 9e-6  # from optimizer study, close to 1e-5

# here's where the transfer magic happens...

optimizer = optim.AdamW(model_system.model.parameters(), lr=lr)


# Decay LR by a factor of 0.1 every 7 epochs
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, "min")

scaler = torch.cuda.amp.GradScaler()

learner = Learner(
    model_system=model_system,
    optimizer=optimizer,
    scaler=scaler,
    scheduler=scheduler,
    device=DEVICE,
)

learner.train_model(num_epochs=NUM_EPOCHS)

learner.metrics.plot_loss_curves()
