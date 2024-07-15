import torch
from tqdm import tqdm

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Learner:
    def __init__(
        self, loader, model, optimizer, loss_fn, scaler, scheduler, device=DEVICE
    ):
        self.loader = loader
        self.model = model
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
            self.scheduler.step()  # has to be after the optimizer step

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
            train_accuracy = self.check_accuracy(loader=self.loader["train"])
            val_accuracy = self.check_accuracy(loader=self.loader["val"])
            self.metrics["train_loss"].append(train_loss)
            self.metrics["val_loss"].append(val_loss)
            self.metrics["train_acc"].append(train_accuracy)
            self.metrics["val_acc"].append(val_accuracy)
