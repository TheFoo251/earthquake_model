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
from learner import Learner

# other files
from full_dataset import get_loaders
from torch_utils import plot_loss_curves

cudnn.benchmark = True

NUM_EPOCHS = 30
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PATCH_SZ, BATCH_SZ = 256, 16  # lower batch size?


# --- METRICS ---


model_weights = models.ConvNeXt_Base_Weights.DEFAULT
dataloaders = get_loaders(PATCH_SZ, BATCH_SZ, transforms=transforms.CONVNEXT, split=0.8)


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
scheduler = lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=math.floor(
        len(dataloaders["train"]) / BATCH_SZ
    ),  # updates each batch, each epoch...
)


scaler = torch.cuda.amp.GradScaler()

learner = Learner(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    scaler=scaler,
    scheduler=scheduler,
    device=DEVICE,
)

learner.train_model(num_epochs=NUM_EPOCHS)

plot_loss_curves(learner.metrics)
