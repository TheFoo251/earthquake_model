import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn


# other files
from models import ConvNextSystem
from learner import Learner


# check for CUDA
if not torch.cuda.is_available():
    print("CUDA isn't working!!")
    exit()

NUM_EPOCHS = 30
DEVICE = torch.device("cuda:0")
PATCH_SZ, BATCH_SZ = 256, 16  # lower batch size?
cudnn.benchmark = True


model_system = ConvNextSystem(PATCH_SZ, BATCH_SZ)

lr = 9e-6  # from optimizer study, close to 1e-5

# here's where the transfer magic happens...

optimizer = optim.AdamW(model_system.model.parameters(), lr=lr)


# Decay LR by a factor of 0.1 every 7 epochs
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, "min")

scaler = torch.amp.Gradscaler("cuda")

learner = Learner(
    model_system=model_system,
    optimizer=optimizer,
    scaler=scaler,
    scheduler=scheduler,
    device=DEVICE,
)

learner.train_model(num_epochs=NUM_EPOCHS)

learner.metrics.plot_loss_curves()
