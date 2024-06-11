import torch
import torchvision
from src.torch_dataset import EarthquakeDataset
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

# TODO -- research tuning models from a checkpoint
def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    
def get_loaders(
    img_dir,
    mask_dir,
    batch_size,
    num_workers=4,
    pin_memory=True,
    transforms=None
):
    full_ds = EarthquakeDataset(
        image_dir=img_dir,
        mask_dir=mask_dir,
        transform=transforms
    )
    num_train = int(len(full_ds) * 0.9)
    train_ds, val_ds = torch.utils.data.random_split(
                                        full_ds,
                                        [num_train, len(full_ds) - num_train],
                                        )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers, #TODO look up 'workers'
        pin_memory=pin_memory, #TODO look up pin memory
        shuffle=True,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )
    
    return train_loader, val_loader
    
def check_accuracy(loader, model, device="cuda"):
    """
    simplistic accuracy checker
    """
    
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (pred*y).sum()) / (
                (preds + y).sum() + 1e-8
            )
            
    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()
    
def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            torchvision.utils.save_image(
                preds, f"{folder}/{idx}_pred.png"
            )
            torchvision.utils.save_image(
                y.unsqueeze(1),
                f"{folder}/{idx}_real.png"
            )
        
    model.train()
    
    
    
    
        