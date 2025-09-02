from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

def get_dataloaders(processed_root: str | Path, img_size: int = 128, batch_size: int = 64):
    processed_root = Path(processed_root)
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tfm = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    train_ds = datasets.ImageFolder(processed_root / "train", transform=tfm)
    val_ds = datasets.ImageFolder(processed_root / "val", transform=tfm)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, val_loader, train_ds, val_ds