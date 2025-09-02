from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from src.utils.logger import get_logger
from src.datasets.loaders import get_dataloaders
from src.models.cnn import ResNet18Binary
from src.utils.metrics import evaluate_classifier
from src.utils.plots import plot_curves, save_json

PROCESSED_DIR = Path("data/processed/catsdogs_128")
CHECK_DIR = Path("checkpoints/week2")
OUT_DIR = Path("outputs/week2")
LOG_FILE = Path("logs/week2_train.log")

EPOCHS = 5
BATCH = 64
LR = 1e-3

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = get_logger("week2", LOG_FILE)
    logger.info("Week 2: RL Agent (REINFORCE) Development - Baseline training started")
    
    train_loader, val_loader, train_ds, val_ds = get_dataloaders(PROCESSED_DIR, img_size=128, batch_size=BATCH)
    logger.info(f"Train images: {len(train_ds)}, Val images: {len(val_ds)}, Device: {device}")
    
    model = ResNet18Binary(pretrained=True).to(device)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=LR)
    
    train_losses, val_accs = [], []
    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    
    for ep in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        
        ep_loss = running / len(train_ds)
        acc, cm = evaluate_classifier(model, val_loader, device)
        logger.info(f"Epoch {ep}/{EPOCHS} - Loss: {ep_loss:.4f} - ValAcc: {acc:.4f}")
        
        torch.save(model.state_dict(), CHECK_DIR / f"resnet18_epoch{ep}.pth")
        train_losses.append(ep_loss)
        val_accs.append(acc)
    
    save_json({"train_loss": train_losses, "val_acc": val_accs}, OUT_DIR / "metrics.json")
    plot_curves({"ValAcc": val_accs}, "Epoch", "Accuracy", "Week2: Baseline Accuracy", OUT_DIR / "plots/val_acc.png")
    plot_curves({"TrainLoss": train_losses}, "Epoch", "Loss", "Week2: Training Loss", OUT_DIR / "plots/train_loss.png")
    
    logger.info(f"Saved checkpoints to {CHECK_DIR} and outputs to {OUT_DIR}")