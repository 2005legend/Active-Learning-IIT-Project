from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
import numpy as np
from src.utils.logger import get_logger
from src.datasets.loaders import get_dataloaders
from src.models.cnn import ResNet18Binary
from src.utils.metrics import evaluate_classifier
from src.al.strategies import random_sampling, entropy_sampling
from src.utils.plots import plot_curves, save_json

PROCESSED_DIR = Path("data/processed/catsdogs_128")
CHECK_DIR = Path("checkpoints/week3")
OUT_DIR = Path("outputs/week3")
LOG_FILE = Path("logs/week3_active_learning.log")

INIT_LABELED = 1000
ACQ_SIZE = 500
ROUNDS = 8  # total new labels = 1000 + 8*500 = 5000 labeled
EPOCHS_PER_ROUND = 3
BATCH = 64
LR = 1e-3
SEED = 42

if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = get_logger("week3", LOG_FILE)
    logger.info("Week 3: Evaluation & Comparisons - Active Learning baselines started")
    
    train_loader_full, val_loader, train_ds, val_ds = get_dataloaders(PROCESSED_DIR, img_size=128, batch_size=BATCH)
    N = len(train_ds)
    all_idx = np.arange(N)
    rng.shuffle(all_idx)
    
    def run_strategy(name: str, selector_fn):
        labeled = set(all_idx[:INIT_LABELED].tolist())
        unlabeled = set(all_idx[INIT_LABELED:].tolist())
        val_curve = []
        
        for rd in range(ROUNDS + 1):  # round 0 = initial
            # Create loaders over labeled subset
            labeled_list = sorted(list(labeled))
            train_subset = Subset(train_ds, labeled_list)
            train_loader = torch.utils.data.DataLoader(train_subset, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)
            
            # Fresh model each round (typical AL retrain)
            model = ResNet18Binary(pretrained=True).to(device)
            crit = nn.CrossEntropyLoss()
            opt = optim.Adam(model.parameters(), lr=LR)
            
            # Train few epochs
            for ep in range(EPOCHS_PER_ROUND):
                model.train()
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad()
                    loss = crit(model(x), y)
                    loss.backward()
                    opt.step()
            
            # Eval
            acc, _ = evaluate_classifier(model, val_loader, device)
            val_curve.append(acc)
            logger.info(f"{name} Round {rd}/{ROUNDS} - Labeled:{len(labeled)} - ValAcc:{acc:.4f}")
            
            torch.save(model.state_dict(), CHECK_DIR / f"{name.lower()}_round{rd}.pth")
            
            if rd == ROUNDS:
                break
            
            # Select next ACQ_SIZE from unlabeled
            unlabeled_list = sorted(list(unlabeled))
            
            # Build temporary loader for unlabeled to score uncertainty
            unl_loader = torch.utils.data.DataLoader(Subset(train_ds, unlabeled_list), batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)
            
            model.eval()
            probs, indices = [], []
            with torch.no_grad():
                for (x, _), idx_batch in zip(unl_loader, np.array_split(unlabeled_list, len(unl_loader))):
                    x = x.to(device)
                    p = torch.softmax(model(x), dim=1).cpu()
                    probs.append(p)
                    indices.append(torch.tensor(idx_batch))
            
            probs = torch.cat(probs, dim=0)
            indices = torch.cat(indices, dim=0)
            
            if name == "Random":
                pick = random_sampling(indices, ACQ_SIZE)
            else:
                pick = entropy_sampling(probs, indices, ACQ_SIZE)
            
            new_idx = set(pick.tolist())
            labeled |= new_idx
            unlabeled -= new_idx
        
        return val_curve
    
    (CHECK_DIR).mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "plots").mkdir(parents=True, exist_ok=True)
    
    curves = {}
    curves["Random"] = run_strategy("Random", random_sampling)
    curves["Uncertainty"] = run_strategy("Uncertainty", entropy_sampling)
    
    save_json(curves, OUT_DIR / "curves.json")
    plot_curves(curves, "Round", "Val Accuracy", "Week3: Baselines Comparison", OUT_DIR / "plots/baselines.png")
    
    logger.info(f"Saved checkpoints to {CHECK_DIR} and outputs to {OUT_DIR}")