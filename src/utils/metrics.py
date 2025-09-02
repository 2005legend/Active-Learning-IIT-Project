import torch
from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate_classifier(model, dataloader, device):
    model.eval()
    all_y, all_p = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu()
            all_p.extend(preds.tolist())
            all_y.extend(y.tolist())
    
    acc = accuracy_score(all_y, all_p)
    cm = confusion_matrix(all_y, all_p)
    return acc, cm