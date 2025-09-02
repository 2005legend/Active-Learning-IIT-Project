import torch
import torch.nn.functional as F

def calculate_entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Calculate entropy for uncertainty measurement."""
    return -(probs * torch.log(probs + eps)).sum(dim=1)

def calculate_margin(logits: torch.Tensor) -> torch.Tensor:
    """Calculate margin between top two predictions."""
    sorted_logits, _ = torch.sort(logits, dim=1, descending=True)
    return sorted_logits[:, 0] - sorted_logits[:, 1]

def get_confidence_metrics(model, dataloader, device):
    """Extract confidence metrics (entropy, margin) for policy state."""
    model.eval()
    entropies, margins, features = [], [], []
    
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            
            # Calculate confidence metrics
            entropy = calculate_entropy(probs)
            margin = calculate_margin(logits)
            
            # Extract features for policy
            feats = model.extract_features(x)
            
            entropies.append(entropy.cpu())
            margins.append(margin.cpu())
            features.append(feats.cpu())
    
    return {
        'entropy': torch.cat(entropies),
        'margin': torch.cat(margins), 
        'features': torch.cat(features)
    }