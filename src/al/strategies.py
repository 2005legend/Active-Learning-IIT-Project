import torch

def random_sampling(unlabeled_indices: torch.Tensor, k: int) -> torch.Tensor:
    perm = torch.randperm(len(unlabeled_indices))
    return unlabeled_indices[perm[:k]]

def entropy_sampling(probs: torch.Tensor, unlabeled_indices: torch.Tensor, k: int) -> torch.Tensor:
    # probs: (N,2) softmax probabilities for unlabeled set
    eps = 1e-8
    ent = -(probs * (probs + eps).log()).sum(dim=1)  # (N,)
    topk = torch.topk(ent, k=k).indices
    return unlabeled_indices[topk]