import torch
import numpy as np
from pathlib import Path
import json

def track_learning_curve(accuracies: list, labeled_counts: list, method_name: str, output_dir: Path):
    """Track and save learning curve data."""
    curve_data = {
        'method': method_name,
        'accuracies': accuracies,
        'labeled_counts': labeled_counts,
        'samples_to_90_percent': None
    }
    
    # Find samples needed to reach 90% accuracy
    for i, acc in enumerate(accuracies):
        if acc >= 0.90:
            curve_data['samples_to_90_percent'] = labeled_counts[i]
            break
    
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"{method_name.lower()}_curve.json", "w") as f:
        json.dump(curve_data, f, indent=2)
    
    return curve_data

def calculate_policy_entropy(policy_probs: torch.Tensor) -> float:
    """Calculate entropy of policy distribution to measure convergence."""
    eps = 1e-8
    entropy = -(policy_probs * torch.log(policy_probs + eps)).sum().item()
    return entropy

def evaluate_sample_efficiency(curves: dict, target_accuracy: float = 0.90):
    """Compare sample efficiency across different methods."""
    efficiency_report = {}
    
    for method, data in curves.items():
        accuracies = data['accuracies']
        labeled_counts = data['labeled_counts']
        
        # Find samples needed to reach target accuracy
        samples_needed = None
        for i, acc in enumerate(accuracies):
            if acc >= target_accuracy:
                samples_needed = labeled_counts[i]
                break
        
        efficiency_report[method] = {
            'samples_to_target': samples_needed,
            'final_accuracy': accuracies[-1] if accuracies else 0,
            'improvement_rate': (accuracies[-1] - accuracies[0]) / len(accuracies) if len(accuracies) > 1 else 0
        }
    
    return efficiency_report