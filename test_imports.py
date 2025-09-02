#!/usr/bin/env python3
"""
Test script to verify all imports work correctly.
Run this to catch import errors before running the main scripts.
"""

def test_imports():
    """Test all project imports."""
    print("Testing imports...")
    
    try:
        # Test basic dependencies
        import torch
        import torchvision
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.model_selection import train_test_split
        from PIL import Image
        print("✅ Basic dependencies imported successfully")
        
        # Test project modules
        from src.datasets.preprocess import preprocess_raw_dataset
        from src.datasets.loaders import get_dataloaders
        from src.models.cnn import ResNet18Binary
        from src.al.strategies import random_sampling, entropy_sampling
        from src.rl.policy import PolicyNetwork
        from src.utils.logger import get_logger
        from src.utils.metrics import evaluate_classifier
        from src.utils.plots import save_json, plot_curves
        from src.utils.confidence import get_confidence_metrics, calculate_entropy
        from src.utils.evaluation import track_learning_curve, calculate_policy_entropy
        print("✅ All project modules imported successfully")
        
        # Test model instantiation
        model = ResNet18Binary(pretrained=False)  # Don't download weights for test
        policy = PolicyNetwork(in_dim=512, hidden=256)
        print("✅ Models can be instantiated")
        
        print("\n🎉 All imports working correctly!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    if not success:
        print("\n💡 Try running: pip install -r requirements.txt")
        exit(1)