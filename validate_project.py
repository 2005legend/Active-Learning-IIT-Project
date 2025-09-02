#!/usr/bin/env python3
"""
Comprehensive validation script for Policy Gradient Active Learning project.
This script performs all necessary checks before running the main scripts.
"""

import sys
import os
import importlib
from pathlib import Path

class ProjectValidator:
    def __init__(self):
        self.errors = []
        self.warnings = []
    
    def check_python_version(self):
        """Check Python version compatibility."""
        if sys.version_info < (3, 9):
            self.errors.append(f"Python {sys.version_info.major}.{sys.version_info.minor} detected. Python 3.9+ required.")
            return False
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        return True
    
    def check_dependencies(self):
        """Check all required dependencies."""
        dependencies = {
            'torch': 'PyTorch',
            'torchvision': 'TorchVision', 
            'numpy': 'NumPy',
            'sklearn': 'scikit-learn',
            'matplotlib': 'Matplotlib',
            'PIL': 'Pillow',
            'pandas': 'Pandas',
            'pathlib': 'pathlib (built-in)',
            'json': 'json (built-in)',
            'logging': 'logging (built-in)',
            'shutil': 'shutil (built-in)'
        }
        
        missing = []
        for module, name in dependencies.items():
            try:
                importlib.import_module(module)
                print(f"✅ {name}")
            except ImportError:
                print(f"❌ {name} - MISSING")
                missing.append(name)
                self.errors.append(f"Missing dependency: {name}")
        
        return len(missing) == 0
    
    def check_project_structure(self):
        """Validate project file structure."""
        required_structure = {
            'src/': 'Source code directory',
            'src/__init__.py': 'Package init file',
            'src/datasets/': 'Dataset utilities',
            'src/datasets/__init__.py': 'Dataset package init',
            'src/datasets/preprocess.py': 'Data preprocessing',
            'src/datasets/loaders.py': 'Data loaders',
            'src/models/': 'Model definitions',
            'src/models/__init__.py': 'Models package init',
            'src/models/cnn.py': 'CNN model',
            'src/al/': 'Active learning strategies',
            'src/al/__init__.py': 'AL package init',
            'src/al/strategies.py': 'AL strategies',
            'src/rl/': 'Reinforcement learning',
            'src/rl/__init__.py': 'RL package init',
            'src/rl/policy.py': 'Policy network',
            'src/utils/': 'Utility functions',
            'src/utils/__init__.py': 'Utils package init',
            'src/utils/logger.py': 'Logging utilities',
            'src/utils/metrics.py': 'Evaluation metrics',
            'src/utils/plots.py': 'Plotting utilities',
            'src/utils/confidence.py': 'Confidence metrics',
            'src/utils/evaluation.py': 'Evaluation tools',
            'main_week1.py': 'Week 1 main script',
            'main_week2.py': 'Week 2 main script',
            'main_week3.py': 'Week 3 main script',
            'main_week4.py': 'Week 4 main script',
            'requirements.txt': 'Dependencies list',
            'README.md': 'Project documentation',
            'model_card.md': 'Model documentation'
        }
        
        missing = []
        for path, description in required_structure.items():
            if not Path(path).exists():
                print(f"❌ {path} - {description}")
                missing.append(path)
                self.errors.append(f"Missing file: {path}")
            else:
                print(f"✅ {path}")
        
        return len(missing) == 0
    
    def check_imports(self):
        """Test all project imports."""
        try:
            # Test project modules
            from src.datasets.preprocess import preprocess_raw_dataset
            from src.datasets.loaders import get_dataloaders
            from src.models.cnn import ResNet18Binary
            from src.al.strategies import random_sampling, entropy_sampling
            from src.rl.policy import PolicyNetwork
            from src.utils.logger import get_logger
            from src.utils.metrics import evaluate_classifier
            from src.utils.plots import save_json, plot_curves
            from src.utils.confidence import get_confidence_metrics
            from src.utils.evaluation import track_learning_curve
            
            print("✅ All project imports successful")
            return True
            
        except ImportError as e:
            print(f"❌ Import error: {e}")
            self.errors.append(f"Import error: {e}")
            return False
    
    def check_model_instantiation(self):
        """Test model creation."""
        try:
            from src.models.cnn import ResNet18Binary
            from src.rl.policy import PolicyNetwork
            
            # Test without downloading pretrained weights
            model = ResNet18Binary(pretrained=False)
            policy = PolicyNetwork(in_dim=512, hidden=256)
            
            print("✅ Models can be instantiated")
            return True
            
        except Exception as e:
            print(f"❌ Model instantiation error: {e}")
            self.errors.append(f"Model error: {e}")
            return False
    
    def check_dataset_configuration(self):
        """Check dataset path configuration."""
        try:
            with open('main_week1.py', 'r') as f:
                content = f.read()
                if 'C:\\\\Users\\\\sidaa\\\\IIT Project\\\\data\\\\dogs-vs-cats\\\\train' in content:
                    print("⚠️  Using example dataset path")
                    self.warnings.append("Dataset path needs to be configured in main_week1.py")
                    return False
                else:
                    print("✅ Dataset path appears customized")
                    return True
        except Exception as e:
            print(f"❌ Error checking dataset config: {e}")
            self.errors.append(f"Dataset config error: {e}")
            return False
    
    def check_directories(self):
        """Check if output directories exist or can be created."""
        required_dirs = [
            'data/processed',
            'logs',
            'checkpoints',
            'outputs'
        ]
        
        for directory in required_dirs:
            try:
                Path(directory).mkdir(parents=True, exist_ok=True)
                print(f"✅ Directory: {directory}")
            except Exception as e:
                print(f"❌ Cannot create directory {directory}: {e}")
                self.errors.append(f"Directory error: {directory}")
                return False
        
        return True
    
    def run_validation(self):
        """Run complete validation."""
        print("🔍 POLICY GRADIENT ACTIVE LEARNING - PROJECT VALIDATION")
        print("=" * 60)
        
        checks = [
            ("Python Version", self.check_python_version),
            ("Dependencies", self.check_dependencies),
            ("Project Structure", self.check_project_structure),
            ("Import Tests", self.check_imports),
            ("Model Tests", self.check_model_instantiation),
            ("Dataset Config", self.check_dataset_configuration),
            ("Directories", self.check_directories)
        ]
        
        results = {}
        for name, check_func in checks:
            print(f"\n📋 {name}:")
            print("-" * 30)
            results[name] = check_func()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 VALIDATION SUMMARY")
        print("=" * 60)
        
        passed = sum(results.values())
        total = len(results)
        
        if self.errors:
            print("❌ ERRORS:")
            for error in self.errors:
                print(f"   • {error}")
            print()
        
        if self.warnings:
            print("⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"   • {warning}")
            print()
        
        print(f"✅ Passed: {passed}/{total} checks")
        
        if self.errors:
            print("\n❌ VALIDATION FAILED")
            print("Fix the errors above before running the project.")
            return False
        else:
            print("\n🎉 VALIDATION SUCCESSFUL!")
            print("\n🚀 Ready to run:")
            print("   python main_week1.py")
            print("   python main_week2.py")
            print("   python main_week3.py") 
            print("   python main_week4.py")
            
            if self.warnings:
                print("\n💡 Note: Address warnings for optimal experience.")
            
            return True

def main():
    validator = ProjectValidator()
    success = validator.run_validation()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()