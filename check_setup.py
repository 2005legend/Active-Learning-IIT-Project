#!/usr/bin/env python3
"""
Setup checker for Policy Gradient Active Learning project.
Run this before executing the main scripts to ensure all dependencies are available.
"""

import sys
import importlib
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 9):
        print(f"❌ Python {sys.version_info.major}.{sys.version_info.minor} detected. Python 3.9+ required.")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} - OK")
    return True

def check_dependencies():
    """Check if all required packages are installed."""
    required_packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('numpy', 'NumPy'),
        ('sklearn', 'scikit-learn'),
        ('matplotlib', 'Matplotlib'),
        ('PIL', 'Pillow'),
        ('pandas', 'Pandas'),
    ]
    
    missing = []
    for package, name in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {name} - OK")
        except ImportError:
            print(f"❌ {name} - MISSING")
            missing.append(name)
    
    return missing

def check_project_structure():
    """Check if all required project files exist."""
    required_files = [
        'src/datasets/preprocess.py',
        'src/datasets/loaders.py',
        'src/models/cnn.py',
        'src/al/strategies.py',
        'src/rl/policy.py',
        'src/utils/logger.py',
        'src/utils/metrics.py',
        'src/utils/plots.py',
        'src/utils/confidence.py',
        'src/utils/evaluation.py',
        'main_week1.py',
        'main_week2.py',
        'main_week3.py',
        'main_week4.py',
    ]
    
    missing = []
    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"❌ {file_path} - MISSING")
            missing.append(file_path)
        else:
            print(f"✅ {file_path} - OK")
    
    return missing

def check_dataset_path():
    """Check if dataset path is configured."""
    try:
        with open('main_week1.py', 'r') as f:
            content = f.read()
            if 'C:\\\\Users\\\\sidaa\\\\IIT Project\\\\data\\\\dogs-vs-cats\\\\train' in content:
                print("⚠️  Dataset path still uses default example path.")
                print("   Please edit RAW_DATASET_DIR in main_week1.py to point to your actual dataset.")
                return False
            else:
                print("✅ Dataset path appears to be customized")
                return True
    except Exception as e:
        print(f"❌ Error checking dataset path: {e}")
        return False

def main():
    print("🔍 Checking Policy Gradient Active Learning Project Setup...\n")
    
    # Check Python version
    print("1. Python Version:")
    python_ok = check_python_version()
    print()
    
    # Check dependencies
    print("2. Dependencies:")
    missing_deps = check_dependencies()
    print()
    
    # Check project structure
    print("3. Project Structure:")
    missing_files = check_project_structure()
    print()
    
    # Check dataset configuration
    print("4. Dataset Configuration:")
    dataset_ok = check_dataset_path()
    print()
    
    # Summary
    print("📋 SETUP SUMMARY:")
    print("=" * 50)
    
    if not python_ok:
        print("❌ Python version incompatible")
        sys.exit(1)
    
    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        print("   Install with: pip install -r requirements.txt")
        sys.exit(1)
    
    if missing_files:
        print(f"❌ Missing project files: {len(missing_files)} files")
        sys.exit(1)
    
    if not dataset_ok:
        print("⚠️  Dataset path needs configuration")
        print("   Edit RAW_DATASET_DIR in main_week1.py before running")
    
    print("✅ Setup check completed successfully!")
    print("\n🚀 Ready to run:")
    print("   python main_week1.py")
    print("   python main_week2.py") 
    print("   python main_week3.py")
    print("   python main_week4.py")

if __name__ == "__main__":
    main()