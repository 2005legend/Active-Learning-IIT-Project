#!/usr/bin/env python3
"""
Installation script for Policy Gradient Active Learning project.
This script will install all required dependencies.
"""

import subprocess
import sys
import os

def run_command(command):
    """Run a command and return success status."""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def install_requirements():
    """Install requirements from requirements.txt."""
    print("🔧 Installing Python dependencies...")
    
    # Check if requirements.txt exists
    if not os.path.exists('requirements.txt'):
        print("❌ requirements.txt not found!")
        return False
    
    # Install requirements
    success, output = run_command(f"{sys.executable} -m pip install -r requirements.txt")
    
    if success:
        print("✅ Dependencies installed successfully!")
        return True
    else:
        print(f"❌ Installation failed: {output}")
        return False

def install_pytorch():
    """Install PyTorch with appropriate configuration."""
    print("🔧 Installing PyTorch...")
    
    # Check if CUDA is available
    try:
        import torch
        if torch.cuda.is_available():
            print("✅ PyTorch with CUDA already installed")
            return True
    except ImportError:
        pass
    
    # Install CPU version (safer for compatibility)
    print("Installing PyTorch CPU version...")
    success, output = run_command(f"{sys.executable} -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
    
    if success:
        print("✅ PyTorch installed successfully!")
        return True
    else:
        print(f"❌ PyTorch installation failed: {output}")
        return False

def create_directories():
    """Create necessary project directories."""
    print("📁 Creating project directories...")
    
    directories = [
        'data/processed',
        'logs',
        'checkpoints/week1',
        'checkpoints/week2', 
        'checkpoints/week3',
        'checkpoints/week4',
        'outputs/week1',
        'outputs/week2/plots',
        'outputs/week3/plots',
        'outputs/week4/plots',
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created: {directory}")
    
    return True

def main():
    """Main installation process."""
    print("🚀 Setting up Policy Gradient Active Learning Project...\n")
    
    # Create directories
    if not create_directories():
        print("❌ Failed to create directories")
        sys.exit(1)
    
    print()
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements")
        sys.exit(1)
    
    print()
    
    # Test installation
    print("🧪 Testing installation...")
    try:
        import torch
        import torchvision
        import numpy
        import sklearn
        import matplotlib
        import PIL
        print("✅ All packages imported successfully!")
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        sys.exit(1)
    
    print("\n🎉 Installation completed successfully!")
    print("\n📝 Next steps:")
    print("1. Edit RAW_DATASET_DIR in main_week1.py to point to your dataset")
    print("2. Run: python check_setup.py")
    print("3. Run: python test_imports.py")
    print("4. Start with: python main_week1.py")

if __name__ == "__main__":
    main()