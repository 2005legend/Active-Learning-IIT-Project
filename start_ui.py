#!/usr/bin/env python3
"""
Startup script for Animal Classification AI UI
Checks dependencies and launches Streamlit application
"""

import sys
import subprocess
import importlib
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = {
        'streamlit': 'streamlit',
        'plotly': 'plotly',
        'PIL': 'pillow',
        'requests': 'requests',
        'torch': 'torch',
        'torchvision': 'torchvision'
    }
    
    missing_packages = []
    
    for package, pip_name in required_packages.items():
        try:
            if package == 'PIL':
                importlib.import_module('PIL')
            else:
                importlib.import_module(package)
            print(f"✅ {package} - OK")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(pip_name)
    
    return missing_packages

def install_packages(packages):
    """Install missing packages"""
    if not packages:
        return True
    
    print(f"\n📦 Installing missing packages: {', '.join(packages)}")
    
    try:
        for package in packages:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def check_project_structure():
    """Check if project structure is correct"""
    required_files = [
        "src/models/cnn.py",
        "src/services/llm_service.py",
        "app.py",
        "config/llm_config.yaml"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
            print(f"❌ {file_path} - Missing")
        else:
            print(f"✅ {file_path} - OK")
    
    return missing_files

def check_model_checkpoints():
    """Check if trained models are available"""
    checkpoint_dirs = [
        "checkpoints/week2",
        "checkpoints/week3", 
        "checkpoints/week4"
    ]
    
    available_models = []
    
    for checkpoint_dir in checkpoint_dirs:
        if Path(checkpoint_dir).exists():
            checkpoint_files = list(Path(checkpoint_dir).glob("*.pth"))
            if checkpoint_files:
                available_models.append(f"{checkpoint_dir}: {len(checkpoint_files)} models")
                print(f"✅ {checkpoint_dir} - {len(checkpoint_files)} models")
            else:
                print(f"⚠️ {checkpoint_dir} - Directory exists but no models")
        else:
            print(f"❌ {checkpoint_dir} - Missing")
    
    return available_models

def launch_streamlit():
    """Launch Streamlit application"""
    print("\n🚀 Launching Animal Classification AI UI...")
    
    try:
        # Check if app.py exists
        if not Path("app.py").exists():
            print("❌ app.py not found. Please ensure you're in the correct directory.")
            return False
        
        # Launch Streamlit
        print("🌐 Opening Streamlit application...")
        print("📱 The UI will open in your default web browser")
        print("🔗 If it doesn't open automatically, go to: http://localhost:8501")
        print("\n💡 Press Ctrl+C to stop the application")
        
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
        return True
        
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
        return True
    except Exception as e:
        print(f"❌ Error launching Streamlit: {e}")
        return False

def main():
    """Main startup function"""
    print("🐾 Animal Classification AI - UI Startup")
    print("=" * 50)
    
    # Check project structure
    print("\n📁 Checking project structure...")
    missing_files = check_project_structure()
    
    if missing_files:
        print(f"\n❌ Missing required files: {', '.join(missing_files)}")
        print("Please ensure you're in the correct project directory.")
        return False
    
    # Check dependencies
    print("\n🔍 Checking dependencies...")
    missing_packages = check_dependencies()
    
    if missing_packages:
        print(f"\n📦 Missing packages detected: {', '.join(missing_packages)}")
        
        install_choice = input("Would you like to install missing packages? (y/n): ").lower().strip()
        
        if install_choice in ['y', 'yes']:
            if not install_packages(missing_packages):
                print("❌ Failed to install packages. Please install manually:")
                print(f"pip install {' '.join(missing_packages)}")
                return False
        else:
            print("❌ Cannot proceed without required packages.")
            return False
    
    # Check model checkpoints
    print("\n🤖 Checking model availability...")
    available_models = check_model_checkpoints()
    
    if not available_models:
        print("\n⚠️ Warning: No trained models found.")
        print("The UI will work but classification features may not function.")
        print("Consider running the training scripts first:")
        print("  python main_week2.py  # Train baseline model")
        print("  python main_week3.py  # Train active learning baselines")
        print("  python main_week4.py  # Train REINFORCE policy")
    
    # Launch application
    print("\n" + "=" * 50)
    return launch_streamlit()

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1) 