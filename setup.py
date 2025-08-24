#!/usr/bin/env python3
"""
Setup script for Stable Diffusion ControlNet Black & White to Color Pipeline
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}:")
        print(f"Command: {command}")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def check_gpu():
    """Check if CUDA GPU is available."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🔥 CUDA GPU detected: {gpu_name}")
            return True
        else:
            print("💻 No CUDA GPU detected, will use CPU (slower)")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed yet, will check GPU after installation")
        return None

def create_directories():
    """Create necessary directories."""
    directories = [
        "input_images",
        "output_images", 
        "model_cache",
        "examples"
    ]
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created directory: {directory}")
        else:
            print(f"📁 Directory already exists: {directory}")

def install_dependencies():
    """Install Python dependencies."""
    if not run_command("pip install --upgrade pip", "Upgrading pip"):
        return False
    
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        return False
    
    return True

def verify_installation():
    """Verify that the installation was successful."""
    print("\n🔍 Verifying installation...")
    
    try:
        import torch
        import diffusers
        import transformers
        import controlnet_aux
        import gradio
        
        print("✅ All core dependencies installed successfully")
        
        # Check GPU again after PyTorch installation
        if torch.cuda.is_available():
            print(f"🔥 CUDA GPU available: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("💻 Running on CPU (GPU acceleration not available)")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Main setup function."""
    print("🎨 Stable Diffusion ControlNet Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check GPU availability (before PyTorch installation)
    check_gpu()
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    if not install_dependencies():
        print("\n❌ Installation failed. Please check the errors above.")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("\n❌ Installation verification failed.")
        sys.exit(1)
    
    # Success message
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\n🚀 Next steps:")
    print("1. Place black & white images in the 'input_images' folder")
    print("2. Run the web interface: python run.py --web")
    print("3. Or process images directly: python run.py --single your_image.jpg")
    print("4. Check examples: python run.py --examples")
    print("\n📖 For more information, see README.md")
    print("💡 Need help? Check the troubleshooting section in README.md")

if __name__ == "__main__":
    main()
