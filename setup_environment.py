#!/usr/bin/env python3
"""
CamFlow Environment Setup Script
KISS, YAGNI, SOLID principles - minimal setup for 2-image motion estimation

Usage:
    python setup_environment.py

This script creates a conda/venv environment and installs all dependencies needed
to run blind_inference.py successfully.
"""

import os
import sys
import subprocess
import logging

def run_command(cmd, description):
    """Run command with error handling"""
    print(f"⚙️  {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False

def main():
    print("🎯 CamFlow Environment Setup")
    print("="*50)
    
    # Check if we're in a virtual environment  
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
        use_pip_directly = True
    else:
        print("⚠️  No virtual environment detected")
        print("Creating new virtual environment...")
        if not run_command("python -m venv camflow_env", "Create virtual environment"):
            return 1
        use_pip_directly = False
    
    # Determine pip command
    if use_pip_directly:
        pip_cmd = "pip"
    else:
        pip_cmd = "camflow_env/bin/pip" if os.name != 'nt' else "camflow_env\\Scripts\\pip"
    
    # Install PyTorch GPU version first
    print("\n🔥 Installing PyTorch (GPU version)...")
    torch_cmd = f"{pip_cmd} install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
    if not run_command(torch_cmd, "Install PyTorch GPU"):
        print("⚠️  GPU installation failed, trying CPU version...")
        torch_cmd = f"{pip_cmd} install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
        if not run_command(torch_cmd, "Install PyTorch CPU"):
            return 1
    
    # Install remaining requirements
    print("\n📦 Installing remaining dependencies...")
    requirements = [
        "opencv-python",
        "Pillow", 
        "imageio",
        "timm",
        "kornia",
        "accelerate", 
        "einops",
        "numpy",
        "h5py",
        "termcolor",
        "coloredlogs",
        "tqdm",
        "PyYAML",
        "huggingface_hub",
        "safetensors",
        "psutil",
        "packaging"
    ]
    
    for req in requirements:
        cmd = f"{pip_cmd} install -i https://pypi.tuna.tsinghua.edu.cn/simple/ {req}"
        if not run_command(cmd, f"Install {req}"):
            print(f"⚠️  Failed to install {req}, continuing...")
    
    # Test the installation
    print("\n🧪 Testing installation...")
    test_cmd = "python blind_inference.py --help" if use_pip_directly else "camflow_env/bin/python blind_inference.py --help"
    if run_command(test_cmd, "Test blind_inference.py"):
        print("\n🎉 Environment setup completed successfully!")
        print("\n📋 Next steps:")
        if not use_pip_directly:
            print("1. Activate environment: source camflow_env/bin/activate")
        print("2. Download data: python download_data.py --minimal")
        print("3. Run inference: python blind_inference.py")
    else:
        print("\n❌ Installation test failed. Please check the output above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())