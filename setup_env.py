#!/usr/bin/env python3
"""
Setup script for CamFlow environment.
Creates virtual environment and installs minimal dependencies.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, check=True):
    """Run command and handle errors."""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=check, 
                              capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False

def setup_environment(env_name="camflow_env", use_conda=False):
    """Setup virtual environment and install dependencies."""
    
    print(f"Setting up CamFlow environment: {env_name}")
    
    # Create virtual environment
    if use_conda:
        if not run_command(f"conda create -n {env_name} python=3.8 -y"):
            print("Failed to create conda environment")
            return False
        activate_cmd = f"conda activate {env_name}"
        pip_cmd = f"conda run -n {env_name} pip"
    else:
        if not run_command(f"python3 -m venv {env_name}"):
            print("Failed to create virtual environment")
            return False
        activate_cmd = f"source {env_name}/bin/activate"
        pip_cmd = f"{env_name}/bin/pip"
    
    print(f"Virtual environment created: {env_name}")
    
    # Upgrade pip
    if not run_command(f"{pip_cmd} install --upgrade pip"):
        print("Warning: Failed to upgrade pip")
    
    # Install requirements
    if not run_command(f"{pip_cmd} install -r requirements.txt"):
        print("Failed to install requirements")
        print("Trying with alternative PyTorch installation...")
        
        # Try installing core packages individually
        core_packages = [
            "numpy>=1.19.0",
            "opencv-python>=4.5.0", 
            "tqdm>=4.62.0",
            "imageio>=2.9.0",
            "matplotlib>=3.3.0"
        ]
        
        for package in core_packages:
            if not run_command(f"{pip_cmd} install '{package}'"):
                print(f"Warning: Failed to install {package}")
        
        return False
    
    print(f"\n✅ Setup complete!")
    print(f"To activate environment:")
    if use_conda:
        print(f"  conda activate {env_name}")
    else:
        print(f"  source {env_name}/bin/activate")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Setup CamFlow environment")
    parser.add_argument("--name", default="camflow_env", 
                       help="Environment name (default: camflow_env)")
    parser.add_argument("--conda", action="store_true",
                       help="Use conda instead of venv")
    
    args = parser.parse_args()
    
    if not Path("requirements.txt").exists():
        print("Error: requirements.txt not found!")
        sys.exit(1)
    
    success = setup_environment(args.name, args.conda)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 