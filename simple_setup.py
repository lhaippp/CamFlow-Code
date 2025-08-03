#!/usr/bin/env python3
"""
Simple setup script for CamFlow environment.
Following KISS, YAGNI and SOLID principles.
"""

import subprocess
import sys

def run_command(cmd):
    """Run command and return success status."""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=True, text=True)
        print("✓ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {e.stderr}")
        return False

def install_minimal_deps():
    """Install minimal dependencies needed to run eval_main.py."""
    
    # Core dependencies that are likely missing
    minimal_deps = [
        "accelerate",
        "timm", 
        "einops",
        "kornia",
        "h5py",
        "huggingface_hub",
        "coloredlogs",
        "termcolor",
        "prettytable"
    ]
    
    print("Installing minimal dependencies...")
    
    for dep in minimal_deps:
        if not run_command(f"pip install {dep}"):
            print(f"Warning: Failed to install {dep}")
    
    # Try to install pyiqa with fallback
    print("Installing pyiqa...")
    if not run_command("pip install pyiqa"):
        print("Warning: pyiqa not available, IQA metrics will be disabled")
        print("You can run without --enable_iqa flag")

def main():
    print("=== CamFlow Simple Setup ===")
    print("Following KISS, YAGNI and SOLID principles")
    print()
    
    # Check if we're in the right directory
    import os
    if not all([os.path.exists(f) for f in ["eval_main.py", "requirements.txt"]]):
        print("Error: Please run this script from the CamFlow-Code directory")
        sys.exit(1)
    
    install_minimal_deps()
    
    print("\n=== Setup Complete ===")
    print("You can now try running:")
    print("python eval_main.py --model_dir experiments/CAHomo/ --restore_file experiments/CAHomo/HEM.pth")
    print("\nOr without IQA metrics:")
    print("python eval_main.py --model_dir experiments/CAHomo/ --restore_file experiments/CAHomo/HEM.pth --enable_iqa")

if __name__ == "__main__":
    main() 