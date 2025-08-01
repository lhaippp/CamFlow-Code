#!/usr/bin/env python3
"""
Script to download GHOF-Cam benchmark and pre-trained weights from Hugging Face.

This script downloads the dataset and model weights from:
https://huggingface.co/datasets/Lhaippp/CamFlow-ICCV25/tree/main
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm

try:
    from huggingface_hub import snapshot_download, hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("Warning: huggingface_hub not found. Please install it with: pip install huggingface_hub")

def download_with_hf_hub(repo_id, local_dir, repo_type="dataset", specific_files=None):
    """Download using huggingface_hub library."""
    if not HF_HUB_AVAILABLE:
        raise ImportError("huggingface_hub is required. Install with: pip install huggingface_hub")
    
    print(f"Downloading from {repo_id} to {local_dir}")
    
    if specific_files:
        # Download specific files
        for file_path in specific_files:
            print(f"Downloading {file_path}...")
            hf_hub_download(
                repo_id=repo_id,
                filename=file_path,
                local_dir=local_dir,
                repo_type=repo_type
            )
    else:
        # Download entire repository
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            repo_type=repo_type
        )
    
    print(f"Download completed to: {local_dir}")

def download_with_git():
    """Fallback method using git clone."""
    repo_url = "https://huggingface.co/datasets/Lhaippp/CamFlow-ICCV25"
    local_dir = "data/CamFlow-ICCV25"
    
    print(f"Downloading using git clone from {repo_url}")
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Clone the repository
    cmd = f"git clone {repo_url} {local_dir}"
    print(f"Running: {cmd}")
    result = os.system(cmd)
    
    if result == 0:
        print(f"Download completed to: {local_dir}")
    else:
        print("Git clone failed. Please check your internet connection and git installation.")
        return False
    
    return True

def setup_directories():
    """Create necessary directories for data and experiments."""
    directories = [
        "data",
        "data/GHOF-Cam",
        "experiments",
        "experiments/CAHomo"
    ]
    
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

def main():
    parser = argparse.ArgumentParser(description="Download CamFlow data from Hugging Face")
    parser.add_argument("--method", choices=["hf_hub", "git"], default="hf_hub",
                      help="Download method (default: hf_hub)")
    parser.add_argument("--data_dir", default="data", 
                      help="Directory to store downloaded data (default: data)")
    parser.add_argument("--specific_files", nargs="+", 
                      help="Download only specific files (optional)")
    parser.add_argument("--setup_dirs", action="store_true",
                      help="Setup directory structure")
    
    args = parser.parse_args()
    
    # Setup directories if requested
    if args.setup_dirs:
        setup_directories()
    
    # Create data directory
    os.makedirs(args.data_dir, exist_ok=True)
    
    repo_id = "Lhaippp/CamFlow-ICCV25"
    local_dir = os.path.join(args.data_dir, "CamFlow-ICCV25")
    
    try:
        if args.method == "hf_hub":
            download_with_hf_hub(repo_id, local_dir, specific_files=args.specific_files)
        elif args.method == "git":
            download_with_git()
        
        print("\n" + "="*50)
        print("Download completed successfully!")
        print("="*50)
        print(f"Data location: {local_dir}")
        print("\nNext steps:")
        print("1. Check the downloaded data structure")
        print("2. Copy pre-trained weights to experiments/CAHomo/ if needed")
        print("3. Run evaluation: python eval_main.py --model_dir experiments/CAHomo/ --restore_file experiments/CAHomo/HEM.pth")
        
    except Exception as e:
        print(f"Error during download: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have internet connection")
        print("2. Install huggingface_hub: pip install huggingface_hub")
        print("3. Try git method: python download_data.py --method git")
        print("4. Check if you need Hugging Face authentication for this dataset")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 