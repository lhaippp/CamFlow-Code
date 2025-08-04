# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CamFlow is a computer vision research project for estimating 2D camera motion with hybrid motion basis. It implements neural networks using PyTorch for homography estimation and optical flow computation. The project follows KISS, YAGNI, and SOLID principles for maintainable code.

## Environment Setup and Dependencies

### Data Download
```bash
python download_data.py
```

## Key Commands

### Evaluation (Main Usage)
```bash
# Basic evaluation
python eval_main.py --model_dir data/CamFlow-ICCV25 --restore_file data/CamFlow-ICCV25/ckpt.pth

# With IQA metrics (requires pyiqa)
python eval_main.py --model_dir data/CamFlow-ICCV25 --restore_file data/CamFlow-ICCV25/ckpt.pth --enable_iqa

### Dependency Management
```bash
# Install requirements
pip install -r requirements.txt

# If torch installation fails, run simple_setup.py which installs core packages individually
```

## Code Architecture

### Core Components

- **eval_main.py**: Main evaluation entry point with argument parsing and environment setup
- **model/net.py**: Core neural network implementation with Swin Transformer backbone
- **model/swin_multi.py**: Swin Transformer architecture for multi-scale feature extraction
- **model/DLT.py**: Direct Linear Transform solver for homography estimation
- **dataset/data_loader.py**: Data loading pipeline with homography and flow operations
- **evaluators.py**: Evaluation framework with GHOFEvaluator and IQAEvaluator classes

### Key Modules

- **common/**: Shared utilities including manager, utils, and flow operations
- **model/**: Neural network architectures and computer vision utilities
- **loss/**: Loss function implementations
- **utils_operations/**: Specialized operations for flow and homography computation
- **dataset/**: Data loading and preprocessing

### Architecture Pattern

The codebase follows a modular design with clear separation of concerns:
- **Manager pattern**: `common/manager.py` handles training state and metrics
- **Evaluator pattern**: `evaluators.py` implements different evaluation strategies (GHOF, IQA)
- **Factory pattern**: `model/net.py` uses `fetch_net()` for model instantiation
- **Data pipeline**: `dataset/data_loader.py` provides `fetch_dataloader()` for data loading

### Dependencies and Integrations

- **PyTorch ecosystem**: torch, torchvision, timm, kornia
- **Computer vision**: opencv-python, imageio
- **Acceleration**: accelerate library for distributed training
- **Optional IQA**: pyiqa for image quality assessment (graceful degradation if missing)
- **Utilities**: einops, h5py, tqdm, coloredlogs

### Important Implementation Details

- Uses Accelerate library for distributed training and mixed precision
- Implements custom tensor erosion/dilation operations in model/net.py
- Graceful handling of optional dependencies (pyiqa for IQA metrics)
- Follows defensive programming with proper error handling in setup scripts
- Swin Transformer backbone for multi-scale feature extraction
- DLT solver for robust homography estimation

### Configuration

- Model parameters stored in JSON format (params.json in model directories)
- Command-line argument parsing with sensible defaults
- Support for both conda and venv environments
- Modular setup scripts for different deployment scenarios