# CamFlow Development Guide

## Overview
CamFlow is a camera motion flow estimation research project following KISS, YAGNI, and SOLID principles.

## Quick Start

### Environment Setup
```bash
# Simple setup (minimal dependencies)
python simple_setup.py

# Full setup (all dependencies)  
python setup_env.py
```

### Data Download
```bash
python download_data.py
```

### Evaluation
```bash
# Basic evaluation
python eval_main.py --model_dir data/CamFlow-ICCV25 --restore_file data/CamFlow-ICCV25/ckpt.pth

# With IQA metrics
python eval_main.py --model_dir data/CamFlow-ICCV25 --restore_file data/CamFlow-ICCV25/ckpt.pth --enable_iqa
```

## Code Organization

### Core Components
- `eval_main.py` - Main evaluation entry point
- `evaluators.py` - GHOF and IQA evaluation classes
- `common/manager.py` - Model and training management
- `common/utils.py` - Shared utilities
- `model/` - Neural network models
- `dataset/` - Data loading
- `loss/` - Loss functions

### Evaluation Pipeline
1. **GHOFEvaluator**: Computes EPE and PME metrics across 5 weather categories
2. **IQAEvaluator**: Computes PSNR, SSIM, LPIPS image quality metrics