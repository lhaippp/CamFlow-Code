# Estimating 2D Camera Motion with Hybrid Motion Basis

## Quick Start

### Dataset Download
Download the complete dataset from Hugging Face:
```bash
# Using Hugging Face Hub
pip install huggingface_hub
huggingface-cli download Lhaippp/CamFlow-ICCV25 --repo-type dataset --local-dir data
```

### Inference With Given Images
```bash
python blind_inference.py
```

### Data Structure
```
data/
├── basis_24.pt      # Motion basis
├── ckpt.pth         # Model weights  
├── params.json      # Model configuration
└── test_imgs/       # Test image pairs
    ├── img1.png
    └── img2.png
```

## Dataset
- **Source**: [Lhaippp/CamFlow-ICCV25](https://huggingface.co/datasets/Lhaippp/CamFlow-ICCV25)
- **Contents**: Pre-trained model, motion basis, configuration, and test images
- **Usage**: Complete setup for camera motion estimation inference

## Alternative Setup

### AI Coding

✅ **Verified with AI Assistance**: This repository has been successfully tested and reproduced using AI coding tools.

For AI-assisted development reference, the project follows KISS, YAGNI, and SOLID principles for easy automation and reproduction.

### Environment Setup
```bash
# Simple setup (minimal dependencies)
python simple_setup.py
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