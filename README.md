# Estimating 2D Camera Motion with Hybrid Motion Basis

## 🌐 Web Demo (Try Now!)

**🚀 Interactive Demo**: Test CamFlow camera motion estimation directly in your browser!

**🔗 https://huggingface.co/spaces/Lhaippp/CamFlow-ICCV**

## Quick Start

### AI Coding

✅ **Verified with AI Assistance**: This repository has been successfully tested and reproduced using AI coding tools.

For AI-assisted development reference, the project follows KISS, YAGNI, and SOLID principles for easy automation and reproduction.

### Dataset Download
Download the complete dataset from Hugging Face:
```bash
# Using Hugging Face Hub
pip install huggingface_hub
huggingface-cli download Lhaippp/CamFlow-ICCV25 --repo-type dataset --local-dir data
```
- **Source**: [Lhaippp/CamFlow-ICCV25](https://huggingface.co/datasets/Lhaippp/CamFlow-ICCV25)
- **Contents**: Pre-trained model, motion basis, configuration, and test images
```
data/
├── basis_24.pt      # Motion basis
├── ckpt.pth         # Model weights  
├── params.json      # Model configuration
└── test_imgs/       # Test image pairs
    ├── img1.png
    └── img2.png
```

### Environment Setup
```bash
# Simple setup (minimal dependencies)
python simple_setup.py
```

### Inference With Given Images
```bash
# Run with default paths (displays all parameters and paths)
python blind_inference.py

# Specify custom model or image directory
python blind_inference.py --model_path custom.pth --imgs_dir custom_images/

# Use direct image paths (recommended for flexibility)
python blind_inference.py --image_paths /path/to/img1.png /path/to/img2.png
```

### Evaluation
```bash
# Basic evaluation
python eval_main.py --model_dir data/CamFlow-ICCV25 --restore_file data/CamFlow-ICCV25/ckpt.pth

# With IQA metrics
python eval_main.py --model_dir data/CamFlow-ICCV25 --restore_file data/CamFlow-ICCV25/ckpt.pth --enable_iqa
```