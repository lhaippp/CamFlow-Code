# Estimating 2D Camera Motion with Hybrid Motion Basis

**📄 Project Page**: For detailed information, results, and methodology, visit our project page:

**🔗 https://lhaippp.github.io/CamFlow/**

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

### Qualitative Comparison
We provide qualitative results of all comparison methods on GHOF-Cam dataset on Hugging Face for easy reproduction and comparison. Use the provided script to create comparison visualizations:

```bash
# Generate comparison GIFs for all methods
python create_comparison_gif.py --folder comparison_methods
```

**Important Note**: The image arrangement differs between directories:
- **'I' (Identity) directory**: Images are arranged as `[im1, im2]`
- **All other comparison methods**: Images are arranged as `[im2_warp, im1_warp]`

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