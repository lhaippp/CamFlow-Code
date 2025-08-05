# Estimating 2D Camera Motion with Hybrid Motion Basis

**📄 Project Page**: For detailed information, results, and methodology, visit our project page:

**🔗 https://lhaippp.github.io/CamFlow/**

## 🌐 Web Demo (Try Now!)

**🚀 Interactive Demo**: Test CamFlow camera motion estimation directly in your browser!

**🔗 https://huggingface.co/spaces/Lhaippp/CamFlow-ICCV**


## Quick Start

### AI Coding

✅ **Verified with AI Assistance**: This repository has been successfully tested and reproduced using AI coding tools.

For AI-assisted development reference, the project follows KISS (Keep It Simple), YAGNI (You Aren't Gonna Need It), and SOLID principles for maintainable code and easy automation.

### Dataset Download

**Quick Start (Minimal Download - ~130MB):**
```bash
# Download only essential files for basic motion estimation
python download_data.py --minimal
```

**Complete Dataset Download (~5.5GB):**
```bash
# Download everything including comparison methods and full benchmark
python download_data.py
```

**Alternative with Hugging Face CLI:**
```bash
# Using Hugging Face Hub (downloads everything)
pip install huggingface_hub
huggingface-cli download Lhaippp/CamFlow-ICCV25 --repo-type dataset --local-dir data
```

- **Source**: [Lhaippp/CamFlow-ICCV25](https://huggingface.co/datasets/Lhaippp/CamFlow-ICCV25)
- **Minimal Contents**: Essential files for 2-image motion estimation
```
data/CamFlow-ICCV25/
├── basis_24.pt      # Motion basis (35MB)
├── ckpt.pth         # Model weights (93MB)
├── params.json      # Model configuration
└── test_imgs/       # Test image pairs
    ├── img1.png
    └── img2.png
```
- **Full Contents**: Includes comparison_methods.zip (2.56GB) and GHOF-Cam.npy (2.8GB) for benchmarking

### Environment Setup

**Option 1: Automated Setup (Recommended)**
```bash
# One-command setup with virtual environment
python setup_environment.py
source camflow_env/bin/activate  # Activate the environment
```

<details>
<summary><b>Option 2: Manual Setup (Click to expand)</b></summary>

```bash
# Create virtual environment
python -m venv camflow_env
source camflow_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Or install PyTorch GPU version specifically
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

</details>

### Inference With Given Images
```bash
# Run with default settings (basis_24.pt, ckpt.pth, params.json) on test images (img1.png, img2.png)
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

### Qualitative Comparison
We provide qualitative results of all comparison methods on GHOF-Cam dataset on Hugging Face for easy reproduction and comparison. Use the provided script to create comparison visualizations:

```bash
# Generate comparison GIFs for all methods
python create_comparison_gif.py --folder comparison_methods
```

**Important Note**: The image arrangement differs between directories:
- **'I' (Identity) directory**: Images are arranged as `[im1, im2]`
- **All other comparison methods**: Images are arranged as `[im2_warp, im1_warp]`

## Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@article{li2025estimating,
  title={Estimating 2D Camera Motion with Hybrid Motion Basis},
  author={Li, Haipeng and Zhou, Tianhao and Yang, Zhanglei and Wu, Yi and Chen, Yan and Mao, Zijing and Cheng, Shen and Zeng, Bing and Liu, Shuaicheng},
  journal={arXiv preprint arXiv:2507.22480},
  year={2025}
}
```