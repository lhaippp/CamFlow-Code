# CamFlow: Estimating 2D Camera Motion with Hybrid Motion Basis

<div align="center">

[![Project Page](https://img.shields.io/badge/Project-Page-blue?style=for-the-badge&logo=github)](https://lhaippp.github.io/CamFlow/)
[![Demo](https://img.shields.io/badge/🚀-Interactive_Demo-orange?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/Lhaippp/CamFlow-ICCV)
[![arXiv](https://img.shields.io/badge/arXiv-2507.22480-red?style=for-the-badge&logo=arxiv)](https://arxiv.org/abs/2507.22480)
[![Dataset](https://img.shields.io/badge/🤗-Dataset-yellow?style=for-the-badge&logo=huggingface)](https://huggingface.co/datasets/Lhaippp/CamFlow-ICCV25)

</div>

> **CamFlow** presents a novel approach for 2D camera motion estimation using hybrid motion basis decomposition. Our method achieves state-of-the-art performance on camera motion estimation tasks while maintaining computational efficiency.

---

## 🚀 Quick Start

### 📦 Data Acquisition

<details>
<summary><b>Option A: Minimal Download (130MB)</b> — Essential files only</summary>

```bash
python download_data.py --minimal
```

**Dataset Structure:**
```
data/CamFlow-ICCV25/
├── basis_24.pt      # Motion basis (35MB)
├── ckpt.pth         # Model weights (93MB) 
├── params.json      # Model configuration
└── test_imgs/       # Test image pairs
    ├── img1.png
    └── img2.png
```
</details>

<details>
<summary><b>Option B: Complete Dataset (5.5GB)</b> — Full benchmark suite</summary>

```bash
python download_data.py
```

**Additional Contents:** `comparison_methods.zip` (2.56GB), `GHOF-Cam.npy` (2.8GB)
</details>

<details>
<summary><b>Option C: Hugging Face CLI</b> — Alternative download method</summary>

```bash
pip install huggingface_hub
huggingface-cli download Lhaippp/CamFlow-ICCV25 --repo-type dataset --local-dir data
```
</details>

**Data Source:** [🤗 Lhaippp/CamFlow-ICCV25](https://huggingface.co/datasets/Lhaippp/CamFlow-ICCV25)

### ⚙️ Environment Setup

```bash
# Automated setup (recommended)
python setup_environment.py && source camflow_env/bin/activate
```

<details>
<summary><b>Manual Setup</b> — Advanced users</summary>

```bash
# Create virtual environment
python -m venv camflow_env && source camflow_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Optional: GPU-accelerated PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
</details>

### 🎯 Inference

**Basic Usage:**
```bash
python blind_inference.py  # Uses test images from data/test_imgs/
```

**Advanced Usage:**
```bash
# Custom image paths
python blind_inference.py --image_paths /path/to/img1.png /path/to/img2.png

# Custom model directory  
python blind_inference.py --model_path custom.pth --imgs_dir custom_images/
```

---

## 📊 Evaluation & Benchmarking

### Model Evaluation
```bash
# Standard evaluation
python eval_main.py --model_dir data/CamFlow-ICCV25 --restore_file data/CamFlow-ICCV25/ckpt.pth

# With Image Quality Assessment metrics
python eval_main.py --model_dir data/CamFlow-ICCV25 --restore_file data/CamFlow-ICCV25/ckpt.pth --enable_iqa
```

### Comparative Analysis
Generate visual comparisons with baseline methods:

```bash
python create_comparison_gif.py --folder comparison_methods
```

> **Note:** Image arrangements vary by method:
> - **Identity method**: `[img1, img2]`  
> - **Other methods**: `[img2_warped, img1_warped]`

---

## 📚 Citation

If you find CamFlow useful in your research, please consider citing:

```bibtex
@article{li2025estimating,
  title={Estimating 2D Camera Motion with Hybrid Motion Basis},
  author={Li, Haipeng and Zhou, Tianhao and Yang, Zhanglei and Wu, Yi and Chen, Yan and Mao, Zijing and Cheng, Shen and Zeng, Bing and Liu, Shuaicheng},
  journal={arXiv preprint arXiv:2507.22480},
  year={2025}
}
```

---

## 🙏 Acknowledgements

We gratefully acknowledge the following contributions that made this work possible:

- **[GyroFlowPlus](https://github.com/lhaippp/GyroFlowPlus)** for providing the foundational dataset infrastructure
- **[DeepHomography](https://github.com/JirongZhang/DeepHomography)** for pioneering deep homography estimation and providing essential training data  
- **[BasesHomo](https://github.com/megvii-research/BasesHomo)** for their elegant and effective motion basis decomposition ideas
- **[HomoGAN](https://github.com/megvii-research/HomoGAN)** and **[DMHomo](https://github.com/lhaippp/DMHomo)** for the 8-basis foundation models and data augmentation strategies

Special thanks to our co-authors for their invaluable contributions, the anonymous reviewers for their constructive feedback, and the Area Chair for recognizing the merit of this work.

---

<div align="center">

**CamFlow** — Bridging the gap between traditional computer vision and modern deep learning approaches.

[![GitHub stars](https://img.shields.io/github/stars/lhaippp/CamFlow?style=social)](https://github.com/lhaippp/CamFlow)

</div>