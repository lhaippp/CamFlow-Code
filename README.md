ta# CamFlow: Camera Motion Flow Estimation

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)

## Overview

CamFlow is a deep learning framework for camera motion flow estimation, designed to handle challenging scenarios including adverse weather conditions (fog, rain, snow) and low-light environments. The model combines homography estimation with optical flow computation to provide robust camera motion analysis.

## Key Features

- **Multi-scenario Robustness**: Handles regular, foggy, dark, rainy, and snowy conditions
- **Dual-task Learning**: Simultaneous homography and optical flow estimation
- **Transformer-based Architecture**: Leverages Swin Transformer for feature extraction
- **Comprehensive Evaluation**: GHOF (Geometric Homography Optical Flow) and IQA (Image Quality Assessment) metrics

## Architecture

The model consists of:
- **Swin Transformer Backbone**: Multi-scale feature extraction
- **Homography Estimation Module**: 8-parameter homography matrix prediction
- **Optical Flow Module**: Dense pixel-wise motion estimation
- **Mask Prediction**: Occlusion and validity region detection

## Installation

```bash
# Clone the repository
git clone https://github.com/lhaippp/CamFlow-Code.git
cd CamFlow-Code
```

## Usage

### Evaluation

```bash
# Basic evaluation
python eval_main.py --model_dir experiments/CAHomo/ --restore_file experiments/CAHomo/HEM.pth

# Enable IQA metrics
python eval_main.py --model_dir experiments/CAHomo/ --restore_file experiments/CAHomo/HEM.pth --enable_iqa
```

### Configuration

Modify `experiments/params.json` to adjust model parameters:

```json
{
    "net_type": "CamFlow",
    "train_batch_size": 8,
    "eval_batch_size": 8,
    "crop_size": [320, 576],
    "embed_dim": 24,
    "num_heads": [3, 12, 24]
}
```

## Evaluation Metrics

### GHOF Metrics
- **EPE (End Point Error)**: Average pixel-wise flow error
- **PME (Projection Motion Error)**: Homography-based motion error

### IQA Metrics
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **LPIPS**: Learned Perceptual Image Patch Similarity

## Dataset

The model is evaluated on the GHOF dataset with five categories:
- **RE**: Regular conditions
- **Fog**: Foggy weather
- **Dark**: Low-light conditions  
- **Rain**: Rainy weather
- **Snow**: Snowy conditions

## Citation

If you find this work useful, please cite:

```bibtex
@article{camflow2024,
  title={CamFlow: Camera Motion Flow Estimation for Adverse Weather Conditions},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with PyTorch and Accelerate
- Uses Swin Transformer architecture
- Evaluated on GHOF dataset 