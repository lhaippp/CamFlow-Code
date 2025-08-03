# Estimating 2D Camera Motion with Hybrid Motion Basis

## Quick Start

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