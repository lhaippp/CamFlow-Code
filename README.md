# CamFlow: Camera Motion Flow Estimation

A deep learning approach for estimating 2D camera motion using hybrid motion basis.

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/lhaippp/CamFlow-Code.git
cd CamFlow-Code

# 2. Setup environment (recommended)
python setup_env.py

# 3. Activate environment
source camflow_env/bin/activate

# 4. Download data
python download_data.py

# 5. Run evaluation
python eval_main.py --model_dir experiments/CAHomo/ --restore_file experiments/CAHomo/HEM.pth
```

## Prerequisites

- Python 3.7+
- PyTorch 1.8+
- CUDA (recommended)

## Installation

### Option 1: Automated Setup (Recommended)
```bash
git clone https://github.com/lhaippp/CamFlow-Code.git
cd CamFlow-Code
python setup_env.py
source camflow_env/bin/activate
```

### Option 2: Manual Installation
```bash
git clone https://github.com/lhaippp/CamFlow-Code.git
cd CamFlow-Code
python3 -m venv camflow_env
source camflow_env/bin/activate
pip install -r requirements.txt
```

### Option 3: Using Conda
```bash
git clone https://github.com/lhaippp/CamFlow-Code.git
cd CamFlow-Code
python setup_env.py --conda
conda activate camflow_env
```

## Data Setup

Download the CamFlow dataset:

```bash
python download_data.py
```

This downloads data from [Hugging Face](https://huggingface.co/datasets/Lhaippp/CamFlow-ICCV25) to `./data/CamFlow-ICCV25/`.

## Usage

### Basic Evaluation
```bash
python eval_main.py --model_dir experiments/CAHomo/ --restore_file experiments/CAHomo/HEM.pth
```

### Evaluation with IQA Metrics
```bash
python eval_main.py --model_dir experiments/CAHomo/ --restore_file experiments/CAHomo/HEM.pth --enable_iqa
```

## Citation

```bibtex
@inproceedings{Li2025Estimating2C,
  title={Estimating 2D Camera Motion with Hybrid Motion Basis},
  author={Haipeng Li and Tianhao Zhou and Zhanglei Yang and Yi Wu and Yan Chen and Zijing Mao and Shen Cheng and Bing Zeng and Shuaicheng Liu},
  year={2025},
  url={https://api.semanticscholar.org/CorpusID:280391787}
}
```