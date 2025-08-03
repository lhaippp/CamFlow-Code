# CamFlow: Camera Motion Flow Estimation

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