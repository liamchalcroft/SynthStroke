<div align="center">

# SynthStroke
### Deep Learning Stroke Lesion Segmentation with Synthetic Data

[![Paper](https://img.shields.io/badge/Paper-MELBA%202025-blue?style=for-the-badge&logo=arxiv)](http://dx.doi.org/10.59275/j.melba.2025-f3g6)
[![SPM Toolbox](https://img.shields.io/badge/SPM-Toolbox-orange?style=for-the-badge&logo=github)](https://github.com/liamchalcroft/SynthStrokeSPM)
[![Python](https://img.shields.io/badge/Python-3.10+-green?style=for-the-badge&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-red?style=for-the-badge)](LICENSE)

*Python implementation of "[Synthetic Data for Robust Stroke Segmentation](http://dx.doi.org/10.59275/j.melba.2025-f3g6)" published in Machine Learning for Biomedical Imaging (MELBA) 2025.*

</div>

---

## About

This repository contains the implementation of our MELBA 2025 paper on synthetic data generation for stroke lesion segmentation. The method uses synthetic data to improve model generalization across different imaging protocols and patient populations.

**Features:**
- Synthetic data generation pipeline using healthy brain MRI
- Multi-tissue segmentation (lesions and healthy brain tissue)
- Mixed precision training with configurable loss functions
- Test-time augmentation for inference

**Paper**: Chalcroft, L., Pappas, I., Price, C. J., & Ashburner, J. (2025). [Synthetic Data for Robust Stroke Segmentation](http://dx.doi.org/10.59275/j.melba.2025-f3g6). *Machine Learning for Biomedical Imaging*, 3, 317–346.

---

## Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/username/synthstroke.git
   cd synthstroke
   ```

2. **Set up environment**
   ```bash
   conda create -n synthstroke python=3.10
   conda activate synthstroke
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Training Models

<details>
<summary><b>Baseline Model (Real Data Only)</b></summary>

Train a baseline model using only real stroke imaging data:

```bash
python train.py \
    --name baseline_model \
    --logdir ./ \
    --baseline \
    --l2 50 \
    --patch 128 \
    --amp \
    --epochs 500 \
    --epoch_length 200 \
    --lr 0.001 \
    --val_interval 2
```

**Parameters:**
- `--baseline`: Use real stroke images (no synthetic data)
- `--l2 50`: L2 loss for first 50 epochs, then switches to Dice loss
- `--patch 128`: Random crop size for training patches
- `--amp`: Enable automatic mixed precision training
- `--val_interval 2`: Validate and save weights every 2 epochs

</details>

<details>
<summary><b>SynthStroke Model (With Synthetic Data)</b></summary>

Train the model with synthetic data augmentation:

```bash
python train.py \
    --name synthstroke_model \
    --logdir ./ \
    --mbhealthy \
    --fade \
    --lesion_weight 2 \
    --l2 50 \
    --patch 128 \
    --amp \
    --epochs 500 \
    --epoch_length 200 \
    --lr 0.001 \
    --val_interval 2
```

**Key Features:**
- `--mbhealthy`: Enable MultiBrain healthy tissue segmentation
- `--fade`: Apply intensity non-uniformity fields to simulate penumbra
- `--lesion_weight 2`: Increase lesion class weight for better sensitivity

</details>

### Model Inference

<details>
<summary><b>Prediction on New Data</b></summary>

Run inference on new stroke MRI scans:

```bash
python test.py \
    --weights ./synthstroke_model/checkpoint.pt \
    --tta \
    --mb \
    --patch 128 \
    --savedir /path/to/output/ \
    --files "/path/to/input/*.nii.gz"
```

**Options:**
- `--tta`: Enable test-time augmentation
- `--mb`: Output multi-brain tissue labels alongside lesions
- `--files`: Path pattern or text file with input paths

</details>

### SPM Integration

For MATLAB/SPM users, check out our **[SPM Toolbox](https://github.com/liamchalcroft/SynthStrokeSPM)** for seamless integration with SPM preprocessing pipelines.

---

## Pre-trained Models

| Model | Description | Status |
|-------|-------------|--------|
| **SynthStroke** | Model trained with synthetic data | Coming Soon |
| **Baseline** | Model trained on real data only | Coming Soon |

Pre-trained weights will be made available.

---

## Support

For issues or questions, please [open an issue](https://github.com/username/synthstroke/issues) on GitHub.

---

## Citation

If you use SynthStroke in your research, please cite:

```bibtex
@article{Chalcroft2025,
  title = {Synthetic Data for Robust Stroke Segmentation},
  volume = {3},
  ISSN = {2766-905X},
  url = {http://dx.doi.org/10.59275/j.melba.2025-f3g6},
  DOI = {10.59275/j.melba.2025-f3g6},
  number = {August 2025},
  journal = {Machine Learning for Biomedical Imaging},
  publisher = {Machine Learning for Biomedical Imaging},
  author = {Chalcroft, Liam and Pappas, Ioannis and Price, Cathy J. and Ashburner, John},
  year = {2025},
  month = aug,
  pages = {317–346}
}
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
