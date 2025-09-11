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
- Test-time augmentation (TTA) for improved inference accuracy
- Hugging Face Hub integration with `PyTorchModelHubMixin`
- 6 pre-trained models available for immediate use
- Easy model loading and inference via `synthstroke_model.py`

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
   git clone https://github.com/liamchalcroft/synthstroke.git
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

## SynthStroke Model Library

The `synthstroke_model.py` file provides a complete Python interface for using our models:

### Key Features

- **PyTorchModelHubMixin Integration**: Seamless loading from Hugging Face Hub
- **Multiple Model Variants**: Support for all 6 model types (baseline, synth, synth_pseudo, synth_plus, qatlas, qsynth)
- **Test-Time Augmentation**: Built-in flip-based TTA for improved accuracy
- **Automatic Device Handling**: Smart GPU/CPU device management
- **Input Validation**: Robust error checking and validation
- **Model Information**: Detailed metadata and parameter counts

### Installation

The model library is included in this repository and requires the same dependencies as the training code.

---

## Pre-trained Models

| Model | Description | Hugging Face |
|-------|-------------|--------------|
| **SynthStroke Baseline** | Model trained on real ATLAS T1w data | [🤗 synthstroke-baseline](https://huggingface.co/liamchalcroft/synthstroke-baseline) |
| **SynthStroke Synth** | Multi-tissue segmentation with synthetic data | [🤗 synthstroke-synth](https://huggingface.co/liamchalcroft/synthstroke-synth) |
| **SynthStroke SynthPseudo** | Synthetic data + pseudo-label augmentation | [🤗 synthstroke-synth-pseudo](https://huggingface.co/liamchalcroft/synthstroke-synth-pseudo) |
| **SynthStroke SynthPlus** | Synthetic data + real multi-dataset training | [🤗 synthstroke-synth-plus](https://huggingface.co/liamchalcroft/synthstroke-synth-plus) |
| **SynthStroke qATLAS** | qMRI-based model trained on synthetic parameters | [🤗 synthstroke-qatlas](https://huggingface.co/liamchalcroft/synthstroke-qatlas) |
| **SynthStroke qSynth** | qMRI-constrained synthetic data training | [🤗 synthstroke-qsynth](https://huggingface.co/liamchalcroft/synthstroke-qsynth) |

### Using Pre-trained Models

The `synthstroke_model.py` provides easy access to all pre-trained models using Hugging Face's `PyTorchModelHubMixin`:

#### Quick Start

```python
import torch
from synthstroke_model import SynthStrokeModel

# Load any model from Hugging Face Hub
model = SynthStrokeModel.from_pretrained("liamchalcroft/synthstroke-baseline")

# Prepare your MRI data (T1-weighted, shape: [batch, 1, H, W, D])
mri_volume = torch.randn(1, 1, 192, 192, 192)

# Run inference with optional Test-Time Augmentation
predictions = model.predict_segmentation(mri_volume, use_tta=True)

# For baseline model: get lesion probability map (channel 1)
lesion_probs = predictions[:, 1]

# For multi-tissue models (synth, synth_pseudo, synth_plus, qsynth):
# Get all tissue probability maps
background = predictions[:, 0]    # Background
gray_matter = predictions[:, 1]   # Gray Matter
white_matter = predictions[:, 2]  # White Matter
partial_volume = predictions[:, 3]  # Gray/White Partial Volume
csf = predictions[:, 4]          # Cerebro-Spinal Fluid
stroke = predictions[:, 5]       # Stroke Lesion
```

#### Model Information

```python
# Get detailed model information
info = model.get_model_info()
print(f"Model type: {info['model_type']}")
print(f"Input channels: {info['input_channels']}")
print(f"Output channels: {info['output_channels']}")
print(f"TTA support: {info['tta_support']}")
print(f"Parameters: {info['parameters']:,}")
```

#### Available Model Configurations

```python
from synthstroke_model import (
    create_baseline_model,      # 2-class: Background + Stroke
    create_synth_model,         # 6-class: Multi-tissue + Stroke
    create_synth_pseudo_model,  # 6-class: With pseudo-labels
    create_synth_plus_model,    # 6-class: Multi-dataset training
    create_qatlas_model,        # 2-class: qMRI-based
    create_qsynth_model         # 6-class: qMRI-constrained
)

# Create models locally (without downloading from Hub)
baseline_model = create_baseline_model()
synth_model = create_synth_model()
```

#### Test-Time Augmentation (TTA)

All models support flip-based TTA for improved inference accuracy:

```python
# Enable TTA for more robust predictions
predictions_with_tta = model.predict_segmentation(mri_volume, use_tta=True)
# This uses 8 augmentations (original + 7 flipped versions) and averages results
```

### Model Architecture Details

- **Framework**: MONAI UNet with PyTorch
- **Input**: 3D MRI volumes (T1-weighted for most models, qMRI parameters for qATLAS/qSynth)
- **Architecture**: 3D UNet with configurable channels and strides
- **Training**: Mixed precision (AMP) with custom loss functions
- **Inference**: Optional Test-Time Augmentation support

For detailed model specifications, see the individual model cards on Hugging Face Hub.

---

## Support

For issues or questions, please [open an issue](https://github.com/liamchalcroft/synthstroke/issues) on GitHub.

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
