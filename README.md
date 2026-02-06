# RBPD
# RBPD: A Robust Black-box Proactive Defense Framework against Face Swapping

This is the official PyTorch implementation of the paper:  **"RBPD: A Robust Black-box Proactive Defense Framework against Face Swapping"**.
DOI: https://doi.org/10.5281/zenodo.18503183
This Digital Object Identifier (DOI) represents the permanent archival version of the RBPD framework (initial version), ensuring the long-term accessibility and traceability of our research artifacts.

## 📖 Overview
RBPD (Robust Black-box Proactive Defense) is a two-stage generative framework designed to protect source images from malicious face swapping. By integrating semantic awareness and texture guidance, it embeds imperceptible but highly robust adversarial perturbations that disrupt identity feature extraction in unknown black-box scenarios.



### Key Contributions:
* **Two-Stage Generation**: Semantic-Aware Encoder (SAE) + Texture-Guided Decoder (TGD) for precise localization, followed by Dual-stream Fusion Encoder (DFE) + Multi-scale Aggregation Decoder (MAD) for robustness enhancement.
* **MAA Strategy**: A Meta-learning Adaptive Attack strategy that adaptively balances gradients from multiple heterogeneous identity extractors to achieve cross-model generalization.
* **DDV Metric**: A novel Distortion Defense Variation (DDV) metric to quantitatively assess defense stability under various image distortions.

## 🛠️ Requirements
The framework is implemented using PyTorch 2.5.1. Install the dependencies via:

```bash
pip install -r requirements.txt

```

Note: Tested on Ubuntu 16.04.6 and NVIDIA Tesla A100.

## 📥 Pre-trained Models & Weights

To use this framework, you need to prepare the following weights from their respective official repositories. We do not provide these weights directly:

### 1. Identity Feature Extractors (for Training & Evaluation)

* **ArcFace**: [Official Implementation](https://github.com/deepinsight/insightface)
* **FaceNet**: [Official Implementation](https://www.google.com/search?q=https://github.com/davidsandberg/facenet-pytorch)
* **MagFace**: [Official Implementation](https://www.google.com/search?q=https://github.com/Irving-S/MagFace)
* **AdaFace**: [Official Implementation](https://github.com/mk-minchul/AdaFace)

### 2. Face Swapping Models (for Black-box Testing)

* **SimSwap**: [Official Implementation](https://github.com/neuralchen/SimSwap)
* **E4S**: [Official Implementation](https://www.google.com/search?q=https://github.com/deepfake-face-swapping/E4S)
* **DiffSwap**: [Official Implementation](https://www.google.com/search?q=https://github.com/Muzic-Code/DiffSwap)

## 🚀 Usage

### Step 1: Stage-I Training (Semantic-Guided Attack)

Train the initial perturbations using the SAE and TGD modules.

```bash
python train_stage1.py --epochs 20 --batch_size 256

```

### Step 2: Stage-II Training (Robustness Enhancement)

Enhance the perturbations' resistance to distortions using the DFE and MAD modules.

```bash
python train_stage2.py --resume ./checkpoints/ars_stage1_epoch_20.pth

```

### Step 3: Evaluation

The code supports calculating identity matching accuracy (Top-1/Top-5) and the proposed **DDV** score.

## ✉️ Contact

For any questions regarding the RBPD framework, please contact:
**Chenxi Huang** - [2023211501@stu.ppsuc.edu.cn](mailto:2023211501@stu.ppsuc.edu.cn)




