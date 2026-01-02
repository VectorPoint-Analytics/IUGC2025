# Adversarially Fine-tuned Self-Supervised Framework for Automated Landmark Detection in Intrapartum Ultrasound (IUGC 2025)

<!-- TAG: Add a banner image (optional) -->
<!-- e.g., ![banner](assets/banner.png) -->

This repository contains the implementation of our MICCAI 2025 submission for the **Intrapartum Ultrasound Grand Challenge (IUGC) 2025**, focused on **automated anatomical landmark detection** in transperineal intrapartum ultrasound and **Angle of Progression (AoP)** estimation.

---

## 🧠 Overview

Accurate monitoring of fetal head progression during labor is clinically important, but current practice often requires manual annotation of landmarks. We propose a fully automated pipeline that combines:

- **Self-supervised pretraining** on unlabeled standard-plane ultrasound images (to learn strong anatomical priors)
- An **attention-enhanced decoder** for improved spatial localization
- **Adversarial fine-tuning (PatchGAN-style discriminator)** to encourage anatomically plausible predictions

The model detects **three landmarks**:
- **PS1**: distal edge of pubic symphysis  
- **PS2**: proximal edge of pubic symphysis  
- **FH**: most distal point of fetal head

These landmarks are then used to estimate AoP.

<img width="928" height="646" alt="image" src="https://github.com/user-attachments/assets/72e5ef79-34eb-4082-b6c3-5412bfe79471" />

FIGURE: Pipeline overview 

---

## ⭐ Key Results

Our best configuration (MoCoV2 + CBAM + adversarial guidance) achieves:

- **Mean Radial Error (MRE): 25.66 px**
- **AoP Mean Absolute Error (MAE): 8.54°**

<img width="928" height="769" alt="image" src="https://github.com/user-attachments/assets/4060d03e-0cfa-44c2-ae9c-3d06e7b4626d" />

---

## 📁 Repository Structure

The repo currently includes the following key files/scripts:

- `moco_v2.py` — MoCoV2-style self-supervised components
- `heatmap_dataset.py` — dataset utilities for heatmaps / landmark learning
- `models.py` — model definitions (encoder/decoder heads)
- `finetuning.py` — supervised fine-tuning / training loop (landmark detection)
- `inference.py` — inference + prediction export
- `utils.py` — helper functions
- `adversarial-hrnet.ipynb` — notebook experiment / demo
- `config.py` — central configuration

---

## 🔧 Setup

### 1) Clone
```bash
git clone https://github.com/zaid-24/MICCAI2025.git
cd MICCAI2025
```

### 2) Create environment
> Recommended: Python 3.9+ and a CUDA-enabled PyTorch install.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install -U pip
pip install -r requirements.txt
```

## 🗂️ Data Preparation

We use the official **IUGC 2025** dataset, which includes 31,421 training images (300 labeled standard-plane images; the rest unlabeled), plus withheld validation/test sets evaluated via the challenge server.
<img width="915" height="280" alt="image" src="https://github.com/user-attachments/assets/5de84606-48ea-477a-86f8-d3e5b3b01837" />

A suggested local structure:

```text
data/
  labeled/
    images/
      *.png / *.jpg
    annotations.csv   # (PS1, PS2, FH coordinates + AoP if available)
  unlabeled_sp/
    images/
      *.png / *.jpg   # unlabeled standard-plane images for SSL pretraining
  splits/
    train.txt
    val.txt
```

✅ **Tip:** Update all dataset paths in `config.py`.

---

## 🏋️ Training

### Stage A — Self-supervised pretraining (MoCoV2)
We pretrain on **unlabeled standard-plane images** (e.g., standard-plane images inside the unlabeled set).

```bash
# Example (update flags/paths to match your scripts)
python finetuning.py \
  --stage pretrain \
  --config config.py
```
<img width="1096" height="520" alt="image" src="https://github.com/user-attachments/assets/b46e25fb-d70e-44ce-907f-d51c2862a213" />


### Stage B — Supervised fine-tuning (landmark detection)
We fine-tune on labeled standard-plane images with three landmarks (PS1, PS2, FH).

```bash
python finetuning.py \
  --stage finetune \
  --config config.py
```

### Stage C — Adversarial fine-tuning (PatchGAN guidance)
Adversarial loss encourages **anatomically plausible** landmark heatmaps.

```bash
python finetuning.py \
  --stage adversarial \
  --config config.py
```

---

## 🔍 Inference

Run inference to generate landmark predictions and AoP estimation:

```bash
python inference.py \
  --config config.py \
  --weights /path/to/checkpoint.pth \
  --input /path/to/images \
  --output outputs/
```

Expected outputs (recommended):
- predicted landmark coordinates (PS1, PS2, FH)
- predicted heatmaps (optional)
- AoP prediction / computed AoP
- visual overlays for quick inspection

---

## 📐 Evaluation

We report:
- **Mean Radial Error (MRE)** for landmark localization
- **AoP MAE (degrees)** for clinical relevance

---

## 🧩 Method Notes (Implementation-Aligned)

A common training objective for this setup combines multiple terms to balance precision and plausibility, such as:
- heatmap regression loss
- coordinate loss (e.g., MRE-like term)
- adversarial loss (LSGAN/PatchGAN style)
- entropy penalty to encourage sharp/unimodal heatmaps

---

## 📌 Reproducibility Checklist

- [ ] Set random seeds (PyTorch/NumPy)
- [ ] Log experiment configs (paths, hyperparameters)
- [ ] Save checkpoints + best model on training metric
- [ ] Export inference outputs (JSON/CSV) + overlays
- [ ] Document data split strategy

---

## 📎 Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{krishna2025iugc_landmarks,
  title     = {Adversarially Fine-tuned Self-Supervised Framework for Automated Landmark Detection in Intrapartum Ultrasound},
  author    = {Anirvan Krishna and Zaid Ahmed Khan},
  booktitle = {MICCAI 2025 (Intrapartum Ultrasound Grand Challenge Workshop/Proceedings)},
  year      = {2025},
}
---

## 👥 Authors / Contact

- **Zaid Ahmed Khan** — ik241168@kgpian.iitkgp.ac.in  
- **Anirvan Krishna** — anirvankrishna@kgpian.iitkgp.ac.in  

---

## 🙏 Acknowledgements

- Intrapartum Ultrasound Grand Challenge (IUGC) 2025 organizers and dataset providers.

