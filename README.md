# 🦴 Knee CT Scan Feature Extraction and Comparison Pipeline

## 📌 Overview

This project implements a full pipeline for **feature extraction** and **similarity comparison** between anatomical regions of the knee from 3D CT scan data. The primary goal is to assess how similar the **Tibia**, **Femur**, and **Background** regions are by leveraging a **3D-inflated DenseNet121 model** pretrained on 2D ImageNet data.

---

## 🗂️ Project Structure


---

## 🧪 Pipeline Summary

### Step 1: Segmentation-Based Splitting
- Separate CT mask into three classes:
  - **Tibia** (label: 1)
  - **Femur** (label: 2)
  - **Background** (all others)

### Step 2: Convert 2D Pretrained Model to 3D
- Load pretrained **DenseNet121** from `torchvision`.
- Convert all `Conv2D` layers to `Conv3D` by:
  - Repeating weights along depth.
  - Normalizing by the depth factor.

### Step 3: Feature Extraction
- Pass region volumes through 3D CNN.
- Extract feature maps from:
  - Last layer
  - Third-last layer
  - Fifth-last layer
- Apply Global Average Pooling to get N-dimensional vectors.

### Step 4: Feature Comparison
- Compute **cosine similarity** between:
  - Tibia ↔ Femur
  - Tibia ↔ Background
  - Femur ↔ Background

### Step 5: Save Results
- Save all similarity scores into `cosine_similarities.csv`.

---

## 📊 Sample Output (`cosine_similarities.csv`)
| Pair              | Fifth-Last | Third-Last | Last |
|-------------------|------------|------------|------|
| Tibia-Femur       | 0.78       | 0.84       | 0.88 |
| Tibia-Background  | 0.32       | 0.36       | 0.41 |
| Femur-Background  | 0.28       | 0.34       | 0.39 |

---

## 🧠 Requirements

Install dependencies using:

```bash
pip install torch torchvision nibabel scikit-image matplotlib pandas
