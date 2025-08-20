---
title: Brain Tumor Segmentation
emoji: 🧠
colorFrom: red
colorTo: blue
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
---

# Brain Tumor Segmentation AI

This demo uses a U-Net deep learning model to segment brain tumors from MRI scans.

## How to use:
1. Upload a brain MRI file (NIfTI format: .nii or .nii.gz)
2. Select the slice you want to analyze
3. View the AI predictions

## Model details:
- Architecture: U-Net
- Training data: BraTS dataset
- Performance: 0.82 Dice score
