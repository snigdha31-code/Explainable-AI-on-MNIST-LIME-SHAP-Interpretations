# XAI MNIST — LIME + SHAP Explanations

**xai_mnist_lime_shap.py** trains a small CNN on MNIST and produces side-by-side LIME and SHAP explanations for a selected test image.

## Features
- Train a small CNN on MNIST (or load pre-trained weights)
- Generate LIME explanations for a test image
- Generate SHAP explanations (DeepExplainer/GradientExplainer depending on backend)
- Save visualization heatmaps and comparison plots

## Requirements
- Python 3.8+
- See `requirements.txt` for Python packages

## Installation (recommended)
```bash
# create a virtual environment
python -m venv venv
source venv/bin/activate    # macOS / Linux
# venv\Scripts\activate     # Windows (PowerShell)

pip install -r requirements.txt
