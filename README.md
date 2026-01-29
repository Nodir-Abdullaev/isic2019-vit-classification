# ISIC 2019 Skin Lesion Classification

This repository contains a PyTorch-based implementation for multi-class skin lesion classification using the ISIC 2019 dataset.
The project was developed as a structured machine learning pipeline with an emphasis on clarity, reproducibility, and experimental flexibility, rather than as a single end-to-end script.

The initial baseline model is a Vision Transformer (ViT), with the intention to later compare its performance against alternative architectures under identical data and training conditions.

---

## Motivation

Skin lesion classification is a highly imbalanced and clinically sensitive problem, where overall accuracy alone is insufficient.
This project focuses on:

- clean dataset handling and label processing
- explicit train/validation/test separation
- metrics suitable for imbalanced medical data
- reproducible training and evaluation

The codebase is intentionally modular to make future experiments (e.g. different models or loss functions) easy to integrate.

---

## Dataset

The project uses the ISIC 2019 Challenge dataset, which consists of dermoscopic images labeled into 8 diagnostic categories.

- Images and original CSV files are not included in the repository
- Dataset paths are defined locally via configuration
- Labels are converted from one-hot encoding to integer class indices during preprocessing

Expected local structure (example):

```
ISIC-images2019/
 ├── ISIC_0000000.jpg
 ├── ISIC_0000001.jpg
 └── ...
ISIC_2019_Training_GroundTruth.csv
```

---

## Repository Structure

```
.
├── src/
│   ├── config.py              # Global configuration and paths
│   ├── data_preprocessing.py  # Label cleaning and dataset split
│   ├── dataset.py             # PyTorch Dataset implementation
│   ├── model.py               # Model construction utilities
│   ├── train.py               # Training loop and early stopping
│   ├── evaluate.py            # Final evaluation and metrics
│   └── utils.py               # Transforms, metrics, visualization
│
├── notebooks/
│   └── Project_Demo_Run.ipynb # Demonstration notebook
│
├── requirements.txt
└── README.md
```

---

## Environment

The code was developed and tested with:

- Python ≥ 3.10
- PyTorch ≥ 2.x (CUDA-enabled build)
- NVIDIA GPU recommended

Dependencies can be installed via:

```
pip install -r requirements.txt
```

---

## Configuration

All dataset paths and global parameters are defined in:

```
src/config.py
```

This includes:

- random seed
- device selection (CPU / CUDA)
- dataset locations
- number of classes
- project root paths

No hard-coded paths are used inside the training or dataset logic.

---

## Data Preprocessing

Before training, the dataset is prepared using:

```
python src/data_preprocessing.py
```

This step performs:

- removal of invalid or unused labels
- conversion from one-hot labels to class indices
- verification that image files exist on disk
- stratified splitting into train, validation, and test sets

The resulting CSV files are saved locally and reused across experiments.

---

## Training

Model training is launched with:

```
python src/train.py
```

Key aspects of the training setup:

- Model: ViT-Base (patch size 16), pretrained on ImageNet
- Optimizer: AdamW
- Loss: Cross-entropy with class weighting
- Mixed precision training (AMP)
- Gradient clipping
- Early stopping based on validation loss

The best-performing model (on validation data) is saved automatically.

---

## Evaluation

After training, the model is evaluated on the held-out test set using:

- accuracy
- macro-averaged F1 score
- confusion matrix

This allows analysis of per-class behavior, which is especially important for imbalanced medical datasets.

---

## Notes

- `num_workers=0` is intentionally used in DataLoaders to avoid multiprocessing issues on Windows
- Mixed precision is enabled to reduce GPU memory usage
- The code prioritizes readability and explicitness over maximal performance optimizations

---

## Disclaimer

This project is intended for research and educational purposes only.
It is not a medical diagnostic system and should not be used for clinical decision-making.

---

## Author

This project was developed as a structured machine learning study focused on medical image classification.

---

