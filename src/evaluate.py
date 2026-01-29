"""
evaluate.py
Evaluation script for ISIC 2019 Skin Lesion Classification
- Loads a trained model (`best_model.pth`)
- Computes metrics (Accuracy, Macro F1)
- Plots confusion matrix
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import ISIC2019Dataset
from src.utils import get_val_transforms, compute_metrics, plot_confusion_matrix
import config
from transformers import ViTForImageClassification

# =========================
# 1. Setup
# =========================

device = config.DEVICE
torch.manual_seed(config.SEED)

# =========================
# 2. Dataset & DataLoader
# =========================

dataset = ISIC2019Dataset(
    csv_path=config.TEST_CSV,
    images_dir=config.IMAGES_DIR,
    transform=get_val_transforms(),
    strict=True
)

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# =========================
# 3. Load Model
# =========================

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=config.NUM_CLASSES,
    ignore_mismatched_sizes=True
).to(device)

model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# =========================
# 4. Evaluation
# =========================

true_labels, pred_labels = [], []

with torch.no_grad():
    for batch in tqdm(dataloader, desc="Evaluating"):
        images, labels = batch['image'].to(device), batch['label'].to(device)
        outputs = model(images).logits
        preds = torch.argmax(outputs, dim=1)
        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(preds.cpu().numpy())

# Metrics
metrics = compute_metrics(true_labels, pred_labels)
print("\nTest Metrics:")
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Macro F1: {metrics['macro_f1']:.4f}")

# Confusion Matrix
plot_confusion_matrix(true_labels, pred_labels)
