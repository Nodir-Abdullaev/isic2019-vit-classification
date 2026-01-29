"""
utils.py
Helper functions for ISIC 2019 ML project:
- torchvision transforms
- evaluation metrics
- confusion matrix plotting
- image visualization
- automatic batch size finder
"""

from typing import Callable
import torch
import numpy as np
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
import config  # для NUM_CLASSES, CLASS_NAMES, DEVICE

# =========================
# Runtime sanity checks
# =========================

if len(config.CLASS_NAMES) != config.NUM_CLASSES:
    raise ValueError(
        f"CLASS_NAMES length ({len(config.CLASS_NAMES)}) "
        f"does not match NUM_CLASSES ({config.NUM_CLASSES})"
    )

# =========================
# 1. Transforms
# =========================

def get_train_transforms(image_size: int = 224) -> Callable:
    """Transforms for training (with augmentations)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=90),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

def get_val_transforms(image_size: int = 224) -> Callable:
    """Transforms for validation/testing (only resize + normalize)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

def find_optimal_batch_size(model, initial_size: int = 32, image_shape=(3, 224, 224)) -> int:
    """
    Автоматический подбор максимального batch size, который помещается в VRAM.
    model: torch model
    initial_size: стартовый размер батча
    image_shape: форма одного изображения (C,H,W)
    Возвращает: максимальный batch size, который помещается на GPU
    """
    model.eval()
    batch_size = initial_size

    while batch_size >= 1:
        try:
            dummy_input = torch.randn(batch_size, *image_shape).to(config.DEVICE)
            with torch.no_grad():
                _ = model(dummy_input).logits
            torch.cuda.empty_cache()
            return batch_size
        except RuntimeError as e:
            if 'out of memory' in str(e):
                torch.cuda.empty_cache()
                batch_size = batch_size // 2
            else:
                raise e
    raise ValueError("Недостаточно VRAM даже для batch_size=1")


# =========================
# 2. Metrics
# =========================

def compute_metrics(true_labels, pred_labels):
    """Compute per-class and macro metrics."""
    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average=None, zero_division=0
    )
    macro_f1 = f1.mean()
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1,
    }

def plot_confusion_matrix(true_labels, pred_labels, normalize: str = "true"):
    """Plot confusion matrix with class labels."""
    cm = confusion_matrix(true_labels, pred_labels, normalize=normalize)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[config.CLASS_NAMES[i] for i in range(config.NUM_CLASSES)]
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


# =========================
# 3. Visualization
# =========================

def show_batch(images, labels, num_images: int = 8):
    """Display a batch of images with labels."""
    plt.figure(figsize=(15, 5))

    for i in range(min(num_images, len(images))):
        plt.subplot(1, num_images, i + 1)

        img = images[i].permute(1, 2, 0).cpu().numpy()
        # unnormalize
        img = np.clip(
            img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]),
            0,
            1,
        )

        plt.imshow(img)
        plt.axis("off")
        plt.title(config.CLASS_NAMES[labels[i].item()], fontsize=8)

    plt.tight_layout()
    plt.show()
