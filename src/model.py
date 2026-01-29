"""
model.py
Vision Transformer (ViT) factory for ISIC 2019 baseline.
"""

import torch
from transformers import ViTForImageClassification
from src.config import DEVICE, NUM_CLASSES

def build_vit_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> torch.nn.Module:
    """
    Build a ViT model for image classification.

    Args:
        num_classes (int): Number of target classes.
        pretrained (bool): Whether to load pretrained weights from ImageNet.

    Returns:
        torch.nn.Module: ViT model on DEVICE.
    """
    model_name = "google/vit-base-patch16-224"

    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True if pretrained else False
    )

    return model.to(DEVICE)


def find_optimal_batch_size(model: torch.nn.Module, initial_size: int = 32, image_size: int = 224) -> int:
    """
    Automatically find the largest batch size that fits in GPU memory.

    Args:
        model (torch.nn.Module): Model to test.
        initial_size (int): Starting batch size.
        image_size (int): Image height and width (square).

    Returns:
        int: Maximum batch size that fits in VRAM.
    """
    model.eval()
    batch_size = initial_size

    while batch_size >= 1:
        try:
            dummy_input = torch.randn(batch_size, 3, image_size, image_size, device=DEVICE)
            with torch.no_grad():
                _ = model(dummy_input)
            torch.cuda.empty_cache()
            return batch_size
        except RuntimeError as e:
            if 'out of memory' in str(e):
                torch.cuda.empty_cache()
                batch_size = batch_size // 2
            else:
                raise e

    raise ValueError("Not enough VRAM even for batch_size=1")
