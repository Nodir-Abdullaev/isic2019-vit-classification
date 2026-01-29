"""
ISIC 2019 Skin Lesion Classification Dataset

Source of images: official ISIC 2019 images
Source of labels: Kaggle one-hot CSV (normalized beforehand)
Metadata: optional, NOT used as ground truth

This Dataset is designed to be:
- explicit
- reproducible
- robust to missing/corrupted files
- suitable for research-style ML pipeline
"""

from pathlib import Path
from typing import Optional, Callable, Dict, Any

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset


class ISIC2019Dataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        images_dir: Path,
        transform: Optional[Callable] = None,
        strict: bool = True,
    ):
        """
        Args:
            csv_path (Path): Path to normalized CSV with columns:
                - image_id (str)  -> e.g. ISIC_1234567
                - label (int)     -> numeric class label
            images_dir (Path): Directory with ISIC images (.jpg)
            transform (Callable, optional): torchvision transforms
            strict (bool): if True, raises error on missing images
        """

        self.csv_path = Path(csv_path)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.strict = strict

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images dir not found: {self.images_dir}")

        self.df = pd.read_csv(self.csv_path)

        required_columns = {"image_id", "label"}
        missing = required_columns - set(self.df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        self.df = self.df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, image_id: str) -> Image.Image:
        img_path = self.images_dir / f"{image_id}.jpg"

        if not img_path.exists():
            msg = f"Image not found: {img_path}"
            if self.strict:
                raise FileNotFoundError(msg)
            else:
                return Image.new("RGB", (224, 224))

        return Image.open(img_path).convert("RGB")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        image_id = row["image_id"]
        label = int(row["label"])

        image = self._load_image(image_id)

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "image_id": image_id,
        }
