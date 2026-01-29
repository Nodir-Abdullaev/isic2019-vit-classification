import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import ViTForImageClassification
from tqdm import tqdm
from pathlib import Path
import numpy as np

# =========================
# Configuration
# =========================
NUM_WORKERS = 0
BATCH_SIZE = 16

torch.backends.cudnn.benchmark = True
scaler = torch.amp.GradScaler('cuda')

from dataset import ISIC2019Dataset
from utils import (
    get_train_transforms,
    get_val_transforms,
    compute_metrics,
    plot_confusion_matrix
)
import config


def main():
    # =========================
    # 1. Initialization
    # =========================
    torch.manual_seed(config.SEED)
    device = config.DEVICE

    train_csv_path = config.TRAIN_CSV
    val_csv_path = config.VAL_CSV
    test_csv_path = config.TEST_CSV

    model_save_path = Path(config.PROJECT_ROOT) / "models" / "best_model.pth"
    model_save_path.parent.mkdir(exist_ok=True, parents=True)

    print(f"Device: {device}")
    print(f"DataLoader workers: {NUM_WORKERS}")

    # =========================
    # 2. Datasets and loaders
    # =========================
    train_dataset = ISIC2019Dataset(
        csv_path=train_csv_path,
        images_dir=config.IMAGES_DIR,
        transform=get_train_transforms()
    )

    val_dataset = ISIC2019Dataset(
        csv_path=val_csv_path,
        images_dir=config.IMAGES_DIR,
        transform=get_val_transforms()
    )

    test_dataset = ISIC2019Dataset(
        csv_path=test_csv_path,
        images_dir=config.IMAGES_DIR,
        transform=get_val_transforms()
    )

    num_classes = int(train_dataset.df['label'].max()) + 1

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    print("Dataset statistics")
    print(f"Train: {len(train_dataset)}")
    print(f"Validation: {len(val_dataset)}")
    print(f"Test: {len(test_dataset)}")
    print(f"Number of classes: {num_classes}")

    # =========================
    # 3. Model
    # =========================
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    ).to(device)

    # =========================
    # 4. Optimization setup
    # =========================
    optimizer = AdamW(model.parameters(), lr=2e-5)

    all_labels = train_dataset.df['label'].values
    class_counts = np.bincount(all_labels)
    class_weights = 1.0 / (torch.tensor(class_counts, dtype=torch.float) + 1e-6)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # =========================
    # 5. Training loop
    # =========================
    EPOCHS = 10
    PATIENCE = 3
    best_val_loss = float('inf')
    no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for batch in pbar:
            if isinstance(batch, dict):
                inputs = batch['image'].to(device, non_blocking=True)
                labels = batch['label'].to(device, non_blocking=True)
            else:
                inputs = batch[0].to(device, non_blocking=True)
                labels = batch[1].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                outputs = model(inputs).logits
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # =========================
        # Validation
        # =========================
        model.eval()
        val_loss = 0.0
        val_true, val_preds = [], []

        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, dict):
                    inputs = batch['image'].to(device)
                    labels = batch['label'].to(device)
                else:
                    inputs = batch[0].to(device)
                    labels = batch[1].to(device)

                with torch.amp.autocast('cuda'):
                    outputs = model(inputs).logits
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_true.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        metrics = compute_metrics(val_true, val_preds)

        print(f"Validation loss: {avg_val_loss:.4f}")
        print(f"Validation macro F1: {metrics.get('macro_f1', 0):.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve = 0
            torch.save(model.state_dict(), model_save_path)
            print("Best model saved")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print("Early stopping triggered")
                break

    # =========================
    # 6. Final evaluation
    # =========================
    print("Final evaluation on test set")

    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    test_true, test_preds = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            if isinstance(batch, dict):
                inputs = batch['image'].to(device)
                labels = batch['label'].to(device)
            else:
                inputs = batch[0].to(device)
                labels = batch[1].to(device)

            outputs = model(inputs).logits
            test_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            test_true.extend(labels.cpu().numpy())

    test_metrics = compute_metrics(test_true, test_preds)
    print(f"Test accuracy: {test_metrics.get('accuracy', 0):.4f}")

    plot_confusion_matrix(test_true, test_preds)


if __name__ == "__main__":
    main()
