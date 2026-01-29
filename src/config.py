"""
Global configuration file for ISIC 2019 Skin Lesion Classification project.

This file is the SINGLE source of truth for:
- dataset paths
- experiment parameters
- reproducibility settings

It must be editable without touching the rest of the codebase.
"""

import torch
from pathlib import Path

# =========================
# Project root
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# =========================
# 1. INPUT DATA (Read-Only)
# =========================
# Absolute paths to your local storage (D: Drive)
IMAGES_DIR = Path(r"D:\AAAAAAAAAAAAAAAA\Dataset\ISIC-images2019")
SOURCE_CSV_PATH = Path(r"D:\AAAAAAAAAAAAAAAA\Dataset\ISIC_2019_Training_GroundTruth.csv")

# =========================
# 2. OUTPUT ARTIFACTS (To be generated)
# =========================
# These files DO NOT exist yet. They will be created by src/data_preprocessing.py
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"

TRAIN_CSV = ANNOTATIONS_DIR / "train.csv"
VAL_CSV = ANNOTATIONS_DIR / "val.csv"
TEST_CSV = ANNOTATIONS_DIR / "test.csv"

# =========================
# Model & Training Parameters
# =========================

# Classes: MEL, NV, BCC, AK, BKL, DF, VASC, SCC
# 'UNK' is removed during preprocessing, so we have 8 classes.
NUM_CLASSES = 8 
CLASS_NAMES = [
    "MEL",
    "NV",
    "BCC",
    "AK",
    "BKL",
    "DF",
    "VASC",
    "SCC",
]

IMAGE_SIZE = 224

# Hardware Settings
BATCH_SIZE = 32 # Safe for RTX 3070 (8GB VRAM)
NUM_WORKERS = 6 # Set to physical cores count

# Device selection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False

# =========================
# Reproducibility
# =========================

SEED = 42

# =========================
# Runtime Checks
# =========================

def check_input_paths():
    """
    Verifies that essential READ-ONLY paths exist.
    Run this at the start of preprocessing or training.
    """
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"CRITICAL: Images directory not found at {IMAGES_DIR}")
    
    if not SOURCE_CSV_PATH.exists():
        raise FileNotFoundError(f"CRITICAL: Source CSV not found at {SOURCE_CSV_PATH}")

    # Create output directory for annotations if it doesn't exist
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"--- Configuration Loaded ---")
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Input source: {SOURCE_CSV_PATH}")
