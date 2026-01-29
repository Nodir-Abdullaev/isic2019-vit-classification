import pandas as pd
from sklearn.model_selection import train_test_split
import sys

# Add 'src' directory to the module search path to import config
sys.path.append('./src')
import config


def run_preprocessing():
    """
    Main preprocessing pipeline:
    1. Loads the raw GroundTruth CSV.
    2. Cleans the labels (removes UNK, converts One-Hot to Class Index).
    3. Verifies image existence on the external drive.
    4. Performs a stratified split for reproducibility.
    5. Saves normalized annotations compatible with dataset.py
       (image_id, label).
    """

    # 0. Sanity check input paths
    config.check_input_paths()

    print("Loading source CSV...")
    df = pd.read_csv(config.SOURCE_CSV_PATH)

    # Kaggle CSV uses column name "image"
    if "image" not in df.columns:
        raise ValueError("Expected column 'image' not found in source CSV")

    # Rename immediately to enforce project-wide contract
    df = df.rename(columns={"image": "image_id"})

    # 1. Remove UNK class
    if "UNK" in df.columns:
        df = df[df["UNK"] == 0].drop(columns=["UNK"])
        print("UNK class removed.")

    # 2. Define class columns (all except image_id)
    class_columns = [c for c in df.columns if c != "image_id"]

    if len(class_columns) != config.NUM_CLASSES:
        print(
            f"Warning: NUM_CLASSES={config.NUM_CLASSES}, "
            f"but found {len(class_columns)} class columns"
        )

    class_to_idx = {cls: idx for idx, cls in enumerate(class_columns)}

    # One-hot → single numeric label
    df["label"] = df[class_columns].idxmax(axis=1).map(class_to_idx)

    # Keep only normalized columns
    df = df[["image_id", "label"]]

    # 3. Verify image existence
    print(f"Checking image files in {config.IMAGES_DIR}...")
    df["exists"] = df["image_id"].apply(
        lambda x: (config.IMAGES_DIR / f"{x}.jpg").exists()
    )

    missing_count = (~df["exists"]).sum()
    if missing_count > 0:
        print(f"Warning: {missing_count} images not found. Removing them.")
        df = df[df["exists"]]

    df = df.drop(columns=["exists"]).reset_index(drop=True)

    # 4. Stratified split: 80 / 10 / 10
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=config.SEED,
        stratify=df["label"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=config.SEED,
        stratify=temp_df["label"],
    )

    # 5. Save normalized annotations
    train_df.to_csv(config.TRAIN_CSV, index=False)
    val_df.to_csv(config.VAL_CSV, index=False)
    test_df.to_csv(config.TEST_CSV, index=False)

    print("-" * 40)
    print("Preprocessing completed successfully.")
    print(
        f"Train: {len(train_df)} | "
        f"Val: {len(val_df)} | "
        f"Test: {len(test_df)}"
    )
    print(f"Saved to: {config.ANNOTATIONS_DIR}")


if __name__ == "__main__":
    run_preprocessing()
