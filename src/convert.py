from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


SplitResult = Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]


def binary_to_image(filepath: str | Path, output_path: str | Path) -> Path:
    """Convert a binary file into a grayscale PNG image with fixed width."""
    src = Path(filepath)
    dst = Path(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    raw = src.read_bytes()
    data = np.frombuffer(raw, dtype=np.uint8)

    width = 256
    height = int(math.ceil(len(data) / width)) if len(data) else 1
    padded = np.zeros(height * width, dtype=np.uint8)
    padded[: len(data)] = data
    image_array = padded.reshape((height, width))

    Image.fromarray(image_array, mode="L").save(dst)
    return dst


def _is_presplit_layout(raw_path: Path) -> bool:
    expected = [raw_path / "train", raw_path / "val", raw_path / "test"]
    if not all(p.is_dir() for p in expected):
        return False

    # Require at least one class folder in each split to treat it as pre-split.
    return all(any(child.is_dir() for child in p.iterdir()) for p in expected)


def _collect_from_class_folders(root: Path) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    for class_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        label = class_dir.name
        for img in class_dir.rglob("*.png"):
            if img.is_file():
                records.append((str(img.resolve()), label))
    return records


def _collect_from_presplit(raw_path: Path) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    train_records = _collect_from_class_folders(raw_path / "train")
    val_records = _collect_from_class_folders(raw_path / "val")
    test_records = _collect_from_class_folders(raw_path / "test")
    return train_records, val_records, test_records


def _records_to_df(records: List[Tuple[str, str]], label_map: Dict[str, int]) -> pd.DataFrame:
    df = pd.DataFrame(records, columns=["filepath", "label"])
    df["label_idx"] = df["label"].map(label_map).astype(int)
    return df


def _print_summary(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    total = len(train_df) + len(val_df) + len(test_df)
    print(f"Total samples: {total}")
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples:   {len(val_df)}")
    print(f"Test samples:  {len(test_df)}")

    merged = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
    print("Samples per class:")
    for label, count in merged["label"].value_counts().sort_index().items():
        print(f"  {label}: {count}")


def build_dataset_splits(
    raw_dir: str | Path,
    output_dir: str | Path,
    splits: Sequence[float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> SplitResult:
    """Build train/val/test CSVs and label map from raw dataset folders.

    Supports two layouts:
    1) Standard layout: raw_dir/<class_name>/*.png
       -> performs stratified split using 'splits'.
    2) Pre-split layout: raw_dir/train|val|test/<class_name>/*.png
       -> uses existing split folders as-is.
    """
    if len(splits) != 3 or not math.isclose(sum(splits), 1.0, rel_tol=1e-6):
        raise ValueError("splits must contain exactly 3 values that sum to 1.0")

    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists() or not raw_path.is_dir():
        raise FileNotFoundError(f"Raw directory not found: {raw_path}")

    if _is_presplit_layout(raw_path):
        print("Detected pre-split dataset layout (train/val/test).")
        train_records, val_records, test_records = _collect_from_presplit(raw_path)

        all_labels = sorted(
            set([label for _, label in train_records + val_records + test_records])
        )
        label_map = {label: idx for idx, label in enumerate(all_labels)}

        train_df = _records_to_df(train_records, label_map)
        val_df = _records_to_df(val_records, label_map)
        test_df = _records_to_df(test_records, label_map)
    else:
        print("Detected class-folder dataset layout. Building stratified splits.")
        records = _collect_from_class_folders(raw_path)
        if not records:
            raise RuntimeError(f"No .png files found under {raw_path}")

        labels = sorted({label for _, label in records})
        label_map = {label: idx for idx, label in enumerate(labels)}

        df = pd.DataFrame(records, columns=["filepath", "label"])
        df["label_idx"] = df["label"].map(label_map).astype(int)

        train_size, val_size, test_size = splits
        train_df, temp_df = train_test_split(
            df,
            test_size=(1.0 - train_size),
            random_state=seed,
            stratify=df["label_idx"],
        )

        val_ratio_in_temp = val_size / (val_size + test_size)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1.0 - val_ratio_in_temp),
            random_state=seed,
            stratify=temp_df["label_idx"],
        )

        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

    train_csv = out_path / "train.csv"
    val_csv = out_path / "val.csv"
    test_csv = out_path / "test.csv"

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    label_map_path = out_path / "label_map.json"
    with label_map_path.open("w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    _print_summary(train_df, val_df, test_df)
    return train_df, val_df, test_df, label_map


if __name__ == "__main__":
    train_df, val_df, test_df, label_map = build_dataset_splits("data/raw", "data/splits")

    print("\nLabel map:")
    for class_name, class_idx in sorted(label_map.items(), key=lambda x: x[1]):
        print(f"  {class_idx:02d}: {class_name}")
