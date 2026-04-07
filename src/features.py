from __future__ import annotations

import os
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import hog
from skimage.measure import shannon_entropy
from tqdm import tqdm


def extract_features_single(image_path):
    image_path = Path(image_path)

    with Image.open(image_path) as image:
        image = image.convert("L")
        img = np.array(image, dtype=np.uint8)

    resized = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)

    hog_features = hog(
        resized,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True,
    )

    mean_value = float(np.mean(resized))
    std_value = float(np.std(resized))
    entropy_value = float(shannon_entropy(resized))
    zero_ratio = float(np.mean(resized == 0))
    byte_uniformity = float(np.clip(1.0 - (std_value / 128.0), 0.0, 1.0))

    extra_features = np.array(
        [mean_value, std_value, entropy_value, zero_ratio, byte_uniformity],
        dtype=np.float32,
    )

    return np.concatenate([hog_features.astype(np.float32), extra_features])


def extract_features_from_csv(csv_path, cache_path=None):
    csv_path = Path(csv_path)
    cache_path = Path(cache_path) if cache_path is not None else None

    if cache_path is not None and cache_path.exists():
        cached = joblib.load(cache_path)
        return cached["X"], cached["y"]

    dataframe = pd.read_csv(csv_path)
    features = []
    labels = []

    for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc=f"Extracting {csv_path.stem}"):
        features.append(extract_features_single(row["filepath"]))
        labels.append(int(row["label_idx"]))

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=int)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"X": X, "y": y}, cache_path)

    return X, y


def build_all_features(splits_dir, cache_dir="data/splits"):
    splits_path = Path(splits_dir)
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    train_cache = cache_path / "features_train.pkl"
    val_cache = cache_path / "features_val.pkl"
    test_cache = cache_path / "features_test.pkl"

    train = extract_features_from_csv(splits_path / "train.csv", cache_path=train_cache)
    val = extract_features_from_csv(splits_path / "val.csv", cache_path=val_cache)
    test = extract_features_from_csv(splits_path / "test.csv", cache_path=test_cache)

    return {"train": train, "val": val, "test": test}


if __name__ == "__main__":
    feature_data = build_all_features("data/splits")
    X_train, _ = feature_data["train"]
    X_val, _ = feature_data["val"]
    X_test, _ = feature_data["test"]
    print(X_train.shape, X_val.shape, X_test.shape)
