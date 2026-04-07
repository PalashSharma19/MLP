from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150


def _sanitize_name(name: str) -> str:
    simplified = name.lower()
    if "cnn" in simplified:
        return "CNN"
    if "random forest" in simplified or simplified == "rf":
        return "RF"
    if "svm" in simplified:
        return "SVM"
    if "k-nn" in simplified or "knn" in simplified:
        return "kNN"
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def plot_confusion_matrices(results_dict, label_map, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    class_names = [label_map[idx] for idx in sorted(label_map.keys())]

    saved = []
    for model_name, metrics in results_dict.items():
        y_true = np.asarray(metrics.get("all_labels", []), dtype=int)
        y_pred = np.asarray(metrics.get("all_preds", []), dtype=int)
        if y_true.size == 0 or y_pred.size == 0:
            continue

        matrix = np.zeros((len(class_names), len(class_names)), dtype=float)
        for true_idx, pred_idx in zip(y_true, y_pred):
            if 0 <= true_idx < len(class_names) and 0 <= pred_idx < len(class_names):
                matrix[true_idx, pred_idx] += 1.0

        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        normalized = matrix / row_sums

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            normalized,
            cmap="Blues",
            ax=ax,
            cbar=True,
            square=True,
            xticklabels=class_names,
            yticklabels=class_names,
        )
        ax.set_title(f"Confusion Matrix - {model_name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)
        fig.tight_layout()

        save_path = output_path / f"cm_{_sanitize_name(model_name)}.png"
        fig.savefig(save_path)
        plt.close(fig)
        saved.append(save_path)

    return saved


def plot_model_comparison(results_dict, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not results_dict:
        return None

    metric_names = ["accuracy", "f1_weighted", "f1_macro"]
    metric_labels = ["Accuracy", "F1 Weighted", "F1 Macro"]
    model_names = list(results_dict.keys())

    values = np.array(
        [[results_dict[name].get(metric, np.nan) for metric in metric_names] for name in model_names],
        dtype=float,
    )

    x = np.arange(len(metric_names))
    width = 0.8 / max(len(model_names), 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, model_name in enumerate(model_names):
        offsets = x - 0.4 + width / 2 + idx * width
        bars = ax.bar(offsets, values[idx], width=width, label=model_name)
        for bar in bars:
            height = bar.get_height()
            if np.isfinite(height):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.005,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0.0, 1.05)
    ax.set_title("CNN vs Traditional ML: Performance Comparison")
    ax.legend()
    fig.tight_layout()

    save_path = output_path / "model_comparison.png"
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_training_curves(log_path, output_dir):
    log_path = Path(log_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not log_path.exists():
        return None

    with log_path.open("r", encoding="utf-8") as f:
        logs = json.load(f)

    if not logs:
        return None

    df = pd.DataFrame(logs)
    phase1_count = int((df["phase"] == "phase1").sum())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(np.arange(1, len(df) + 1), df["train_loss"], label="Train Loss")
    axes[0].plot(np.arange(1, len(df) + 1), df["val_loss"], label="Val Loss")
    axes[0].axvline(phase1_count + 0.5, linestyle="--", color="gray", linewidth=1)
    axes[0].set_title("Loss over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(np.arange(1, len(df) + 1), df["train_acc"], label="Train Accuracy")
    axes[1].plot(np.arange(1, len(df) + 1), df["val_acc"], label="Val Accuracy")
    axes[1].axvline(phase1_count + 0.5, linestyle="--", color="gray", linewidth=1)
    axes[1].set_title("Accuracy over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend()

    fig.tight_layout()
    save_path = output_path / "training_curves.png"
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_sample_images(raw_dir, label_map, output_dir, n_per_class=1):
    del n_per_class
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    class_names = [label_map[idx] for idx in sorted(label_map.keys())]
    image_paths = []
    for class_name in class_names:
        # Support pre-split raw layout by searching train/val/test first, then fallback.
        candidates = list(raw_path.glob(f"train/{class_name}/*.png"))
        if not candidates:
            candidates = list(raw_path.glob(f"val/{class_name}/*.png"))
        if not candidates:
            candidates = list(raw_path.glob(f"test/{class_name}/*.png"))
        if not candidates:
            candidates = list(raw_path.glob(f"{class_name}/*.png"))
        image_paths.append(candidates[0] if candidates else None)

    n_cols = 5
    n_rows = int(math.ceil(len(class_names) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)

    for idx, class_name in enumerate(class_names):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        path = image_paths[idx]
        if path is not None and path.exists():
            with Image.open(path) as image:
                ax.imshow(image.convert("L"), cmap="gray")
        else:
            ax.text(0.5, 0.5, "Missing", ha="center", va="center")
        ax.set_title(class_name, fontsize=8)
        ax.axis("off")

    for idx in range(len(class_names), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")

    fig.tight_layout()
    save_path = output_path / "sample_images.png"
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def generate_all_plots(results_dict, splits_dir, log_path, raw_dir):
    output_dir = Path("outputs/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    splits_path = Path(splits_dir)
    with (splits_path / "label_map.json").open("r", encoding="utf-8") as f:
        forward_label_map = json.load(f)
    label_map = {int(idx): name for name, idx in forward_label_map.items()}

    saved_files = []

    for path in plot_confusion_matrices(results_dict, label_map, output_dir):
        print(f"Saved: {path}")
        saved_files.append(path)

    comparison_path = plot_model_comparison(results_dict, output_dir)
    if comparison_path is not None:
        print(f"Saved: {comparison_path}")
        saved_files.append(comparison_path)

    curves_path = plot_training_curves(log_path, output_dir)
    if curves_path is not None:
        print(f"Saved: {curves_path}")
        saved_files.append(curves_path)

    samples_path = plot_sample_images(raw_dir, label_map, output_dir)
    if samples_path is not None:
        print(f"Saved: {samples_path}")
        saved_files.append(samples_path)

    return saved_files


def _load_results_for_plots(results_csv_path, ml_results_path):
    results = {}

    results_csv_path = Path(results_csv_path)
    if results_csv_path.exists():
        df = pd.read_csv(results_csv_path)
        for _, row in df.iterrows():
            model_name = row["Model"]
            results[model_name] = {
                "accuracy": float(row.get("Test Accuracy", np.nan)),
                "f1_weighted": float(row.get("F1 Weighted", np.nan)),
                "f1_macro": float(row.get("F1 Macro", np.nan)),
                "all_preds": [],
                "all_labels": [],
            }

    ml_results_path = Path(ml_results_path)
    if ml_results_path.exists():
        with ml_results_path.open("r", encoding="utf-8") as f:
            ml_data = json.load(f)
        for model_name, payload in ml_data.items():
            if model_name not in results:
                results[model_name] = {}
            results[model_name]["accuracy"] = float(payload.get("test_accuracy", np.nan))
            report = payload.get("classification_report", {})
            weighted = report.get("weighted avg", {}) if isinstance(report, dict) else {}
            macro = report.get("macro avg", {}) if isinstance(report, dict) else {}
            results[model_name]["f1_weighted"] = float(weighted.get("f1-score", np.nan))
            results[model_name]["f1_macro"] = float(macro.get("f1-score", np.nan))
            results[model_name]["all_preds"] = payload.get("all_preds", [])
            results[model_name]["all_labels"] = payload.get("all_labels", [])

    return results


if __name__ == "__main__":
    results_dict = _load_results_for_plots("outputs/results.csv", "outputs/models/ml_results.json")

    with Path("data/splits/label_map.json").open("r", encoding="utf-8") as f:
        label_map_forward = json.load(f)
    _ = {int(v): k for k, v in label_map_forward.items()}

    generate_all_plots(
        results_dict=results_dict,
        splits_dir="data/splits",
        log_path="outputs/models/cnn_training_log.json",
        raw_dir="data/raw",
    )
    print("All plots saved to outputs/plots/")
