from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import get_dataloaders, get_num_classes
from src.features import build_all_features
from src.model import build_resnet18


def _safe_f1(y_true, y_pred, average):
    return float(f1_score(y_true, y_pred, average=average, zero_division=0))


def _safe_roc_auc(y_true, y_pred_or_proba):
    try:
        if y_pred_or_proba.ndim == 1:
            return float(roc_auc_score(y_true, y_pred_or_proba))
        return float(roc_auc_score(y_true, y_pred_or_proba, multi_class="ovr"))
    except Exception:
        return None


def _load_label_map(splits_dir: str | Path) -> Dict[int, str]:
    splits_path = Path(splits_dir)
    with (splits_path / "label_map.json").open("r", encoding="utf-8") as f:
        label_map = json.load(f)
    return {int(idx): name for name, idx in label_map.items()}


def evaluate_cnn(model_path, splits_dir, device=None):
    model_path = Path(model_path)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if not model_path.exists():
        raise FileNotFoundError(f"CNN checkpoint not found: {model_path}")

    num_classes = get_num_classes(splits_dir)
    checkpoint = torch.load(model_path, map_location=device)
    model = build_resnet18(num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    dataloaders = get_dataloaders(splits_dir, batch_size=32, num_workers=0)
    test_loader = dataloaders["test"]

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            all_preds.extend(predictions.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    all_preds_array = np.asarray(all_preds, dtype=int)
    all_labels_array = np.asarray(all_labels, dtype=int)
    return {
        "accuracy": float(accuracy_score(all_labels_array, all_preds_array)),
        "f1_weighted": _safe_f1(all_labels_array, all_preds_array, average="weighted"),
        "f1_macro": _safe_f1(all_labels_array, all_preds_array, average="macro"),
        "confusion_matrix": confusion_matrix(all_labels_array, all_preds_array).tolist(),
        "all_preds": all_preds_array.tolist(),
        "all_labels": all_labels_array.tolist(),
        "classification_report": classification_report(all_labels_array, all_preds_array, zero_division=0),
        "roc_auc": None,
    }


def evaluate_ml_model(model_path, X_test, y_test):
    model = joblib.load(model_path)
    predictions = model.predict(X_test)

    probabilities = None
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(X_test)
        except Exception:
            probabilities = None

    all_preds = np.asarray(predictions, dtype=int)
    all_labels = np.asarray(y_test, dtype=int)
    roc_auc = _safe_roc_auc(all_labels, probabilities if probabilities is not None else all_preds)

    return {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "f1_weighted": _safe_f1(all_labels, all_preds, average="weighted"),
        "f1_macro": _safe_f1(all_labels, all_preds, average="macro"),
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
        "all_preds": all_preds.tolist(),
        "all_labels": all_labels.tolist(),
        "classification_report": classification_report(all_labels, all_preds, zero_division=0),
        "roc_auc": roc_auc,
    }


def _format_results_table(results_df: pd.DataFrame) -> str:
    if results_df.empty:
        return "No evaluation results available."
    return results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}")


def run_full_evaluation(splits_dir="data/splits", models_dir="outputs/models"):
    splits_path = Path(splits_dir)
    models_path = Path(models_dir)

    results: Dict[str, Dict[str, Any]] = {}
    label_map = _load_label_map(splits_path)

    cnn_path = models_path / "resnet18_best.pth"
    if cnn_path.exists():
        print("Evaluating CNN...")
        results["CNN ResNet-18"] = evaluate_cnn(cnn_path, splits_path)
    else:
        print(f"Skipping CNN evaluation; missing checkpoint: {cnn_path}")

    feature_data = build_all_features(splits_path)
    _, _ = feature_data["train"]
    X_test, y_test = feature_data["test"]

    ml_models = {
        "Random Forest": models_path / "rf_model.pkl",
        "SVM (RBF)": models_path / "svm_model.pkl",
        "k-NN": models_path / "knn_model.pkl",
    }

    for model_name, model_path in ml_models.items():
        if model_path.exists():
            print(f"Evaluating {model_name}...")
            results[model_name] = evaluate_ml_model(model_path, X_test, y_test)
        else:
            print(f"Skipping {model_name}; missing model file: {model_path}")

    if not results:
        print("No trained models found. Evaluation skipped.")
        return {}

    results_df = pd.DataFrame(
        [
            {
                "Model": model_name,
                "Test Accuracy": metrics["accuracy"],
                "F1 Weighted": metrics["f1_weighted"],
                "F1 Macro": metrics["f1_macro"],
            }
            for model_name, metrics in results.items()
        ]
    )

    winner_columns = []
    for metric_name in ["Test Accuracy", "F1 Weighted", "F1 Macro"]:
        best_model = results_df.loc[results_df[metric_name].idxmax(), "Model"]
        winner_columns.append((metric_name, best_model))

    def _winner_marker(model_name: str) -> str:
        winners = [metric for metric, best_model in winner_columns if best_model == model_name]
        return "*" if winners else ""

    results_df["Winner"] = results_df["Model"].map(_winner_marker)

    results_path = Path("outputs/results.csv")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)

    print("\nEvaluation results")
    print(_format_results_table(results_df))

    cnn_key = "CNN ResNet-18"
    best_ml_key = None
    ml_candidates = {name: metrics for name, metrics in results.items() if name != cnn_key}
    if ml_candidates:
        best_ml_key = max(ml_candidates, key=lambda name: ml_candidates[name]["accuracy"])

    if cnn_key in results:
        print(f"\nClassification report - {cnn_key}")
        print(results[cnn_key]["classification_report"])

    if best_ml_key is not None:
        print(f"\nClassification report - {best_ml_key}")
        print(results[best_ml_key]["classification_report"])

    return {
        "results_df": results_df,
        "results": results,
        "label_map": label_map,
        "best_ml_model": best_ml_key,
        "feature_cache": str(Path(splits_path) / "features_test.pkl"),
        "y_test": y_test.tolist(),
    }


if __name__ == "__main__":
    run_full_evaluation()
    print("Evaluation complete.")
