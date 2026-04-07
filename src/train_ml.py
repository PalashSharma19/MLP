from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import build_all_features


def train_random_forest(X_train, y_train, X_val, y_val):
    start_time = time.time()
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    val_predictions = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)
    time_taken = time.time() - start_time
    return model, val_accuracy, time_taken


def train_svm(X_train, y_train, X_val, y_val):
    print("Training SVM (this may take several minutes on the full dataset)...")
    start_time = time.time()
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    val_predictions = pipeline.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)
    time_taken = time.time() - start_time
    return pipeline, val_accuracy, time_taken


def train_knn(X_train, y_train, X_val, y_val):
    start_time = time.time()
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=5, metric="euclidean", n_jobs=-1)),
        ]
    )
    pipeline.fit(X_train, y_train)
    val_predictions = pipeline.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)
    time_taken = time.time() - start_time
    return pipeline, val_accuracy, time_taken


def _save_model(model, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


def _evaluate_model(model, X_test, y_test):
    test_predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, test_predictions)
    report = classification_report(y_test, test_predictions, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, test_predictions)
    return {
        "accuracy": float(accuracy),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "all_preds": test_predictions.astype(int).tolist(),
        "all_labels": np.asarray(y_test, dtype=int).tolist(),
    }


def train_all_ml(splits_dir="data/splits", output_dir="outputs/models"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    feature_data = build_all_features(splits_dir)
    X_train, y_train = feature_data["train"]
    X_val, y_val = feature_data["val"]
    X_test, y_test = feature_data["test"]

    results = {}

    print("Training Random Forest...")
    rf_model, rf_val_acc, rf_time = train_random_forest(X_train, y_train, X_val, y_val)
    _save_model(rf_model, output_path / "rf_model.pkl")
    rf_test = _evaluate_model(rf_model, X_test, y_test)
    results["Random Forest"] = {
        "val_accuracy": float(rf_val_acc),
        "test_accuracy": rf_test["accuracy"],
        "train_time": float(rf_time),
        "classification_report": rf_test["classification_report"],
        "confusion_matrix": rf_test["confusion_matrix"],
        "all_preds": rf_test["all_preds"],
        "all_labels": rf_test["all_labels"],
    }

    print("Training SVM...")
    svm_model, svm_val_acc, svm_time = train_svm(X_train, y_train, X_val, y_val)
    _save_model(svm_model, output_path / "svm_model.pkl")
    svm_test = _evaluate_model(svm_model, X_test, y_test)
    results["SVM (RBF)"] = {
        "val_accuracy": float(svm_val_acc),
        "test_accuracy": svm_test["accuracy"],
        "train_time": float(svm_time),
        "classification_report": svm_test["classification_report"],
        "confusion_matrix": svm_test["confusion_matrix"],
        "all_preds": svm_test["all_preds"],
        "all_labels": svm_test["all_labels"],
    }

    print("Training k-NN...")
    knn_model, knn_val_acc, knn_time = train_knn(X_train, y_train, X_val, y_val)
    _save_model(knn_model, output_path / "knn_model.pkl")
    knn_test = _evaluate_model(knn_model, X_test, y_test)
    results["k-NN"] = {
        "val_accuracy": float(knn_val_acc),
        "test_accuracy": knn_test["accuracy"],
        "train_time": float(knn_time),
        "classification_report": knn_test["classification_report"],
        "confusion_matrix": knn_test["confusion_matrix"],
        "all_preds": knn_test["all_preds"],
        "all_labels": knn_test["all_labels"],
    }

    results_path = output_path / "ml_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    table = pd.DataFrame(
        [
            {
                "Model": model_name,
                "Val Acc": metrics["val_accuracy"],
                "Test Acc": metrics["test_accuracy"],
                "Train Time": metrics["train_time"],
            }
            for model_name, metrics in results.items()
        ]
    )

    print("\nML comparison table")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    return results


if __name__ == "__main__":
    train_all_ml()
    print("ML training complete.")
