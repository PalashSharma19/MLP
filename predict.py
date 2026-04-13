from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.convert import binary_to_image
from src.features import extract_features_single
from src.model import build_resnet18


SUPPORTED_BINARY_EXTENSIONS = {".exe", ".dll", ".bat"}
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
SUPPORTED_EXTENSIONS = SUPPORTED_BINARY_EXTENSIONS | SUPPORTED_IMAGE_EXTENSIONS
MODEL_FILES = {
    "rf": Path("outputs/models/rf_model.pkl"),
    "svm": Path("outputs/models/svm_model.pkl"),
    "knn": Path("outputs/models/knn_model.pkl"),
}
DEFAULT_BENIGN_THRESHOLD = 0.80


def load_label_maps(label_map_path: Path) -> tuple[dict[str, int], dict[int, str]]:
    if not label_map_path.exists():
        raise FileNotFoundError(f"Missing label map: {label_map_path}")

    with label_map_path.open("r", encoding="utf-8") as f:
        forward = json.load(f)

    inverse = {int(idx): name for name, idx in forward.items()}
    return forward, inverse


def print_all_classes(index_to_name: dict[int, str]) -> None:
    print("Available classes:")
    for idx in sorted(index_to_name.keys()):
        print(f"{idx:02d}: {index_to_name[idx]}")


def _validate_input_file(file_path: Path) -> None:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Not a file: {file_path}")
    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported extension '{file_path.suffix}'. Allowed: {allowed}")


def _is_image_file(file_path: Path) -> bool:
    return file_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def _sample_image_path(file_path: Path, temp_dir: Path) -> Path:
    if _is_image_file(file_path):
        return file_path

    temp_png = temp_dir / "input.png"
    binary_to_image(file_path, temp_png)
    return temp_png


def _cnn_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def predict_cnn_result(file_path: Path, index_to_name: dict[int, str]) -> dict[str, Any]:
    checkpoint_path = Path("outputs/models/resnet18_best.pth")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing CNN checkpoint: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18(num_classes=len(index_to_name))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    with tempfile.TemporaryDirectory() as temp_dir:
        sample_path = _sample_image_path(file_path, Path(temp_dir))

        with Image.open(sample_path) as image:
            image = image.convert("L").convert("RGB")
            tensor = _cnn_transform()(image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    top_indices = np.argsort(probabilities)[::-1][:3]
    best_idx = int(top_indices[0])
    best_name = index_to_name.get(best_idx, str(best_idx))
    best_prob = float(probabilities[best_idx])

    top_predictions = [
        {
            "rank": rank,
            "index": int(idx),
            "label": index_to_name.get(int(idx), str(int(idx))),
            "probability": float(probabilities[int(idx)]),
        }
        for rank, idx in enumerate(top_indices, start=1)
    ]

    return {
        "model": "cnn",
        "input_kind": "image" if _is_image_file(file_path) else "binary",
        "prediction_type": "malware-family",
        "verdict": "malware",
        "predicted_index": best_idx,
        "predicted_class": best_name,
        "confidence": best_prob,
        "top_predictions": top_predictions,
        "note": "Current trained models are malware-family classifiers. They do not yet produce a true benign-vs-malware verdict.",
    }


def predict_with_cnn(file_path: Path, index_to_name: dict[int, str]) -> None:
    result = predict_cnn_result(file_path, index_to_name)
    print(f"Predicted class: {result['predicted_class']} ({result['confidence']:.2%})")
    print("Top-3 predictions:")
    for item in result["top_predictions"]:
        print(f"{item['rank']}. {item['label']}: {item['probability']:.2%}")


def _confidence_from_model(model, feature_vector: np.ndarray, prediction_idx: int) -> float | None:
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(feature_vector)
            return float(np.max(proba))
        except Exception:
            return None
    return None


def _apply_threshold_heuristic(result: dict[str, Any], benign_threshold: float) -> dict[str, Any]:
    threshold = float(np.clip(benign_threshold, 0.0, 1.0))
    confidence = result.get("confidence")

    if confidence is None:
        result["heuristic_verdict"] = "unknown"
        result["heuristic_reason"] = "Model did not provide confidence score for thresholding."
    elif float(confidence) < threshold:
        result["heuristic_verdict"] = "benign"
        result["heuristic_reason"] = (
            f"Confidence {float(confidence):.2%} is below threshold {threshold:.2%}."
        )
    else:
        result["heuristic_verdict"] = "malware"
        result["heuristic_reason"] = (
            f"Confidence {float(confidence):.2%} meets/exceeds threshold {threshold:.2%}."
        )

    result["heuristic_threshold"] = threshold
    result["verdict"] = result["heuristic_verdict"]
    result["note"] = (
        "Demo heuristic: low confidence is treated as benign. "
        "This is not a true benign classifier and can miss novel malware."
    )
    return result


def predict_ml_result(file_path: Path, model_name: str, index_to_name: dict[int, str]) -> dict[str, Any]:
    model_path = MODEL_FILES[model_name]
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    model = joblib.load(model_path)

    with tempfile.TemporaryDirectory() as temp_dir:
        sample_path = _sample_image_path(file_path, Path(temp_dir))
        features = extract_features_single(sample_path).astype(np.float32).reshape(1, -1)

    prediction = model.predict(features)
    pred_idx = int(prediction[0])
    pred_name = index_to_name.get(pred_idx, str(pred_idx))

    confidence = _confidence_from_model(model, features, pred_idx)
    return {
        "model": model_name,
        "input_kind": "image" if _is_image_file(file_path) else "binary",
        "prediction_type": "malware-family",
        "verdict": "malware",
        "predicted_index": pred_idx,
        "predicted_class": pred_name,
        "confidence": confidence,
        "note": "Current trained models are malware-family classifiers. They do not yet produce a true benign-vs-malware verdict.",
    }


def predict_with_ml(file_path: Path, model_name: str, index_to_name: dict[int, str]) -> None:
    result = predict_ml_result(file_path, model_name, index_to_name)
    if result["confidence"] is not None:
        print(f"Predicted class: {result['predicted_class']} ({float(result['confidence']):.2%})")
    else:
        print(f"Predicted class: {result['predicted_class']}")


def predict_file(
    file_path: str | Path,
    model_name: str = "cnn",
    splits_dir: str | Path = "data/splits",
    benign_threshold: float = DEFAULT_BENIGN_THRESHOLD,
) -> dict[str, Any]:
    target_file = Path(file_path)
    _validate_input_file(target_file)

    _, index_to_name = load_label_maps(Path(splits_dir) / "label_map.json")

    if model_name == "cnn":
        result = predict_cnn_result(target_file, index_to_name)
        return _apply_threshold_heuristic(result, benign_threshold)
    if model_name in MODEL_FILES:
        result = predict_ml_result(target_file, model_name, index_to_name)
        return _apply_threshold_heuristic(result, benign_threshold)
    raise ValueError(f"Unsupported model: {model_name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-file malware family inference")
    parser.add_argument("--file", type=str, help="Path to .exe/.dll/.bat file")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "rf", "svm", "knn"])
    parser.add_argument("--benign-threshold", type=float, default=DEFAULT_BENIGN_THRESHOLD)
    parser.add_argument("--list-classes", action="store_true", help="Print class list and exit")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    label_map_path = Path("data/splits/label_map.json")
    _, index_to_name = load_label_maps(label_map_path)

    if args.list_classes:
        print_all_classes(index_to_name)
        return

    if not args.file:
        raise ValueError("--file is required unless --list-classes is provided")

    target_file = Path(args.file)
    _validate_input_file(target_file)
    result = predict_file(
        target_file,
        model_name=args.model,
        splits_dir="data/splits",
        benign_threshold=args.benign_threshold,
    )
    confidence = result.get("confidence")
    if confidence is not None:
        print(f"Predicted class: {result['predicted_class']} ({float(confidence):.2%})")
    else:
        print(f"Predicted class: {result['predicted_class']}")
    print(f"Heuristic verdict: {result['heuristic_verdict']}")
    print(f"Heuristic reason: {result['heuristic_reason']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Prediction failed: {exc}")
        sys.exit(1)
