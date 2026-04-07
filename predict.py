from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.convert import binary_to_image
from src.features import extract_features_single
from src.model import build_resnet18


SUPPORTED_EXTENSIONS = {".exe", ".dll", ".bat"}
MODEL_FILES = {
    "rf": Path("outputs/models/rf_model.pkl"),
    "svm": Path("outputs/models/svm_model.pkl"),
    "knn": Path("outputs/models/knn_model.pkl"),
}


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


def _cnn_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def predict_with_cnn(file_path: Path, index_to_name: dict[int, str]) -> None:
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
        temp_png = Path(temp_dir) / "input.png"
        binary_to_image(file_path, temp_png)

        with Image.open(temp_png) as image:
            image = image.convert("L").convert("RGB")
            tensor = _cnn_transform()(image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    top_indices = np.argsort(probabilities)[::-1][:3]
    best_idx = int(top_indices[0])
    best_name = index_to_name.get(best_idx, str(best_idx))
    best_prob = float(probabilities[best_idx])

    print(f"Predicted class: {best_name} ({best_prob:.2%})")
    print("Top-3 predictions:")
    for rank, idx in enumerate(top_indices, start=1):
        class_name = index_to_name.get(int(idx), str(int(idx)))
        print(f"{rank}. {class_name}: {float(probabilities[int(idx)]):.2%}")


def _confidence_from_model(model, feature_vector: np.ndarray, prediction_idx: int) -> float | None:
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(feature_vector)
            return float(np.max(proba))
        except Exception:
            return None
    return None


def predict_with_ml(file_path: Path, model_name: str, index_to_name: dict[int, str]) -> None:
    model_path = MODEL_FILES[model_name]
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    model = joblib.load(model_path)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_png = Path(temp_dir) / "input.png"
        binary_to_image(file_path, temp_png)
        features = extract_features_single(temp_png).astype(np.float32).reshape(1, -1)

    prediction = model.predict(features)
    pred_idx = int(prediction[0])
    pred_name = index_to_name.get(pred_idx, str(pred_idx))

    confidence = _confidence_from_model(model, features, pred_idx)
    if confidence is not None:
        print(f"Predicted class: {pred_name} ({confidence:.2%})")
    else:
        print(f"Predicted class: {pred_name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-file malware family inference")
    parser.add_argument("--file", type=str, help="Path to .exe/.dll/.bat file")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "rf", "svm", "knn"])
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

    if args.model == "cnn":
        predict_with_cnn(target_file, index_to_name)
    else:
        predict_with_ml(target_file, args.model, index_to_name)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Prediction failed: {exc}")
        sys.exit(1)
