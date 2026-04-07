from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision  # noqa: F401
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import get_dataloaders, get_num_classes
from src.model import build_resnet18, unfreeze_all


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(loader, desc="Train", leave=False)
    for images, labels in progress:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += batch_size

        progress.set_postfix(loss=loss.item())

    avg_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Eval", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += batch_size

        all_preds.extend(predictions.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy, np.asarray(all_preds, dtype=int), np.asarray(all_labels, dtype=int)


def _save_checkpoint(path, model, best_val_acc, num_classes, phase):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "best_val_acc": best_val_acc,
        "num_classes": num_classes,
        "phase": phase,
    }
    torch.save(checkpoint, path)


def _format_summary_row(row):
    return (
        f"{row['phase']:<7} "
        f"{row['epoch']:>5} "
        f"{row['train_loss']:.4f} "
        f"{row['train_acc']:.4f} "
        f"{row['val_loss']:.4f} "
        f"{row['val_acc']:.4f}"
    )


def train_cnn(
    splits_dir="data/splits",
    output_dir="outputs/models",
    phase1_epochs=10,
    phase2_epochs=15,
    batch_size=32,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataloaders = get_dataloaders(splits_dir, batch_size=batch_size, num_workers=0)
    num_classes = get_num_classes(splits_dir)

    criterion = nn.CrossEntropyLoss()
    logs = []
    best_val_acc = 0.0

    # Phase 1: train the classification head only.
    model = build_resnet18(num_classes).to(device)
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    phase1_best = 0.0
    phase1_checkpoint = output_path / "resnet18_phase1.pth"

    for epoch in range(1, phase1_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, dataloaders["train"], criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, dataloaders["val"], criterion, device)
        scheduler.step()

        logs.append(
            {
                "epoch": epoch,
                "phase": "phase1",
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
            }
        )

        print(
            f"[phase1] epoch {epoch:02d}/{phase1_epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > phase1_best:
            phase1_best = val_acc
            _save_checkpoint(phase1_checkpoint, model, phase1_best, num_classes, "phase1")

    # Phase 2: fine-tune the full model.
    if phase1_checkpoint.exists():
        checkpoint = torch.load(phase1_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    model = unfreeze_all(model).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(phase2_epochs, 1))

    phase2_best = 0.0
    final_checkpoint = output_path / "resnet18_best.pth"

    for epoch in range(1, phase2_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, dataloaders["train"], criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, dataloaders["val"], criterion, device)
        scheduler.step()

        logs.append(
            {
                "epoch": epoch,
                "phase": "phase2",
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
            }
        )

        print(
            f"[phase2] epoch {epoch:02d}/{phase2_epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > phase2_best:
            phase2_best = val_acc
            _save_checkpoint(final_checkpoint, model, phase2_best, num_classes, "phase2")

    best_val_acc = max(phase1_best, phase2_best)

    log_path = output_path / "cnn_training_log.json"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)

    print("\nTraining summary")
    print("phase   epoch train_loss train_acc val_loss val_acc")
    for row in logs:
        print(_format_summary_row(row))
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")

    return best_val_acc


if __name__ == "__main__":
    train_cnn()
    print("CNN training complete.")
