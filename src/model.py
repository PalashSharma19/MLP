from __future__ import annotations

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, ResNet18_Weights, efficientnet_b0, resnet18


def build_resnet18(num_classes, dropout=0.5):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, num_classes))
    return model


def build_efficientnet(num_classes, dropout=0.5):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(1280, num_classes))
    return model


def build_baseline_cnn(num_classes):
    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Linear(128 * 4 * 4, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes),
    )
    return model


def unfreeze_all(model):
    for parameter in model.parameters():
        parameter.requires_grad = True
    return model


def _count_params(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


if __name__ == "__main__":
    num_classes = 25

    baseline = build_baseline_cnn(num_classes)
    resnet = build_resnet18(num_classes)
    efficientnet = build_efficientnet(num_classes)

    for name, model in [
        ("baseline_cnn", baseline),
        ("resnet18", resnet),
        ("efficientnet_b0", efficientnet),
    ]:
        total, trainable = _count_params(model)
        print(f"{name}: total={total:,} trainable={trainable:,}")
