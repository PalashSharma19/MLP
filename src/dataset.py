from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class MalwareDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.dataframe = pd.read_csv(csv_path)
        self.filepaths = self.dataframe["filepath"].tolist()
        self.labels = self.dataframe["label_idx"].astype(int).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        label_idx = int(self.labels[idx])

        with Image.open(filepath) as image:
            image = image.convert("L").convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label_idx


def get_dataloaders(splits_dir, batch_size=32, num_workers=2):
    splits_path = Path(splits_dir)

    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = MalwareDataset(splits_path / "train.csv", transform=train_transform)
    val_dataset = MalwareDataset(splits_path / "val.csv", transform=val_transform)
    test_dataset = MalwareDataset(splits_path / "test.csv", transform=val_transform)

    loaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }

    return loaders


def get_num_classes(splits_dir):
    splits_path = Path(splits_dir)
    with (splits_path / "label_map.json").open("r", encoding="utf-8") as f:
        label_map = json.load(f)
    return len(label_map)


if __name__ == "__main__":
    dataloaders = get_dataloaders("data/splits")
    images, labels = next(iter(dataloaders["train"]))
    print(images.shape, labels.shape)