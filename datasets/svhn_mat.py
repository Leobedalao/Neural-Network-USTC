"""SVHN dataset utilities for MAT files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset

from utils.io import ensure_dir, load_npy, save_npy


def remap_labels(labels: np.ndarray) -> np.ndarray:
    """Map the SVHN label 10 back to 0."""
    remapped = labels.astype(np.int64).reshape(-1)
    remapped[remapped == 10] = 0
    return remapped


def load_svhn_mat(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load images and labels from an SVHN MAT file."""
    mat = loadmat(path)
    images = mat["X"].transpose(3, 2, 0, 1).astype(np.float32) / 255.0
    labels = remap_labels(mat["y"])
    return images, labels


def build_label_histogram(labels: np.ndarray, num_classes: int) -> dict[str, int]:
    """Count samples per class using string keys for JSON compatibility."""
    counts = np.bincount(labels.astype(np.int64), minlength=num_classes)
    return {str(index): int(value) for index, value in enumerate(counts.tolist())}


def build_split_stats(
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    test_labels: np.ndarray,
    num_classes: int,
) -> dict[str, dict[str, int | dict[str, int]]]:
    """Build dataset statistics for train, validation, and test splits."""
    return {
        "train": {
            "num_samples": int(len(train_labels)),
            "class_counts": build_label_histogram(train_labels, num_classes),
        },
        "val": {
            "num_samples": int(len(val_labels)),
            "class_counts": build_label_histogram(val_labels, num_classes),
        },
        "test": {
            "num_samples": int(len(test_labels)),
            "class_counts": build_label_histogram(test_labels, num_classes),
        },
    }


def split_train_val(
    labels: np.ndarray,
    val_ratio: float,
    seed: int,
    split_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Create or load deterministic train/validation indices."""
    split_dir = Path(split_dir)
    ensure_dir(split_dir)
    train_idx_path = split_dir / "train_idx.npy"
    val_idx_path = split_dir / "val_idx.npy"

    if train_idx_path.exists() and val_idx_path.exists():
        return load_npy(train_idx_path), load_npy(val_idx_path)

    rng = np.random.default_rng(seed)
    indices = np.arange(len(labels))
    rng.shuffle(indices)

    val_size = int(len(labels) * val_ratio)
    val_indices = np.sort(indices[:val_size])
    train_indices = np.sort(indices[val_size:])

    save_npy(train_idx_path, train_indices)
    save_npy(val_idx_path, val_indices)
    return train_indices, val_indices


class SVHNMatDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Torch dataset for preloaded SVHN MAT arrays."""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        normalize: bool = False,
        mean: Iterable[float] | None = None,
        std: Iterable[float] | None = None,
    ) -> None:
        self.images = torch.from_numpy(images).float()
        self.labels = torch.from_numpy(labels).long()
        self.normalize = normalize
        self.mean = torch.tensor(list(mean or [0.5, 0.5, 0.5]), dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(list(std or [0.5, 0.5, 0.5]), dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[index]
        if self.normalize:
            image = (image - self.mean) / self.std
        label = self.labels[index]
        return image, label


@dataclass
class DataLoaders:
    """Container for train, validation, and test loaders."""

    train: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    val: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    test: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    stats: dict[str, dict[str, int | dict[str, int]]] | None = None


def build_dataloaders(cfg) -> DataLoaders:
    """Build train, validation, and test dataloaders from config."""
    train_images, train_labels = load_svhn_mat(cfg.paths.train_mat)
    test_images, test_labels = load_svhn_mat(cfg.paths.test_mat)

    train_indices, val_indices = split_train_val(
        labels=train_labels,
        val_ratio=cfg.data.val_ratio,
        seed=cfg.train.seed,
        split_dir=cfg.paths.split_dir,
    )

    dataset_kwargs = {
        "normalize": bool(cfg.data.normalize),
        "mean": list(cfg.data.mean),
        "std": list(cfg.data.std),
    }
    train_dataset = SVHNMatDataset(
        train_images[train_indices],
        train_labels[train_indices],
        **dataset_kwargs,
    )
    val_dataset = SVHNMatDataset(
        train_images[val_indices],
        train_labels[val_indices],
        **dataset_kwargs,
    )
    test_dataset = SVHNMatDataset(test_images, test_labels, **dataset_kwargs)

    common_loader_kwargs = {
        "num_workers": int(cfg.data.num_workers),
        "pin_memory": bool(cfg.data.pin_memory and torch.cuda.is_available()),
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        **common_loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.eval.batch_size),
        shuffle=False,
        **common_loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(cfg.eval.batch_size),
        shuffle=False,
        **common_loader_kwargs,
    )
    stats = build_split_stats(
        train_labels=train_labels[train_indices],
        val_labels=train_labels[val_indices],
        test_labels=test_labels,
        num_classes=int(cfg.data.num_classes),
    )
    return DataLoaders(train=train_loader, val=val_loader, test=test_loader, stats=stats)
