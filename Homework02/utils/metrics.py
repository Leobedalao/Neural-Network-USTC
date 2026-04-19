"""Metric helpers for classification experiments."""

from __future__ import annotations

import numpy as np
import torch


def compute_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute the mean classification accuracy."""
    if len(targets) == 0:
        return 0.0
    return float((predictions == targets).mean())


def compute_accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute batch accuracy from logits and labels."""
    predictions = torch.argmax(logits, dim=1)
    return float((predictions == targets).float().mean().item())


def compute_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Compute a confusion matrix using fixed class order 0..num_classes-1."""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, prediction in zip(targets, predictions, strict=True):
        matrix[int(target), int(prediction)] += 1
    return matrix


def compute_per_class_accuracy(confusion_matrix: np.ndarray) -> dict[int, float]:
    """Compute per-class accuracy from a confusion matrix."""
    per_class: dict[int, float] = {}
    for class_idx in range(confusion_matrix.shape[0]):
        total = confusion_matrix[class_idx].sum()
        correct = confusion_matrix[class_idx, class_idx]
        per_class[class_idx] = float(correct / total) if total > 0 else 0.0
    return per_class


def detach_predictions(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert logits and targets to NumPy arrays."""
    predictions = torch.argmax(logits, dim=1).detach().cpu().numpy()
    labels = targets.detach().cpu().numpy()
    return predictions, labels
