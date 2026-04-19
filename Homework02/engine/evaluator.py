"""Evaluation helpers for SVHN experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.metrics import (
    compute_accuracy,
    compute_confusion_matrix,
    compute_per_class_accuracy,
    detach_predictions,
)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> dict[str, Any]:
    """Evaluate a model on a dataloader."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_predictions: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = criterion(logits, targets)

        batch_size = int(targets.size(0))
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

        predictions, labels = detach_predictions(logits, targets)
        all_predictions.append(predictions)
        all_targets.append(labels)

    predictions = (
        np.concatenate(all_predictions) if all_predictions else np.array([], dtype=np.int64)
    )
    targets = np.concatenate(all_targets) if all_targets else np.array([], dtype=np.int64)
    avg_loss = total_loss / max(total_samples, 1)
    accuracy = compute_accuracy(predictions, targets)
    confusion_matrix = compute_confusion_matrix(predictions, targets, num_classes=num_classes)
    per_class_accuracy = compute_per_class_accuracy(confusion_matrix)
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "predictions": predictions,
        "targets": targets,
        "confusion_matrix": confusion_matrix,
        "per_class_accuracy": per_class_accuracy,
    }
