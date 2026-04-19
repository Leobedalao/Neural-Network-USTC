"""Plot helpers for experiment artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.io import ensure_dir


def build_confusion_annotations(confusion_matrix: np.ndarray) -> list[list[str]]:
    """Convert a confusion matrix into per-cell annotation strings."""
    return [[str(int(value)) for value in row] for row in confusion_matrix.tolist()]


def plot_training_curves(
    history: list[dict[str, float]],
    save_path: str | Path,
    dpi: int = 150,
) -> None:
    """Plot train/validation loss and accuracy curves."""
    if not history:
        return

    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    val_loss = [entry["val_loss"] for entry in history]
    train_acc = [entry["train_acc"] for entry in history]
    val_acc = [entry["val_acc"] for entry in history]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=dpi)
    axes[0].plot(epochs, train_loss, label="train_loss")
    axes[0].plot(epochs, val_loss, label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="train_acc")
    axes[1].plot(epochs, val_acc, label="val_acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    figure.tight_layout()
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    figure.savefig(save_path)
    plt.close(figure)


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    save_path: str | Path,
    class_names: list[str] | None = None,
    dpi: int = 150,
) -> None:
    """Plot a confusion matrix heatmap with numeric annotations."""
    figure, axis = plt.subplots(figsize=(6, 5), dpi=dpi)
    image = axis.imshow(confusion_matrix, cmap="Blues")
    axis.set_title("Confusion Matrix")
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    tick_labels = class_names or [str(index) for index in range(confusion_matrix.shape[0])]
    axis.set_xticks(range(confusion_matrix.shape[0]), tick_labels)
    axis.set_yticks(range(confusion_matrix.shape[0]), tick_labels)

    annotations = build_confusion_annotations(confusion_matrix)
    threshold = float(confusion_matrix.max()) / 2.0 if confusion_matrix.size else 0.0
    for row_index, row in enumerate(annotations):
        for col_index, value in enumerate(row):
            text_color = "white" if confusion_matrix[row_index, col_index] > threshold else "black"
            axis.text(
                col_index,
                row_index,
                value,
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )

    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    figure.savefig(save_path)
    plt.close(figure)


def plot_per_class_accuracy(
    per_class_accuracy: dict[int, float] | dict[str, float],
    save_path: str | Path,
    dpi: int = 150,
) -> None:
    """Plot a per-class accuracy bar chart."""
    class_labels = [str(label) for label in per_class_accuracy.keys()]
    values = [float(value) for value in per_class_accuracy.values()]

    figure, axis = plt.subplots(figsize=(8, 4), dpi=dpi)
    bars = axis.bar(class_labels, values, color="#4C78A8")
    axis.set_title("Per-Class Accuracy")
    axis.set_xlabel("Class")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0.0, 1.05)

    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.02,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    figure.tight_layout()
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    figure.savefig(save_path)
    plt.close(figure)
