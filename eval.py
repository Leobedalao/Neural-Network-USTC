"""Evaluate a saved checkpoint on the SVHN test set."""

from __future__ import annotations

import argparse
from pathlib import Path

from torch import nn

from datasets.svhn_mat import build_dataloaders
from engine.evaluator import evaluate
from models import build_model
from train import load_config, resolve_device
from utils.checkpoint import load_checkpoint
from utils.io import ensure_dir, save_json, save_npy
from utils.plot import plot_confusion_matrix, plot_per_class_accuracy


def parse_args() -> argparse.Namespace:
    """Parse evaluation arguments."""
    parser = argparse.ArgumentParser(description="Evaluate SVHN classification checkpoints.")
    parser.add_argument("--config", required=True, help="Path to a Python config file.")
    parser.add_argument("--checkpoint", required=True, help="Path to a checkpoint file.")
    parser.add_argument("--output-dir", default="", help="Optional output directory override.")
    return parser.parse_args()


def main() -> None:
    """Load a checkpoint and evaluate it on the test split."""
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(str(cfg.train.device))
    output_dir = (
        Path(args.output_dir) if args.output_dir else Path(args.checkpoint).resolve().parent
    )
    ensure_dir(output_dir)

    dataloaders = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = evaluate(
        model=model,
        loader=dataloaders.test,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        num_classes=int(cfg.data.num_classes),
    )
    payload = {
        "test_loss": float(metrics["loss"]),
        "test_accuracy": float(metrics["accuracy"]),
        "per_class_accuracy": {
            str(key): value for key, value in metrics["per_class_accuracy"].items()
        },
    }
    save_json(output_dir / "eval_metrics.json", payload)
    save_json(output_dir / "per_class_accuracy.json", payload["per_class_accuracy"])
    save_npy(output_dir / "confusion_matrix.npy", metrics["confusion_matrix"])
    plot_confusion_matrix(metrics["confusion_matrix"], output_dir / "confusion_matrix.png")
    plot_per_class_accuracy(payload["per_class_accuracy"], output_dir / "per_class_accuracy.png")


if __name__ == "__main__":
    main()
