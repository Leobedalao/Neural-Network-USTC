"""Train SVHN models from a configuration file."""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

from datasets.svhn_mat import build_dataloaders
from engine.evaluator import evaluate
from engine.trainer import fit
from models import build_model
from utils.checkpoint import load_checkpoint
from utils.io import ensure_dir, save_csv, save_json, save_npy
from utils.logger import create_logger
from utils.plot import plot_confusion_matrix, plot_per_class_accuracy, plot_training_curves
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train SVHN classification models.")
    parser.add_argument("--config", required=True, help="Path to a Python config file.")
    return parser.parse_args()


def load_config(config_path: str | Path):
    """Load a ConfigDict from a Python config module."""
    config_path = Path(config_path)
    spec = importlib.util.spec_from_file_location("svhn_config", config_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Unable to load config from {config_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config()


def resolve_device(device_name: str) -> torch.device:
    """Resolve the configured device string."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def config_to_dict(cfg) -> dict[str, Any]:
    """Convert a ConfigDict into a plain dictionary."""
    return json.loads(cfg.to_json_best_effort())


def build_optimizer(cfg, model: nn.Module) -> SGD:
    """Build the configured optimizer."""
    if str(cfg.optimizer.name).lower() != "sgd":
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer.name}")
    return SGD(
        model.parameters(),
        lr=float(cfg.optimizer.lr),
        momentum=float(cfg.optimizer.momentum),
        weight_decay=float(cfg.optimizer.weight_decay),
        nesterov=bool(cfg.optimizer.nesterov),
    )


def build_scheduler(cfg, optimizer: SGD):
    """Build the configured scheduler if enabled."""
    if str(cfg.scheduler.name).lower() == "none":
        return None
    if str(cfg.scheduler.name).lower() == "step":
        return StepLR(
            optimizer,
            step_size=int(cfg.scheduler.step_size),
            gamma=float(cfg.scheduler.gamma),
        )
    raise ValueError(f"Unsupported scheduler: {cfg.scheduler.name}")


def main() -> None:
    """Execute the full train/validate/test pipeline."""
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.train.seed))

    run_id = cfg.experiment.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ensure_dir(Path(cfg.paths.output_root) / cfg.experiment.name / run_id)
    logger = create_logger(output_dir / "train.log")
    logger.info("loading config from %s", args.config)

    dataloaders = build_dataloaders(cfg)
    model = build_model(cfg)
    device = resolve_device(str(cfg.train.device))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    config_dict = config_to_dict(cfg)
    save_json(output_dir / "config.json", config_dict)
    if dataloaders.stats is not None:
        save_json(output_dir / "dataset_stats.json", dataloaders.stats)
        for split_name, split_stats in dataloaders.stats.items():
            logger.info(
                "dataset split=%s num_samples=%s class_counts=%s",
                split_name,
                split_stats["num_samples"],
                split_stats["class_counts"],
            )

    training_result = fit(
        model=model,
        train_loader=dataloaders.train,
        val_loader=dataloaders.val,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=int(cfg.train.epochs),
        output_dir=output_dir,
        logger=logger,
        num_classes=int(cfg.data.num_classes),
        scheduler=scheduler,
        config=config_dict,
    )

    history = training_result["history"]
    save_json(output_dir / "history.json", history)
    save_csv(output_dir / "history.csv", history)
    plot_training_curves(history, output_dir / "curves.png", dpi=int(cfg.plot.dpi))

    best_checkpoint = load_checkpoint(output_dir / "best.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_metrics = evaluate(
        model=model,
        loader=dataloaders.test,
        criterion=criterion,
        device=device,
        num_classes=int(cfg.data.num_classes),
    )

    metrics_payload = {
        "test_loss": float(test_metrics["loss"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "per_class_accuracy": {
            str(key): value for key, value in test_metrics["per_class_accuracy"].items()
        },
    }
    save_json(output_dir / "test_metrics.json", metrics_payload)
    save_json(output_dir / "per_class_accuracy.json", metrics_payload["per_class_accuracy"])
    save_npy(output_dir / "confusion_matrix.npy", test_metrics["confusion_matrix"])
    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        output_dir / "confusion_matrix.png",
    )
    plot_per_class_accuracy(
        metrics_payload["per_class_accuracy"],
        output_dir / "per_class_accuracy.png",
        dpi=int(cfg.plot.dpi),
    )
    logger.info("test loss=%.4f acc=%.4f", test_metrics["loss"], test_metrics["accuracy"])


if __name__ == "__main__":
    main()
