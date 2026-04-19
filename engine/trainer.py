"""Training loop implementation for SVHN experiments."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from engine.evaluator import evaluate
from utils.checkpoint import save_checkpoint


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
) -> tuple[float, float]:
    """Train a single epoch and return loss and accuracy."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress = tqdm(loader, desc=f"train {epoch}", leave=False)
    for images, targets in progress:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = int(targets.size(0))
        total_loss += float(loss.item()) * batch_size
        predictions = torch.argmax(logits, dim=1)
        total_correct += int((predictions == targets).sum().item())
        total_samples += batch_size
        progress.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    logger.info("epoch=%s split=train loss=%.4f acc=%.4f", epoch, avg_loss, avg_acc)
    return avg_loss, avg_acc


def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
    num_classes: int,
) -> tuple[float, float, dict[str, Any]]:
    """Validate a single epoch and return summary statistics."""
    metrics = evaluate(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        num_classes=num_classes,
    )
    logger.info(
        "epoch=%s split=val loss=%.4f acc=%.4f",
        epoch,
        metrics["loss"],
        metrics["accuracy"],
    )
    return float(metrics["loss"]), float(metrics["accuracy"]), metrics


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    epochs: int,
    output_dir: str | Path,
    logger: logging.Logger,
    num_classes: int,
    scheduler: LRScheduler | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the full training loop and persist checkpoints."""
    output_dir = Path(output_dir)
    history: list[dict[str, float]] = []
    best_val_acc = float("-inf")
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            logger=logger,
        )
        val_loss, val_acc, _ = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            logger=logger,
            num_classes=num_classes,
        )

        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
        }
        history.append(entry)

        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "config": config or {},
        }
        save_checkpoint(output_dir / "last.pt", payload)

        is_better = val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss)
        if is_better:
            best_val_acc = val_acc
            best_val_loss = val_loss
            payload["best_val_acc"] = best_val_acc
            save_checkpoint(output_dir / "best.pt", payload)

        if scheduler is not None:
            scheduler.step()

    return {
        "history": history,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
    }
