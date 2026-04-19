"""Reporting helpers for aggregating experiment results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from utils.io import load_json, save_csv

SUMMARY_FIELDS = [
    "model_name",
    "run_id",
    "run_dir",
    "epochs",
    "batch_size",
    "learning_rate",
    "best_val_acc",
    "test_loss",
    "test_accuracy",
]


def _safe_load_json(path: Path) -> dict[str, Any] | list[dict[str, Any]] | None:
    if not path.exists():
        return None
    return load_json(path)


def collect_experiment_rows(outputs_dir: str | Path) -> list[dict[str, Any]]:
    """Collect summary rows from experiment output directories."""
    outputs_dir = Path(outputs_dir)
    rows: list[dict[str, Any]] = []

    for metrics_path in sorted(outputs_dir.glob("*/*/test_metrics.json")):
        run_dir = metrics_path.parent
        config = _safe_load_json(run_dir / "config.json") or {}
        history = _safe_load_json(run_dir / "history.json") or []
        metrics = load_json(metrics_path)

        best_val_acc = None
        if isinstance(history, list) and history:
            best_val_acc = max(float(item["val_acc"]) for item in history if "val_acc" in item)

        rows.append(
            {
                "model_name": str(run_dir.parent.name),
                "run_id": str(run_dir.name),
                "run_dir": str(run_dir),
                "epochs": int(config.get("train", {}).get("epochs", 0)),
                "batch_size": int(config.get("train", {}).get("batch_size", 0)),
                "learning_rate": float(config.get("optimizer", {}).get("lr", 0.0)),
                "best_val_acc": best_val_acc,
                "test_loss": float(metrics.get("test_loss", 0.0)),
                "test_accuracy": float(metrics.get("test_accuracy", 0.0)),
            }
        )

    return rows


def export_experiment_summary_csv(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    """Export experiment summary rows to CSV."""
    normalized_rows = [{field: row.get(field, "") for field in SUMMARY_FIELDS} for row in rows]
    save_csv(output_path, normalized_rows)
