"""Checkpoint helpers for training and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from utils.io import ensure_dir


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    """Save a checkpoint payload to disk."""
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint payload from disk."""
    return torch.load(Path(path), map_location=map_location, weights_only=False)
