"""Filesystem helpers for experiment artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, payload: Any) -> None:
    """Serialize a JSON payload to disk."""
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    """Load JSON content from disk."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of dictionaries to CSV."""
    if not rows:
        return

    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_npy(path: str | Path, array: np.ndarray) -> None:
    """Persist an array to disk."""
    path = Path(path)
    ensure_dir(path.parent)
    np.save(path, array)


def load_npy(path: str | Path) -> np.ndarray:
    """Load an array from disk."""
    return np.load(Path(path))
