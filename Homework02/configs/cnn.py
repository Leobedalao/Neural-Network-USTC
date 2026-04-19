"""CNN experiment configuration."""

from __future__ import annotations

from ml_collections import ConfigDict

from configs.default import get_config as get_default_config


def get_config() -> ConfigDict:
    """Return the CNN experiment configuration."""
    config = get_default_config()
    config.experiment.name = "cnn"
    config.model.name = "cnn"
    config.model.base_channels = 32
    config.model.dropout = 0.3
    return config
