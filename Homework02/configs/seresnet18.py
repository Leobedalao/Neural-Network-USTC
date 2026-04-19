"""SE-ResNet-18 experiment configuration."""

from __future__ import annotations

from ml_collections import ConfigDict

from configs.default import get_config as get_default_config


def get_config() -> ConfigDict:
    """Return the SE-ResNet-18 experiment configuration."""
    config = get_default_config()
    config.experiment.name = "seresnet18"
    config.model.name = "seresnet18"
    config.model.base_channels = 32
    config.model.reduction = 8
    return config
