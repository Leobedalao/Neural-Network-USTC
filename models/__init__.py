"""Model builders for SVHN classification architectures."""

from __future__ import annotations

from torch import nn

from models.cnn import BasicCNN
from models.resnet import resnet111
from models.seresnet import seresnet111


def build_model(cfg) -> nn.Module:
    """Build a model from the experiment configuration."""
    model_name = str(cfg.model.name).lower()
    num_classes = int(cfg.model.num_classes)
    in_channels = int(cfg.model.in_channels)
    base_channels = int(cfg.model.base_channels)
    dropout = float(cfg.model.dropout)
    reduction = int(cfg.model.reduction)

    if model_name == "cnn":
        return BasicCNN(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            dropout=dropout,
        )
    if model_name in {"resnet111", "resnet18"}:
        return resnet111(num_classes=num_classes, base_channels=base_channels)
    if model_name in {"seresnet111", "seresnet18"}:
        return seresnet111(
            num_classes=num_classes,
            base_channels=base_channels,
            reduction=reduction,
        )

    raise ValueError(f"Unsupported model name: {cfg.model.name}")
