"""Baseline CNN model for SVHN classification."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """A simple convolutional block with batch normalization and pooling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the convolutional block."""
        return self.block(inputs)


class BasicCNN(nn.Module):
    """Baseline CNN for 32x32 image classification."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        base_channels: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, base_channels),
            ConvBlock(base_channels, base_channels * 2),
            ConvBlock(base_channels * 2, base_channels * 4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 4 * 4 * 4, base_channels * 8),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(base_channels * 8, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return class logits for the input batch."""
        features = self.features(inputs)
        return self.classifier(features)
