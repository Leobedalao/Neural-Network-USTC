"""Squeeze-and-Excitation module."""

from __future__ import annotations

import torch
from torch import nn


class SEBlock(nn.Module):
    """Channel-wise squeeze-and-excitation block."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Reweight channels using global context."""
        batch_size, channels, _, _ = inputs.shape
        weights = self.pool(inputs).view(batch_size, channels)
        weights = self.fc(weights).view(batch_size, channels, 1, 1)
        return inputs * weights
