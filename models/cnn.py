from __future__ import annotations

import torch
from torch import nn


class PlainBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class BasicCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        base_channels: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.layer1 = PlainBlock(base_channels, base_channels, stride=1)
        self.layer2 = PlainBlock(base_channels, base_channels * 2, stride=2)
        self.layer3 = PlainBlock(base_channels * 2, base_channels * 4, stride=2)
        self.layer4 = PlainBlock(base_channels * 4, base_channels * 8, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 8, base_channels * 8),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(base_channels * 8, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.stem(inputs)
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        outputs = self.layer4(outputs)
        outputs = self.avgpool(outputs)
        return self.classifier(outputs)
