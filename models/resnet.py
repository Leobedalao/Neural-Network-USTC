"""ResNet models adapted for 32x32 images."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    """Return a 3x3 convolution with padding."""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    """Basic residual block for CIFAR/SVHN-style ResNet."""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the residual block."""
        identity = inputs

        outputs = self.conv1(inputs)
        outputs = self.bn1(outputs)
        outputs = self.relu(outputs)

        outputs = self.conv2(outputs)
        outputs = self.bn2(outputs)

        if self.downsample is not None:
            identity = self.downsample(inputs)

        outputs += identity
        outputs = self.relu(outputs)
        return outputs


class ResNet(nn.Module):
    """ResNet backbone adapted to 32x32 images."""

    def __init__(
        self,
        block: type[BasicBlock],
        layers: list[int],
        num_classes: int = 10,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        self.in_channels = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(block, base_channels, layers[0], stride=1)
        self.layer2 = self._make_layer(block, base_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, base_channels * 4, layers[2], stride=2)
        #self.layer4 = self._make_layer(block, base_channels * 8, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 4 * block.expansion, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _make_layer(
        self,
        block: type[BasicBlock],
        out_channels: int,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = [block(self.in_channels, out_channels, stride=stride, downsample=downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return class logits."""
        outputs = self.stem(inputs)
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        #outputs = self.layer4(outputs)
        outputs = self.avgpool(outputs)
        outputs = torch.flatten(outputs, 1)
        return self.fc(outputs)


def _build_resnet(
    layers: list[int],
    num_classes: int,
    base_channels: int,
    block_factory: Callable[..., BasicBlock] | type[BasicBlock] = BasicBlock,
) -> ResNet:
    return ResNet(
        block_factory,
        layers=layers,
        num_classes=num_classes,
        base_channels=base_channels,
    )


def resnet111(num_classes: int = 10, base_channels: int = 32) -> ResNet:
    """Build a 32x32-adapted ResNet with stage depths [1, 1, 1]."""
    return _build_resnet([1, 1, 1], num_classes=num_classes, base_channels=base_channels)


def resnet18(num_classes: int = 10, base_channels: int = 32) -> ResNet:
    """Legacy alias for the historical resnet18 name used in this project."""
    return resnet111(num_classes=num_classes, base_channels=base_channels)


