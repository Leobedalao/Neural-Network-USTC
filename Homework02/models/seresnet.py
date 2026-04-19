"""SE-ResNet models adapted for 32x32 images."""

from __future__ import annotations

from torch import nn

from models.resnet import BasicBlock, ResNet, conv3x3
from models.se import SEBlock


class SEBasicBlock(BasicBlock):
    """Residual block augmented with squeeze-and-excitation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        reduction: int = 8,
    ) -> None:
        super().__init__(in_channels, out_channels, stride=stride, downsample=downsample)
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels, reduction=reduction)

    def forward(self, inputs):
        identity = inputs

        outputs = self.conv1(inputs)
        outputs = self.bn1(outputs)
        outputs = self.relu(outputs)

        outputs = self.conv2(outputs)
        outputs = self.bn2(outputs)
        outputs = self.se(outputs)

        if self.downsample is not None:
            identity = self.downsample(inputs)

        outputs += identity
        outputs = self.relu(outputs)
        return outputs


def _make_block(reduction: int):
    def factory(
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ):
        return SEBasicBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            downsample=downsample,
            reduction=reduction,
        )

    factory.expansion = SEBasicBlock.expansion
    return factory


def seresnet111(num_classes: int = 10, base_channels: int = 64, reduction: int = 16) -> ResNet:
    """Build a 32x32-adapted SE-ResNet with stage depths [1, 1, 1]."""
    return ResNet(
        block=_make_block(reduction),
        layers=[1, 1, 1],
        num_classes=num_classes,
        base_channels=base_channels,
    )


def seresnet18(num_classes: int = 10, base_channels: int = 64, reduction: int = 16) -> ResNet:
    """Legacy alias for the historical seresnet18 name used in this project."""
    return seresnet111(num_classes=num_classes, base_channels=base_channels, reduction=reduction)

