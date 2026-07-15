"""Compact convolutional network used by the main CIFAR-10 experiment."""

from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig


class CompactCNN(nn.Module):
    """A small CNN that preserves some position information before classification."""

    def __init__(self, *, channels: tuple[int, ...] = (32, 64, 128), dropout: float = 0.25):
        super().__init__()
        if channels != (32, 64, 128):
            raise ValueError("CompactCNN currently expects channels=(32, 64, 128)")

        blocks: list[nn.Module] = []
        input_channels = 3
        for output_channels in channels:
            blocks.extend(
                [
                    nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
            input_channels = output_channels

        self.features = nn.Sequential(*blocks)
        self.spatial_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1] * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 10),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.classifier(self.spatial_pool(features))


def build_model(config: ModelConfig) -> nn.Module:
    if config.name != "compact_cnn":
        raise ValueError(f"Unknown model: {config.name}")
    return CompactCNN(channels=config.channels, dropout=config.dropout)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
