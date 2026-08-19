"""Attention U-Net selected from the gas-prediction experiments."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn


class GasPrediction(NamedTuple):
    concentration: torch.Tensor
    sigma: torch.Tensor
    source_position: torch.Tensor


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class AttentionGate(nn.Module):
    def __init__(self, gate_channels: int, skip_channels: int, hidden_channels: int):
        super().__init__()
        self.gate = nn.Conv2d(gate_channels, hidden_channels, 1, bias=False)
        self.skip = nn.Conv2d(skip_channels, hidden_channels, 1, bias=False)
        self.mask = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(hidden_channels, 1, 1), nn.Sigmoid())

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return skip * self.mask(self.gate(gate) + self.skip(skip))


class PhysicsInformedAttentionUNet(nn.Module):
    """Predict a gas map, normalized dispersion sigma, and source position.

    Input channels are the obstacle map and sparsely measured gas map. Spatial
    dimensions must be divisible by 16. Physics is imposed during training by
    :class:`gas_predict.losses.PhysicsInformedLoss`, using the model's gas,
    sigma, and source-position outputs together.
    """

    def __init__(self, in_channels: int = 2, base_channels: int = 64):
        super().__init__()
        channels = [base_channels * 2**i for i in range(5)]
        self.encoders = nn.ModuleList(
            [DoubleConv(in_channels, channels[0])]
            + [DoubleConv(channels[i - 1], channels[i]) for i in range(1, 4)]
        )
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(channels[3], channels[4])
        self.ups = nn.ModuleList(
            [nn.ConvTranspose2d(channels[i], channels[i - 1], 2, stride=2) for i in range(4, 0, -1)]
        )
        self.attention = nn.ModuleList(
            [AttentionGate(channels[i], channels[i], channels[i] // 2) for i in range(3, -1, -1)]
        )
        self.decoders = nn.ModuleList(
            [DoubleConv(channels[i] * 2, channels[i]) for i in range(3, -1, -1)]
        )
        self.concentration_head = nn.Conv2d(channels[0], 1, 1)
        self.sigma_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels[0], 1), nn.Softplus())
        self.position_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels[0], 2), nn.Sigmoid())

    def forward(self, inputs: torch.Tensor) -> GasPrediction:
        if inputs.ndim != 4 or inputs.shape[-2] % 16 or inputs.shape[-1] % 16:
            raise ValueError("Expected BCHW input with height and width divisible by 16")
        skips = []
        x = inputs
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, attention, decoder, skip in zip(self.ups, self.attention, self.decoders, reversed(skips)):
            x = up(x)
            x = decoder(torch.cat([x, attention(x, skip)], dim=1))
        return GasPrediction(
            concentration=self.concentration_head(x),
            sigma=self.sigma_head(x),
            source_position=self.position_head(x),
        )


# Short backwards-compatible name used by the original notebooks.
AttentionUNet = PhysicsInformedAttentionUNet
