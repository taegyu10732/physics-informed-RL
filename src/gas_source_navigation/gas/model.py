"""Checkpoint-compatible PI-Attention-UNet from the research notebook."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn


class GasPrediction(NamedTuple):
    concentration: torch.Tensor
    sigma: torch.Tensor
    source_position: torch.Tensor
    attention_maps: dict[str, torch.Tensor]


class DoubleConv(nn.Module):
    """The notebook's convolution block.

    Attribute names and convolution biases intentionally match the saved
    research state dictionaries (``enc1.block.0.weight``, and so on).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class AttentionGate(nn.Module):
    """Attention gate used by the selected ``_21`` research notebook."""

    def __init__(self, gate_channels: int, skip_channels: int, hidden_channels: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(skip_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self, gate: torch.Tensor, skip: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = self.psi(self.relu(self.W_g(gate) + self.W_x(skip)))
        return skip * mask, mask


class AttentionUNet(nn.Module):
    """Predict gas concentration, normalized sigma, and source position.

    The module layout is kept compatible with the ``pi_kernel_100to300_*``
    checkpoints imported into the downstream RL experiments. Spatial
    dimensions must be divisible by 16.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        base_channels: int = 64,
    ):
        super().__init__()
        c1, c2, c3, c4, c5 = (base_channels * 2**index for index in range(5))

        self.enc1 = DoubleConv(in_channels, c1)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(c1, c2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(c2, c3)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(c3, c4)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(c4, c5)

        self.up4 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.att4 = AttentionGate(c4, c4, c3)
        self.dec4 = DoubleConv(c4 * 2, c4)

        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.att3 = AttentionGate(c3, c3, c2)
        self.dec3 = DoubleConv(c3 * 2, c3)

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.att2 = AttentionGate(c2, c2, c1)
        self.dec2 = DoubleConv(c2 * 2, c2)

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.att1 = AttentionGate(c1, c1, max(1, c1 // 2))
        self.dec1 = DoubleConv(c1 * 2, c1)

        # This layout matches the PI-U-Net checkpoint imported into the
        # downstream RL project (published as
        # ``pi_attention_unet_indoor_gas_v1.pt``).
        self.final_conv = nn.Sequential(
            nn.Conv2d(c1, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.final_conv_sigma = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c1, 1),
            nn.ReLU(),
        )
        self.final_conv_pos = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c1, 2),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> GasPrediction:
        if inputs.ndim != 4 or inputs.shape[-2] % 16 or inputs.shape[-1] % 16:
            raise ValueError("Expected BCHW input with height and width divisible by 16")

        e1 = self.enc1(inputs)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        bottleneck = self.bottleneck(self.pool4(e4))

        d4 = self.up4(bottleneck)
        e4_attention, attention4 = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4_attention], dim=1))

        d3 = self.up3(d4)
        e3_attention, attention3 = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3_attention], dim=1))

        d2 = self.up2(d3)
        e2_attention, attention2 = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2_attention], dim=1))

        d1 = self.up1(d2)
        e1_attention, attention1 = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1_attention], dim=1))

        return GasPrediction(
            concentration=self.final_conv(d1),
            sigma=self.final_conv_sigma(d1),
            source_position=self.final_conv_pos(d1),
            attention_maps={
                "att4": attention4,
                "att3": attention3,
                "att2": attention2,
                "att1": attention1,
            },
        )


# Descriptive name retained for the refactored package API.
PhysicsInformedAttentionUNet = AttentionUNet
