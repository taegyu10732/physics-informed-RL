"""Policy feature extractors used by the gas-source navigation agent."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class Shared3DStackCNN(nn.Module):
    """Learn spatial and temporal features from stacked multi-scale maps."""

    def __init__(self, in_channels: int, out_channels: int = 32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class StackedMapFeaturesExtractor(BaseFeaturesExtractor):
    """Fuse map stacks with LiDAR and the PI-U-Net source estimate.

    ``coverage``, ``obstacles``, and ``gas`` each have shape
    ``(batch, stacks, num_maps, height, width)``. The three map families are
    concatenated as channels and processed by a 3-D CNN over the stack depth.
    The resulting map query attends to projected LiDAR and predicted-source
    tokens before the final feature projection.

    The extra keyword arguments are retained for compatibility with the
    experiment metadata created by ``Paper_main_pi_unet``.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 256,
        map_size: int | None = None,
        num_maps: int | None = None,
        lidar_rays: int | None = None,
        stacks: int | None = None,
        grouped_convs: bool = False,
        frontier_observation: bool = False,
        **_: Any,
    ):
        del map_size, num_maps, lidar_rays, stacks, grouped_convs, frontier_observation
        if features_dim % 4:
            raise ValueError("features_dim must be divisible by four for cross-attention")
        required = {"coverage", "obstacles", "gas", "lidar", "pred_pos"}
        missing = required.difference(observation_space.spaces)
        if missing:
            raise ValueError(f"observation space is missing: {sorted(missing)}")

        super().__init__(observation_space, features_dim)
        map_shape = observation_space["coverage"].shape
        if len(map_shape) != 4:
            raise ValueError("map observations must have shape (stacks, num_maps, H, W)")
        inferred_num_maps = map_shape[1]
        inferred_stacks = map_shape[0]
        inferred_lidar_rays = observation_space["lidar"].shape[-1]
        self.map_channels = inferred_num_maps * 3

        self.shared_cnn = Shared3DStackCNN(self.map_channels, 32)
        self.map_projector = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, features_dim),
            nn.ReLU(inplace=True),
        )
        self.lidar_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(inferred_stacks * inferred_lidar_rays, 64),
            nn.ReLU(inplace=True),
        )
        self.source_extractor = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(inplace=True),
        )
        self.lidar_projector = nn.Linear(64, features_dim)
        self.source_projector = nn.Linear(32, features_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=features_dim,
            num_heads=4,
            batch_first=True,
        )
        self.fused_extractor = nn.Sequential(
            nn.Linear(features_dim * 2, features_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        maps = torch.cat(
            [
                observations["coverage"],
                observations["obstacles"],
                observations["gas"],
            ],
            dim=2,
        )
        maps = maps.permute(0, 2, 1, 3, 4).contiguous()
        map_features_3d = self.shared_cnn(maps)
        map_features = self.map_projector(map_features_3d.mean(dim=2))

        lidar_features = self.lidar_projector(
            self.lidar_extractor(observations["lidar"])
        )
        source_features = self.source_projector(
            self.source_extractor(observations["pred_pos"])
        )
        key_value = torch.stack([lidar_features, source_features], dim=1)
        attended, _ = self.cross_attention(
            map_features.unsqueeze(1), key_value, key_value, need_weights=False
        )
        return self.fused_extractor(
            torch.cat([map_features, attended.squeeze(1)], dim=1)
        )


# Descriptive compatibility name used by the cleaned training entry point.
GasMapFeaturesExtractor = StackedMapFeaturesExtractor
