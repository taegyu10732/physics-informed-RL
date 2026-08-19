"""Neural-network components used by the gas-source navigation agent."""

from __future__ import annotations

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class GasMapFeaturesExtractor(BaseFeaturesExtractor):
    """Encode gas/obstacle maps, lidar, and the recent robot trajectory."""

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        map_channels = observation_space["map"].shape[0]
        lidar_size = observation_space["lidar"].shape[0]

        self.map_encoder = nn.Sequential(
            nn.Conv2d(map_channels, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
        )
        self.trajectory_encoder = nn.GRU(2, 32, batch_first=True)
        self.fusion = nn.Sequential(
            nn.Linear(64 * 4 * 4 + lidar_size + 32, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        map_features = self.map_encoder(observations["map"])
        trajectory = observations["trajectory"]
        _, trajectory_features = self.trajectory_encoder(trajectory)
        features = torch.cat(
            [map_features, observations["lidar"], trajectory_features[-1]], dim=1
        )
        return self.fusion(features)


# A small compatibility alias for checkpoints/configs created before the cleanup.
StackedMapFeaturesExtractor = GasMapFeaturesExtractor
