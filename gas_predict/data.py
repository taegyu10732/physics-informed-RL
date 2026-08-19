"""Dataset adapter for the preprocessed indoor gas experiments."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class GasFieldDataset(Dataset):
    """Load obstacle, sparse measurement, gas field, sigma, and source labels."""

    required_keys = {
        "obstacle_maps",
        "masked_maps",
        "true_gas_maps",
        "sigma_true",
        "position_true",
    }

    def __init__(self, path: str | Path):
        with Path(path).open("rb") as stream:
            data = pickle.load(stream)
        missing = self.required_keys - data.keys()
        if missing:
            raise ValueError(f"Dataset is missing keys: {sorted(missing)}")
        lengths = {len(data[key]) for key in self.required_keys}
        if len(lengths) != 1:
            raise ValueError("Dataset fields must have the same length")
        self.data = data

    def __len__(self) -> int:
        return len(self.data["obstacle_maps"])

    def __getitem__(self, index: int):
        obstacle = np.asarray(self.data["obstacle_maps"][index], dtype=np.float32)
        if obstacle.max(initial=0.0) > 1.0:
            obstacle = obstacle / 255.0
        sparse_gas = np.asarray(self.data["masked_maps"][index], dtype=np.float32)
        gas = np.asarray(self.data["true_gas_maps"][index], dtype=np.float32)
        model_input = torch.from_numpy(np.stack([obstacle, sparse_gas]))
        return (
            model_input,
            torch.from_numpy(gas[None]),
            torch.as_tensor(self.data["sigma_true"][index], dtype=torch.float32).reshape(1),
            torch.as_tensor(self.data["position_true"][index], dtype=torch.float32).reshape(2),
        )
