"""PI-Attention-UNet inference used by the RL environment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import torch
from torch import nn

from gas_source_navigation.gas.model import AttentionUNet, GasPrediction


@dataclass(frozen=True)
class GasMapEstimate:
    """One reconstruction result in the environment map coordinates."""

    concentration: np.ndarray
    sigma: float
    source_position: np.ndarray


class GasMapPredictor:
    """Load a research checkpoint and reconstruct a gas map from sparse data."""

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        *,
        model: nn.Module | None = None,
        device: str = "auto",
        inference_size: int = 320,
    ):
        if inference_size <= 0 or inference_size % 16:
            raise ValueError("inference_size must be positive and divisible by 16")
        if model is None and checkpoint is None:
            raise ValueError("checkpoint is required when model is not supplied")

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.inference_size = inference_size
        self.model = model if model is not None else AttentionUNet()
        if checkpoint is not None:
            self._load_checkpoint(Path(checkpoint))
        self.model.to(self.device).eval()

    def _load_checkpoint(self, checkpoint: Path) -> None:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Gas predictor checkpoint not found: {checkpoint}")
        loaded: Any = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if isinstance(loaded, Mapping) and "state_dict" in loaded:
            loaded = loaded["state_dict"]
        if not isinstance(loaded, Mapping):
            raise ValueError(f"Checkpoint does not contain a state dict: {checkpoint}")
        state_dict = {
            key.removeprefix("module."): value
            for key, value in loaded.items()
        }
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as error:
            raise ValueError(
                f"Checkpoint is incompatible with the PI-Attention-UNet: {checkpoint}"
            ) from error

    def predict(
        self, obstacle_map: np.ndarray, sparse_concentration_map: np.ndarray
    ) -> GasMapEstimate:
        if obstacle_map.ndim != 2 or sparse_concentration_map.shape != obstacle_map.shape:
            raise ValueError("obstacle and sparse gas maps must be equally-sized 2-D arrays")

        model_shape = (self.inference_size, self.inference_size)
        obstacle = cv2.resize(
            obstacle_map.astype(np.float32), model_shape, interpolation=cv2.INTER_NEAREST
        )
        sparse = cv2.resize(
            sparse_concentration_map.astype(np.float32),
            model_shape,
            interpolation=cv2.INTER_NEAREST,
        )
        inputs = torch.from_numpy(np.stack([obstacle, sparse])[None]).to(self.device)

        with torch.inference_mode():
            output = self.model(inputs)
        prediction = self._coerce_prediction(output)
        concentration = prediction.concentration[0, 0].detach().cpu().numpy()
        concentration = np.clip(concentration, 0.0, 1.0)
        concentration = cv2.resize(
            concentration,
            (obstacle_map.shape[1], obstacle_map.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        concentration[obstacle_map > 0] = 0.0

        return GasMapEstimate(
            concentration=concentration,
            sigma=float(prediction.sigma.reshape(-1)[0].detach().cpu()),
            source_position=(
                prediction.source_position.reshape(-1, 2)[0].detach().cpu().numpy().astype(np.float32)
            ),
        )

    @staticmethod
    def _coerce_prediction(output: Any) -> GasPrediction:
        if isinstance(output, GasPrediction):
            return output
        if isinstance(output, (tuple, list)) and len(output) >= 3:
            attention_maps = output[3] if len(output) > 3 else {}
            return GasPrediction(output[0], output[1], output[2], attention_maps)
        raise TypeError("Gas model must return concentration, sigma, and source position")
