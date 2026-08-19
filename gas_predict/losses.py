"""Finite-difference physics loss from the PI-Attention-UNet experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as functional
from torch import nn

from gas_predict.model import GasPrediction

Direction = Literal["x", "y"]
DifferenceMode = Literal["forward", "backward"]


def directional_derivative(
    field: torch.Tensor,
    direction: Direction = "x",
    mode: DifferenceMode = "forward",
) -> torch.Tensor:
    """Apply the notebook's one-sided finite difference to a 2-D field.

    The last/first value is replicated at the boundary exactly as in the
    selected ``_21`` notebook. Both ``(H, W)`` and ``(B, H, W)`` are accepted.
    """
    if field.ndim not in (2, 3):
        raise ValueError("field must have shape (H, W) or (B, H, W)")
    if direction not in ("x", "y") or mode not in ("forward", "backward"):
        raise ValueError("invalid direction or finite-difference mode")

    squeeze = field.ndim == 2
    values = field[None, None] if squeeze else field[:, None]
    if direction == "x":
        if mode == "forward":
            difference = values[..., 1:] - values[..., :-1]
            derivative = functional.pad(difference, (0, 1, 0, 0), mode="replicate")
        else:
            difference = values[..., :-1] - values[..., 1:]
            derivative = functional.pad(difference, (1, 0, 0, 0), mode="replicate")
    elif mode == "forward":
        difference = values[..., 1:, :] - values[..., :-1, :]
        derivative = functional.pad(difference, (0, 0, 0, 1), mode="replicate")
    else:
        difference = values[..., :-1, :] - values[..., 1:, :]
        derivative = functional.pad(difference, (0, 0, 1, 0), mode="replicate")
    derivative = derivative[:, 0]
    return derivative[0] if squeeze else derivative


def quadrant_directional_physics_loss(
    concentration: torch.Tensor,
    normalized_sigma: torch.Tensor,
    source_position: torch.Tensor,
    *,
    sigma_scale: float = 350.0,
    min_sigma: float = 16.666,
    max_sigma: float = 350.0,
) -> torch.Tensor:
    """Enforce Gaussian decay with source-directed one-sided differences.

    Each source-centered quadrant chooses its own forward/backward difference,
    matching ``physics_loss_quadrant_directional`` from the research notebook.
    ``normalized_sigma`` is converted back to pixel units using 350 by default.
    """
    if concentration.ndim != 4 or concentration.shape[1] != 1:
        raise ValueError("concentration must have shape (B, 1, H, W)")
    batch, _, height, width = concentration.shape
    sigma = normalized_sigma.reshape(batch).mul(sigma_scale).clamp(min_sigma, max_sigma)
    if source_position.shape != (batch, 2):
        raise ValueError("source_position must have shape (B, 2)")

    dtype, device = concentration.dtype, concentration.device
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=dtype, device=device),
        torch.arange(width, dtype=dtype, device=device),
        indexing="ij",
    )
    total = concentration.new_zeros(())
    for index in range(batch):
        field = concentration[index, 0]
        x0 = source_position[index, 0] * width
        y0 = source_position[index, 1] * height
        dx, dy = xx - x0, yy - y0
        quadrants = (
            ((dx >= 0) & (dy >= 0), "forward", "forward"),
            ((dx < 0) & (dy >= 0), "backward", "forward"),
            ((dx < 0) & (dy < 0), "backward", "backward"),
            ((dx >= 0) & (dy < 0), "forward", "backward"),
        )
        sample_loss = concentration.new_zeros(())
        for mask, x_mode, y_mode in quadrants:
            derivative_x = directional_derivative(field, "x", x_mode)
            derivative_y = directional_derivative(field, "y", y_mode)
            decay_x = dx.abs() * field / sigma[index].square()
            decay_y = dy.abs() * field / sigma[index].square()
            if mask.any():
                sample_loss = sample_loss + (derivative_x[mask] + decay_x[mask]).square().mean()
                sample_loss = sample_loss + (derivative_y[mask] + decay_y[mask]).square().mean()
        total = total + sample_loss / 4.0
    return total / batch


@dataclass(frozen=True)
class LossTerms:
    total: torch.Tensor
    reconstruction: torch.Tensor
    sigma: torch.Tensor
    position: torch.Tensor
    physics: torch.Tensor


class PhysicsInformedLoss(nn.Module):
    """Combined supervised and finite-difference loss for PI-Attention-UNet.

    Set ``reconstruction_weight=0`` to reproduce the notebook's final
    ``physics_dwa_no_data`` objective, where reconstruction is monitored but
    not included in the optimized total.
    """

    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        sigma_weight: float = 1.0,
        position_weight: float = 1.0,
        physics_weight: float = 1.0,
    ):
        super().__init__()
        self.weights = (
            reconstruction_weight,
            sigma_weight,
            position_weight,
            physics_weight,
        )

    def forward(
        self,
        prediction: GasPrediction,
        target_concentration: torch.Tensor,
        target_sigma: torch.Tensor,
        target_position: torch.Tensor,
    ) -> LossTerms:
        reconstruction = functional.mse_loss(prediction.concentration, target_concentration)
        sigma = functional.mse_loss(prediction.sigma.reshape_as(target_sigma), target_sigma)
        position = functional.mse_loss(prediction.source_position, target_position)
        physics = quadrant_directional_physics_loss(
            prediction.concentration, prediction.sigma, prediction.source_position
        )
        terms = (reconstruction, sigma, position, physics)
        total = sum(weight * term for weight, term in zip(self.weights, terms))
        return LossTerms(total, reconstruction, sigma, position, physics)


# Compatibility name for code copied from an earlier cleanup revision.
gas_physics_loss = quadrant_directional_physics_loss
