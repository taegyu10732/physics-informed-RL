"""Obstacle-aware indoor gas dispersion used by the experiments."""

from __future__ import annotations

from collections import deque

import numpy as np


def geodesic_distance_map(source: tuple[int, int], occupancy_map: np.ndarray) -> np.ndarray:
    """Return four-neighbor shortest-path distances through free indoor space."""
    occupancy = np.asarray(occupancy_map, dtype=bool)
    if occupancy.ndim != 2:
        raise ValueError("occupancy_map must be a 2-D array")
    row, column = source
    if not (0 <= row < occupancy.shape[0] and 0 <= column < occupancy.shape[1]):
        raise ValueError("source is outside occupancy_map")
    if occupancy[row, column]:
        raise ValueError("source must be in free space")

    distances = np.full(occupancy.shape, np.inf, dtype=np.float32)
    distances[row, column] = 0.0
    queue = deque([(row, column)])
    while queue:
        current_row, current_column = queue.popleft()
        next_distance = distances[current_row, current_column] + 1.0
        for next_row, next_column in (
            (current_row - 1, current_column),
            (current_row + 1, current_column),
            (current_row, current_column - 1),
            (current_row, current_column + 1),
        ):
            if (
                0 <= next_row < occupancy.shape[0]
                and 0 <= next_column < occupancy.shape[1]
                and not occupancy[next_row, next_column]
                and np.isinf(distances[next_row, next_column])
            ):
                distances[next_row, next_column] = next_distance
                queue.append((next_row, next_column))
    return distances


def indoor_gas_concentration(
    source: tuple[int, int],
    occupancy_map: np.ndarray,
    *,
    source_strength: float = 1.0,
    sigma: float = 200.0,
) -> np.ndarray:
    """Generate the obstacle-aware Gaussian concentration field.

    ``sigma`` is expressed in grid cells, as in the original notebooks.
    Obstacles and free-space regions unreachable from the source receive zero.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    distances = geodesic_distance_map(source, occupancy_map)
    concentration = source_strength * np.exp(-(distances**2) / (2.0 * sigma**2))
    concentration[~np.isfinite(distances)] = 0.0
    concentration[np.asarray(occupancy_map, dtype=bool)] = 0.0
    return concentration.astype(np.float32)
