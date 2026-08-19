"""Gymnasium environment for PI-U-Net-assisted gas-source navigation.

The environment is the cleaned counterpart of the downstream
``Paper_main_pi_unet`` experiment: local gas measurements are reconstructed by
the pretrained PI-U-Net, and an RL policy receives stacked robot-relative
coverage, obstacle, and reconstructed-gas maps plus LiDAR and the predicted
source position. The simulator's true gas field is used only for sensor values,
reward, and termination.
"""

from __future__ import annotations

import math
from pathlib import Path
import random
from typing import Protocol

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gas_source_navigation.gas.dispersion import indoor_gas_concentration
from gas_source_navigation.gas.inference import GasMapEstimate


DEFAULT_MAP_DIR = Path(__file__).resolve().parents[1] / "assets" / "maps"


class GasPredictor(Protocol):
    """Interface between the pretrained PI-U-Net and the RL environment."""

    def predict(
        self, obstacle_map: np.ndarray, sparse_concentration_map: np.ndarray
    ) -> GasMapEstimate: ...


class MowerEnv(gym.Env):
    """Continuous-control gas-source navigation environment.

    ``MowerEnv`` is retained as the class name so old experiment metadata still
    identifies the environment from which this task was derived.
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 10}

    def __init__(
        self,
        map_dir: str | Path | None = None,
        gas_predictor: GasPredictor | None = None,
        eval: bool = False,
        input_size: int = 32,
        num_maps: int = 4,
        scale_factor: float = 4.0,
        stacks: int = 1,
        meters_per_pixel: float = 0.0375,
        step_size: float = 0.5,
        max_lin_vel: float = 0.26,
        max_ang_vel: float = 1.0,
        mower_radius: float = 0.15,
        lidar_rays: int = 24,
        lidar_range: float = 3.5,
        gas_sigma: tuple[float, float] = (100.0, 300.0),
        gas_sensor_radius: int = 5,
        unobserved_gas_value: float = 0.001,
        goal_concentration: float = 0.98,
        max_episode_steps: int = 10_000,
        collision_reward: float = -10.0,
        goal_reward: float = 100.0,
        step_reward: float = -0.1,
        gas_reward_scale: float = 1.5,
        coverage_reward_scale: float = 1.0,
        overlap_reward_scale: float = 0.0,
        seed: int | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        if input_size < 16:
            raise ValueError("input_size must be at least 16")
        if num_maps < 1 or stacks < 1:
            raise ValueError("num_maps and stacks must be positive")
        if scale_factor < 1:
            raise ValueError("scale_factor must be at least one")
        if not 0 < goal_concentration <= 1:
            raise ValueError("goal_concentration must be in (0, 1]")
        if gas_predictor is None:
            raise ValueError("gas_predictor is required so RL never observes the true gas map")
        if gas_sensor_radius < 0:
            raise ValueError("gas_sensor_radius must be non-negative")
        if not 0 <= unobserved_gas_value <= 1:
            raise ValueError("unobserved_gas_value must be in [0, 1]")

        self.map_dir = DEFAULT_MAP_DIR if map_dir is None else Path(map_dir)
        self.gas_predictor = gas_predictor
        self.eval_mode = eval
        self.input_size = input_size
        self.num_maps = num_maps
        self.scale_factor = scale_factor
        self.stacks = stacks
        self.meters_per_pixel = meters_per_pixel
        self.pixels_per_meter = 1.0 / meters_per_pixel
        self.initial_step_size = step_size
        self.step_size = step_size
        self.max_lin_vel = max_lin_vel
        self.max_ang_vel = max_ang_vel
        self.mower_radius = mower_radius
        self.lidar_rays = lidar_rays
        self.lidar_range = lidar_range
        self.gas_sigma = gas_sigma
        self.gas_sensor_radius = gas_sensor_radius
        self.unobserved_gas_value = unobserved_gas_value
        self.goal_concentration = goal_concentration
        self.max_episode_steps = max_episode_steps
        self.collision_reward = collision_reward
        self.goal_reward = goal_reward
        self.step_reward = step_reward
        self.gas_reward_scale = gas_reward_scale
        self.coverage_reward_scale = coverage_reward_scale
        self.overlap_reward_scale = overlap_reward_scale
        self._rng = np.random.default_rng(seed)
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        self.render_mode = render_mode

        prefix = "eval_" if eval else "train_"
        self.map_files = sorted(self.map_dir.glob(f"{prefix}*.png"))
        if not self.map_files:
            raise FileNotFoundError(f"No {prefix}*.png maps found in {self.map_dir}")

        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        map_shape = (stacks, num_maps, input_size, input_size)
        self.observation_space = spaces.Dict(
            {
                "coverage": spaces.Box(0.0, 1.0, map_shape, np.float32),
                "obstacles": spaces.Box(0.0, 1.0, map_shape, np.float32),
                "gas": spaces.Box(0.0, 1.0, map_shape, np.float32),
                "lidar": spaces.Box(0.0, 1.0, (stacks, lidar_rays), np.float32),
                "pred_pos": spaces.Box(0.0, 1.0, (2,), np.float32),
            }
        )
        self._map_index = 0
        self.elapsed_steps = 0
        self.num_collisions = 0

    def seed(self, seed: int | None = None) -> list[int | None]:
        self._rng = np.random.default_rng(seed)
        random.seed(seed)
        return [seed]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        del options
        if seed is not None:
            self.seed(seed)
        self._load_map(self.map_files[self._map_index])
        self._map_index = (self._map_index + 1) % len(self.map_files)

        free = np.argwhere(self.obstacle_map == 0)
        source_index = int(self._rng.integers(len(free)))
        safe = self._safe_start_positions(free)
        start_index = int(self._rng.integers(len(safe)))
        self.source_position = free[source_index].astype(np.float32)
        self.position_p = safe[start_index].astype(np.float32)
        self.heading = float(self._rng.uniform(-np.pi, np.pi))
        self.gas_sigma_value = float(self._rng.uniform(*self.gas_sigma))
        self.true_concentration_map = self._gas_map(
            self.source_position, self.gas_sigma_value
        )
        self.sparse_concentration_map = np.full(
            self.obstacle_map.shape, self.unobserved_gas_value, dtype=np.float32
        )
        self.gas_measurement_mask = np.zeros(self.obstacle_map.shape, dtype=bool)
        self.coverage_map = np.zeros(self.obstacle_map.shape, dtype=np.float32)
        self.overlap_map = np.zeros(self.obstacle_map.shape, dtype=np.float32)
        self._update_coverage(self.position_p, self.position_p)
        self._measure_gas()
        self._update_gas_prediction()

        self.elapsed_steps = 0
        self.num_collisions = 0
        self.step_size = self.initial_step_size
        self.path = [self.position_p.copy()]
        frame = self._current_frame()
        self._frames = {
            key: np.repeat(value[None], self.stacks, axis=0)
            for key, value in frame.items()
        }
        return self._observation(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(1)
        steering = float(np.clip(action[0], -1.0, 1.0))
        old_position = self.position_p.copy()
        self.heading = (self.heading + steering * self.max_ang_vel * self.step_size) % (
            2 * np.pi
        )
        distance_p = self.max_lin_vel * self.step_size * self.pixels_per_meter
        candidate = self.position_p + distance_p * np.array(
            [np.cos(self.heading), np.sin(self.heading)], dtype=np.float32
        )

        collided = self._collides(candidate)
        if collided:
            self.num_collisions += 1
        else:
            self.position_p = candidate

        newly_covered, overlap = self._update_coverage(old_position, self.position_p)
        self.elapsed_steps += 1
        self.path.append(self.position_p.copy())
        self._measure_gas()
        self._update_gas_prediction()
        self._append_frame(self._current_frame())

        concentration = self._current_concentration()
        distance_to_source = float(np.linalg.norm(self.position_p - self.source_position))
        diagonal = float(np.linalg.norm(self.obstacle_map.shape))
        distance_reward = -distance_to_source / diagonal
        swept_area = max(
            1.0,
            2 * self.mower_radius * self.pixels_per_meter * max(distance_p, 1.0),
        )
        coverage_reward = self.coverage_reward_scale * min(newly_covered / swept_area, 2.0)
        overlap_reward = -self.overlap_reward_scale * overlap / swept_area
        reward = (
            self.step_reward
            + self.gas_reward_scale * concentration
            + distance_reward
            + coverage_reward
            + overlap_reward
        )
        if collided:
            reward += self.collision_reward

        reached_source = concentration > self.goal_concentration
        timed_out = self.elapsed_steps >= self.max_episode_steps
        if reached_source:
            reward += self.goal_reward

        # Adaptive time step used by the downstream Paper_main_pi_unet task.
        self.step_size = float(
            1.0 - 0.8 / (1.0 + np.exp(-20.0 * (concentration - 0.7)))
        )
        info = {
            "concentration": concentration,
            "distance_to_source": distance_to_source,
            "coverage": float(self.coverage_map.mean()),
            "success": reached_source,
            "predicted_sigma": self.predicted_sigma,
            "predicted_source_position": self.predicted_source_position.copy(),
        }
        if self.render_mode == "human":
            self.render()
        return self._observation(), float(reward), reached_source, timed_out, info

    def _load_map(self, filename: Path) -> None:
        image = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read map: {filename}")
        if image.ndim != 2:
            raise ValueError(f"Map must be grayscale: {filename}")
        self.obstacle_map = (image < 255).astype(np.uint8)
        self.size_p = image.shape[0]

    def _safe_start_positions(self, free: np.ndarray) -> np.ndarray:
        radius = max(1, int(round(self.mower_radius * self.pixels_per_meter)))
        clearance = cv2.distanceTransform(1 - self.obstacle_map, cv2.DIST_L2, 5)
        safe = np.argwhere(clearance > radius)
        return safe if len(safe) else free

    def _gas_map(self, source: np.ndarray, sigma: float) -> np.ndarray:
        return indoor_gas_concentration(
            tuple(np.rint(source).astype(int)), self.obstacle_map, sigma=sigma
        )

    def _collides(self, position: np.ndarray) -> bool:
        radius = max(1, int(round(self.mower_radius * self.pixels_per_meter)))
        y, x = np.rint(position).astype(int)
        height, width = self.obstacle_map.shape
        if y - radius < 0 or x - radius < 0 or y + radius >= height or x + radius >= width:
            return True
        patch = self.obstacle_map[
            y - radius : y + radius + 1,
            x - radius : x + radius + 1,
        ]
        return bool(patch.any())

    def _current_concentration(self) -> float:
        y, x = np.clip(
            np.rint(self.position_p).astype(int),
            0,
            np.array(self.obstacle_map.shape) - 1,
        )
        return float(self.true_concentration_map[y, x])

    def _measure_gas(self) -> None:
        """Add the current circular sensor footprint to the accumulated map."""
        center_y, center_x = np.clip(
            np.rint(self.position_p).astype(int),
            0,
            np.array(self.obstacle_map.shape) - 1,
        )
        radius = self.gas_sensor_radius
        y_min = max(0, center_y - radius)
        y_max = min(self.obstacle_map.shape[0], center_y + radius + 1)
        x_min = max(0, center_x - radius)
        x_max = min(self.obstacle_map.shape[1], center_x + radius + 1)
        yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
        footprint = (yy - center_y) ** 2 + (xx - center_x) ** 2 <= radius**2
        self.gas_measurement_mask[y_min:y_max, x_min:x_max][footprint] = True
        sparse = self.sparse_concentration_map[y_min:y_max, x_min:x_max]
        truth = self.true_concentration_map[y_min:y_max, x_min:x_max]
        sparse[footprint] = truth[footprint]

    def _update_gas_prediction(self) -> None:
        estimate = self.gas_predictor.predict(
            self.obstacle_map, self.sparse_concentration_map
        )
        concentration = np.asarray(estimate.concentration, dtype=np.float32)
        if concentration.shape != self.obstacle_map.shape:
            raise ValueError(
                "gas predictor returned shape "
                f"{concentration.shape}, expected {self.obstacle_map.shape}"
            )
        if not np.isfinite(concentration).all():
            raise ValueError("gas predictor returned non-finite concentration values")
        self.predicted_concentration_map = np.clip(concentration, 0.0, 1.0)
        self.predicted_concentration_map[self.obstacle_map > 0] = 0.0
        self.predicted_sigma = float(estimate.sigma)
        self.predicted_source_position = np.clip(
            np.asarray(estimate.source_position, dtype=np.float32).reshape(2),
            0.0,
            1.0,
        )

    def _update_coverage(
        self, old_position: np.ndarray, new_position: np.ndarray
    ) -> tuple[int, int]:
        swept = np.zeros(self.obstacle_map.shape, dtype=np.uint8)
        thickness = max(1, 2 * int(round(self.mower_radius * self.pixels_per_meter)))
        old_xy = tuple(np.flip(np.rint(old_position).astype(int)))
        new_xy = tuple(np.flip(np.rint(new_position).astype(int)))
        cv2.line(swept, old_xy, new_xy, 1, thickness=thickness)
        swept[self.obstacle_map > 0] = 0
        visited = self.coverage_map > 0
        newly_covered = int(np.count_nonzero((swept > 0) & ~visited))
        overlap = int(np.count_nonzero((swept > 0) & visited))
        self.overlap_map[swept > 0] += 1
        self.coverage_map[swept > 0] = 1
        return newly_covered, overlap

    def _lidar(self) -> np.ndarray:
        result = np.ones(self.lidar_rays, dtype=np.float32)
        max_range_p = self.lidar_range * self.pixels_per_meter
        angles = self.heading + np.linspace(-np.pi, np.pi, self.lidar_rays, endpoint=False)
        for ray, angle in enumerate(angles):
            for distance in np.linspace(1.0, max_range_p, max(2, int(max_range_p))):
                point = self.position_p + distance * np.array(
                    [np.cos(angle), np.sin(angle)]
                )
                y, x = np.rint(point).astype(int)
                if (
                    y < 0
                    or x < 0
                    or y >= self.obstacle_map.shape[0]
                    or x >= self.obstacle_map.shape[1]
                    or self.obstacle_map[y, x]
                ):
                    result[ray] = distance / max_range_p
                    break
        return result

    def _relative_map(self, source: np.ndarray, scale: float, pad_value: float) -> np.ndarray:
        scale = min(scale, float(min(source.shape)))
        resized = cv2.resize(
            source.astype(np.float32),
            (
                max(1, int(round(source.shape[1] / scale))),
                max(1, int(round(source.shape[0] / scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )
        translation_1 = np.eye(3, dtype=np.float32)
        translation_1[0, 2] = -self.position_p[1] / scale
        translation_1[1, 2] = -self.position_p[0] / scale
        rotation = np.eye(3, dtype=np.float32)
        rotation[:2] = cv2.getRotationMatrix2D(
            (0, 0), 90.0 - math.degrees(self.heading), 1.0
        )
        translation_2 = np.eye(3, dtype=np.float32)
        translation_2[0, 2] = self.input_size / 2
        translation_2[1, 2] = self.input_size / 2
        matrix = translation_2 @ rotation @ translation_1
        return cv2.warpAffine(
            resized,
            matrix[:2],
            (self.input_size, self.input_size),
            flags=cv2.INTER_AREA,
            borderValue=pad_value,
        ).astype(np.float32)

    def _multi_scale_map(self, source: np.ndarray, pad_value: float) -> np.ndarray:
        return np.stack(
            [
                self._relative_map(source, self.scale_factor**index, pad_value)
                for index in range(self.num_maps)
            ]
        ).astype(np.float32)

    def _current_frame(self) -> dict[str, np.ndarray]:
        return {
            "coverage": self._multi_scale_map(self.coverage_map, 0.0),
            "obstacles": self._multi_scale_map(self.obstacle_map, 1.0),
            "gas": self._multi_scale_map(self.predicted_concentration_map, 0.0),
            "lidar": self._lidar(),
        }

    def _append_frame(self, frame: dict[str, np.ndarray]) -> None:
        for key, value in frame.items():
            self._frames[key] = np.roll(self._frames[key], -1, axis=0)
            self._frames[key][-1] = value

    def _observation(self) -> dict[str, np.ndarray]:
        return {
            "coverage": self._frames["coverage"].copy(),
            "obstacles": self._frames["obstacles"].copy(),
            "gas": self._frames["gas"].copy(),
            "lidar": self._frames["lidar"].copy(),
            "pred_pos": self.predicted_source_position.copy(),
        }

    def render(self):
        gas = np.uint8(np.clip(self.predicted_concentration_map, 0, 1) * 255)
        image = cv2.applyColorMap(gas, cv2.COLORMAP_VIRIDIS)
        image[self.obstacle_map > 0] = 0
        source = tuple(np.flip(np.rint(self.source_position).astype(int)))
        agent = tuple(np.flip(np.rint(self.position_p).astype(int)))
        cv2.circle(image, source, 3, (0, 0, 255), -1)
        cv2.circle(image, agent, 3, (255, 0, 0), -1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.render_mode == "human":
            cv2.imshow("Gas source navigation", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        return image if self.render_mode == "rgb_array" else None

    def close(self):
        cv2.destroyAllWindows()
