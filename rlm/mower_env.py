"""Gym environment for locating a simulated gas source.

This is the project-specific core extracted from the original coverage-path
planning environment. It intentionally supports one task: navigate around map
obstacles until the measured gas concentration reaches the target threshold.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
import random

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gas_predict.dispersion import indoor_gas_concentration


class MowerEnv(gym.Env):
    """Continuous-control gas-source navigation environment.

    The legacy class name is retained so existing experiment metadata remains
    loadable. Observations contain a two-channel global map (obstacles and gas),
    normalized lidar ranges, and the most recent normalized positions.
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 10}

    def __init__(
        self,
        map_dir: str | Path = "maps",
        eval: bool = False,
        input_size: int = 64,
        meters_per_pixel: float = 0.0375,
        step_size: float = 0.5,
        max_lin_vel: float = 0.26,
        max_ang_vel: float = 1.0,
        mower_radius: float = 0.15,
        lidar_rays: int = 24,
        lidar_range: float = 3.5,
        trajectory_length: int = 200,
        gas_sigma: tuple[float, float] = (100.0, 300.0),
        goal_concentration: float = 0.98,
        max_episode_steps: int = 10_000,
        collision_reward: float = -10.0,
        goal_reward: float = 100.0,
        step_reward: float = -0.1,
        seed: int | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        if input_size < 16:
            raise ValueError("input_size must be at least 16")
        if not 0 < goal_concentration <= 1:
            raise ValueError("goal_concentration must be in (0, 1]")

        self.map_dir = Path(map_dir)
        self.eval_mode = eval
        self.input_size = input_size
        self.meters_per_pixel = meters_per_pixel
        self.pixels_per_meter = 1.0 / meters_per_pixel
        self.base_step_size = step_size
        self.step_size = step_size
        self.max_lin_vel = max_lin_vel
        self.max_ang_vel = max_ang_vel
        self.mower_radius = mower_radius
        self.lidar_rays = lidar_rays
        self.lidar_range = lidar_range
        self.trajectory_length = trajectory_length
        self.gas_sigma = gas_sigma
        self.goal_concentration = goal_concentration
        self.max_episode_steps = max_episode_steps
        self.collision_reward = collision_reward
        self.goal_reward = goal_reward
        self.step_reward = step_reward
        self._rng = np.random.default_rng(seed)
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        self.render_mode = render_mode

        prefix = "eval_" if eval else "train_"
        self.map_files = sorted(self.map_dir.glob(f"{prefix}*.png"))
        if not self.map_files:
            raise FileNotFoundError(f"No {prefix}*.png maps found in {self.map_dir}")

        # The project trains steering only; forward speed is controlled by the
        # concentration-dependent step size below.
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "map": spaces.Box(0.0, 1.0, (2, input_size, input_size), np.float32),
                "lidar": spaces.Box(0.0, 1.0, (lidar_rays,), np.float32),
                "trajectory": spaces.Box(
                    0.0, 1.0, (trajectory_length, 2), np.float32
                ),
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
        if seed is not None:
            self.seed(seed)
        self._load_map(self.map_files[self._map_index])
        self._map_index = (self._map_index + 1) % len(self.map_files)
        free = np.argwhere(self.obstacle_map == 0)
        source_idx, start_idx = self._rng.choice(len(free), size=2, replace=False)
        self.source_position = free[source_idx].astype(np.float32)
        self.position_p = free[start_idx].astype(np.float32)
        self.heading = float(self._rng.uniform(-np.pi, np.pi))
        self.gas_sigma_value = float(self._rng.uniform(*self.gas_sigma))
        self.concentration_map = self._gas_map(self.source_position, self.gas_sigma_value)
        self.trajectory = deque(maxlen=self.trajectory_length)
        normalized = self.position_p / np.array(self.obstacle_map.shape, np.float32)
        self.trajectory.extend([normalized.copy()] * self.trajectory_length)
        self.elapsed_steps = 0
        self.num_collisions = 0
        self.path = [self.position_p.copy()]
        return self._observation(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(1)
        steering = float(np.clip(action[0], -1.0, 1.0))
        self.heading += steering * self.max_ang_vel * self.step_size
        distance_p = self.max_lin_vel * self.step_size * self.pixels_per_meter
        candidate = self.position_p + distance_p * np.array(
            [np.cos(self.heading), np.sin(self.heading)], dtype=np.float32
        )

        collided = self._collides(candidate)
        if collided:
            self.num_collisions += 1
        else:
            self.position_p = candidate

        self.elapsed_steps += 1
        normalized = self.position_p / np.array(self.obstacle_map.shape, np.float32)
        self.trajectory.append(normalized)
        self.path.append(self.position_p.copy())

        concentration = self._current_concentration()
        distance_to_source = np.linalg.norm(self.position_p - self.source_position)
        diagonal = np.linalg.norm(self.obstacle_map.shape)
        reward = self.step_reward + concentration - distance_to_source / diagonal
        if collided:
            reward += self.collision_reward
        reached_source = concentration >= self.goal_concentration
        timed_out = self.elapsed_steps >= self.max_episode_steps
        if reached_source:
            reward += self.goal_reward

        # Slow down close to the source, preserving the behavior of the research code.
        self.step_size = self.base_step_size * (
            1.0
            - 0.5 / (1.0 + np.exp(-30.0 * (concentration - 0.5)))
            - 0.3 / (1.0 + np.exp(-30.0 * (concentration - 0.9)))
        )
        info = {
            "concentration": concentration,
            "distance_to_source": float(distance_to_source),
            "success": reached_source,
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

    def _gas_map(self, source: np.ndarray, sigma: float) -> np.ndarray:
        return indoor_gas_concentration(
            tuple(np.rint(source).astype(int)), self.obstacle_map, sigma=sigma
        )

    def _collides(self, position: np.ndarray) -> bool:
        radius = max(1, int(round(self.mower_radius * self.pixels_per_meter)))
        y, x = np.rint(position).astype(int)
        h, w = self.obstacle_map.shape
        if y - radius < 0 or x - radius < 0 or y + radius >= h or x + radius >= w:
            return True
        patch = self.obstacle_map[y - radius : y + radius + 1, x - radius : x + radius + 1]
        return bool(patch.any())

    def _current_concentration(self) -> float:
        y, x = np.clip(
            np.rint(self.position_p).astype(int),
            0,
            np.array(self.obstacle_map.shape) - 1,
        )
        return float(self.concentration_map[y, x])

    def _lidar(self) -> np.ndarray:
        result = np.ones(self.lidar_rays, dtype=np.float32)
        max_range_p = self.lidar_range * self.pixels_per_meter
        angles = self.heading + np.linspace(-np.pi, np.pi, self.lidar_rays, endpoint=False)
        for ray, angle in enumerate(angles):
            for distance in np.linspace(1.0, max_range_p, max(2, int(max_range_p))):
                point = self.position_p + distance * np.array([np.cos(angle), np.sin(angle)])
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

    def _observation(self) -> dict[str, np.ndarray]:
        obstacle = cv2.resize(
            self.obstacle_map.astype(np.float32),
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_NEAREST,
        )
        gas = cv2.resize(
            self.concentration_map,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_AREA,
        )
        return {
            "map": np.stack([obstacle, gas]).astype(np.float32),
            "lidar": self._lidar(),
            "trajectory": np.asarray(self.trajectory, dtype=np.float32),
        }

    def render(self):
        gas = np.uint8(np.clip(self.concentration_map, 0, 1) * 255)
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
