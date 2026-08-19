import numpy as np

from gas_source_navigation.gas.inference import GasMapEstimate
from gas_source_navigation.rl.environment import MowerEnv


class RecordingPredictor:
    def __init__(self):
        self.sparse_inputs = []

    def predict(self, obstacle_map, sparse_concentration_map):
        self.sparse_inputs.append(sparse_concentration_map.copy())
        value = len(self.sparse_inputs) / 10.0
        return GasMapEstimate(
            concentration=np.full(obstacle_map.shape, value, dtype=np.float32),
            sigma=0.5,
            source_position=np.array([0.25, 0.75], dtype=np.float32),
        )


def test_reset_and_step_match_spaces():
    predictor = RecordingPredictor()
    env = MowerEnv(
        gas_predictor=predictor,
        input_size=32,
        max_episode_steps=2,
        seed=1,
    )
    observation, reset_info = env.reset()
    assert reset_info == {}
    assert env.observation_space.contains(observation)
    assert len(predictor.sparse_inputs) == 1
    assert np.isclose(observation["gas"].max(), 0.1)
    assert observation["coverage"].shape == (1, 4, 32, 32)
    assert observation["obstacles"].shape == (1, 4, 32, 32)
    assert observation["lidar"].shape == (1, 24)
    assert np.array_equal(observation["pred_pos"], np.array([0.25, 0.75]))
    assert "trajectory" not in observation
    assert not np.array_equal(env.predicted_concentration_map, env.true_concentration_map)

    observation, reward, terminated, truncated, info = env.step(np.zeros(1, dtype=np.float32))
    assert env.observation_space.contains(observation)
    assert len(predictor.sparse_inputs) == 2
    assert np.isclose(observation["gas"].max(), 0.2)
    measured_before = predictor.sparse_inputs[0] != env.unobserved_gas_value
    measured_after = predictor.sparse_inputs[1] != env.unobserved_gas_value
    assert measured_after.sum() >= measured_before.sum()
    assert np.isfinite(reward)
    assert {
        "concentration",
        "distance_to_source",
        "success",
        "predicted_sigma",
        "predicted_source_position",
    } <= info.keys()
    assert isinstance(terminated, (bool, np.bool_))
    assert isinstance(truncated, (bool, np.bool_))
    env.close()
