import numpy as np

from rlm.mower_env import MowerEnv


def test_reset_and_step_match_spaces():
    env = MowerEnv(input_size=32, max_episode_steps=2, seed=1)
    observation, reset_info = env.reset()
    assert reset_info == {}
    assert env.observation_space.contains(observation)
    observation, reward, terminated, truncated, info = env.step(np.zeros(1, dtype=np.float32))
    assert env.observation_space.contains(observation)
    assert np.isfinite(reward)
    assert {"concentration", "distance_to_source", "success"} <= info.keys()
    assert isinstance(terminated, (bool, np.bool_))
    assert isinstance(truncated, (bool, np.bool_))
    env.close()
