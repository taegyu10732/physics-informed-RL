import gymnasium as gym
import torch

from gas_source_navigation.rl.architectures import StackedMapFeaturesExtractor


def test_stacked_map_extractor_uses_paper_observation_contract():
    space = gym.spaces.Dict(
        {
            "coverage": gym.spaces.Box(0, 1, (2, 4, 32, 32), dtype=float),
            "obstacles": gym.spaces.Box(0, 1, (2, 4, 32, 32), dtype=float),
            "gas": gym.spaces.Box(0, 1, (2, 4, 32, 32), dtype=float),
            "lidar": gym.spaces.Box(0, 1, (2, 24), dtype=float),
            "pred_pos": gym.spaces.Box(0, 1, (2,), dtype=float),
        }
    )
    extractor = StackedMapFeaturesExtractor(space, features_dim=64)
    observations = {
        key: torch.zeros((3, *subspace.shape), dtype=torch.float32)
        for key, subspace in space.spaces.items()
    }
    output = extractor(observations)
    assert output.shape == (3, 64)
    assert extractor.shared_cnn.layers[0].in_channels == 12
    assert extractor.cross_attention.embed_dim == 64
