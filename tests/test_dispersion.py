import numpy as np
import pytest

from gas_predict.dispersion import geodesic_distance_map, indoor_gas_concentration


def test_geodesic_distance_routes_around_wall():
    occupancy = np.zeros((5, 5), dtype=np.uint8)
    occupancy[:4, 2] = 1
    distances = geodesic_distance_map((0, 0), occupancy)
    assert distances[0, 4] == 12
    assert np.isinf(distances[0, 2])


def test_indoor_concentration_uses_geodesic_distance_and_masks_obstacles():
    occupancy = np.zeros((5, 5), dtype=np.uint8)
    occupancy[:4, 2] = 1
    gas = indoor_gas_concentration((0, 0), occupancy, sigma=2.0)
    assert gas[0, 0] == pytest.approx(1.0)
    assert gas[0, 2] == 0.0
    assert gas[0, 4] == pytest.approx(np.exp(-(12**2) / 8.0))
    assert gas[0, 4] < gas[4, 2]
