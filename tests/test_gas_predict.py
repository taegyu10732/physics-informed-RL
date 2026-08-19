import pytest
import torch

from gas_predict.model import AttentionUNet
from gas_predict.losses import directional_derivative, quadrant_directional_physics_loss


def test_attention_unet_output_shapes():
    model = AttentionUNet(base_channels=8).eval()
    with torch.no_grad():
        prediction = model(torch.zeros(2, 2, 32, 32))
    assert prediction.concentration.shape == (2, 1, 32, 32)
    assert prediction.sigma.shape == (2, 1)
    assert prediction.source_position.shape == (2, 2)
    assert torch.all((prediction.source_position >= 0) & (prediction.source_position <= 1))


def test_attention_unet_rejects_invalid_size():
    with pytest.raises(ValueError):
        AttentionUNet(base_channels=8)(torch.zeros(1, 2, 30, 32))


def test_directional_derivative_matches_notebook_difference():
    field = torch.tensor([[0.0, 1.0, 3.0], [4.0, 6.0, 9.0]])
    assert torch.equal(
        directional_derivative(field, "x", "forward"),
        torch.tensor([[1.0, 2.0, 2.0], [2.0, 3.0, 3.0]]),
    )
    assert torch.equal(
        directional_derivative(field, "y", "backward"),
        torch.tensor([[-4.0, -5.0, -6.0], [-4.0, -5.0, -6.0]]),
    )


def test_quadrant_physics_loss_is_differentiable():
    field = torch.rand(2, 1, 16, 16, requires_grad=True)
    loss = quadrant_directional_physics_loss(
        field, torch.full((2, 1), 0.5), torch.full((2, 2), 0.5)
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert field.grad is not None
