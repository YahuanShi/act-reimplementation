"""
Unit tests for EMAModel.

Validates:
  - Decay schedule increases monotonically from 0 toward max_value
  - After step(), EMA parameters are a blend of initial and updated values
  - Buffers (not just parameters) are copied during step()
  - copy_to() transfers EMA weights back to a model
  - state_dict / load_state_dict roundtrip preserves optimization_step and decay
"""

import torch
import torch.nn as nn
import pytest

from act.model.ema import EMAModel


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.register_buffer("running_sum", torch.zeros(4))


def make_model():
    model = SimpleModel()
    nn.init.ones_(model.linear.weight)
    nn.init.zeros_(model.linear.bias)
    model.running_sum.fill_(0.0)
    return model


def test_decay_increases_monotonically():
    model = make_model()
    ema = EMAModel(model, power=2 / 3, max_value=0.9999)
    decays = [ema.get_decay(s) for s in range(200)]
    for i in range(len(decays) - 1):
        assert decays[i] <= decays[i + 1], f"decay not monotonic at step {i}"


def test_decay_bounded():
    model = make_model()
    ema = EMAModel(model, min_value=0.0, max_value=0.9999)
    for step in [0, 1, 10, 100, 10000]:
        d = ema.get_decay(step)
        assert 0.0 <= d <= 0.9999


def test_ema_step_blends_parameters():
    model = make_model()
    ema = EMAModel(model, power=2 / 3, max_value=0.9999)

    # After 1000 warm-up steps the decay is non-trivial
    for _ in range(1000):
        ema.step(model)

    decay = ema.decay
    # Change model weights to 0
    with torch.no_grad():
        model.linear.weight.fill_(0.0)

    ema.step(model)

    # EMA weight should be decay * 1 + (1-decay) * 0 = decay
    expected = decay
    actual = ema.averaged_model.linear.weight[0, 0].item()
    assert abs(actual - expected) < 1e-5, f"expected {expected:.5f}, got {actual:.5f}"


def test_ema_copies_buffers():
    model = make_model()
    ema = EMAModel(model)

    # Modify the buffer in the live model
    model.running_sum.fill_(42.0)
    ema.step(model)

    assert torch.all(ema.averaged_model.running_sum == 42.0), "buffer not copied"


def test_copy_to_transfers_ema_weights():
    model = make_model()
    ema = EMAModel(model)

    # Train for many steps so EMA diverges from current weights
    for _ in range(500):
        ema.step(model)

    # Change model weights significantly
    with torch.no_grad():
        model.linear.weight.fill_(99.0)

    # copy_to should overwrite model with EMA weights
    ema_weight = ema.averaged_model.linear.weight.clone()
    ema.copy_to(model)
    assert torch.allclose(model.linear.weight, ema_weight), "copy_to did not transfer weights"


def test_state_dict_roundtrip():
    model = make_model()
    ema = EMAModel(model)
    for _ in range(50):
        ema.step(model)

    sd = ema.state_dict()
    ema2 = EMAModel(make_model())
    ema2.load_state_dict(sd)

    assert ema2.optimization_step == ema.optimization_step
    assert abs(ema2.decay - ema.decay) < 1e-8
    assert torch.allclose(
        ema2.averaged_model.linear.weight,
        ema.averaged_model.linear.weight,
    )
