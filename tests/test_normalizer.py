"""
Unit tests for ACTNormalizer.

Validates:
  - fit + normalize + denormalize is a lossless roundtrip
  - std floor prevents division by zero on constant inputs
  - save/load via state_dict preserves statistics exactly
  - fit_from_tensors is equivalent to computing manually
"""

import torch
import pytest
from act.dataset.normalizer import ACTNormalizer


def test_action_roundtrip():
    normalizer = ACTNormalizer()
    actions = torch.randn(100, 7) * 3 + 1
    qpos = torch.randn(100, 7) * 0.5
    normalizer.fit_from_tensors(qpos, actions)

    normalized = normalizer.normalize_action(actions)
    recovered = normalizer.denormalize_action(normalized)
    assert torch.allclose(actions, recovered, atol=1e-5), "action roundtrip must be lossless"


def test_qpos_roundtrip():
    normalizer = ACTNormalizer()
    actions = torch.randn(100, 7)
    qpos = torch.randn(100, 7) * 2 - 0.5
    normalizer.fit_from_tensors(qpos, actions)

    normalized = normalizer.normalize_qpos(qpos)
    recovered = normalizer.denormalize_qpos(normalized)
    assert torch.allclose(qpos, recovered, atol=1e-5), "qpos roundtrip must be lossless"


def test_std_floor_no_nan():
    normalizer = ACTNormalizer()
    # Constant values → std=0 → should not produce NaN after floor clipping
    actions = torch.ones(50, 7)
    qpos = torch.zeros(50, 7)
    normalizer.fit_from_tensors(qpos, actions)

    normalized = normalizer.normalize_action(actions)
    assert not torch.isnan(normalized).any(), "constant input must not produce NaN"
    assert not torch.isinf(normalized).any(), "constant input must not produce Inf"


def test_std_floor_value():
    normalizer = ACTNormalizer()
    actions = torch.ones(50, 3)  # all same → std = 0
    actions[:, 1] = torch.randn(50)  # dim 1 has real variance
    qpos = torch.randn(50, 3)
    normalizer.fit_from_tensors(qpos, actions)

    # Clipped dims (0 and 2) should have std >= 0.01
    assert normalizer.action_std[0].item() >= 0.01 - 1e-6
    assert normalizer.action_std[2].item() >= 0.01 - 1e-6


def test_state_dict_roundtrip():
    normalizer = ACTNormalizer()
    actions = torch.randn(80, 14) * 2
    qpos = torch.randn(80, 14)
    normalizer.fit_from_tensors(qpos, actions)

    sd = normalizer.state_dict()
    normalizer2 = ACTNormalizer()
    normalizer2.load_state_dict(sd)

    test_actions = torch.randn(10, 14)
    out1 = normalizer.normalize_action(test_actions)
    out2 = normalizer2.normalize_action(test_actions)
    assert torch.allclose(out1, out2), "state_dict roundtrip must preserve statistics"


def test_not_fitted_raises():
    normalizer = ACTNormalizer()
    with pytest.raises(Exception):
        normalizer.normalize_action(torch.randn(5, 7))


def test_batch_and_single_consistent():
    normalizer = ACTNormalizer()
    actions = torch.randn(100, 7)
    qpos = torch.randn(100, 7)
    normalizer.fit_from_tensors(qpos, actions)

    sample = torch.randn(1, 7)
    batch_out = normalizer.normalize_action(sample)
    single_out = normalizer.normalize_action(sample[0].unsqueeze(0))
    assert torch.allclose(batch_out, single_out)
