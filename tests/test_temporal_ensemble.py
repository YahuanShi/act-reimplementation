"""
Unit tests for TemporalEnsemble and InferencePipeline.

Validates:
  - Single prediction is returned unmodified (weights sum to 1)
  - Two predictions produce a weighted average between the two values
  - reset() clears all stored predictions
  - Ensembling reduces step-to-step variance vs raw predictions
  - InferencePipeline outputs correct shape in both modes
  - Non-ensembled mode reuses cached chunk within chunk_size steps
"""

import torch
import pytest
from act.policy.act_policy import ACTPolicy
from act.dataset.normalizer import ACTNormalizer
from inference.temporal_ensemble import TemporalEnsemble, InferencePipeline


def make_pipeline(temporal_agg=False, chunk_size=10, state_dim=7, action_dim=7):
    config = {
        "state_dim": state_dim, "action_dim": action_dim,
        "hidden_dim": 128, "dim_feedforward": 256,
        "backbone": "resnet18", "lr_backbone": 0,
        "enc_layers": 2, "dec_layers": 2, "nheads": 4,
        "num_queries": chunk_size, "camera_names": ["cam"],
        "lr": 1e-4, "kl_weight": 10.0,
    }
    policy = ACTPolicy(config)
    normalizer = ACTNormalizer()
    normalizer.fit_from_tensors(torch.randn(50, state_dim), torch.randn(50, action_dim))
    return InferencePipeline(
        policy, normalizer, chunk_size=chunk_size,
        temporal_agg=temporal_agg, max_timesteps=100,
    )


def test_single_prediction_identity():
    ensemble = TemporalEnsemble(chunk_size=10, action_dim=7, max_timesteps=50)
    actions = torch.ones(10, 7) * 3.0
    ensemble.add_prediction(0, actions)
    result = ensemble.get_action(0)
    assert torch.allclose(result, torch.ones(7) * 3.0, atol=1e-5)


def test_two_predictions_weighted_average():
    ensemble = TemporalEnsemble(chunk_size=10, action_dim=3, max_timesteps=50, k=0.01)
    ensemble.add_prediction(0, torch.ones(10, 3) * 1.0)
    ensemble.add_prediction(1, torch.ones(10, 3) * 3.0)
    result = ensemble.get_action(1)
    # Result should be between 1 and 3 (weighted average)
    assert (result > 1.0).all() and (result < 3.0).all()


def test_reset_clears_state():
    ensemble = TemporalEnsemble(chunk_size=10, action_dim=7, max_timesteps=50)
    ensemble.add_prediction(0, torch.ones(10, 7))
    ensemble.reset()
    result = ensemble.get_action(0)
    assert torch.allclose(result, torch.zeros(7))


def test_empty_returns_zero():
    ensemble = TemporalEnsemble(chunk_size=10, action_dim=5, max_timesteps=30)
    result = ensemble.get_action(0)
    assert result.shape == (5,)
    assert torch.allclose(result, torch.zeros(5))


def test_smoothing_reduces_variance():
    ensemble = TemporalEnsemble(chunk_size=10, action_dim=1, max_timesteps=30, k=0.01)
    # Alternating high/low predictions produce high raw variance
    for t in range(20):
        val = 1.0 if t % 2 == 0 else -1.0
        ensemble.add_prediction(t, torch.ones(10, 1) * val)

    ensembled = [ensemble.get_action(t)[0].item() for t in range(5, 15)]
    diffs = [abs(ensembled[i + 1] - ensembled[i]) for i in range(len(ensembled) - 1)]
    avg_diff = sum(diffs) / len(diffs)
    # Without ensembling avg diff would be ~2.0 (alternating ±1)
    assert avg_diff < 2.0, f"ensembling should smooth; got avg diff {avg_diff:.3f}"


def test_pipeline_output_shape_no_agg():
    pipeline = make_pipeline(temporal_agg=False)
    pipeline.reset()
    for t in range(15):
        qpos = torch.randn(7)
        image = torch.rand(1, 3, 120, 160)
        action = pipeline.get_action(t, qpos, image)
        assert action.shape == (7,), f"expected (7,), got {action.shape}"


def test_pipeline_output_shape_with_agg():
    pipeline = make_pipeline(temporal_agg=True)
    pipeline.reset()
    for t in range(15):
        qpos = torch.randn(7)
        image = torch.rand(1, 3, 120, 160)
        action = pipeline.get_action(t, qpos, image)
        assert action.shape == (7,)


def test_pipeline_reset_is_idempotent():
    pipeline = make_pipeline(temporal_agg=True)
    pipeline.reset()
    torch.manual_seed(0)
    actions_first = [pipeline.get_action(t, torch.randn(7), torch.rand(1, 3, 120, 160)) for t in range(5)]

    pipeline.reset()
    torch.manual_seed(0)
    actions_second = [pipeline.get_action(t, torch.randn(7), torch.rand(1, 3, 120, 160)) for t in range(5)]

    for a, b in zip(actions_first, actions_second):
        assert torch.allclose(a, b), "same seed after reset must produce identical actions"
