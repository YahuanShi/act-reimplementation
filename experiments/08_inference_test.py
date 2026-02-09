"""
Step 9.2 — Inference pipeline validation.

Tests temporal ensembling and the inference pipeline:
1. TemporalEnsemble: basic add/get operations and smoothing properties
2. InferencePipeline: both ensembled and non-ensembled modes
3. Smoothing effect: ensembled actions should have smaller step-to-step changes
"""

import torch
from act.policy.act_policy import ACTPolicy
from act.dataset.normalizer import ACTNormalizer
from inference.temporal_ensemble import TemporalEnsemble, InferencePipeline


def test_temporal_ensemble_basic():
    """Test basic ensemble operations."""
    print("=" * 60)
    print("Test 1: TemporalEnsemble Basic Operations")
    print("=" * 60)

    chunk_size = 10
    action_dim = 7
    max_t = 50
    ensemble = TemporalEnsemble(chunk_size, action_dim, max_t)

    # Add a prediction at t=0
    actions_t0 = torch.ones(chunk_size, action_dim)
    ensemble.add_prediction(0, actions_t0)

    # Get action at t=0: should be close to 1.0 (only one prediction)
    action = ensemble.get_action(0)
    print(f"  Action at t=0 (single pred): {action[:3]}")
    assert torch.allclose(action, torch.ones(action_dim), atol=1e-5)

    # Add a prediction at t=1 with different values
    actions_t1 = torch.ones(chunk_size, action_dim) * 2.0
    ensemble.add_prediction(1, actions_t1)

    # Get action at t=1: should be between 1.0 and 2.0
    # (weighted average of prediction from t=0 and t=1)
    action = ensemble.get_action(1)
    print(f"  Action at t=1 (two preds): {action[:3]}")
    assert action[0].item() > 1.0 and action[0].item() < 2.0

    # Reset
    ensemble.reset()
    action = ensemble.get_action(0)
    assert torch.allclose(action, torch.zeros(action_dim))
    print("  Reset works: all zeros")

    print("  PASSED\n")


def test_temporal_ensemble_smoothing():
    """Verify ensembling produces smoother trajectories."""
    print("=" * 60)
    print("Test 2: Temporal Ensemble Smoothing Effect")
    print("=" * 60)

    chunk_size = 10
    action_dim = 3
    max_t = 30
    ensemble = TemporalEnsemble(chunk_size, action_dim, max_t, k=0.01)

    # Add predictions with alternating high/low values to create jitter
    for t in range(20):
        val = 1.0 if t % 2 == 0 else -1.0
        actions = torch.ones(chunk_size, action_dim) * val
        # Add some variation per step within the chunk
        for s in range(chunk_size):
            actions[s] *= (1.0 + 0.1 * s)
        ensemble.add_prediction(t, actions)

    # Collect ensembled actions
    ensembled_actions = []
    for t in range(5, 15):
        action = ensemble.get_action(t)
        ensembled_actions.append(action[0].item())

    # Compute step-to-step differences
    diffs = [abs(ensembled_actions[i+1] - ensembled_actions[i])
             for i in range(len(ensembled_actions)-1)]
    avg_diff = sum(diffs) / len(diffs)

    print(f"  Ensembled actions: {[f'{a:.3f}' for a in ensembled_actions[:5]]}")
    print(f"  Avg step-to-step diff: {avg_diff:.4f}")
    # Without ensembling, diff would be ~2.0 (alternating -1/+1)
    # With ensembling, diff should be much smaller
    assert avg_diff < 2.0, "Ensembling should reduce step-to-step jitter"

    print("  PASSED\n")


def test_inference_pipeline():
    """Test the full inference pipeline in both modes."""
    print("=" * 60)
    print("Test 3: InferencePipeline")
    print("=" * 60)

    config = {
        'state_dim': 7, 'action_dim': 7,
        'hidden_dim': 128, 'dim_feedforward': 256,
        'backbone': 'resnet18', 'lr_backbone': 0,
        'enc_layers': 2, 'dec_layers': 2, 'nheads': 4,
        'num_queries': 10, 'camera_names': ['cam'],
        'lr': 1e-4, 'kl_weight': 10.0,
    }

    policy = ACTPolicy(config)
    normalizer = ACTNormalizer()
    normalizer.fit_from_tensors(torch.randn(50, 7), torch.randn(50, 7))

    # Test without temporal ensembling
    pipeline_no_agg = InferencePipeline(
        policy, normalizer, chunk_size=10, temporal_agg=False, max_timesteps=50)
    pipeline_no_agg.reset()

    actions_no_agg = []
    for t in range(20):
        qpos = torch.randn(7)
        image = torch.rand(1, 3, 120, 160)
        action = pipeline_no_agg.get_action(t, qpos, image)
        actions_no_agg.append(action)
        assert action.shape == (7,), f"Expected (7,), got {action.shape}"

    print(f"  No ensembling: {len(actions_no_agg)} actions, shape {actions_no_agg[0].shape}")

    # Test with temporal ensembling
    pipeline_agg = InferencePipeline(
        policy, normalizer, chunk_size=10, temporal_agg=True, max_timesteps=50)
    pipeline_agg.reset()

    actions_agg = []
    for t in range(20):
        qpos = torch.randn(7)
        image = torch.rand(1, 3, 120, 160)
        action = pipeline_agg.get_action(t, qpos, image)
        actions_agg.append(action)
        assert action.shape == (7,), f"Expected (7,), got {action.shape}"

    print(f"  With ensembling: {len(actions_agg)} actions, shape {actions_agg[0].shape}")
    print("  PASSED\n")


if __name__ == "__main__":
    test_temporal_ensemble_basic()
    test_temporal_ensemble_smoothing()
    test_inference_pipeline()
    print("All inference tests passed!")
