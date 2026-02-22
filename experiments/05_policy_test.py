"""
Step 6.2 — ACTPolicy validation experiment.

Tests the ACTPolicy wrapper to verify:
1. compute_loss returns correct loss dict with positive values
2. predict_action returns correct output shape
3. Optimizer is correctly configured with separate backbone lr
4. Loss decreases after one gradient step (basic sanity check)
"""

import torch

from act.policy.act_policy import ACTPolicy


def make_policy_config():
    return {
        "state_dim": 7,
        "action_dim": 7,
        "hidden_dim": 256,
        "dim_feedforward": 1024,
        "backbone": "resnet18",
        "lr_backbone": 1e-5,
        "enc_layers": 4,
        "dec_layers": 4,
        "nheads": 8,
        "num_queries": 50,
        "camera_names": ["cam"],
        "lr": 1e-4,
        "kl_weight": 10.0,
    }


def test_compute_loss():
    """Test training loss computation."""
    print("=" * 60)
    print("Test 1: compute_loss")
    print("=" * 60)

    config = make_policy_config()
    policy = ACTPolicy(config)

    qpos = torch.randn(2, 7)
    image = torch.rand(2, 1, 3, 240, 320)  # [0, 1] range
    actions = torch.randn(2, 50, 7)
    is_pad = torch.zeros(2, 50, dtype=torch.bool)

    loss_dict = policy.compute_loss(qpos, image, actions, is_pad)

    print(f"  l1:   {loss_dict['l1'].item():.4f}")
    print(f"  kl:   {loss_dict['kl'].item():.4f}")
    print(f"  loss: {loss_dict['loss'].item():.4f}")

    assert "l1" in loss_dict and "kl" in loss_dict and "loss" in loss_dict
    assert loss_dict["loss"].item() > 0, "Loss should be positive"
    assert loss_dict["l1"].item() > 0, "L1 should be positive"
    assert loss_dict["kl"].item() >= 0, "KL should be non-negative"

    print("  PASSED\n")


def test_predict_action():
    """Test inference action prediction."""
    print("=" * 60)
    print("Test 2: predict_action")
    print("=" * 60)

    config = make_policy_config()
    policy = ACTPolicy(config)
    policy.eval()

    qpos = torch.randn(2, 7)
    image = torch.rand(2, 1, 3, 240, 320)

    a_hat = policy.predict_action(qpos, image)

    print(f"  a_hat shape: {a_hat.shape}")
    assert a_hat.shape == (2, 50, 7)

    print("  PASSED\n")


def test_optimizer():
    """Test optimizer configuration."""
    print("=" * 60)
    print("Test 3: Optimizer Configuration")
    print("=" * 60)

    config = make_policy_config()
    policy = ACTPolicy(config)
    optimizer = policy.configure_optimizers()

    print(f"  Optimizer type: {type(optimizer).__name__}")
    print(f"  Param groups: {len(optimizer.param_groups)}")
    print(f"  Group 0 lr: {optimizer.param_groups[0]['lr']}")
    print(f"  Group 1 lr (backbone): {optimizer.param_groups[1]['lr']}")

    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["lr"] == 1e-4
    assert optimizer.param_groups[1]["lr"] == 1e-5

    print("  PASSED\n")


def test_loss_decreases():
    """Verify loss decreases after one optimization step."""
    print("=" * 60)
    print("Test 4: Loss Decrease After One Step")
    print("=" * 60)

    config = make_policy_config()
    policy = ACTPolicy(config)
    optimizer = policy.configure_optimizers()

    qpos = torch.randn(4, 7)
    image = torch.rand(4, 1, 3, 240, 320)
    actions = torch.randn(4, 50, 7)
    is_pad = torch.zeros(4, 50, dtype=torch.bool)

    # First forward pass
    loss_dict_1 = policy.compute_loss(qpos, image, actions, is_pad)
    loss_1 = loss_dict_1["loss"]
    loss_1.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Second forward pass
    loss_dict_2 = policy.compute_loss(qpos, image, actions, is_pad)
    loss_2 = loss_dict_2["loss"]

    print(f"  Loss before: {loss_1.item():.4f}")
    print(f"  Loss after:  {loss_2.item():.4f}")
    print(f"  Decreased: {loss_2.item() < loss_1.item()}")

    # Loss should decrease on the same batch (overfitting test)
    assert loss_2.item() < loss_1.item(), "Loss should decrease after one step on the same batch"

    print("  PASSED\n")


if __name__ == "__main__":
    test_compute_loss()
    test_predict_action()
    test_optimizer()
    test_loss_decreases()
    print("All policy tests passed!")
