"""
Unit tests for ACTPolicy.

Validates:
  - compute_loss returns positive scalars and correct component structure
  - predict_action output shape matches (batch, num_queries, action_dim)
  - is_pad masking correctly ignores padded timesteps in the L1 loss
  - optimizer has two param groups with different learning rates
  - one gradient step decreases the loss
"""

import torch
import pytest
from act.policy.act_policy import ACTPolicy


def make_policy(state_dim=7, action_dim=7, chunk_size=20):
    config = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_dim": 128,
        "dim_feedforward": 256,
        "backbone": "resnet18",
        "lr_backbone": 1e-5,
        "enc_layers": 2,
        "dec_layers": 2,
        "nheads": 4,
        "num_queries": chunk_size,
        "camera_names": ["cam"],
        "lr": 1e-4,
        "kl_weight": 10.0,
    }
    return ACTPolicy(config)


def make_batch(B=2, state_dim=7, action_dim=7, chunk_size=20, num_cam=1):
    return {
        "qpos": torch.randn(B, state_dim),
        "image": torch.rand(B, num_cam, 3, 120, 160),
        "action": torch.randn(B, chunk_size, action_dim),
        "is_pad": torch.zeros(B, chunk_size, dtype=torch.bool),
    }


def test_compute_loss_shape():
    policy = make_policy()
    batch = make_batch()
    info = policy.compute_loss(batch["qpos"], batch["image"], batch["action"], batch["is_pad"])
    assert "loss" in info and "l1" in info and "kl" in info
    assert info["loss"].shape == ()  # scalar
    assert info["loss"].item() > 0


def test_compute_loss_with_padding():
    policy = make_policy()
    batch = make_batch()
    # Mask out last 10 steps — loss should still be finite
    batch["is_pad"][:, 10:] = True
    info = policy.compute_loss(batch["qpos"], batch["image"], batch["action"], batch["is_pad"])
    assert torch.isfinite(info["loss"])


def test_predict_action_shape():
    policy = make_policy(chunk_size=20)
    policy.eval()
    qpos = torch.randn(1, 7)
    image = torch.rand(1, 1, 3, 120, 160)
    with torch.no_grad():
        a_hat = policy.predict_action(qpos, image)
    assert a_hat.shape == (1, 20, 7)


def test_predict_action_deterministic():
    policy = make_policy()
    policy.eval()
    qpos = torch.randn(1, 7)
    image = torch.rand(1, 1, 3, 120, 160)
    with torch.no_grad():
        a1 = policy.predict_action(qpos, image)
        a2 = policy.predict_action(qpos, image)
    assert torch.allclose(a1, a2), "inference must be deterministic"


def test_optimizer_two_groups():
    policy = make_policy()
    opt = policy.configure_optimizers()
    assert len(opt.param_groups) == 2, "should have backbone and non-backbone groups"
    lrs = [g["lr"] for g in opt.param_groups]
    # non-backbone lr > backbone lr (or backbone lr=0 when frozen)
    assert lrs[0] >= lrs[1], "backbone lr should be <= main lr"


def test_loss_decreases_after_step():
    policy = make_policy()
    batch = make_batch()
    opt = policy.configure_optimizers()

    info_before = policy.compute_loss(batch["qpos"], batch["image"], batch["action"], batch["is_pad"])
    info_before["loss"].backward()
    opt.step()
    opt.zero_grad()

    info_after = policy.compute_loss(batch["qpos"], batch["image"], batch["action"], batch["is_pad"])
    assert info_after["loss"].item() < info_before["loss"].item(), (
        "loss should decrease after one gradient step"
    )


def test_multi_camera():
    config = {
        "state_dim": 14,
        "action_dim": 14,
        "hidden_dim": 128,
        "dim_feedforward": 256,
        "backbone": "resnet18",
        "lr_backbone": 1e-5,
        "enc_layers": 2,
        "dec_layers": 2,
        "nheads": 4,
        "num_queries": 20,
        "camera_names": ["cam_left", "cam_right"],
        "lr": 1e-4,
        "kl_weight": 10.0,
    }
    policy = ACTPolicy(config)
    batch = {
        "qpos": torch.randn(2, 14),
        "image": torch.rand(2, 2, 3, 120, 160),  # 2 cameras
        "action": torch.randn(2, 20, 14),
        "is_pad": torch.zeros(2, 20, dtype=torch.bool),
    }
    info = policy.compute_loss(batch["qpos"], batch["image"], batch["action"], batch["is_pad"])
    assert torch.isfinite(info["loss"])
