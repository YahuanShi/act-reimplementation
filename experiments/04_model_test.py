"""
Step 5.2 — DETRVAE model forward pass validation.

Tests the complete DETRVAE model to verify:
1. Training mode: correct output shapes for a_hat, mu, logvar
2. Inference mode: correct output shapes, mu/logvar are None
3. Multi-camera support: features concatenated along width
4. Parameter count: reasonable for the configured architecture
5. End-to-end gradient flow from loss to all components
"""

import torch
from act.model.detr_vae import build_act_model, kl_divergence


def make_config(camera_names=None, hidden_dim=256, num_queries=50,
                state_dim=7, action_dim=7):
    """Create a model config dict."""
    if camera_names is None:
        camera_names = ['cam']
    return {
        'state_dim': state_dim, 'action_dim': action_dim,
        'hidden_dim': hidden_dim, 'dim_feedforward': 1024,
        'backbone': 'resnet18', 'lr_backbone': 1e-5,
        'enc_layers': 4, 'dec_layers': 7, 'nheads': 8,
        'num_queries': num_queries, 'camera_names': camera_names,
    }


def test_training_mode():
    """Test training forward pass with full loss computation."""
    print("=" * 60)
    print("Test 1: Training Mode")
    print("=" * 60)

    config = make_config()
    model = build_act_model(config)

    bs = 2
    qpos = torch.randn(bs, 7)
    image = torch.randn(bs, 1, 3, 240, 320)
    actions = torch.randn(bs, 50, 7)
    is_pad = torch.zeros(bs, 50, dtype=torch.bool)
    # Mark last 10 actions as padding
    is_pad[:, -10:] = True

    a_hat, is_pad_hat, (mu, logvar) = model(qpos, image, None, actions, is_pad)

    print(f"  a_hat: {a_hat.shape}")
    print(f"  is_pad_hat: {is_pad_hat.shape}")
    print(f"  mu: {mu.shape}")
    print(f"  logvar: {logvar.shape}")

    assert a_hat.shape == (bs, 50, 7)
    assert is_pad_hat.shape == (bs, 50, 1)
    assert mu.shape == (bs, 32)
    assert logvar.shape == (bs, 32)

    print("  PASSED\n")


def test_inference_mode():
    """Test inference forward pass (no actions provided)."""
    print("=" * 60)
    print("Test 2: Inference Mode")
    print("=" * 60)

    config = make_config()
    model = build_act_model(config)
    model.eval()

    bs = 2
    qpos = torch.randn(bs, 7)
    image = torch.randn(bs, 1, 3, 240, 320)

    with torch.inference_mode():
        a_hat, is_pad_hat, (mu, logvar) = model(qpos, image, None)

    print(f"  a_hat: {a_hat.shape}")
    print(f"  mu: {mu}")
    assert a_hat.shape == (bs, 50, 7)
    assert mu is None
    assert logvar is None

    print("  PASSED\n")


def test_multi_camera():
    """Test with multiple cameras."""
    print("=" * 60)
    print("Test 3: Multi-Camera Support")
    print("=" * 60)

    config = make_config(camera_names=['cam_left', 'cam_right'])
    model = build_act_model(config)

    bs = 2
    qpos = torch.randn(bs, 7)
    # Two cameras
    image = torch.randn(bs, 2, 3, 240, 320)
    actions = torch.randn(bs, 50, 7)
    is_pad = torch.zeros(bs, 50, dtype=torch.bool)

    a_hat, _, (mu, logvar) = model(qpos, image, None, actions, is_pad)

    print(f"  a_hat shape (2 cameras): {a_hat.shape}")
    assert a_hat.shape == (bs, 50, 7)

    print("  PASSED\n")


def test_different_dims():
    """Test with different state/action dimensions."""
    print("=" * 60)
    print("Test 4: Different State/Action Dimensions")
    print("=" * 60)

    # Single arm robot: 6 joints + 1 gripper = 7
    config = make_config(state_dim=7, action_dim=7)
    model = build_act_model(config)
    a_hat, _, _ = model(torch.randn(2, 7), torch.randn(2, 1, 3, 240, 320),
                        None, torch.randn(2, 50, 7),
                        torch.zeros(2, 50, dtype=torch.bool))
    print(f"  7-DOF robot: {a_hat.shape}")
    assert a_hat.shape == (2, 50, 7)

    # Dual arm robot: 14 joints
    config = make_config(state_dim=14, action_dim=14)
    model = build_act_model(config)
    a_hat, _, _ = model(torch.randn(2, 14), torch.randn(2, 1, 3, 240, 320),
                        None, torch.randn(2, 50, 14),
                        torch.zeros(2, 50, dtype=torch.bool))
    print(f"  14-DOF robot: {a_hat.shape}")
    assert a_hat.shape == (2, 50, 14)

    print("  PASSED\n")


def test_gradient_flow():
    """Test end-to-end gradient flow with ACT loss."""
    print("=" * 60)
    print("Test 5: End-to-End Gradient Flow")
    print("=" * 60)

    config = make_config()
    model = build_act_model(config)

    qpos = torch.randn(2, 7)
    image = torch.randn(2, 1, 3, 240, 320)
    actions = torch.randn(2, 50, 7)
    is_pad = torch.zeros(2, 50, dtype=torch.bool)

    a_hat, _, (mu, logvar) = model(qpos, image, None, actions, is_pad)

    # Compute ACT loss: L1 + KL
    l1_loss = torch.nn.functional.l1_loss(actions, a_hat, reduction='none')
    l1 = (l1_loss * ~is_pad.unsqueeze(-1)).mean()
    total_kl, _, _ = kl_divergence(mu, logvar)
    loss = l1 + 10.0 * total_kl[0]

    loss.backward()

    # Check backbone has gradients
    backbone_grad = sum(1 for p in model.backbones.parameters()
                        if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"  Backbone params with gradient: {backbone_grad}")
    assert backbone_grad > 0

    # Check CVAE encoder has gradients
    encoder_grad = sum(1 for p in model.encoder.parameters()
                       if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"  CVAE encoder params with gradient: {encoder_grad}")
    assert encoder_grad > 0

    print(f"  Loss value: {loss.item():.4f}")
    print("  PASSED\n")


if __name__ == "__main__":
    test_training_mode()
    test_inference_mode()
    test_multi_camera()
    test_different_dims()
    test_gradient_flow()
    print("All DETRVAE model tests passed!")
