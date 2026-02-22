"""
Step 4.2 — CVAE encoder validation experiment.

Tests the CVAE encoder components of DETRVAE:
1. Latent variable shape and distribution properties
2. KL divergence correctness (non-negative, zero for standard normal)
3. Reparameterization gradient flow
4. Prior sampling (z=0 at inference)
"""

import torch

from act.model.detr_vae import build_act_model, kl_divergence, reparametrize


def test_cvae_latent():
    """Verify CVAE encoder produces correct latent shapes."""
    print("=" * 60)
    print("Test 1: CVAE Latent Variable")
    print("=" * 60)

    config = {
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
    }
    model = build_act_model(config)

    qpos = torch.randn(2, 7)
    image = torch.randn(2, 1, 3, 240, 320)
    actions = torch.randn(2, 50, 7)
    is_pad = torch.zeros(2, 50, dtype=torch.bool)

    _, _, (mu, logvar) = model(qpos, image, None, actions, is_pad)

    print(f"  mu shape: {mu.shape}")
    print(f"  logvar shape: {logvar.shape}")
    assert mu.shape == (2, 32), f"Expected (2, 32), got {mu.shape}"
    assert logvar.shape == (2, 32), f"Expected (2, 32), got {logvar.shape}"

    print("  PASSED\n")


def test_kl_divergence():
    """Verify KL divergence properties."""
    print("=" * 60)
    print("Test 2: KL Divergence")
    print("=" * 60)

    # KL should be non-negative
    mu = torch.randn(4, 32)
    logvar = torch.randn(4, 32)
    total_kl, dim_kl, mean_kl = kl_divergence(mu, logvar)

    print(f"  Total KL: {total_kl.item():.4f}")
    print(f"  Mean KL:  {mean_kl.item():.4f}")
    print(f"  Dim-wise KL shape: {dim_kl.shape}")
    assert total_kl.item() >= 0, "KL divergence should be non-negative"
    assert dim_kl.shape == (32,)

    # KL(N(0,I) || N(0,I)) should be 0
    mu_zero = torch.zeros(4, 32)
    logvar_zero = torch.zeros(4, 32)
    kl_zero, _, _ = kl_divergence(mu_zero, logvar_zero)
    print(f"  KL(N(0,I) || N(0,I)): {kl_zero.item():.6f}")
    assert abs(kl_zero.item()) < 1e-5, "KL should be ~0 for standard normal"

    print("  PASSED\n")


def test_reparameterization_gradient():
    """Verify gradients flow through reparameterization trick."""
    print("=" * 60)
    print("Test 3: Reparameterization Gradient Flow")
    print("=" * 60)

    mu = torch.randn(4, 32, requires_grad=True)
    logvar = torch.randn(4, 32, requires_grad=True)

    z = reparametrize(mu, logvar)
    loss = z.sum()
    loss.backward()

    print(f"  mu gradient exists: {mu.grad is not None}")
    print(f"  logvar gradient exists: {logvar.grad is not None}")
    assert mu.grad is not None, "No gradient for mu"
    assert logvar.grad is not None, "No gradient for logvar"

    print("  PASSED\n")


def test_inference_prior():
    """Verify inference uses z=0 (prior mean)."""
    print("=" * 60)
    print("Test 4: Inference Prior (z=0)")
    print("=" * 60)

    config = {
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
    }
    model = build_act_model(config)
    model.eval()

    qpos = torch.randn(2, 7)
    image = torch.randn(2, 1, 3, 240, 320)

    with torch.inference_mode():
        a_hat, _, (mu, logvar) = model(qpos, image, None)

    print(f"  mu: {mu}")
    print(f"  logvar: {logvar}")
    print(f"  a_hat shape: {a_hat.shape}")
    assert mu is None, "mu should be None at inference"
    assert logvar is None, "logvar should be None at inference"
    assert a_hat.shape == (2, 50, 7)

    print("  PASSED\n")


if __name__ == "__main__":
    test_cvae_latent()
    test_kl_divergence()
    test_reparameterization_gradient()
    test_inference_prior()
    print("All CVAE tests passed!")
