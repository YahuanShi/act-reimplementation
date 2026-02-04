"""
Step 3.2 — Backbone validation experiment.

Tests the ResNet18 visual backbone to verify:
1. Output feature map dimensions match expected 32x downsampling
2. Position encoding shape matches feature map spatial dims
3. FrozenBatchNorm2d parameters are truly frozen (no gradients)
4. Multiple image sizes produce correctly shaped outputs
"""

import torch
from act.model.backbone import build_backbone, FrozenBatchNorm2d


def test_feature_map_dimensions():
    """Verify feature map dimensions for different input sizes."""
    print("=" * 60)
    print("Test 1: Feature Map Dimensions")
    print("=" * 60)

    backbone = build_backbone('resnet18', hidden_dim=256)

    test_cases = [
        # (H, W) -> expected (H/32, W/32)
        ((480, 640), (15, 20)),
        ((240, 320), (8, 10)),
        ((256, 256), (8, 8)),
    ]

    for (H, W), (eH, eW) in test_cases:
        img = torch.randn(2, 3, H, W)
        features, pos = backbone(img)

        feat = features[0]
        p = pos[0]
        print(f"  Input: ({H}, {W})")
        print(f"    Feature: {feat.shape} (expected (2, 512, {eH}, {eW}))")
        print(f"    Pos enc: {p.shape} (expected (1, 256, {eH}, {eW}))")

        assert feat.shape == (2, 512, eH, eW), \
            f"Expected (2, 512, {eH}, {eW}), got {feat.shape}"
        assert p.shape == (1, 256, eH, eW), \
            f"Expected (1, 256, {eH}, {eW}), got {p.shape}"

    print("  PASSED\n")


def test_frozen_batchnorm():
    """Verify FrozenBatchNorm2d does not have learnable parameters."""
    print("=" * 60)
    print("Test 2: FrozenBatchNorm2d")
    print("=" * 60)

    fbn = FrozenBatchNorm2d(64)

    # Should have no parameters (all are buffers)
    params = list(fbn.parameters())
    print(f"  Learnable parameters: {len(params)}")
    assert len(params) == 0, f"Expected 0 parameters, got {len(params)}"

    # Should have 4 buffers
    buffers = dict(fbn.named_buffers())
    print(f"  Buffers: {list(buffers.keys())}")
    assert set(buffers.keys()) == {'weight', 'bias', 'running_mean', 'running_var'}

    # Forward pass should work
    x = torch.randn(2, 64, 8, 8)
    y = fbn(x)
    assert y.shape == x.shape

    print("  PASSED\n")


def test_backbone_gradients():
    """Verify backbone feature extraction supports gradient flow."""
    print("=" * 60)
    print("Test 3: Backbone Gradient Flow")
    print("=" * 60)

    backbone = build_backbone('resnet18', hidden_dim=256, lr_backbone=1e-5)

    img = torch.randn(2, 3, 240, 320, requires_grad=True)
    features, pos = backbone(img)

    loss = features[0].sum()
    loss.backward()

    print(f"  Input gradient exists: {img.grad is not None}")
    assert img.grad is not None, "No gradient for input image"

    print("  PASSED\n")


if __name__ == "__main__":
    test_feature_map_dimensions()
    test_frozen_batchnorm()
    test_backbone_gradients()
    print("All backbone tests passed!")
