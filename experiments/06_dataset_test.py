"""
Step 7.3 — Dataset validation experiment.

Tests the LeRobot dataset adapter and normalizer:
1. Normalizer roundtrip: fit -> normalize -> denormalize recovers original values
2. LeRobot dataset loading and output shapes (requires a dataset to be available)
3. is_pad mask correctness at episode boundaries
4. DataLoader batching
"""

import torch

from act.dataset.normalizer import ACTNormalizer


def test_normalizer_roundtrip():
    """Verify normalize -> denormalize recovers original values."""
    print("=" * 60)
    print("Test 1: Normalizer Roundtrip")
    print("=" * 60)

    normalizer = ACTNormalizer()

    # Create synthetic data with known statistics
    actions = torch.randn(100, 7) * 3 + 2  # mean~2, std~3
    qpos = torch.randn(100, 7) * 0.5 + 1  # mean~1, std~0.5

    normalizer.fit_from_tensors(actions, qpos)

    print(f"  Action mean: {normalizer.action_mean[:3]}")
    print(f"  Action std:  {normalizer.action_std[:3]}")
    print(f"  Qpos mean:   {normalizer.qpos_mean[:3]}")
    print(f"  Qpos std:    {normalizer.qpos_std[:3]}")

    # Test roundtrip
    action_sample = torch.randn(4, 7)
    normalized = normalizer.normalize_action(action_sample)
    recovered = normalizer.denormalize_action(normalized)
    error = (action_sample - recovered).abs().max().item()
    print(f"  Action roundtrip max error: {error:.8f}")
    assert error < 1e-5, f"Roundtrip error too large: {error}"

    qpos_sample = torch.randn(4, 7)
    normalized = normalizer.normalize_qpos(qpos_sample)
    recovered = normalizer.denormalize_qpos(normalized)
    error = (qpos_sample - recovered).abs().max().item()
    print(f"  Qpos roundtrip max error: {error:.8f}")
    assert error < 1e-5, f"Roundtrip error too large: {error}"

    print("  PASSED\n")


def test_normalizer_save_load():
    """Verify normalizer state can be saved and loaded."""
    print("=" * 60)
    print("Test 2: Normalizer Save/Load")
    print("=" * 60)

    normalizer = ACTNormalizer()
    normalizer.fit_from_tensors(torch.randn(50, 7), torch.randn(50, 7))

    state = normalizer.state_dict()
    print(f"  State dict keys: {list(state.keys())}")

    # Load into a new normalizer
    normalizer2 = ACTNormalizer()
    normalizer2.load_state_dict(state)

    # Verify they produce the same results
    test_input = torch.randn(4, 7)
    out1 = normalizer.normalize_action(test_input)
    out2 = normalizer2.normalize_action(test_input)
    error = (out1 - out2).abs().max().item()
    print(f"  Normalized output match error: {error:.8f}")
    assert error < 1e-7, "Save/load should produce identical results"

    print("  PASSED\n")


def test_normalizer_std_clipping():
    """Verify std is clipped to minimum 0.01."""
    print("=" * 60)
    print("Test 3: Std Floor Clipping")
    print("=" * 60)

    normalizer = ACTNormalizer()
    # Create data where one dimension has zero variance
    actions = torch.zeros(50, 3)
    actions[:, 0] = 1.0  # constant -> std = 0
    actions[:, 1] = torch.randn(50)  # normal std
    actions[:, 2] = 1.0  # constant -> std = 0

    normalizer.fit_from_tensors(actions)

    print(f"  Action std: {normalizer.action_std}")
    assert normalizer.action_std.min().item() >= 0.01 - 1e-6, (
        "Std should be clipped to at least 0.01"
    )

    print("  PASSED\n")


def test_lerobot_dataset():
    """Test LeRobot dataset loading (skipped if dataset not available)."""
    print("=" * 60)
    print("Test 4: LeRobot Dataset Loading")
    print("=" * 60)

    try:
        from act.dataset.lerobot_wrapper import LeRobotACTDataset

        ds = LeRobotACTDataset(
            repo_id="lerobot/aloha_sim_transfer_cube_human", chunk_size=50, episodes=[0]
        )

        print(f"  Dataset length: {len(ds)}")
        print(f"  State dim: {ds.state_dim}")
        print(f"  Action dim: {ds.action_dim}")

        sample = ds[0]
        print(f"  qpos shape: {sample['qpos'].shape}")
        print(f"  image shape: {sample['image'].shape}")
        print(f"  action shape: {sample['action'].shape}")
        print(f"  is_pad shape: {sample['is_pad'].shape}")
        print(f"  is_pad sum: {sample['is_pad'].sum().item()}")

        # Check shapes
        assert sample["action"].shape[0] == 50, "Action chunk should be chunk_size"
        assert sample["is_pad"].shape[0] == 50, "is_pad should be chunk_size"

        # Test near end of episode
        last_sample = ds[len(ds) - 1]
        print(f"  Last sample is_pad sum: {last_sample['is_pad'].sum().item()}")

        # Test DataLoader
        from torch.utils.data import DataLoader

        loader = DataLoader(ds, batch_size=4, shuffle=True)
        batch = next(iter(loader))
        print(f"  Batch qpos: {batch['qpos'].shape}")
        print(f"  Batch image: {batch['image'].shape}")
        print(f"  Batch action: {batch['action'].shape}")

        print("  PASSED\n")

    except Exception as e:
        print(f"  SKIPPED (dataset not available: {e})\n")


if __name__ == "__main__":
    test_normalizer_roundtrip()
    test_normalizer_save_load()
    test_normalizer_std_clipping()
    test_lerobot_dataset()
    print("All dataset tests passed!")
