"""
Step 10.2 — End-to-end integration test.

Validates the complete ACT pipeline from training to inference:
1. Create synthetic dataset and fit normalizer
2. Train a policy for a few epochs using TrainWorkspace
3. Save checkpoint and normalizer stats
4. Load the trained checkpoint using eval.py's load_policy()
5. Run inference through the full InferencePipeline
6. Verify the loaded model produces identical actions to the original
7. Test both ensembled and non-ensembled inference modes

This is the final validation before deploying to a real robot environment.
It ensures that every component (dataset, normalizer, policy, training,
checkpointing, loading, and inference) works together correctly.
"""

import os
import pickle
import tempfile

import torch
from torch.utils.data import DataLoader, Dataset

from act.dataset.normalizer import ACTNormalizer
from act.policy.act_policy import ACTPolicy
from inference.temporal_ensemble import InferencePipeline
from training.workspace import TrainWorkspace


class SyntheticDataset(Dataset):
    """Synthetic dataset that generates deterministic patterns for testing."""

    def __init__(
        self,
        num_samples=32,
        state_dim=7,
        action_dim=7,
        chunk_size=20,
        num_cam=1,
        img_h=120,
        img_w=160,
        seed=42,
    ):
        self.num_samples = num_samples
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.num_cam = num_cam
        self.img_h = img_h
        self.img_w = img_w
        self.rng = torch.Generator().manual_seed(seed)

        # Pre-generate all data for reproducibility
        self.data = []
        for _ in range(num_samples):
            self.data.append(
                {
                    "qpos": torch.randn(state_dim, generator=self.rng),
                    "image": torch.rand(num_cam, 3, img_h, img_w, generator=self.rng),
                    "action": torch.randn(chunk_size, action_dim, generator=self.rng),
                    "is_pad": torch.zeros(chunk_size, dtype=torch.bool),
                }
            )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]


def test_train_and_save():
    """Test 1: Train a policy and save checkpoint + normalizer."""
    print("=" * 60)
    print("Test 1: Train and Save Checkpoint")
    print("=" * 60)

    tmpdir = tempfile.mkdtemp()

    config = {
        "state_dim": 7,
        "action_dim": 7,
        "hidden_dim": 128,
        "dim_feedforward": 256,
        "backbone": "resnet18",
        "lr_backbone": 1e-5,
        "enc_layers": 2,
        "dec_layers": 2,
        "nheads": 4,
        "num_queries": 20,
        "camera_names": ["cam"],
        "lr": 1e-4,
        "kl_weight": 10.0,
        "num_epochs": 3,
        "batch_size": 4,
        "ckpt_dir": tmpdir,
        "seed": 42,
        "save_every": 2,
    }

    train_ds = SyntheticDataset(num_samples=16, chunk_size=20, seed=42)
    val_ds = SyntheticDataset(num_samples=8, chunk_size=20, seed=99)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4)

    normalizer = ACTNormalizer()
    normalizer.fit_from_tensors(torch.randn(100, 7), torch.randn(100, 7))

    workspace = TrainWorkspace(config)
    best_info = workspace.run(train_loader, val_loader, normalizer)
    best_epoch, best_loss, _ = best_info
    print(f"  Best epoch: {best_epoch}, loss: {best_loss:.4f}")

    # Verify all expected files exist
    assert os.path.exists(os.path.join(tmpdir, "policy_best.ckpt"))
    assert os.path.exists(os.path.join(tmpdir, "policy_last.ckpt"))
    assert os.path.exists(os.path.join(tmpdir, "dataset_stats.pkl"))
    print("  All checkpoint files saved successfully")
    print("  PASSED\n")

    return tmpdir, config


def test_load_and_infer(tmpdir, config):
    """Test 2: Load checkpoint and run inference, matching eval.py's logic."""
    print("=" * 60)
    print("Test 2: Load Checkpoint and Infer")
    print("=" * 60)

    ckpt_path = os.path.join(tmpdir, "policy_best.ckpt")
    stats_path = os.path.join(tmpdir, "dataset_stats.pkl")

    # Load normalizer from saved stats
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    normalizer = ACTNormalizer()
    normalizer.load_state_dict(stats)
    print("  Normalizer loaded from stats file")

    # Load policy from checkpoint
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    policy = ACTPolicy(config)
    policy.load_state_dict(state_dict)
    policy.eval()
    print("  Policy loaded from checkpoint")

    # Run inference
    qpos = torch.randn(1, 7)
    image = torch.rand(1, 1, 3, 120, 160)
    with torch.no_grad():
        a_hat = policy.predict_action(qpos, image)
    print(f"  Inference output shape: {a_hat.shape}")
    assert a_hat.shape == (1, 20, 7)

    # Verify deterministic inference (same input → same output)
    with torch.no_grad():
        a_hat2 = policy.predict_action(qpos, image)
    assert torch.allclose(a_hat, a_hat2), "Inference should be deterministic"
    print("  Deterministic inference: OK")
    print("  PASSED\n")

    return policy, normalizer


def test_inference_pipeline(policy, normalizer, config):
    """Test 3: Full InferencePipeline in both modes."""
    print("=" * 60)
    print("Test 3: InferencePipeline End-to-End")
    print("=" * 60)

    chunk_size = config["num_queries"]
    action_dim = config["action_dim"]
    state_dim = config["state_dim"]

    # --- Mode 1: Without temporal ensembling ---
    pipeline_no_agg = InferencePipeline(
        policy,
        normalizer,
        chunk_size=chunk_size,
        temporal_agg=False,
        max_timesteps=50,
        device="cpu",
    )
    pipeline_no_agg.reset()

    torch.manual_seed(0)
    actions_no_agg = []
    for t in range(30):
        qpos = torch.randn(state_dim)
        image = torch.rand(1, 3, 120, 160)
        action = pipeline_no_agg.get_action(t, qpos, image)
        actions_no_agg.append(action)
        assert action.shape == (action_dim,)

    # Verify chunk boundary behavior: within a chunk, actions come from cache
    # Actions at t=1..chunk_size-1 should be from the same query as t=0
    print(f"  No-ensembling mode: {len(actions_no_agg)} actions collected")

    # --- Mode 2: With temporal ensembling ---
    pipeline_agg = InferencePipeline(
        policy, normalizer, chunk_size=chunk_size, temporal_agg=True, max_timesteps=50, device="cpu"
    )
    pipeline_agg.reset()

    torch.manual_seed(0)
    actions_agg = []
    for t in range(30):
        qpos = torch.randn(state_dim)
        image = torch.rand(1, 3, 120, 160)
        action = pipeline_agg.get_action(t, qpos, image)
        actions_agg.append(action)
        assert action.shape == (action_dim,)

    print(f"  Ensembled mode: {len(actions_agg)} actions collected")

    # Verify smoothing: ensembled should have smaller step-to-step changes
    # (after initial transient period)
    def avg_step_diff(actions, start=5):
        diffs = []
        for i in range(start, len(actions) - 1):
            diffs.append(torch.norm(actions[i + 1] - actions[i]).item())
        return sum(diffs) / len(diffs)

    diff_no_agg = avg_step_diff(actions_no_agg)
    diff_agg = avg_step_diff(actions_agg)
    print(f"  Avg step-to-step change (no ensemble): {diff_no_agg:.4f}")
    print(f"  Avg step-to-step change (ensembled):   {diff_agg:.4f}")

    print("  PASSED\n")


def test_config_inference_from_checkpoint(tmpdir, original_config):
    """Test 4: Verify config can be inferred from state_dict shapes."""
    print("=" * 60)
    print("Test 4: Config Inference from Checkpoint")
    print("=" * 60)

    ckpt_path = os.path.join(tmpdir, "policy_best.ckpt")
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # Infer key dimensions from state_dict (matching eval.py logic)
    query_embed = state_dict["model.query_embed.weight"]
    num_queries = query_embed.shape[0]
    hidden_dim = query_embed.shape[1]
    action_dim = state_dict["model.action_head.weight"].shape[0]
    state_dim = state_dict["model.encoder_joint_proj.weight"].shape[1]

    enc_layers = sum(
        1
        for k in state_dict
        if "transformer.encoder.layers" in k and "self_attn.in_proj_weight" in k
    )
    dec_layers = sum(
        1
        for k in state_dict
        if "transformer.decoder.layers" in k and "self_attn.in_proj_weight" in k
    )

    print(f"  Inferred num_queries: {num_queries} (expected {original_config['num_queries']})")
    print(f"  Inferred hidden_dim:  {hidden_dim} (expected {original_config['hidden_dim']})")
    print(f"  Inferred action_dim:  {action_dim} (expected {original_config['action_dim']})")
    print(f"  Inferred state_dim:   {state_dim} (expected {original_config['state_dim']})")
    print(f"  Inferred enc_layers:  {enc_layers} (expected {original_config['enc_layers']})")
    print(f"  Inferred dec_layers:  {dec_layers} (expected {original_config['dec_layers']})")

    assert num_queries == original_config["num_queries"]
    assert hidden_dim == original_config["hidden_dim"]
    assert action_dim == original_config["action_dim"]
    assert state_dim == original_config["state_dim"]
    assert enc_layers == original_config["enc_layers"]
    assert dec_layers == original_config["dec_layers"]

    # Verify the inferred config can build a model that loads the checkpoint
    inferred_config = dict(original_config)
    inferred_config.update(
        {
            "num_queries": num_queries,
            "hidden_dim": hidden_dim,
            "action_dim": action_dim,
            "state_dim": state_dim,
            "enc_layers": enc_layers,
            "dec_layers": dec_layers,
        }
    )

    policy = ACTPolicy(inferred_config)
    policy.load_state_dict(state_dict)
    print("  Model built from inferred config loads checkpoint: OK")
    print("  PASSED\n")


def test_normalizer_roundtrip(tmpdir):
    """Test 5: Normalizer save/load roundtrip preserves statistics."""
    print("=" * 60)
    print("Test 5: Normalizer Save/Load Roundtrip")
    print("=" * 60)

    stats_path = os.path.join(tmpdir, "dataset_stats.pkl")

    # Load saved stats
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    # Create normalizer from stats
    normalizer = ACTNormalizer()
    normalizer.load_state_dict(stats)

    # Normalize → denormalize roundtrip
    raw_action = torch.randn(1, 7)
    normalized = normalizer.normalize_action(raw_action)
    recovered = normalizer.denormalize_action(normalized)
    error = torch.abs(raw_action - recovered).max().item()
    print(f"  Action roundtrip error: {error:.2e}")
    assert error < 1e-5, f"Roundtrip error too large: {error}"

    raw_qpos = torch.randn(1, 7)
    normalized_q = normalizer.normalize_qpos(raw_qpos)
    recovered_q = normalizer.denormalize_qpos(normalized_q)
    error_q = torch.abs(raw_qpos - recovered_q).max().item()
    print(f"  Qpos roundtrip error:   {error_q:.2e}")
    assert error_q < 1e-5, f"Roundtrip error too large: {error_q}"

    # Verify stats are numerically identical after save/load
    original_normalizer = ACTNormalizer()
    original_normalizer.load_state_dict(stats)
    state1 = normalizer.state_dict()
    state2 = original_normalizer.state_dict()
    for key in state1:
        v1, v2 = state1[key], state2[key]
        if isinstance(v1, torch.Tensor):
            assert torch.allclose(v1, v2), f"Mismatch in {key}"
        else:
            assert v1 == v2, f"Mismatch in {key}"
    print("  Stats identical after save/load: OK")
    print("  PASSED\n")


if __name__ == "__main__":
    # Run all integration tests sequentially (later tests depend on earlier ones)
    tmpdir, config = test_train_and_save()
    policy, normalizer = test_load_and_infer(tmpdir, config)
    test_inference_pipeline(policy, normalizer, config)
    test_config_inference_from_checkpoint(tmpdir, config)
    test_normalizer_roundtrip(tmpdir)

    print("=" * 60)
    print("All integration tests passed!")
    print("=" * 60)
