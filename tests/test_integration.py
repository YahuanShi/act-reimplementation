"""
Integration tests — full pipeline: train → save → load → infer.

Run: pytest tests/ -v

Covers:
  - TrainWorkspace produces self-contained inference checkpoints (weights + config + stats)
  - Checkpoint can be reloaded and produces identical inference outputs
  - Config can be inferred from state_dict tensor shapes (fallback for old checkpoints)
  - checkpoint_full.pt is written periodically; step checkpoints resume correctly
  - max_steps caps training mid-epoch; legacy checkpoint directories still load
  - EvalRunner + MockEnv completes n_episodes episodes without errors
"""

import json
import os
import pickle
import tempfile

import torch
from torch.utils.data import DataLoader, Dataset

from act.dataset.normalizer import ACTNormalizer
from act.policy.act_policy import ACTPolicy
from act.policy.loader import infer_config_from_state_dict, load_checkpoint
from evaluation.runner import EvalRunner, MockEnv
from training.workspace import TrainWorkspace


class _SyntheticDataset(Dataset):
    def __init__(self, n=16, state_dim=7, action_dim=7, chunk_size=20, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.data = [
            {
                "qpos": torch.randn(state_dim, generator=g),
                "image": torch.rand(1, 3, 120, 160, generator=g),
                "action": torch.randn(chunk_size, action_dim, generator=g),
                "is_pad": torch.zeros(chunk_size, dtype=torch.bool),
            }
            for _ in range(n)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def _base_config(tmpdir, epochs=3):
    return {
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
        "num_epochs": epochs,
        "batch_size": 4,
        "output_dir": tmpdir,
        "seed": 42,
        "save_every_epochs": 2,
    }


def test_training_produces_checkpoints():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _base_config(tmpdir)
        train_ds = _SyntheticDataset(n=16)
        val_ds = _SyntheticDataset(n=8, seed=1)
        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=4)

        normalizer = ACTNormalizer()
        normalizer.fit_from_tensors(torch.randn(100, 7), torch.randn(100, 7))

        workspace = TrainWorkspace(config)
        best_epoch, best_loss, _ = workspace.run(train_loader, val_loader, normalizer)

        assert os.path.exists(os.path.join(tmpdir, "policy_best.pt"))
        assert os.path.exists(os.path.join(tmpdir, "policy_final.pt"))
        assert best_loss < float("inf")
        # Inference checkpoints are self-contained: weights + config + normalizer
        state_dict, cfg, _ = load_checkpoint(os.path.join(tmpdir, "policy_best.pt"))
        assert (
            cfg["num_queries"] == config["num_queries"] and "model.query_embed.weight" in state_dict
        )


def test_checkpoint_reload_deterministic():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _base_config(tmpdir)
        train_ds = _SyntheticDataset(n=16)
        val_ds = _SyntheticDataset(n=8, seed=1)
        normalizer = ACTNormalizer()
        normalizer.fit_from_tensors(torch.randn(100, 7), torch.randn(100, 7))

        workspace = TrainWorkspace(config)
        workspace.run(
            DataLoader(train_ds, batch_size=4),
            DataLoader(val_ds, batch_size=4),
            normalizer,
        )

        state_dict, _, _ = load_checkpoint(tmpdir)  # directory → policy_best.pt
        policy = ACTPolicy(config)
        policy.load_state_dict(state_dict)
        policy.eval()

        qpos = torch.randn(1, 7)
        image = torch.rand(1, 1, 3, 120, 160)
        with torch.no_grad():
            a1 = policy.predict_action(qpos, image)
            a2 = policy.predict_action(qpos, image)

        assert torch.allclose(a1, a2), "same input must produce same output"
        assert a1.shape == (1, 20, 7)


def test_config_inferred_from_state_dict():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _base_config(tmpdir)
        train_ds = _SyntheticDataset(n=16)
        val_ds = _SyntheticDataset(n=8, seed=1)
        normalizer = ACTNormalizer()
        normalizer.fit_from_tensors(torch.randn(100, 7), torch.randn(100, 7))

        workspace = TrainWorkspace(config)
        workspace.run(
            DataLoader(train_ds, batch_size=4),
            DataLoader(val_ds, batch_size=4),
            normalizer,
        )

        sd, _, _ = load_checkpoint(os.path.join(tmpdir, "policy_best.pt"))
        inferred = infer_config_from_state_dict(sd)
        for key in ("num_queries", "hidden_dim", "action_dim", "state_dim"):
            assert inferred[key] == config[key], key
        assert inferred["enc_layers"] == config["enc_layers"]
        assert inferred["dec_layers"] == config["dec_layers"]


def test_normalizer_stats_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _base_config(tmpdir)
        normalizer = ACTNormalizer()
        normalizer.fit_from_tensors(torch.randn(100, 7), torch.randn(100, 7))
        workspace = TrainWorkspace(config)
        workspace.run(
            DataLoader(_SyntheticDataset(n=16), batch_size=4),
            DataLoader(_SyntheticDataset(n=8, seed=1), batch_size=4),
            normalizer,
        )

        _, _, loaded = load_checkpoint(os.path.join(tmpdir, "policy_final.pt"))

        sample = torch.randn(1, 7)
        assert torch.allclose(
            normalizer.normalize_action(sample),
            loaded.normalize_action(sample),
            atol=1e-6,
        )


def test_periodic_full_checkpoint_and_step_resume():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _base_config(tmpdir, epochs=3)
        config.update(save_every_epochs=1, save_every_steps=2, max_keep_checkpoints=2)
        normalizer = ACTNormalizer()
        normalizer.fit_from_tensors(torch.randn(100, 7), torch.randn(100, 7))
        train_loader = DataLoader(_SyntheticDataset(n=16), batch_size=4)  # 4 steps/epoch
        val_loader = DataLoader(_SyntheticDataset(n=8, seed=1), batch_size=4)

        TrainWorkspace(config).run(train_loader, val_loader, normalizer)

        # checkpoint_full.pt is written periodically, epoch/step copies are pruned
        assert os.path.exists(os.path.join(tmpdir, "checkpoint_full.pt"))
        step_ckpts = sorted(f for f in os.listdir(tmpdir) if f.startswith("checkpoint_step"))
        assert step_ckpts == ["checkpoint_step10.pt", "checkpoint_step12.pt"]

        # A mid-epoch step checkpoint must be resumable and re-run its epoch
        # (step 10 = 2nd batch of epoch 2, which covers steps 9-12)
        step_path = os.path.join(tmpdir, "checkpoint_step10.pt")
        ckpt = torch.load(step_path, map_location="cpu", weights_only=False)
        assert ckpt["epoch"] == 2 and ckpt["epoch_complete"] is False

        resume_dir = os.path.join(tmpdir, "resume")
        resume_cfg = _base_config(resume_dir, epochs=3)
        resume_cfg["resume"] = step_path
        TrainWorkspace(resume_cfg).run(train_loader, val_loader, normalizer)
        final = torch.load(
            os.path.join(resume_dir, "checkpoint_full.pt"), map_location="cpu", weights_only=False
        )
        # Epoch 2 is re-run from step 10: 10 + 4 = 14 steps
        assert final["global_step"] == 14
        assert final["epoch"] == 2


def test_eval_runner_mock_env():
    config = {
        "state_dim": 7,
        "action_dim": 7,
        "hidden_dim": 128,
        "dim_feedforward": 256,
        "backbone": "resnet18",
        "lr_backbone": 0,
        "enc_layers": 2,
        "dec_layers": 2,
        "nheads": 4,
        "num_queries": 10,
        "camera_names": ["cam"],
        "lr": 1e-4,
        "kl_weight": 10.0,
    }
    policy = ACTPolicy(config)
    normalizer = ACTNormalizer()
    normalizer.fit_from_tensors(torch.randn(50, 7), torch.randn(50, 7))

    runner = EvalRunner(
        env_factory=lambda: MockEnv(state_dim=7, action_dim=7, episode_len=20),
        policy=policy,
        normalizer=normalizer,
        chunk_size=10,
        temporal_agg=False,
        n_episodes=3,
        max_steps=20,
    )
    metrics = runner.run()
    assert metrics["n_episodes"] == 3
    assert "mean_reward" in metrics
    assert "success_rate" in metrics


def test_max_steps_overrides_epochs():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _base_config(tmpdir, epochs=100)
        config.update(max_steps=6, save_every_steps=1, max_keep_checkpoints=0)
        normalizer = ACTNormalizer()
        normalizer.fit_from_tensors(torch.randn(100, 7), torch.randn(100, 7))
        TrainWorkspace(config).run(
            DataLoader(_SyntheticDataset(n=16), batch_size=4),  # 4 steps/epoch
            DataLoader(_SyntheticDataset(n=8, seed=1), batch_size=4),
            normalizer,
        )
        # 6 steps = 1.5 epochs: exactly 6 step checkpoints, none beyond max_steps
        steps = sorted(int(f[15:-3]) for f in os.listdir(tmpdir) if f.startswith("checkpoint_step"))
        assert steps == [1, 2, 3, 4, 5, 6]


def test_legacy_checkpoint_directory_loads():
    """Checkpoints from before the self-contained format: bare state_dict + side files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _base_config(tmpdir)
        normalizer = ACTNormalizer()
        normalizer.fit_from_tensors(torch.randn(100, 7), torch.randn(100, 7))
        policy = ACTPolicy(config)
        torch.save(policy.state_dict(), os.path.join(tmpdir, "policy_best.ckpt"))
        with open(os.path.join(tmpdir, "dataset_stats.pkl"), "wb") as f:
            pickle.dump(normalizer.state_dict(), f)
        with open(os.path.join(tmpdir, "config.json"), "w") as f:
            json.dump(config, f)

        state_dict, cfg, loaded = load_checkpoint(tmpdir)
        assert cfg["hidden_dim"] == config["hidden_dim"]
        assert state_dict.keys() == policy.state_dict().keys()
        sample = torch.randn(1, 7)
        assert torch.allclose(normalizer.normalize_action(sample), loaded.normalize_action(sample))
