"""
Integration tests — full pipeline: train → save → load → infer.

Run: pytest tests/ -v

Covers:
  - TrainWorkspace produces valid checkpoint and normalizer files
  - Checkpoint can be reloaded and produces identical inference outputs
  - Config can be inferred from state_dict tensor shapes (eval.py flow)
  - checkpoint_full.pt is written periodically; step checkpoints resume correctly
  - EvalRunner + MockEnv completes n_test episodes without errors
"""

import os
import pickle
import tempfile

import torch
from torch.utils.data import DataLoader, Dataset

from act.dataset.normalizer import ACTNormalizer
from act.policy.act_policy import ACTPolicy
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
        "ckpt_dir": tmpdir,
        "seed": 42,
        "save_every": 2,
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

        assert os.path.exists(os.path.join(tmpdir, "policy_best.ckpt"))
        assert os.path.exists(os.path.join(tmpdir, "policy_last.ckpt"))
        assert os.path.exists(os.path.join(tmpdir, "dataset_stats.pkl"))
        assert best_loss < float("inf")


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

        state_dict = torch.load(
            os.path.join(tmpdir, "policy_best.ckpt"),
            map_location="cpu",
            weights_only=True,
        )
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

        sd = torch.load(
            os.path.join(tmpdir, "policy_best.ckpt"),
            map_location="cpu",
            weights_only=True,
        )

        # Replicate eval.py's config inference logic
        num_queries = sd["model.query_embed.weight"].shape[0]
        hidden_dim = sd["model.query_embed.weight"].shape[1]
        action_dim = sd["model.action_head.weight"].shape[0]
        state_dim = sd["model.encoder_joint_proj.weight"].shape[1]
        enc_layers = sum(
            1 for k in sd if "transformer.encoder.layers" in k and "self_attn.in_proj_weight" in k
        )
        dec_layers = sum(
            1 for k in sd if "transformer.decoder.layers" in k and "self_attn.in_proj_weight" in k
        )

        assert num_queries == config["num_queries"]
        assert hidden_dim == config["hidden_dim"]
        assert action_dim == config["action_dim"]
        assert state_dim == config["state_dim"]
        assert enc_layers == config["enc_layers"]
        assert dec_layers == config["dec_layers"]


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

        with open(os.path.join(tmpdir, "dataset_stats.pkl"), "rb") as f:
            stats = pickle.load(f)
        loaded = ACTNormalizer()
        loaded.load_state_dict(stats)

        sample = torch.randn(1, 7)
        assert torch.allclose(
            normalizer.normalize_action(sample),
            loaded.normalize_action(sample),
            atol=1e-6,
        )


def test_periodic_full_checkpoint_and_step_resume():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _base_config(tmpdir, epochs=3)
        config.update(save_every=1, save_every_steps=2, max_keep_ckpts=2)
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
        n_test=3,
        max_steps=20,
    )
    metrics = runner.run()
    assert metrics["n_episodes"] == 3
    assert "mean_reward" in metrics
    assert "success_rate" in metrics
