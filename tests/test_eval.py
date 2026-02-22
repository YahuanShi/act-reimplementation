"""
Tests for evaluation rollouts and policy image augmentation.

Validates:
  - Success comes from info['is_success'], not from the episode ending
    (regression: time-limit truncation used to count as success)
  - Rollouts stop on termination and respect max_steps
  - Crop augmentation: random in train mode, deterministic center crop in eval
  - AlohaSimEnv returns ACT-format observations (skipped without gym-aloha)
"""

import numpy as np
import pytest
import torch

from act.dataset.normalizer import ACTNormalizer
from act.policy.act_policy import ACTPolicy
from evaluation.runner import EvalRunner, MockEnv, run_episode
from inference.temporal_ensemble import InferencePipeline


def make_config(**overrides):
    config = {
        "state_dim": 7,
        "action_dim": 7,
        "hidden_dim": 64,
        "dim_feedforward": 128,
        "backbone": "resnet18",
        "lr_backbone": 0,
        "enc_layers": 1,
        "dec_layers": 1,
        "nheads": 4,
        "num_queries": 10,
        "camera_names": ["cam"],
        "lr": 1e-4,
        "kl_weight": 10.0,
    }
    config.update(overrides)
    return config


def make_pipeline(policy):
    normalizer = ACTNormalizer()
    normalizer.fit_from_tensors(torch.randn(50, 7), torch.randn(50, 7))
    return InferencePipeline(policy, normalizer, chunk_size=10, max_timesteps=50)


class _SucceedsAtStep(MockEnv):
    """Reports success and terminates at a fixed step."""

    def __init__(self, success_step, **kwargs):
        super().__init__(**kwargs)
        self.success_step = success_step

    def step(self, action):
        obs, reward, _, truncated, _ = super().step(action)
        success = self._step == self.success_step
        return obs, reward, success, truncated, {"is_success": success}


def test_timeout_is_not_success():
    pipeline = make_pipeline(ACTPolicy(make_config()))
    result = run_episode(MockEnv(episode_len=5), pipeline, max_steps=20)
    assert result["steps"] == 5
    assert result["success"] is False


def test_success_stops_episode():
    pipeline = make_pipeline(ACTPolicy(make_config()))
    result = run_episode(_SucceedsAtStep(3, episode_len=20), pipeline, max_steps=20)
    assert result["success"] is True
    assert result["steps"] == 3


def test_runner_success_rate_and_videos():
    policy = ACTPolicy(make_config())
    normalizer = ACTNormalizer()
    normalizer.fit_from_tensors(torch.randn(50, 7), torch.randn(50, 7))
    runner = EvalRunner(
        env_factory=lambda: _SucceedsAtStep(2, episode_len=4),
        policy=policy,
        normalizer=normalizer,
        chunk_size=10,
        n_test=2,
        max_steps=4,
        num_videos=1,
    )
    metrics = runner.run()
    assert metrics["success_rate"] == 1.0
    # step 0 frame + one per step, only for the first episode
    assert len(metrics["results"][0]["frames"]) == 3
    assert metrics["results"][1]["frames"] == []


def test_crop_random_in_train_center_in_eval():
    policy = ACTPolicy(make_config(crop_size=[100, 140], color_jitter=True))
    image = torch.rand(2, 1, 3, 120, 160)

    policy.train()
    assert policy._preprocess_images(image).shape == (2, 1, 3, 100, 140)

    policy.eval()
    out1 = policy._preprocess_images(image)
    out2 = policy._preprocess_images(image)
    assert out1.shape == (2, 1, 3, 100, 140)
    assert torch.equal(out1, out2), "eval mode must be deterministic (center crop, no jitter)"


def test_augmented_policy_trains_and_infers():
    policy = ACTPolicy(make_config(crop_size=[100, 140], color_jitter=True))
    policy.train()
    loss = policy.compute_loss(
        torch.randn(2, 7),
        torch.rand(2, 1, 3, 120, 160),
        torch.randn(2, 10, 7),
        torch.zeros(2, 10, dtype=torch.bool),
    )["loss"]
    assert torch.isfinite(loss)
    policy.eval()
    assert policy.predict_action(torch.randn(1, 7), torch.rand(1, 1, 3, 120, 160)).shape == (
        1,
        10,
        7,
    )


def test_aloha_env_obs_format():
    pytest.importorskip("gym_aloha")
    from evaluation.aloha_env import AlohaSimEnv

    try:
        env = AlohaSimEnv(camera_names=["observation.images.top"], max_steps=5)
        obs, info = env.reset(seed=0)
    except Exception as e:  # no EGL/OSMesa rendering available
        pytest.skip(f"MuJoCo rendering unavailable: {e}")

    assert obs["qpos"].shape == (14,)
    assert obs["image"].shape == (1, 3, 480, 640)
    assert obs["image"].dtype == np.float32
    assert 0.0 <= obs["image"].min() and obs["image"].max() <= 1.0
    assert env.render().shape == (480, 640, 3)

    obs, reward, terminated, truncated, info = env.step(obs["qpos"])
    assert "is_success" in info
    env.close()
