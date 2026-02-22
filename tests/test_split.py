"""
Unit tests for the episode-level train/val split in train.py.

Validates:
  - no episode contributes frames to both train and val
  - every frame lands in exactly one split
  - the split is reproducible for a fixed seed
  - at least one episode always stays in train
"""

import pytest

from train import split_by_episode

# 5 episodes of uneven length, frames stored contiguously like LeRobot
EPISODES = [0] * 4 + [1] * 3 + [2] * 5 + [3] * 2 + [4] * 6


def test_no_episode_leakage():
    train_idx, val_idx, val_eps = split_by_episode(EPISODES, val_ratio=0.4, seed=0)
    train_eps = {EPISODES[i] for i in train_idx}
    assert train_eps.isdisjoint(val_eps)
    assert {EPISODES[i] for i in val_idx} == set(val_eps)
    assert len(val_eps) == 2


def test_covers_all_frames():
    train_idx, val_idx, _ = split_by_episode(EPISODES, val_ratio=0.2, seed=0)
    assert sorted(train_idx + val_idx) == list(range(len(EPISODES)))


def test_deterministic_for_seed():
    assert split_by_episode(EPISODES, 0.2, seed=3) == split_by_episode(EPISODES, 0.2, seed=3)


def test_keeps_a_training_episode():
    _, _, val_eps = split_by_episode([0, 0, 1, 1], val_ratio=0.9, seed=0)
    assert len(val_eps) == 1


def test_single_episode_rejected():
    with pytest.raises(ValueError):
        split_by_episode([0, 0, 0], val_ratio=0.1, seed=0)
