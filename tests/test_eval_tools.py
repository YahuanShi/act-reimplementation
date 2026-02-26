"""
Tests for evaluation / deployment helpers that run without hardware.

Validates:
  - evaluation.offline.evaluate_offline reports finite per-dim L1 and writes plots
  - inference.resolve_camera_order follows config camera_names, fails on missing cams
  - inference._center_crop_resize produces the training resolution for any aspect ratio
"""

import json
import os
import tempfile

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from act.dataset.normalizer import ACTNormalizer
from act.policy.act_policy import ACTPolicy
from act.policy.temporal_ensemble import InferencePipeline
from evaluation.offline import evaluate_offline
from inference import _center_crop_resize, resolve_camera_order

CONFIG = {
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
    "camera_names": ["observation.images.exterior"],
    "lr": 1e-4,
    "kl_weight": 10.0,
}


class _EpisodeDataset(Dataset):
    """Two short episodes, last frames padded like LeRobotACTDataset."""

    def __init__(self):
        self.episode_indices = [3] * 6 + [5] * 4

    def __len__(self):
        return len(self.episode_indices)

    def __getitem__(self, idx):
        is_pad = torch.zeros(10, dtype=torch.bool)
        is_pad[7:] = True
        return {
            "qpos": torch.randn(7),
            "image": torch.rand(1, 3, 64, 64),
            "action": torch.randn(10, 7),
            "is_pad": is_pad,
        }


def test_evaluate_offline_outputs():
    normalizer = ACTNormalizer()
    normalizer.fit_from_tensors(torch.randn(50, 7), torch.randn(50, 7))
    pipeline = InferencePipeline(ACTPolicy(CONFIG), normalizer, chunk_size=10)

    with tempfile.TemporaryDirectory() as tmp:
        m = evaluate_offline(
            pipeline, CONFIG, _EpisodeDataset(), [3, 5], plot_dir=tmp, batch_size=4
        )
        assert len(m["l1_first_step_per_dim"]) == 7
        assert len(m["l1_by_chunk_step"]) == 10
        assert np.isfinite(m["l1_chunk"])
        # padded chunk steps have no valid samples → reported as 0
        assert m["l1_by_chunk_step"][-1] == 0.0
        assert json.loads(json.dumps(m))["n_frames"] == 10  # JSON-serialisable
        assert os.path.exists(os.path.join(tmp, "open_loop_ep3.png"))
        assert os.path.exists(os.path.join(tmp, "open_loop_ep5.png"))


def test_camera_order_follows_config():
    names = ["observation.images.wrist", "observation.images.exterior"]
    assert resolve_camera_order(names, ["exterior", "wrist"]) == ["wrist", "exterior"]
    with pytest.raises(ValueError):
        resolve_camera_order(names, ["exterior"])


@pytest.mark.parametrize("out_hw", [(480, 480), (480, 640), (240, 320)])
def test_center_crop_resize_shape(out_hw):
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    assert _center_crop_resize(frame, out_hw).shape == (*out_hw, 3)
