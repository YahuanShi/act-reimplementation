"""
ACT ↔ robot_sim adapter (evaluation/franka_sim.py).

Skipped when robot_sim (sibling repo ../robot-sim) is not installed, e.g. in CI.
"""

import numpy as np
import pytest
import torch

pytest.importorskip("robot_sim")

from evaluation.franka_sim import ACTSimPolicy  # noqa: E402

CAMS = ["observation.images.agentview", "observation.images.wrist"]


class StubPipeline:
    def __init__(self):
        self.calls = []
        self.resets = 0

    def reset(self):
        self.resets += 1

    def get_action(self, t, qpos, image):
        self.calls.append((t, qpos, image))
        return torch.full((10,), float(t))


def _obs(value):
    img = np.full((8, 8, 3), value, dtype=np.uint8)
    return {
        "observation.images.agentview": img,
        "observation.images.wrist": img // 2,
        "observation.state": np.arange(9, dtype=np.float32),
    }


def test_act_sim_policy_converts_obs_and_counts_steps():
    pipe = StubPipeline()
    policy = ACTSimPolicy(pipe, CAMS)
    policy.reset()
    a0 = policy.act(_obs(255))
    a1 = policy.act(_obs(255))
    assert pipe.resets == 1
    assert a0.shape == (10,) and a1[0] == 1.0  # timestep advances per call
    t, qpos, image = pipe.calls[0]
    assert t == 0 and qpos.shape == (9,)
    assert image.shape == (2, 3, 8, 8)  # (num_cam, C, H, W) in camera_names order
    assert torch.allclose(image[0], torch.ones(3, 8, 8))  # 255 → 1.0
    assert torch.allclose(image[1], torch.full((3, 8, 8), 127 / 255))
    policy.reset()
    policy.act(_obs(0))
    assert pipe.calls[-1][0] == 0  # timestep restarts after reset


def test_act_sim_policy_reports_missing_camera():
    policy = ACTSimPolicy(StubPipeline(), CAMS + ["observation.images.top"])
    with pytest.raises(KeyError):
        policy.act(_obs(0))
