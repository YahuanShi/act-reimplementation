"""
End-to-end test of the real-robot control loop with a fake robot (no hardware).

Same structure as the DP repo's test. Validates:
  - run_episode executes exactly max_steps actions and stops
  - the robot receives a boolean gripper command for every executed action
  - the measured gripper width maps to the training-data convention (1 = open)
  - URRobot converts the degree-valued action to radians before servoJ
  - TrialRecorder writes one video per camera, steps.npz and meta.json
  - the pre-run state check flags dims far outside the training distribution
"""

import json
import os

import numpy as np
import torch

import inference
from act.dataset.normalizer import ACTNormalizer
from act.policy.act_policy import ACTPolicy
from act.policy.temporal_ensemble import InferencePipeline
from evaluation.trial_recorder import TrialRecorder

CONFIG = {
    "state_dim": 7,
    "action_dim": 7,
    "hidden_dim": 32,
    "dim_feedforward": 64,
    "backbone": "resnet18",
    "lr_backbone": 1e-5,
    "enc_layers": 1,
    "dec_layers": 1,
    "nheads": 2,
    "num_queries": 4,
    "camera_names": ["observation.images.exterior", "observation.images.wrist"],
    "lr": 1e-4,
    "kl_weight": 10.0,
}


class FakeRobot:
    def __init__(self):
        self.commands = []

    def get_obs(self):
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        return {
            "state": np.ones(7, dtype=np.float32),
            "exterior_rgb": img,
            "wrist_rgb": img.copy(),
            "sensor_age_s": 0.001,
        }

    def apply_action(self, action, gripper_open):
        self.commands.append((np.asarray(action).copy(), gripper_open))


def _pipeline(temporal_agg=False):
    torch.manual_seed(0)
    normalizer = ACTNormalizer()
    normalizer.fit_from_tensors(torch.rand(100, 7), torch.rand(100, 7))
    return InferencePipeline(
        ACTPolicy(CONFIG).eval(),
        normalizer,
        chunk_size=CONFIG["num_queries"],
        temporal_agg=temporal_agg,
        max_timesteps=20,
    )


def test_run_episode_with_fake_robot(tmp_path):
    robot = FakeRobot()
    recorder = TrialRecorder(str(tmp_path), fps=50, meta={"checkpoint": "fake"})
    steps, timing = inference.run_episode(
        _pipeline(temporal_agg=True),
        robot,
        cam_order=["exterior", "wrist"],
        state_dim=7,
        frequency=50,
        max_steps=9,
        temporal_agg=True,
        no_vis=True,
        recorder=recorder,
    )

    assert steps == 9 and len(robot.commands) == 9
    assert all(isinstance(g, bool) for _, g in robot.commands)
    assert timing.summary()["infer"]["n"] == 9

    trial_dir = recorder.close(success=False)
    for name in ("exterior.mp4", "wrist.mp4", "steps.npz", "meta.json"):
        assert os.path.exists(os.path.join(trial_dir, name)), name
    assert np.load(os.path.join(trial_dir, "steps.npz"))["action"].shape == (9, 7)
    assert json.load(open(os.path.join(trial_dir, "meta.json")))["success"] is False


def test_gripper_state_matches_training_convention():
    # Training data: 1 = open (jaws apart), 0 = closed (holding the object)
    assert inference.gripper_state_from_width(inference.GRIPPER_MAX_MM) == 1.0
    assert inference.gripper_state_from_width(0.0) == 0.0


class _FakeRTDE:
    def __init__(self):
        self.servo_targets = []

    def servoJ(self, q, *args):
        self.servo_targets.append(np.asarray(q))


def test_apply_action_converts_degrees_to_radians():
    robot = inference.URRobot.__new__(inference.URRobot)  # skip hardware __init__
    robot._dt = 1 / 20
    robot._max_joint_vel = 100.0  # no velocity clipping for this check
    robot._last_cmd_rad = inference.HOME_RAD.copy()
    robot._rtde_c = _FakeRTDE()
    robot._gripper = None

    target_deg = np.array(inference.HOME_DEG, dtype=np.float32) + 10.0
    robot.apply_action(np.append(target_deg, 1.0), gripper_open=True)
    np.testing.assert_allclose(robot._rtde_c.servo_targets[0], np.radians(target_deg), atol=1e-5)


def test_state_check_flags_unit_mismatch():
    # Joint ranges (degrees) of the UR pick-and-place data, gripper in {0, 1}
    low = torch.tensor([15.0, -146, -141, -75, -289, -38, 0])
    high = torch.tensor([112.0, -19, -46, 2, -253, 38, 1])
    deg = low + (high - low) * torch.rand(500, 7)
    normalizer = ACTNormalizer()
    normalizer.fit_from_tensors(torch.randn(500, 7), deg)
    assert inference.check_qpos_in_distribution(deg[0].numpy(), normalizer) == []
    # Forgot the rad → deg conversion: joint 4 (≈ -270°) becomes ≈ -4.7 and is flagged
    as_radians = deg[0].numpy().copy()
    as_radians[:6] = np.radians(as_radians[:6])
    assert 4 in inference.check_qpos_in_distribution(as_radians, normalizer)
