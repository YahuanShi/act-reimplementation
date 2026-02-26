"""
Adapter: ACT InferencePipeline → robot_sim's Policy interface (reset / act).

robot_sim (sibling repo ../robot-sim, robosuite Franka Panda) is shared with the
Diffusion Policy reimplementation: both policies run through robot_sim.evaluate with
the same seeds, so their success rates are directly comparable.

robot_sim observations use LeRobot names, which are also what ACT trained on:
  observation.images.agentview / .wrist  (H, W, 3) uint8  →  'image' (num_cam, 3, H, W) in [0, 1]
  observation.state (9,)                                  →  'qpos'
Cameras are stacked in config["camera_names"] order, as during training.

Chunked execution vs temporal ensembling is handled by InferencePipeline unchanged.
Used by eval.py --sim lift|can (same file name and role as in the DP repo).

Requires: uv pip install --python .venv/bin/python -e ../robot-sim "opencv-python<5"
"""

import numpy as np
import torch


class ACTSimPolicy:
    def __init__(self, pipeline, camera_names):
        self.pipeline = pipeline
        self.camera_names = list(camera_names)
        self.t = 0

    def reset(self, seed=None):
        # ACT inference is deterministic (z = 0, dropout off), so seed is unused
        self.pipeline.reset()
        self.t = 0

    def act(self, obs):
        missing = [k for k in self.camera_names + ["observation.state"] if k not in obs]
        if missing:
            raise KeyError(f"Simulator obs lacks keys the policy was trained on: {missing}")
        image = np.stack([obs[k] for k in self.camera_names]).transpose(0, 3, 1, 2)
        image = torch.from_numpy(image.astype(np.float32) / 255.0)
        qpos = torch.from_numpy(np.asarray(obs["observation.state"], dtype=np.float32))
        action = self.pipeline.get_action(self.t, qpos, image)
        self.t += 1
        return action.numpy()
