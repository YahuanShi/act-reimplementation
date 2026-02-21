"""
ALOHA simulation environment adapter for ACT evaluation.

Wraps gym-aloha (MuJoCo, bimanual ViperX arms) so its observations match what
ACT was trained on from the LeRobot ALOHA sim datasets:

  gym-aloha obs                          ACT obs
  ─────────────────────────────────      ───────────────────────────────────
  'agent_pos'   (14,) float              'qpos'  (14,)
  'pixels'/'top' (480, 640, 3) uint8     'image' (num_cam, 3, 480, 640) float [0, 1]

The LeRobot dataset stores the same camera as 'observation.images.top', so the
camera is picked by the last dotted component of each training camera name.

Tasks (both end the episode on success, when reward reaches 4):
  - AlohaTransferCube-v0: right arm picks up the cube, hands it to the left arm
      datasets: lerobot/aloha_sim_transfer_cube_scripted / _human
  - AlohaInsertion-v0:    each arm picks up one part, insert peg into socket
      datasets: lerobot/aloha_sim_insertion_scripted / _human

Rendering is headless: MUJOCO_GL defaults to 'egl' (GPU). Use MUJOCO_GL=osmesa
on machines without EGL.

Requires the sim extra:  uv sync --extra sim
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import gym_aloha  # noqa: E402, F401  (registers the gym_aloha/* envs)
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402


class AlohaSimEnv:
    """
    gym-aloha environment returning ACT-format observations.

    Args:
        task: gym-aloha task id, e.g. "AlohaTransferCube-v0"
        camera_names: camera names from the training config, e.g.
                      ["observation.images.top"]; order defines the image stack
        max_steps: episode time limit (LeRobot datasets use 400 at 50 fps)
    """

    def __init__(self, task="AlohaTransferCube-v0", camera_names=("top",), max_steps=400):
        self.env = gym.make(
            f"gym_aloha/{task}",
            obs_type="pixels_agent_pos",
            max_episode_steps=max_steps,
        )
        self.cams = [name.split(".")[-1] for name in camera_names]
        self._frame = None

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        return self._convert(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(
            np.asarray(action, dtype=np.float32)
        )
        return self._convert(obs), reward, terminated, truncated, info

    def render(self):
        """Last observed frame(s), side by side, (H, W * num_cam, 3) uint8."""
        return self._frame

    def close(self):
        self.env.close()

    def _convert(self, obs):
        pixels = [obs["pixels"][cam] for cam in self.cams]  # each (H, W, 3) uint8
        self._frame = np.concatenate(pixels, axis=1)
        image = np.stack(pixels).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        return {"qpos": obs["agent_pos"].astype(np.float32), "image": image}
