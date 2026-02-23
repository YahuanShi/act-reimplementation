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

Requires: uv pip install --python .venv/bin/python -e ../robot-sim "opencv-python<5"
"""

import numpy as np
import torch


class ACTSimPolicy:
    def __init__(self, pipeline, camera_names):
        self.pipeline = pipeline
        self.camera_names = list(camera_names)
        self.t = 0

    def reset(self):
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


def evaluate_franka(
    pipeline,
    config,
    task,
    n_episodes,
    start_seed,
    image_size,
    video_dir,
    output_json,
    max_steps,
    n_videos=5,
):
    from robot_sim.evaluate import evaluate

    action_space = {10: "abs", 7: "delta"}.get(config["action_dim"])
    if action_space is None:
        raise SystemExit(f"action_dim {config['action_dim']} matches no robot_sim action space")
    return evaluate(
        ACTSimPolicy(pipeline, config["camera_names"]),
        task=task,
        n_episodes=n_episodes,
        start_seed=start_seed,
        image_size=image_size,
        action_space=action_space,
        max_steps=max_steps,
        video_dir=video_dir,
        n_videos=n_videos,
        output_json=output_json,
    )
