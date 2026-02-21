"""
Evaluation runner — rollout a trained ACT policy in an environment.

ACT uses action chunking: at each query step the policy predicts a chunk of
future actions, which are then executed sequentially. Two modes are supported:

  1. Chunked (no temporal ensemble):
       - Query policy every chunk_size steps
       - Execute all predicted actions in the chunk
       - Re-plan at the next chunk boundary

  2. Temporal ensemble:
       - Query policy every step
       - Smooth overlapping predictions with exp(-k*i) weighting
       - Execute only the ensembled action at the current step

Both modes share the same interface: call runner.run(policy) and it returns
per-episode metrics aggregated over n_test rollouts.

Environments must follow the gymnasium interface, with dict observations
already converted to ACT's input format (see evaluation/aloha_env.py):
  obs, info = env.reset(seed=seed)
      obs = { 'qpos': (state_dim,), 'image': (num_cam, 3, H, W) float in [0, 1] }
  obs, reward, terminated, truncated, info = env.step(action)   action: (action_dim,)

An episode counts as a success if the env ever reports info['is_success'].
Termination alone is NOT success: an episode can also end by time limit.
Optionally, env.render() returning an (H, W, 3) uint8 frame enables videos.
"""

import numpy as np
import torch

from inference.temporal_ensemble import InferencePipeline


def run_episode(env, pipeline, max_steps=400, seed=None, record=False):
    """
    Roll out one episode using an InferencePipeline.

    The pipeline handles both chunked and temporal-ensemble modes internally.

    Args:
        env: gymnasium-style environment with ACT-format dict obs
        pipeline: InferencePipeline (already configured with policy + normalizer)
        max_steps: episode length cap
        seed: reset seed (controls the object's initial position)
        record: collect env.render() frames for a video

    Returns:
        dict: { 'total_reward', 'max_reward', 'steps', 'success', 'frames' }
    """
    obs, info = env.reset(seed=seed)
    pipeline.reset()
    total_reward = 0.0
    max_reward = float("-inf")
    success = bool(info.get("is_success", False))
    frames = [env.render()] if record else []
    step = 0

    while step < max_steps:
        qpos = torch.as_tensor(obs["qpos"], dtype=torch.float32)
        image = torch.as_tensor(obs["image"], dtype=torch.float32)
        # image expected as (num_cam, 3, H, W); add cam dim if single camera
        if image.dim() == 3:
            image = image.unsqueeze(0)

        action = pipeline.get_action(step, qpos, image)
        obs, reward, terminated, truncated, info = env.step(action.numpy())
        total_reward += reward
        max_reward = max(max_reward, reward)
        success = success or bool(info.get("is_success", False))
        step += 1
        if record:
            frames.append(env.render())
        if terminated or truncated:
            break

    return {
        "total_reward": float(total_reward),
        "max_reward": float(max_reward),
        "steps": step,
        "success": success,
        "frames": frames,
    }


class EvalRunner:
    """
    Multi-episode evaluation runner for ACT policies.

    One env and one pipeline are built and reused; each episode resets both,
    and episode i is reset with seed + i so results are reproducible.

    Args:
        env_factory: callable that returns a fresh environment instance
        policy: ACTPolicy instance
        normalizer: ACTNormalizer instance
        chunk_size: action chunk length (num_queries)
        temporal_agg: whether to use temporal ensembling
        k: temporal ensemble decay constant
        n_test: number of rollout episodes
        max_steps: maximum steps per episode
        device: torch device for inference
        seed: base reset seed
        num_videos: record frames for the first num_videos episodes
    """

    def __init__(
        self,
        env_factory,
        policy,
        normalizer,
        chunk_size,
        temporal_agg=False,
        k=0.01,
        n_test=50,
        max_steps=400,
        device="cpu",
        seed=0,
        num_videos=0,
    ):
        self.env_factory = env_factory
        self.policy = policy
        self.normalizer = normalizer
        self.chunk_size = chunk_size
        self.temporal_agg = temporal_agg
        self.k = k
        self.n_test = n_test
        self.max_steps = max_steps
        self.device = device
        self.seed = seed
        self.num_videos = num_videos

    def run(self):
        """
        Run n_test episodes and aggregate metrics.

        Returns:
            dict with success rate, reward stats, episode count and per-episode
            results (each with 'frames' if recorded)
        """
        env = self.env_factory()
        pipeline = InferencePipeline(
            self.policy,
            self.normalizer,
            chunk_size=self.chunk_size,
            temporal_agg=self.temporal_agg,
            max_timesteps=self.max_steps,
            k=self.k,
            device=self.device,
        )
        results = []
        for ep_idx in range(self.n_test):
            result = run_episode(
                env,
                pipeline,
                max_steps=self.max_steps,
                seed=self.seed + ep_idx,
                record=ep_idx < self.num_videos,
            )
            results.append(result)
            n_success = sum(r["success"] for r in results)
            print(
                f"  Episode {ep_idx + 1:3d}/{self.n_test}: "
                f"success={result['success']!s:5}  "
                f"max_reward={result['max_reward']:.1f}  "
                f"steps={result['steps']}  "
                f"(running success {n_success}/{len(results)})"
            )
        if hasattr(env, "close"):
            env.close()

        rewards = [r["total_reward"] for r in results]
        return {
            "success_rate": float(np.mean([r["success"] for r in results])),
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "max_reward": float(np.max(rewards)),
            "min_reward": float(np.min(rewards)),
            "n_episodes": len(results),
            "results": results,
        }


class MockEnv:
    """
    Mock environment for testing the evaluation pipeline without real hardware.

    Generates random observations that match the ACT input format:
      - 'qpos':  (state_dim,) joint positions
      - 'image': (num_cam, 3, H, W) camera images in [0, 1]

    The reward is a negative quadratic of the action (shaped to decrease toward 0),
    so a near-zero action policy will score near 0 per step.
    """

    def __init__(
        self,
        state_dim=7,
        action_dim=7,
        num_cam=1,
        img_h=120,
        img_w=160,
        episode_len=50,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_cam = num_cam
        self.img_h = img_h
        self.img_w = img_w
        self.episode_len = episode_len
        self._step = 0

    def reset(self, seed=None):
        self._step = 0
        return self._obs(), {"is_success": False}

    def step(self, action):
        self._step += 1
        obs = self._obs()
        # Reward: negative L2 norm of action (encourage small actions)
        reward = float(-np.sum(action**2) * 0.01)
        # The mock task can't be solved: hitting the time limit is truncation
        truncated = self._step >= self.episode_len
        return obs, reward, False, truncated, {"is_success": False}

    def render(self):
        return np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)

    def _obs(self):
        return {
            "qpos": np.random.randn(self.state_dim).astype(np.float32),
            "image": np.random.rand(self.num_cam, 3, self.img_h, self.img_w).astype(np.float32),
        }
