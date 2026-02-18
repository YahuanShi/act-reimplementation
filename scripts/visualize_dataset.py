"""
Dataset statistics visualization — quick sanity check for LeRobot datasets.

Outputs:
  - Action dimension distributions (histogram per dim)
  - Episode length distribution
  - Sample action trajectory from the first episode

Usage:
    python scripts/visualize_dataset.py --repo_id lerobot/aloha_sim_transfer_cube_human
    python scripts/visualize_dataset.py --repo_id local/my_dataset --root ./data --output_dir ./plots
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def plot_action_distributions(actions: torch.Tensor, output_dir: str):
    """Plot per-dimension action histograms.

    Why this matters for ACT training:
    - Reveals the range of each joint/action dimension before normalization.
    - A very skewed distribution (e.g., most values clustered near 0 with rare
      large values) may cause the mean/std normalizer to produce poor scaling.
      In that case, consider limits-based (min-max) normalization instead.
    - Red dashed line shows the mean — far from the histogram centre indicates skew.
    """
    actions_np = actions.numpy()
    n_dims = actions_np.shape[1]
    cols = min(4, n_dims)
    rows = (n_dims + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.array(axes).flatten() if n_dims > 1 else [axes]

    for i in range(n_dims):
        axes[i].hist(actions_np[:, i], bins=50, color="steelblue", alpha=0.8)
        axes[i].set_title(f"action dim {i}")
        axes[i].set_xlabel("value")
        axes[i].set_ylabel("count")
        axes[i].axvline(
            actions_np[:, i].mean(),
            color="red",
            linestyle="--",
            linewidth=1,
            label=f"mean={actions_np[:, i].mean():.2f}",
        )
        axes[i].legend(fontsize=8)

    for i in range(n_dims, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Action dimension distributions", fontsize=14)
    fig.tight_layout()
    path = os.path.join(output_dir, "action_distributions.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved {path}")


def plot_episode_lengths(ds: LeRobotDataset, output_dir: str):
    """Plot the distribution of episode lengths (in frames).

    LeRobot stores a flat list of frames for all episodes. episode_data_index
    holds the [from, to) frame indices for each episode — this is how the dataset
    knows which frames belong to which episode without storing episode IDs per frame.

    Why this matters:
    - High variance in episode length affects chunk boundary handling (shorter
      episodes may produce many padded chunks if chunk_size >= episode length).
    - Very short episodes (<< chunk_size) are wasted data because the action
      chunk extends past the episode end and must be padded with is_pad=True.
    """
    lengths = [
        ds.episode_data_index["to"][i].item() - ds.episode_data_index["from"][i].item()
        for i in range(ds.num_episodes)
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(lengths, bins=min(30, ds.num_episodes), color="darkorange", alpha=0.8)
    ax.set_xlabel("Episode length (frames)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Episode length distribution  (n={ds.num_episodes}, "
        f"mean={np.mean(lengths):.0f}, min={min(lengths)}, max={max(lengths)})"
    )
    ax.axvline(np.mean(lengths), color="red", linestyle="--", linewidth=1, label="mean")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(output_dir, "episode_lengths.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved {path}")


def plot_sample_trajectory(ds: LeRobotDataset, action_key: str, output_dir: str):
    """Plot the action trajectory of the first episode.

    Useful for:
    - Checking that data loading produced sensible values (not all zeros, not NaN).
    - Seeing which dimensions are active vs. constant (constant dims waste model
      capacity and may need to be removed from action_dim).
    - Understanding the temporal structure: periodic motion, ramp-up/down, etc.
    """
    ep_start = ds.episode_data_index["from"][0].item()
    ep_end = ds.episode_data_index["to"][0].item()
    actions = torch.stack(
        [torch.tensor(ds.hf_dataset[i][action_key]) for i in range(ep_start, ep_end)]
    )
    actions_np = actions.numpy()
    T, n_dims = actions_np.shape

    fig, ax = plt.subplots(figsize=(12, 4))
    for i in range(n_dims):
        ax.plot(range(T), actions_np[:, i], label=f"dim {i}", alpha=0.8)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Action value")
    ax.set_title(f"Action trajectory — episode 0  (T={T}, action_dim={n_dims})")
    ax.legend(loc="upper right", fontsize=8, ncol=max(1, n_dims // 4))
    fig.tight_layout()
    path = os.path.join(output_dir, "sample_trajectory.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved {path}")


def print_summary(ds: LeRobotDataset, action_key: str):
    """Print dataset statistics to help set normalization parameters.

    The action range tells you whether mean/std or min-max normalization is
    more appropriate. A large range with moderate std suggests min-max;
    a Gaussian-looking range suggests z-score normalization.
    """
    actions = torch.tensor(
        np.stack([ds.hf_dataset[i][action_key] for i in range(len(ds.hf_dataset))])
    )
    print(f"  fps:          {ds.fps}")
    print(f"  episodes:     {ds.num_episodes}")
    print(f"  total frames: {len(ds.hf_dataset)}")
    print(f"  action_dim:   {actions.shape[1]}")
    print(f"  action range: [{actions.min():.3f}, {actions.max():.3f}]")
    print(f"  action mean:  {actions.mean(0).tolist()}")
    print(f"  action std:   {actions.std(0).tolist()}")


def main():
    parser = argparse.ArgumentParser(description="Visualize a LeRobot dataset")
    parser.add_argument("--repo_id", required=True, help="LeRobot dataset repo_id")
    parser.add_argument("--root", default=None, help="Local dataset root (optional)")
    parser.add_argument("--action_key", default="action")
    parser.add_argument("--output_dir", default="plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading dataset: {args.repo_id} ...")
    ds = LeRobotDataset(args.repo_id, root=args.root)

    print("\nDataset summary:")
    print_summary(ds, args.action_key)

    print("\nGenerating plots ...")
    actions = torch.tensor(
        np.stack([ds.hf_dataset[i][args.action_key] for i in range(len(ds.hf_dataset))])
    )
    plot_action_distributions(actions, args.output_dir)
    plot_episode_lengths(ds, args.output_dir)
    plot_sample_trajectory(ds, args.action_key, args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
