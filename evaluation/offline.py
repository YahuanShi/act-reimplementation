"""
Offline (open-loop) evaluation — compare predicted action chunks against recorded demos.

Without a simulator or robot this is the cheapest signal of whether the policy learned
anything: for every frame of the held-out episodes, predict a chunk and measure the L1
error against the demonstrated actions, in raw units (per dim, first step and full chunk).

Caveat: open-loop error says nothing about compounding errors, and demonstrations are
multimodal, so a policy can pick a valid-but-different mode and still be scored as
"wrong". Use it to compare checkpoints and catch broken normalization or overfitting
before touching the robot, not as a substitute for closed-loop success rates.
"""

import os

import numpy as np
import torch


@torch.inference_mode()
def evaluate_offline(
    pipeline, config, dataset, episodes, plot_dir=None, batch_size=32, num_workers=2
):
    """
    Open-loop evaluation: predict a chunk at every frame of each episode and
    compare with the demonstrated actions.

    Args:
        pipeline: InferencePipeline from load_policy() (provides policy + normalizer)
        config: model config dict
        dataset: LeRobotACTDataset restricted to the episodes to evaluate
        episodes: list of episode ids (for reporting)
        plot_dir: if given, one plot per episode (predicted vs demonstrated first action)

    Returns:
        metrics dict (raw action units)
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    policy, normalizer, device = pipeline.policy, pipeline.normalizer, pipeline.device
    policy.eval()
    chunk = config["num_queries"]

    ep_ids = np.array(dataset.episode_indices)
    abs_err_sum = 0.0  # (chunk, action_dim) running sum of |error| over valid steps
    valid_count = 0.0
    first_pred, first_gt = [], []

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    for batch in loader:
        qpos = normalizer.normalize_qpos(batch["qpos"].to(device))
        image = batch["image"].to(device)
        gt = batch["action"][:, :chunk].to(device)  # raw units
        valid = (~batch["is_pad"][:, :chunk]).to(device).float().unsqueeze(-1)

        pred = normalizer.denormalize_action(policy.predict_action(qpos, image))
        abs_err_sum = abs_err_sum + ((pred - gt).abs() * valid).sum(0)
        valid_count = valid_count + valid.sum(0)
        first_pred.append(pred[:, 0].cpu())
        first_gt.append(gt[:, 0].cpu())

    l1 = (abs_err_sum / valid_count.clamp(min=1)).cpu()  # (chunk, action_dim)
    first_pred = torch.cat(first_pred).numpy()
    first_gt = torch.cat(first_gt).numpy()

    metrics = {
        "episodes": [int(e) for e in episodes],
        "n_frames": len(dataset),
        "l1_first_step_per_dim": l1[0].tolist(),
        "l1_chunk_per_dim": l1.mean(0).tolist(),
        "l1_first_step": l1[0].mean().item(),
        "l1_chunk": l1.mean().item(),
        # How error grows along the chunk — a steep curve means late chunk
        # steps are unreliable (relevant when executing chunks without ensembling)
        "l1_by_chunk_step": l1.mean(1).tolist(),
    }

    if plot_dir is None:
        return metrics
    os.makedirs(plot_dir, exist_ok=True)
    # Per-episode plot: predicted first action vs demonstration, one subplot per dim
    action_dim = first_gt.shape[1]
    for ep in episodes:
        mask = ep_ids == ep
        fig, axes = plt.subplots(action_dim, 1, figsize=(8, 1.6 * action_dim), sharex=True)
        axes = np.atleast_1d(axes)
        for d in range(action_dim):
            axes[d].plot(first_gt[mask, d], label="demo", lw=1.2)
            axes[d].plot(first_pred[mask, d], label="policy", lw=1.0, alpha=0.8)
            axes[d].set_ylabel(f"a{d}")
        axes[0].legend(loc="upper right")
        axes[-1].set_xlabel("frame")
        fig.suptitle(f"Episode {ep} — open-loop first-step prediction")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"open_loop_ep{ep}.png"))
        plt.close(fig)

    return metrics
