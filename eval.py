"""
Evaluation entry point for ACT.

Usage:
  # ALOHA sim pick-and-place, 50 episodes, save 5 videos (needs: uv sync --extra sim)
  python eval.py --ckpt_dir checkpoints --env aloha --task AlohaTransferCube-v0

  # Same with temporal ensembling
  python eval.py --ckpt_dir checkpoints --env aloha --temporal_agg

  # Shared Franka simulation (robot-sim repo, robosuite) — same seeds as the DP repo
  python eval.py --ckpt_dir checkpoints/lift --env franka --task lift --temporal_agg

  # Pipeline smoke test without a simulator (random observations)
  python eval.py --ckpt_dir checkpoints --env mock --n_episodes 2

  # Open-loop evaluation on held-out demonstrations (no robot / simulator needed)
  python eval.py --ckpt_dir checkpoints --dataset local/ur5_pick_place \\
                 --data_root ./data/ur5_pick_place

This script:
1. Loads a trained ACT policy checkpoint and normalizer stats (load_policy)
2. Without --dataset: rolls out n_episodes in the chosen environment via
   EvalRunner, reports the success rate, saves rollout videos and
   eval_results.json
3. With --dataset: replays the validation episodes recorded in config.json,
   predicts an action chunk at every frame and reports the L1 error against
   the demonstrated actions in raw units (per joint, first step and full
   chunk), plus trajectory plots. This is open-loop — it says nothing about
   compounding errors, but catches broken checkpoints, wrong normalization
   and overfitting before touching the robot.

For closed-loop real robot evaluation (success rate, videos), use
  python inference.py ... --num_episodes N --record_dir videos
"""

import argparse
import json
import os
import pickle

import av
import numpy as np
import torch

from act.dataset.normalizer import ACTNormalizer
from act.policy.act_policy import ACTPolicy
from evaluation.runner import EvalRunner, MockEnv
from inference.temporal_ensemble import InferencePipeline


def load_policy(ckpt_dir, device="cuda", temporal_agg=False, max_timesteps=400, k=0.01):
    """
    Load a trained ACT policy and normalizer for inference.

    Args:
        ckpt_dir: directory containing policy_best.ckpt and dataset_stats.pkl
        device: torch device
        temporal_agg: whether to use temporal ensembling
        max_timesteps: maximum episode length
        k: temporal ensembling decay constant

    Returns:
        pipeline: InferencePipeline ready for inference
        config: the model config dict
    """
    ckpt_path = os.path.join(ckpt_dir, "policy_best.ckpt")
    stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")

    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    assert os.path.exists(stats_path), f"Stats not found: {stats_path}"

    # Load normalizer
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    normalizer = ACTNormalizer()
    normalizer.load_state_dict(stats)

    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)

    # config.json is saved by training/workspace.py alongside every checkpoint.
    # Load it directly to get exact training hyperparameters (nheads, latent_dim,
    # camera_names, etc.). Fall back to shape-inference only for old checkpoints
    # that predate config.json saving.
    config_path = os.path.join(ckpt_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    else:
        print(f"Warning: config.json not found in {ckpt_dir}, inferring from state_dict shapes.")
        config = _infer_config_from_state_dict(state_dict)

    # Build policy and load weights
    policy = ACTPolicy(config)
    policy.load_state_dict(state_dict)
    policy.eval()

    # Build inference pipeline
    pipeline = InferencePipeline(
        policy,
        normalizer,
        chunk_size=config["num_queries"],
        temporal_agg=temporal_agg,
        max_timesteps=max_timesteps,
        k=k,
        device=device,
    )

    return pipeline, config


def _infer_config_from_state_dict(state_dict):
    """
    Reconstruct model config by inspecting state_dict tensor shapes.

    Fallback for checkpoints that don't have a config.json. Note that
    nheads cannot be inferred from weights and is hardcoded to 8, and
    latent_dim defaults to 32 — both match the standard training defaults.
    """
    # hidden_dim from query_embed: (num_queries, hidden_dim)
    query_embed = state_dict["model.query_embed.weight"]
    num_queries = query_embed.shape[0]
    hidden_dim = query_embed.shape[1]

    # action_dim from action_head: (action_dim, hidden_dim)
    action_dim = state_dict["model.action_head.weight"].shape[0]

    # state_dim from encoder_joint_proj: (hidden_dim, state_dim)
    state_dim = state_dict["model.encoder_joint_proj.weight"].shape[1]

    # latent_dim from latent_proj: (latent_dim*2, hidden_dim)
    latent_dim = state_dict["model.latent_proj.weight"].shape[0] // 2

    enc_layers = sum(
        1
        for k in state_dict
        if "transformer.encoder.layers" in k and "self_attn.in_proj_weight" in k
    )
    dec_layers = sum(
        1
        for k in state_dict
        if "transformer.decoder.layers" in k and "self_attn.in_proj_weight" in k
    )

    dim_feedforward = hidden_dim * 4
    for k, v in state_dict.items():
        if "transformer.encoder.layers.0.linear1.weight" in k:
            dim_feedforward = v.shape[0]
            break

    # nheads cannot be inferred from weight shapes alone; 8 is the standard default
    nheads = 8

    # Camera count from backbone keys; names are generic since originals aren't stored
    n_cams = len({k.split(".")[2] for k in state_dict if k.startswith("model.backbones.")})
    camera_names = [f"cam_{i}" for i in range(max(n_cams, 1))]

    return {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_dim": hidden_dim,
        "dim_feedforward": dim_feedforward,
        "backbone": "resnet18",
        "lr_backbone": 0,
        "enc_layers": enc_layers,
        "dec_layers": dec_layers,
        "nheads": nheads,
        "latent_dim": latent_dim,
        "num_queries": num_queries,
        "camera_names": camera_names,
        "lr": 1e-5,
        "kl_weight": 10.0,
    }


def make_env_factory(args, config):
    """Return a callable building the evaluation env named by --env."""
    if args.env == "aloha":
        from evaluation.aloha_env import AlohaSimEnv

        return lambda: AlohaSimEnv(
            task=args.task, camera_names=config["camera_names"], max_steps=args.max_timesteps
        )
    return lambda: MockEnv(
        state_dim=config["state_dim"],
        action_dim=config["action_dim"],
        num_cam=len(config["camera_names"]),
        episode_len=args.max_timesteps,
    )


def save_video(frames, path, fps):
    """Write (H, W, 3) uint8 frames to an H.264 mp4."""
    with av.open(path, mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.height, stream.width = frames[0].shape[:2]
        stream.pix_fmt = "yuv420p"
        for frame in frames:
            container.mux(stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")))
        container.mux(stream.encode())


@torch.inference_mode()
def open_loop_eval(pipeline, config, dataset, episodes, out_dir, batch_size=32):
    """
    Open-loop evaluation: predict a chunk at every frame of each episode and
    compare with the demonstrated actions.

    Args:
        pipeline: InferencePipeline from load_policy() (provides policy + normalizer)
        config: model config dict
        dataset: LeRobotACTDataset restricted to the episodes to evaluate
        episodes: list of episode ids (for reporting)
        out_dir: where to write open_loop_eval.json and plots

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

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
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

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "open_loop_eval.json"), "w") as f:
        json.dump(metrics, f, indent=2)

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
        fig.savefig(os.path.join(out_dir, f"open_loop_ep{ep}.png"))
        plt.close(fig)

    return metrics


def main(args):
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, using CPU")

    pipeline, config = load_policy(
        args.ckpt_dir,
        device=device,
        temporal_agg=args.temporal_agg,
        max_timesteps=args.max_timesteps,
        k=args.k,
    )

    print(f"Loaded policy from {args.ckpt_dir}")
    print(f"  State dim: {config['state_dim']}")
    print(f"  Action dim: {config['action_dim']}")
    print(f"  Chunk size: {config['num_queries']}")
    print(f"  Cameras: {config['camera_names']}")
    print(f"  Temporal ensembling: {args.temporal_agg}")

    if args.dataset:
        from act.dataset.lerobot_wrapper import LeRobotACTDataset

        episodes = args.episodes or config.get("val_episodes")
        if not episodes:
            raise SystemExit("No --episodes given and config.json has no val_episodes")
        dataset = LeRobotACTDataset(
            repo_id=args.dataset,
            root=args.data_root,
            episodes=episodes,
            chunk_size=config["num_queries"],
            image_keys=config["camera_names"],
        )
        out_dir = args.out_dir or os.path.join(args.ckpt_dir, "open_loop_eval")
        print(f"\nOpen-loop evaluation on episodes {episodes} ({len(dataset)} frames)...")
        m = open_loop_eval(pipeline, config, dataset, episodes, out_dir)
        print(f"  L1 first step : {m['l1_first_step']:.4f}")
        print(f"  L1 full chunk : {m['l1_chunk']:.4f}")
        print(
            "  L1 per dim (first step): " + " ".join(f"{v:.4f}" for v in m["l1_first_step_per_dim"])
        )
        print(f"  Results and plots: {out_dir}")
        return

    if args.env == "franka":
        from evaluation.franka_sim import evaluate_franka

        out_dir = args.out_dir or os.path.join(
            args.ckpt_dir, f"eval_franka_{args.task}_{'tagg' if args.temporal_agg else 'chunk'}"
        )
        os.makedirs(out_dir, exist_ok=True)
        evaluate_franka(
            pipeline,
            config,
            task=args.task,
            n_episodes=args.n_episodes,
            start_seed=args.seed,
            image_size=args.image_size,
            video_dir=os.path.join(out_dir, "videos") if args.num_videos else None,
            output_json=os.path.join(out_dir, "eval_results.json"),
            max_steps=args.max_timesteps,
            n_videos=args.num_videos,
        )
        print(f"Results and videos saved to {out_dir}")
        return

    print(f"\nEvaluating {args.n_episodes} episodes in {args.env} ({args.task})...")
    runner = EvalRunner(
        env_factory=make_env_factory(args, config),
        policy=pipeline.policy,
        normalizer=pipeline.normalizer,
        chunk_size=config["num_queries"],
        temporal_agg=args.temporal_agg,
        k=args.k,
        n_test=args.n_episodes,
        max_steps=args.max_timesteps,
        device=device,
        seed=args.seed,
        num_videos=args.num_videos,
    )
    metrics = runner.run()
    print(f"\nSuccess rate: {metrics['success_rate'] * 100:.1f}% ({args.n_episodes} episodes)")

    out_dir = args.out_dir or os.path.join(
        args.ckpt_dir, f"eval_{args.env}_{'tagg' if args.temporal_agg else 'chunk'}"
    )
    os.makedirs(out_dir, exist_ok=True)
    for i, result in enumerate(metrics["results"]):
        frames = result.pop("frames")
        if frames:
            tag = "success" if result["success"] else "fail"
            save_video(frames, os.path.join(out_dir, f"episode_{i:03d}_{tag}.mp4"), args.fps)
    with open(os.path.join(out_dir, "eval_results.json"), "w") as f:
        json.dump({"args": vars(args), **metrics}, f, indent=2)
    print(f"Results and videos saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ACT policy")

    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Directory with policy_best.ckpt and dataset_stats.pkl",
    )
    parser.add_argument("--env", choices=["aloha", "franka", "mock"], default="aloha")
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="aloha: AlohaTransferCube-v0 (default) or AlohaInsertion-v0; franka: lift (default) or can",
    )
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Episode i is reset with seed+i (default: 1000 aloha, 100000 franka = robot_sim default)",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=84,
        help="franka: camera resolution the policy was trained on",
    )
    parser.add_argument("--num_videos", type=int, default=5, help="Save videos of first N episodes")
    parser.add_argument("--fps", type=int, default=50, help="Video fps (ALOHA sim runs at 50 Hz)")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Default: <ckpt_dir>/eval_* (rollouts) or <ckpt_dir>/open_loop_eval (--dataset)",
    )
    parser.add_argument("--temporal_agg", action="store_true", help="Use temporal ensembling")
    parser.add_argument("--max_timesteps", type=int, default=400, help="Episode length cap")
    parser.add_argument("--k", type=float, default=0.01, help="Temporal ensembling decay constant")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--dataset", type=str, default=None, help="LeRobot repo_id for open-loop evaluation"
    )
    parser.add_argument("--data_root", type=str, default=None, help="Local dataset root")
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=None,
        help="Episodes to evaluate (default: val_episodes from config.json)",
    )

    args = parser.parse_args()
    if args.task is None:
        args.task = {"aloha": "AlohaTransferCube-v0", "franka": "lift"}.get(args.env, "mock")
    if args.seed is None:
        args.seed = 100_000 if args.env == "franka" else 1000
    main(args)
