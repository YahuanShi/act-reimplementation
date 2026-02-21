"""
Evaluation entry point for ACT.

Usage:
  # ALOHA sim pick-and-place, 50 episodes, save 5 videos (needs: uv sync --extra sim)
  python eval.py --ckpt_dir checkpoints --env aloha --task AlohaTransferCube-v0

  # Same with temporal ensembling
  python eval.py --ckpt_dir checkpoints --env aloha --temporal_agg

  # Pipeline smoke test without a simulator (random observations)
  python eval.py --ckpt_dir checkpoints --env mock --n_episodes 2

This script:
1. Loads a trained ACT policy checkpoint and normalizer stats
2. Rolls out n_episodes in the chosen environment via EvalRunner
3. Reports the success rate, saves rollout videos and eval_results.json

For real robot deployment use inference.py, which reuses load_policy().
"""

import argparse
import json
import os
import pickle

import av
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
    parser.add_argument("--env", choices=["aloha", "mock"], default="aloha")
    parser.add_argument(
        "--task",
        type=str,
        default="AlohaTransferCube-v0",
        help="gym-aloha task: AlohaTransferCube-v0 or AlohaInsertion-v0",
    )
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1000, help="Episode i is reset with seed+i")
    parser.add_argument("--num_videos", type=int, default=5, help="Save videos of first N episodes")
    parser.add_argument("--fps", type=int, default=50, help="Video fps (ALOHA sim runs at 50 Hz)")
    parser.add_argument("--out_dir", type=str, default=None, help="Default: <ckpt_dir>/eval_*")
    parser.add_argument("--temporal_agg", action="store_true", help="Use temporal ensembling")
    parser.add_argument("--max_timesteps", type=int, default=400, help="Episode length cap")
    parser.add_argument("--k", type=float, default=0.01, help="Temporal ensembling decay constant")
    parser.add_argument("--device", type=str, default="cuda")

    main(parser.parse_args())
