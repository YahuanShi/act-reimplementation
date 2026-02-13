"""
Inference latency benchmark for ACT.

Measures the time cost of policy.predict_action() under different settings:
  - Chunk size (num_queries): how many steps are predicted per query
  - Inference mode: single-chunk (standard) vs temporal ensemble (query every step)
  - Device: CPU vs CUDA

ACT inference is a single transformer forward pass regardless of chunk size,
so chunk_size mainly affects the output tensor size (not compute time).
Temporal ensemble queries the policy at every step, so it costs ~chunk_size×
more compute per episode than standard chunked inference.

Usage:
    python scripts/benchmark_inference.py
    python scripts/benchmark_inference.py --device cuda
    python scripts/benchmark_inference.py --state_dim 14 --action_dim 14 --num_cam 2
"""

import argparse
import time

import torch

from act.policy.act_policy import ACTPolicy
from act.dataset.normalizer import ACTNormalizer
from inference.temporal_ensemble import InferencePipeline


def build_policy(state_dim, action_dim, chunk_size, num_cam, device):
    config = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_dim": 512,
        "dim_feedforward": 3200,
        "backbone": "resnet18",
        "lr_backbone": 0,
        "enc_layers": 4,
        "dec_layers": 7,
        "nheads": 8,
        "num_queries": chunk_size,
        "camera_names": [f"cam_{i}" for i in range(num_cam)],
        "lr": 1e-5,
        "kl_weight": 10.0,
    }
    policy = ACTPolicy(config)
    policy.to(device)
    policy.eval()
    return policy


@torch.no_grad()
def benchmark_single_step(policy, normalizer, state_dim, action_dim, num_cam,
                           chunk_size, device, n_warmup=10, n_repeat=100):
    """Benchmark a single policy.predict_action() call (one chunk query)."""
    qpos = torch.randn(1, state_dim, device=device)
    image = torch.rand(1, num_cam, 3, 240, 320, device=device)

    # Warmup
    for _ in range(n_warmup):
        _ = policy.predict_action(qpos, image)

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_repeat):
        _ = policy.predict_action(qpos, image)
    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    latency_ms = elapsed / n_repeat * 1000
    return latency_ms


@torch.no_grad()
def benchmark_temporal_ensemble(policy, normalizer, state_dim, action_dim, num_cam,
                                chunk_size, device, episode_len=50, n_repeat=5):
    """Benchmark a full episode rollout with temporal ensembling."""
    pipeline = InferencePipeline(
        policy, normalizer, chunk_size=chunk_size,
        temporal_agg=True, max_timesteps=episode_len,
        device=device,
    )

    # Warmup
    pipeline.reset()
    for t in range(episode_len):
        qpos = torch.randn(state_dim)
        image = torch.rand(num_cam, 3, 240, 320)
        pipeline.get_action(t, qpos, image)

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_repeat):
        pipeline.reset()
        for t in range(episode_len):
            qpos = torch.randn(state_dim)
            image = torch.rand(num_cam, 3, 240, 320)
            pipeline.get_action(t, qpos, image)
    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    ms_per_step = elapsed / n_repeat / episode_len * 1000
    return ms_per_step


def main():
    parser = argparse.ArgumentParser(description="Benchmark ACT inference latency")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--state_dim", type=int, default=7)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--num_cam", type=int, default=1)
    parser.add_argument("--episode_len", type=int, default=50)
    parser.add_argument("--n_repeat", type=int, default=50)
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    normalizer = ACTNormalizer()
    normalizer.fit_from_tensors(
        torch.randn(100, args.state_dim),
        torch.randn(100, args.action_dim),
    )

    chunk_sizes = [20, 50, 100]

    print(f"\ndevice: {device}  |  state_dim={args.state_dim}  "
          f"action_dim={args.action_dim}  num_cam={args.num_cam}")
    print(f"{'mode':<22}  {'chunk_size':>10}  {'latency/step':>14}  {'freq':>8}")
    print("-" * 60)

    for chunk_size in chunk_sizes:
        policy = build_policy(
            args.state_dim, args.action_dim, chunk_size, args.num_cam, device
        )
        n_params = sum(p.numel() for p in policy.parameters())

        # Single-step inference
        lat_ms = benchmark_single_step(
            policy, normalizer, args.state_dim, args.action_dim, args.num_cam,
            chunk_size, device, n_repeat=args.n_repeat,
        )
        # Per-step cost: the chunk is queried once every chunk_size steps
        per_step_ms = lat_ms / chunk_size
        freq = 1000 / per_step_ms
        print(f"{'standard chunk':<22}  {chunk_size:>10}  {per_step_ms:>12.2f} ms  {freq:>6.0f} Hz")

    print()

    # Temporal ensemble (chunk_size=100 as default)
    chunk_size = 100
    policy = build_policy(
        args.state_dim, args.action_dim, chunk_size, args.num_cam, device
    )
    ms_per_step = benchmark_temporal_ensemble(
        policy, normalizer, args.state_dim, args.action_dim, args.num_cam,
        chunk_size, device, episode_len=args.episode_len, n_repeat=5,
    )
    freq = 1000 / ms_per_step
    print(f"{'temporal ensemble':<22}  {chunk_size:>10}  {ms_per_step:>12.2f} ms  {freq:>6.0f} Hz")
    print()
    print(f"(params: {n_params:,}  episode_len={args.episode_len})")
    print()


if __name__ == "__main__":
    main()
