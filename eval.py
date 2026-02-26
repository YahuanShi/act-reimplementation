"""
Evaluation entry point — loads a trained ACT checkpoint and evaluates it.

The checkpoint (policy_best.pt / policy_final.pt) contains the config and normalizer
stats, so the model is fully reconstructed without hardcoding dimensions. A training
output directory can be passed instead of a file (→ its policy_best.pt).

Three modes (same as the Diffusion Policy reimplementation):
  offline (default) — open-loop: predicted vs recorded action chunks on held-out LeRobot
                      episodes. Defaults to the val episodes stored in the checkpoint
                      (train.py --val_ratio). Catches broken checkpoints, wrong
                      normalization and overfitting; says nothing about compounding errors.
  --sim TASK        — closed-loop rollouts with fixed seeds, reports the success rate:
                        lift, can, square                       shared Franka simulation
                                                                (robot-sim repo, robosuite)
                        aloha_transfer_cube, aloha_insertion    gym-aloha (uv sync --extra sim)
  --mock            — rollout loop against a random-observation env (pipeline smoke test only).

Chunked execution is the default; --temporal_agg switches to temporal ensembling.

Usage:
    python eval.py --checkpoint outputs/policy_best.pt
    python eval.py --checkpoint outputs/policy_best.pt --episodes 3 17 42 --plot_dir plots
    python eval.py --checkpoint outputs/lift --sim lift --n_episodes 50 --temporal_agg \\
        --video_dir outputs/lift/videos --output_json outputs/lift/eval_lift_tagg.json
    python eval.py --checkpoint outputs/aloha --sim aloha_transfer_cube
    python eval.py --checkpoint outputs/policy_best.pt --mock --n_episodes 2

For closed-loop real robot evaluation, use inference.py --num_episodes N --record_dir trials/
"""

import argparse
import json

import torch

from act.policy.loader import load_policy
from evaluation.runner import EvalRunner, MockEnv

ROBOT_SIM_TASKS = ("lift", "can", "square")
ALOHA_TASKS = {
    "aloha_transfer_cube": "AlohaTransferCube-v0",
    "aloha_insertion": "AlohaInsertion-v0",
}
# Start seeds: robot_sim's default keeps away from the demos' placements
DEFAULT_START_SEED = {"robot_sim": 100_000, "aloha": 1000, "mock": 0}


def run_offline(pipeline, config, args):
    from act.dataset.lerobot_wrapper import LeRobotACTDataset
    from evaluation.offline import evaluate_offline

    repo_id = args.repo_id or config.get("repo_id")
    root = args.root or config.get("root")
    if repo_id is None:
        raise SystemExit("No repo_id in checkpoint config — pass --repo_id / --root")

    episodes = args.episodes or config.get("val_episodes")
    if not episodes:
        raise SystemExit(
            "Checkpoint has no held-out val episodes (trained with --val_ratio 0). "
            "Pass --episodes explicitly; note that scoring training episodes "
            "only measures fit, not generalisation."
        )
    dataset = LeRobotACTDataset(
        repo_id=repo_id,
        root=root,
        episodes=episodes,
        chunk_size=config["num_queries"],
        image_keys=config["camera_names"],
    )
    print(f"Offline eval on episodes {episodes} ({len(dataset)} frames)")

    m = evaluate_offline(
        pipeline,
        config,
        dataset,
        episodes,
        plot_dir=args.plot_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"L1 first step  : {m['l1_first_step']:.4f}")
    print(f"L1 full chunk  : {m['l1_chunk']:.4f}")
    print("L1 per dim     : " + " ".join(f"{v:.3f}" for v in m["l1_first_step_per_dim"]))
    if args.plot_dir:
        print(f"plots in {args.plot_dir}")
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(m, f, indent=2)
        print(f"saved {args.output_json}")


def run_robot_sim(pipeline, config, args):
    from robot_sim.evaluate import evaluate

    from evaluation.sim_policy import ACTSimPolicy

    action_space = {10: "abs", 7: "delta"}.get(config["action_dim"])
    if action_space is None:
        raise SystemExit(f"action_dim {config['action_dim']} matches no robot_sim action space")
    image_size = args.image_size or config.get("image_hw", [84])[0]
    evaluate(
        ACTSimPolicy(pipeline, config["camera_names"]),
        task=args.sim,
        n_episodes=args.n_episodes,
        start_seed=args.start_seed,
        image_size=image_size,
        action_space=action_space,
        max_steps=args.max_steps,
        video_dir=args.video_dir,
        n_videos=args.n_videos,
        output_json=args.output_json,
    )


def run_rollouts(pipeline, config, args, env_factory, fps):
    """ALOHA sim or mock env through the gym-style EvalRunner."""
    runner = EvalRunner(
        env_factory=env_factory,
        policy=pipeline.policy,
        normalizer=pipeline.normalizer,
        chunk_size=config["num_queries"],
        temporal_agg=args.temporal_agg,
        k=args.k,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        device=pipeline.device,
        start_seed=args.start_seed,
        n_videos=args.n_videos if args.video_dir else 0,
    )
    metrics = runner.run()
    k = sum(r["success"] for r in metrics["results"])
    print(f"success {k}/{metrics['n_episodes']} = {metrics['success_rate']:.0%}")

    if args.video_dir:
        import os

        os.makedirs(args.video_dir, exist_ok=True)
    for i, result in enumerate(metrics["results"]):
        frames = result.pop("frames")
        if frames and args.video_dir:
            tag = "success" if result["success"] else "fail"
            path = f"{args.video_dir}/seed{args.start_seed + i}_{tag}.mp4"
            save_video(frames, path, fps)
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({"args": vars(args), **metrics}, f, indent=2)
        print(f"saved {args.output_json}")


def save_video(frames, path, fps):
    """Write (H, W, 3) uint8 frames to an H.264 mp4."""
    import av

    with av.open(path, mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.height, stream.width = frames[0].shape[:2]
        stream.pix_fmt = "yuv420p"
        for frame in frames:
            container.mux(stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")))
        container.mux(stream.encode())


def main():
    parser = argparse.ArgumentParser(description="Evaluate an ACT policy")
    parser.add_argument(
        "--checkpoint", required=True, help="policy_*.pt, or a training output directory"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--temporal_agg", action="store_true", help="Use temporal ensembling")
    parser.add_argument("--k", type=float, default=0.01, help="Temporal ensembling decay constant")
    parser.add_argument("--mock", action="store_true", help="Random-obs smoke test")

    # Closed-loop modes (--sim / --mock)
    parser.add_argument(
        "--sim",
        default=None,
        metavar="TASK",
        choices=list(ROBOT_SIM_TASKS) + list(ALOHA_TASKS),
        help="lift, can, square (robot-sim) or aloha_transfer_cube, aloha_insertion (gym-aloha)",
    )
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument(
        "--start_seed",
        type=int,
        default=None,
        help="Episode i uses start_seed + i (default: 100000 robot-sim, 1000 aloha, 0 mock)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Episode length cap (default: task default; 400 for aloha / mock)",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,
        help="robot-sim camera resolution (default: training resolution from the checkpoint)",
    )
    parser.add_argument("--video_dir", default=None, help="Save videos of the first episodes")
    parser.add_argument("--n_videos", type=int, default=5)

    # Offline mode
    parser.add_argument("--repo_id", default=None, help="Override checkpoint repo_id")
    parser.add_argument("--root", default=None, help="Override checkpoint dataset root")
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--plot_dir", default=None, help="Per-episode prediction plots")
    parser.add_argument("--output_json", default=None, help="Save metrics to JSON")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    if args.sim in ROBOT_SIM_TASKS:
        family = "robot_sim"
    elif args.sim:
        family = "aloha"
    else:
        family = "mock"
    if args.start_seed is None:
        args.start_seed = DEFAULT_START_SEED[family]
    # robot_sim applies its own per-task limit when max_steps is None
    buffer_steps = args.max_steps or (1000 if family == "robot_sim" else 400)
    if family != "robot_sim" and args.max_steps is None:
        args.max_steps = 400

    pipeline, config = load_policy(
        args.checkpoint,
        device=args.device,
        temporal_agg=args.temporal_agg,
        max_timesteps=buffer_steps,
        k=args.k,
    )
    print(
        f"Loaded {args.checkpoint}: state_dim={config['state_dim']} "
        f"action_dim={config['action_dim']} chunk={config['num_queries']} "
        f"cameras={config['camera_names']} temporal_agg={args.temporal_agg}"
    )

    if family == "robot_sim":
        run_robot_sim(pipeline, config, args)
    elif family == "aloha":
        from evaluation.aloha_env import AlohaSimEnv

        task = ALOHA_TASKS[args.sim]
        run_rollouts(
            pipeline,
            config,
            args,
            lambda: AlohaSimEnv(
                task=task, camera_names=config["camera_names"], max_steps=args.max_steps
            ),
            fps=50,
        )
    elif args.mock:
        run_rollouts(
            pipeline,
            config,
            args,
            lambda: MockEnv(
                state_dim=config["state_dim"],
                action_dim=config["action_dim"],
                num_cam=len(config["camera_names"]),
                episode_len=args.max_steps,
            ),
            fps=config.get("fps") or 50,
        )
    else:
        run_offline(pipeline, config, args)


if __name__ == "__main__":
    main()
