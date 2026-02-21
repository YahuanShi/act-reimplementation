"""
Training entry point for ACT.

Usage:
  # Basic (ALOHA sim pick-and-place; ~45k steps)
  python train.py --dataset lerobot/aloha_sim_transfer_cube_scripted \\
                  --chunk_size 100 --epochs 20 --batch_size 8

  # With wandb
  python train.py --dataset lerobot/aloha_sim_transfer_cube_human \\
                  --wandb --wandb_project ACT --wandb_run_name aloha_run1

  # Resume from checkpoint
  python train.py --dataset lerobot/aloha_sim_transfer_cube_human \\
                  --resume checkpoints/checkpoint_full.pt

Output files (in --ckpt_dir):
  policy_best.ckpt      best val loss, inference-ready (state_dict only)
  policy_last.ckpt      final epoch state_dict
  checkpoint_full.pt    full state for --resume (policy + optimizer + epoch)
  checkpoint_epochN.pt  periodic full checkpoints (keeps last --max_keep_ckpts)
  dataset_stats.pkl     normalizer stats
  config.json           all hyperparameters
  train_val_*.png       loss curves
"""

import argparse

import numpy as np
from torch.utils.data import DataLoader, Subset

from act.dataset.lerobot_wrapper import LeRobotACTDataset
from act.dataset.normalizer import ACTNormalizer
from training.workspace import TrainWorkspace


def main(args):
    # ── Dataset ──
    print(f"Loading dataset: {args.dataset}")
    dataset = LeRobotACTDataset(
        repo_id=args.dataset,
        root=args.data_root,
        chunk_size=args.chunk_size,
        episodes=None,
    )
    print(f"  Samples:    {len(dataset)}")
    print(f"  State dim:  {dataset.state_dim}")
    print(f"  Action dim: {dataset.action_dim}")
    print(f"  Image keys: {dataset.image_keys}")

    # ── Train/Val split (by episode) ──
    # Split whole episodes, not frames: neighbouring frames are nearly identical,
    # so a per-frame split puts near-copies of training samples into val and
    # makes val loss (and the policy_best.ckpt choice) misleadingly optimistic.
    ep_of_sample = dataset.episode_indices()
    episodes = np.unique(ep_of_sample)
    n_val_eps = max(1, int(round(len(episodes) * args.val_ratio)))
    assert n_val_eps < len(episodes), "Need at least 2 episodes for a train/val split"
    rng = np.random.default_rng(args.seed)
    val_eps = rng.choice(episodes, size=n_val_eps, replace=False)
    is_val = np.isin(ep_of_sample, val_eps)
    train_ds = Subset(dataset, np.flatnonzero(~is_val).tolist())
    val_ds = Subset(dataset, np.flatnonzero(is_val).tolist())
    print(
        f"  Episodes: {len(episodes) - n_val_eps} train / {n_val_eps} val (val: {sorted(val_eps.tolist())})"
    )
    print(f"  Samples:  {len(train_ds)} train / {len(val_ds)} val")
    steps_per_epoch = -(-len(train_ds) // args.batch_size)
    print(f"  Steps:    {steps_per_epoch}/epoch, {steps_per_epoch * args.epochs} total")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=(args.num_workers > 0),
    )

    # ── Normalizer ──
    print("Computing normalization statistics...")
    normalizer = ACTNormalizer()
    normalizer.fit(dataset.ds)
    print(f"  Action mean: {normalizer.action_mean[:4]}...")
    print(f"  Action std:  {normalizer.action_std[:4]}...")

    # ── Augmentation ──
    crop_size = None
    if args.crop_ratio < 1.0:
        # (num_cam, 3, H, W) -> crop H, W by crop_ratio
        h, w = dataset[0]["image"].shape[-2:]
        crop_size = [int(h * args.crop_ratio), int(w * args.crop_ratio)]
        print(f"  Random crop: {h}x{w} -> {crop_size[0]}x{crop_size[1]} (center crop at eval)")

    # ── Config ──
    config = {
        # Model
        "state_dim": dataset.state_dim,
        "action_dim": dataset.action_dim,
        "hidden_dim": args.hidden_dim,
        "dim_feedforward": args.dim_feedforward,
        "backbone": args.backbone,
        "lr_backbone": args.lr_backbone,
        "enc_layers": args.enc_layers,
        "dec_layers": args.dec_layers,
        "nheads": args.nheads,
        "latent_dim": args.latent_dim,
        "dropout": args.dropout,
        "pre_norm": args.pre_norm,
        "num_queries": args.chunk_size,
        "camera_names": dataset.image_keys,
        "kl_weight": args.kl_weight,
        "crop_size": crop_size,
        "color_jitter": args.color_jitter,
        # Training
        "lr": args.lr,
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "ckpt_dir": args.ckpt_dir,
        "seed": args.seed,
        "save_every": args.save_every,
        "save_every_steps": args.save_every_steps,
        "grad_clip": args.grad_clip,
        "max_keep_ckpts": args.max_keep_ckpts,
        "use_ema": args.use_ema,
        # Resume
        "resume": args.resume,
        # WandB
        "use_wandb": args.wandb,
        "wandb_project": args.wandb_project,
        "wandb_run_name": args.wandb_run_name,
    }

    # ── Train ──
    workspace = TrainWorkspace(config)
    best_info = workspace.run(train_loader, val_loader, normalizer)

    if best_info is not None:
        best_epoch, best_loss, _ = best_info
        print("\nTraining complete.")
        print(f"  Best val loss: {best_loss:.6f} at epoch {best_epoch}")
    print(f"  Checkpoints: {args.ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ACT policy")

    # Dataset
    parser.add_argument("--dataset", type=str, required=True, help="LeRobot dataset repo_id")
    parser.add_argument("--data_root", type=str, default=None, help="Local dataset root directory")
    parser.add_argument("--val_ratio", type=float, default=0.1)

    # Model architecture
    parser.add_argument(
        "--chunk_size", type=int, default=100, help="Action chunk length (num_queries)"
    )
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dim_feedforward", type=int, default=3200)
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--enc_layers", type=int, default=4)
    parser.add_argument("--dec_layers", type=int, default=7)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--latent_dim", type=int, default=32, help="CVAE latent dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Transformer dropout rate")
    parser.add_argument("--pre_norm", action="store_true", help="Use pre-norm in transformer")

    # Training hyperparameters
    # One epoch here = one pass over EVERY frame (original ACT: one random frame
    # per episode), so 2000 epochs from the paper would be ~400x more steps.
    # ALOHA sim (50 eps x 400 frames, bs=8): 1 epoch ≈ 2250 steps; 20 epochs ≈ 45k.
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--kl_weight", type=float, default=10.0)
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping max norm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--crop_ratio",
        type=float,
        default=1.0,
        help="Random-crop images to this fraction of H and W during training (1.0 = off)",
    )
    parser.add_argument(
        "--color_jitter", action="store_true", help="Color jitter augmentation during training"
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        default=True,
        help="Enable EMA of model weights for inference (default: on)",
    )
    parser.add_argument("--no_ema", dest="use_ema", action="store_false", help="Disable EMA")

    # Checkpointing
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument(
        "--save_every", type=int, default=5, help="Save full checkpoint every N epochs"
    )
    parser.add_argument(
        "--save_every_steps",
        type=int,
        default=0,
        help="Save step-level checkpoint every N steps for crash recovery (0 = disabled)",
    )
    parser.add_argument(
        "--max_keep_ckpts",
        type=int,
        default=5,
        help="Max periodic checkpoints to keep (0 = keep all)",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint_full.pt to resume training"
    )

    # WandB
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="ACT")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    main(parser.parse_args())
