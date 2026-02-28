"""
Training entry point for ACT.

Usage:
  # Basic (ALOHA sim pick-and-place; ~45k steps)
  python train.py --repo_id lerobot/aloha_sim_transfer_cube_scripted \\
                  --chunk_size 100 --epochs 20 --batch_size 8

  # Fixed step budget instead of epochs (comparable with the DP repo's --max_steps)
  python train.py --repo_id local/my_data --root ./data/my_data --max_steps 30000

  # With wandb
  python train.py --repo_id lerobot/aloha_sim_transfer_cube_human \\
                  --wandb --wandb_project ACT --wandb_run_name aloha_run1

  # Resume from checkpoint
  python train.py --repo_id lerobot/aloha_sim_transfer_cube_human \\
                  --resume outputs/checkpoint_full.pt

Output files (in --output_dir):
  policy_best.pt        best val loss, inference-ready (weights + config + normalizer)
  policy_final.pt       end of training, same format
  checkpoint_full.pt    full state for --resume (policy + optimizer + epoch)
  checkpoint_epochN.pt  periodic full checkpoints (keeps last --max_keep_checkpoints)
  config.json           all hyperparameters (human-readable copy)
  train_val_*.png       loss curves
"""

import argparse

import torch
from torch.utils.data import DataLoader, Subset

from act.dataset.lerobot_wrapper import LeRobotACTDataset
from act.dataset.normalizer import ACTNormalizer
from training.workspace import TrainWorkspace


def split_by_episode(episode_indices, val_ratio, seed):
    """
    Split frame indices into train/val sets by whole episodes.

    Frames within one demonstration are near-duplicates of each other, so a
    frame-level random split puts almost-identical frames on both sides and
    makes the validation loss (and thus policy_best.pt selection) overly
    optimistic. Holding out entire episodes measures generalisation to unseen
    demonstrations instead.

    Returns:
        (train_frame_indices, val_frame_indices, val_episode_ids)
    """
    episodes = sorted(set(episode_indices))
    if len(episodes) < 2:
        raise ValueError(
            f"Need at least 2 episodes for an episode-level train/val split, got {len(episodes)}"
        )
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(episodes), generator=g).tolist()
    n_val = min(len(episodes) - 1, max(1, round(len(episodes) * val_ratio)))
    val_episodes = sorted(episodes[i] for i in perm[:n_val])

    val_set = set(val_episodes)
    train_idx = [i for i, ep in enumerate(episode_indices) if ep not in val_set]
    val_idx = [i for i, ep in enumerate(episode_indices) if ep in val_set]
    return train_idx, val_idx, val_episodes


def train(args):
    # ── Dataset ──
    print(f"Loading dataset: {args.repo_id}")
    dataset = LeRobotACTDataset(
        repo_id=args.repo_id,
        root=args.root,
        chunk_size=args.chunk_size,
        episodes=args.episodes,
    )
    print(f"  Samples:    {len(dataset)}")
    print(f"  State dim:  {dataset.state_dim}")
    print(f"  Action dim: {dataset.action_dim}")
    print(f"  Image keys: {dataset.image_keys}")

    # ── Train/Val split (by episode) ──
    # Split whole episodes, not frames: neighbouring frames are nearly identical,
    # so a per-frame split puts near-copies of training samples into val and
    # makes val loss (and the policy_best.pt choice) misleadingly optimistic.
    train_idx, val_idx, val_episodes = split_by_episode(
        dataset.episode_indices, args.val_ratio, args.seed
    )
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    n_eps = len(set(dataset.episode_indices))
    print(
        f"  Episodes: {n_eps - len(val_episodes)} train / {len(val_episodes)} val (val: {val_episodes})"
    )
    print(f"  Samples:  {len(train_ds)} train / {len(val_ds)} val")
    steps_per_epoch = -(-len(train_ds) // args.batch_size)
    total = args.max_steps or steps_per_epoch * args.epochs
    print(f"  Steps:    {steps_per_epoch}/epoch, {total} total")

    # Image resolution and control rate the policy is trained on — saved to
    # config.json so inference.py can reproduce the exact preprocessing.
    _, _, img_h, img_w = dataset[0]["image"].shape

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

    if args.crop_ratio < 1.0:
        ch, cw = round(img_h * args.crop_ratio), round(img_w * args.crop_ratio)
        print(f"  Random crop: {img_h}x{img_w} -> {ch}x{cw} (center crop at eval)")

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
        "use_cvae": not args.no_cvae,
        # Augmentation (random crop + color jitter in train, center crop in eval)
        "crop_ratio": args.crop_ratio,
        "color_jitter": args.color_jitter,
        # Data properties (consumed by inference.py consistency checks)
        "image_hw": [img_h, img_w],
        "fps": dataset.fps,
        "val_episodes": val_episodes,
        "repo_id": args.repo_id,
        "root": args.root,
        # Training
        "lr": args.lr,
        "num_epochs": args.epochs,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "output_dir": args.output_dir,
        "seed": args.seed,
        "save_every_epochs": args.save_every_epochs,
        "save_every_steps": args.save_every_steps,
        "grad_clip": args.grad_clip,
        "max_keep_checkpoints": args.max_keep_checkpoints,
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
    print(f"  Checkpoints: {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train ACT policy")

    # Dataset
    parser.add_argument("--repo_id", type=str, required=True, help="LeRobot dataset repo_id")
    parser.add_argument("--root", type=str, default=None, help="Local root for LeRobot dataset")
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=None,
        help="Only use these episode indices (default: all) — handy for quick debug runs",
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.1, help="Fraction of EPISODES held out for validation"
    )

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
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Train for exactly this many gradient steps instead of --epochs",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--kl_weight", type=float, default=10.0)
    parser.add_argument(
        "--no_cvae",
        action="store_true",
        help="Ablation: drop the CVAE encoder (z = 0 in training too, L1 loss only)",
    )
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
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument(
        "--save_every_epochs", type=int, default=5, help="Save full checkpoint every N epochs"
    )
    parser.add_argument(
        "--save_every_steps",
        type=int,
        default=0,
        help="Save step-level checkpoint every N steps for crash recovery (0 = disabled)",
    )
    parser.add_argument(
        "--max_keep_checkpoints",
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

    train(parser.parse_args())


if __name__ == "__main__":
    main()
