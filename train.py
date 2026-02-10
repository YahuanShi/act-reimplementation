"""
Training entry point for ACT.

Usage:
  python train.py --dataset lerobot/aloha_sim_transfer_cube_human \
                  --chunk_size 100 --epochs 2000 --batch_size 8

This script:
1. Loads a LeRobot dataset and splits into train/val
2. Computes normalization statistics
3. Builds the ACTPolicy and TrainWorkspace
4. Runs the full training loop with checkpointing
5. Saves the best model, normalizer stats, and training curves

Output files (in --ckpt_dir):
  - policy_best.ckpt:   best model checkpoint (lowest val loss)
  - policy_last.ckpt:   final model checkpoint
  - dataset_stats.pkl:  normalizer statistics for inference
  - train_val_*.png:    training curve plots
"""

import argparse
import os

import torch
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
    print(f"  Samples: {len(dataset)}")
    print(f"  State dim: {dataset.state_dim}")
    print(f"  Action dim: {dataset.action_dim}")
    print(f"  Image keys: {dataset.image_keys}")

    # ── Train/Val split ──
    n = len(dataset)
    n_val = int(n * args.val_ratio)
    n_train = n - n_val
    indices = torch.randperm(n).tolist()
    train_ds = Subset(dataset, indices[:n_train])
    val_ds = Subset(dataset, indices[n_train:])
    print(f"  Train: {n_train}, Val: {n_val}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        pin_memory=True, num_workers=args.num_workers, prefetch_factor=1)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        pin_memory=True, num_workers=args.num_workers, prefetch_factor=1)

    # ── Normalizer ──
    print("Computing normalization statistics...")
    normalizer = ACTNormalizer()
    normalizer.fit(dataset.ds)
    print(f"  Action mean: {normalizer.action_mean[:4]}...")
    print(f"  Action std:  {normalizer.action_std[:4]}...")

    # ── Config ──
    config = {
        'state_dim': dataset.state_dim,
        'action_dim': dataset.action_dim,
        'hidden_dim': args.hidden_dim,
        'dim_feedforward': args.dim_feedforward,
        'backbone': args.backbone,
        'lr_backbone': args.lr_backbone,
        'enc_layers': args.enc_layers,
        'dec_layers': args.dec_layers,
        'nheads': args.nheads,
        'num_queries': args.chunk_size,
        'camera_names': dataset.image_keys,
        'lr': args.lr,
        'kl_weight': args.kl_weight,
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'ckpt_dir': args.ckpt_dir,
        'seed': args.seed,
        'save_every': args.save_every,
    }

    # ── Train ──
    workspace = TrainWorkspace(config)
    best_info = workspace.run(train_loader, val_loader, normalizer)

    best_epoch, best_loss, _ = best_info
    print(f"\nTraining complete. Best val loss: {best_loss:.6f} at epoch {best_epoch}")
    print(f"Checkpoints saved to: {args.ckpt_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ACT policy')

    # Dataset
    parser.add_argument('--dataset', type=str, required=True,
                        help='LeRobot dataset repo_id')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Local dataset root (optional)')
    parser.add_argument('--val_ratio', type=float, default=0.2)

    # Model architecture
    parser.add_argument('--chunk_size', type=int, default=100)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--dim_feedforward', type=int, default=3200)
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--enc_layers', type=int, default=4)
    parser.add_argument('--dec_layers', type=int, default=7)
    parser.add_argument('--nheads', type=int, default=8)

    # Training
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--lr_backbone', type=float, default=1e-5)
    parser.add_argument('--kl_weight', type=float, default=10.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--save_every', type=int, default=100)

    # Output
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints')

    main(parser.parse_args())
