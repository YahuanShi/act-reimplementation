"""
Training workspace for ACT.

This module implements the complete training pipeline for the ACT policy,
including data loading, optimization, validation, checkpointing, and
training curve visualization.

Training loop structure (following the original ACT implementation):
  for each epoch:
    1. Validation pass (eval mode, no gradients)
    2. Training pass (train mode, gradient updates)
    3. Log metrics and save checkpoints

Why validate first? This gives a baseline validation loss before any training
in that epoch, making it easier to track improvement. It also ensures the
best checkpoint is always evaluated on unseen data before the model has been
updated.

Checkpointing strategy:
  - Best checkpoint: saved whenever validation loss hits a new minimum
  - Periodic checkpoints: saved every N epochs (configurable)
  - Last checkpoint: saved at the end of training
  - Normalizer stats: saved alongside checkpoints for inference

Training curves are saved as PNG plots showing loss, L1, and KL over epochs.
"""

import os
import pickle
from copy import deepcopy

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

from act.policy.act_policy import ACTPolicy
from act.dataset.normalizer import ACTNormalizer


class TrainWorkspace:
    """
    Complete training workspace for ACT.

    Manages the training loop, data loading, optimizer, checkpointing,
    and metric logging.

    Args:
        config: dict with all training and model hyperparameters.
            Model config keys: state_dim, action_dim, hidden_dim, dim_feedforward,
                backbone, lr_backbone, enc_layers, dec_layers, nheads,
                num_queries, camera_names, lr, kl_weight
            Training config keys: num_epochs, batch_size, ckpt_dir, seed,
                val_ratio (default 0.2), save_every (default 100)
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Set random seed for reproducibility
        seed = config.get('seed', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create checkpoint directory
        self.ckpt_dir = config['ckpt_dir']
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def run(self, train_dataloader, val_dataloader, normalizer=None):
        """
        Run the full training loop.

        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            normalizer: ACTNormalizer instance (saved with checkpoints)

        Returns:
            best_ckpt_info: (epoch, val_loss, state_dict) of the best checkpoint
        """
        config = self.config
        num_epochs = config['num_epochs']
        save_every = config.get('save_every', 100)

        # Build policy and optimizer
        policy = ACTPolicy(config)
        policy.to(self.device)
        optimizer = policy.configure_optimizers()

        # Save normalizer stats for inference
        if normalizer is not None:
            stats_path = os.path.join(self.ckpt_dir, 'dataset_stats.pkl')
            with open(stats_path, 'wb') as f:
                pickle.dump(normalizer.state_dict(), f)

        # Training state
        train_history = []
        validation_history = []
        min_val_loss = float('inf')
        best_ckpt_info = None

        for epoch in tqdm(range(num_epochs), desc='Training'):
            # ── Validation ──
            val_summary = self._validate(policy, val_dataloader)
            validation_history.append(val_summary)

            epoch_val_loss = val_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss,
                                  deepcopy(policy.state_dict()))

            # ── Training ──
            epoch_train_losses = self._train_epoch(
                policy, train_dataloader, optimizer)
            train_history.extend(epoch_train_losses)

            epoch_train_summary = _compute_dict_mean(epoch_train_losses)

            # ── Logging ──
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                print(f"\nEpoch {epoch}/{num_epochs}")
                print(f"  Train loss: {epoch_train_summary['loss']:.5f} "
                      f"(l1: {epoch_train_summary['l1']:.5f}, "
                      f"kl: {epoch_train_summary['kl']:.5f})")
                print(f"  Val loss:   {val_summary['loss']:.5f} "
                      f"(l1: {val_summary['l1']:.5f}, "
                      f"kl: {val_summary['kl']:.5f})")

            # ── Periodic checkpoint ──
            if epoch % save_every == 0 and epoch > 0:
                ckpt_path = os.path.join(
                    self.ckpt_dir, f'policy_epoch_{epoch}.ckpt')
                torch.save(policy.state_dict(), ckpt_path)
                self._plot_history(train_history, validation_history,
                                   epoch, num_epochs)

        # ── Final saves ──
        # Save last checkpoint
        last_path = os.path.join(self.ckpt_dir, 'policy_last.ckpt')
        torch.save(policy.state_dict(), last_path)

        # Save best checkpoint
        if best_ckpt_info is not None:
            best_epoch, min_val_loss, best_state = best_ckpt_info
            best_path = os.path.join(self.ckpt_dir, 'policy_best.ckpt')
            torch.save(best_state, best_path)
            print(f"\nBest checkpoint: epoch {best_epoch}, "
                  f"val loss {min_val_loss:.6f}")

        # Save training curves
        self._plot_history(train_history, validation_history,
                           num_epochs, num_epochs)

        return best_ckpt_info

    def _train_epoch(self, policy, dataloader, optimizer):
        """Run one training epoch, return per-batch loss dicts."""
        policy.train()
        epoch_losses = []

        for batch in dataloader:
            qpos = batch['qpos'].to(self.device)
            image = batch['image'].to(self.device)
            actions = batch['action'].to(self.device)
            is_pad = batch['is_pad'].to(self.device)

            # Forward + backward
            loss_dict = policy.compute_loss(qpos, image, actions, is_pad)
            loss = loss_dict['loss']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(_detach_dict(loss_dict))

        return epoch_losses

    def _validate(self, policy, dataloader):
        """Run validation, return averaged loss dict."""
        policy.eval()
        epoch_dicts = []

        with torch.inference_mode():
            for batch in dataloader:
                qpos = batch['qpos'].to(self.device)
                image = batch['image'].to(self.device)
                actions = batch['action'].to(self.device)
                is_pad = batch['is_pad'].to(self.device)

                loss_dict = policy.compute_loss(qpos, image, actions, is_pad)
                epoch_dicts.append(_detach_dict(loss_dict))

        return _compute_dict_mean(epoch_dicts)

    def _plot_history(self, train_history, validation_history,
                      current_epoch, total_epochs):
        """Save training curve plots to the checkpoint directory."""
        if not train_history:
            return

        for key in train_history[0]:
            plot_path = os.path.join(self.ckpt_dir, f'train_val_{key}.png')
            plt.figure()

            train_values = [d[key].item() for d in train_history]
            val_values = [d[key].item() for d in validation_history]

            train_x = np.linspace(0, current_epoch, len(train_values))
            val_x = np.linspace(0, current_epoch, len(val_values))

            plt.plot(train_x, train_values, label='train', alpha=0.7)
            plt.plot(val_x, val_values, label='validation', alpha=0.7)
            plt.xlabel('Epoch')
            plt.ylabel(key)
            plt.title(key)
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()


def _compute_dict_mean(dicts):
    """Average a list of dicts with tensor values."""
    result = {}
    for k in dicts[0]:
        result[k] = sum(d[k] for d in dicts) / len(dicts)
    return result


def _detach_dict(d):
    """Detach all tensor values in a dict."""
    return {k: v.detach() for k, v in d.items()}
