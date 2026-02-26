"""
Training workspace for ACT.

Training loop structure (validate-first, then train each epoch):
  for each epoch:
    1. Validation pass  → log val loss (computed with the EMA weights, i.e.
                          the same weights that get saved as policy_best)
    2. Training pass    → gradient update per batch + EMA step
    3. Log metrics, save checkpoints

Checkpointing strategy (in output_dir):
  - policy_best.pt      : EMA weights at lowest val loss — use for inference
  - policy_final.pt     : EMA weights at the end of training
                          (both self-contained: weights + config + normalizer stats,
                           see act/policy/loader.py)
  - checkpoint_full.pt  : latest full state (policy + EMA + optimizer + epoch/step),
                           overwritten every save_every_epochs and at the end — use for --resume
  - checkpoint_epochN.pt: periodic copies of the full state (keeps last max_keep_checkpoints)
  - checkpoint_stepN.pt : step-level full state every save_every_steps steps
                           (keeps last max_keep_checkpoints), also valid for --resume:
                           the interrupted epoch is re-run from its start
  - config.json         : all hyperparameters, human-readable copy

Training length: num_epochs, or max_steps gradient steps if given (the last epoch is
cut short) — max_steps makes budgets comparable with the step-based DP repo.

Optional wandb logging:
  Set use_wandb=True in config and optionally wandb_project / wandb_run_name.
  Per-step: train/loss, train/l1, train/kl, train/lr, train/grad_norm, train/ema_decay
  Per-epoch: val/loss, val/l1, val/kl, epoch/gpu_mem_gb
"""

import glob
import json
import math
import os
import warnings
from copy import deepcopy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from act.model.ema import EMAModel
from act.policy.act_policy import ACTPolicy


class TrainWorkspace:
    """
    Complete training workspace for ACT.

    Args:
        config: dict with model and training hyperparameters.
            Model keys: state_dim, action_dim, hidden_dim, dim_feedforward,
                backbone, lr_backbone, enc_layers, dec_layers, nheads,
                num_queries, camera_names, lr, kl_weight
            Training keys: num_epochs, max_steps (optional), batch_size, output_dir,
                seed, save_every_epochs (default 5), grad_clip (default 1.0),
                use_wandb (default False), wandb_project (default "ACT"),
                wandb_run_name (default None), resume (default None),
                max_keep_checkpoints (default 5)
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        seed = config.get("seed", 42)
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.output_dir = config["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self, train_dataloader, val_dataloader, normalizer=None):
        """
        Run the full training loop.

        Returns:
            (best_epoch, best_val_loss, best_state_dict)
        """
        config = self.config
        num_epochs = config["num_epochs"]
        max_steps = config.get("max_steps")
        if max_steps:
            num_epochs = math.ceil(max_steps / len(train_dataloader))
        save_every = config.get("save_every_epochs", 5)
        save_every_steps = config.get("save_every_steps", 0)
        grad_clip = config.get("grad_clip", 1.0)
        use_wandb = config.get("use_wandb", False)
        max_keep_ckpts = config.get("max_keep_checkpoints", 5)
        use_ema = config.get("use_ema", True)

        # ── Save config ──
        cfg_path = os.path.join(self.output_dir, "config.json")
        with open(cfg_path, "w") as f:
            json.dump({k: v for k, v in config.items() if _json_serializable(v)}, f, indent=2)

        # ── Build model ──
        policy = ACTPolicy(config)
        policy.to(self.device)
        optimizer = policy.configure_optimizers()

        total_steps = max_steps or len(train_dataloader) * num_epochs
        lr_scheduler = _cosine_schedule(optimizer, total_steps, min_lr_ratio=0.1)

        # ── EMA ──
        ema = EMAModel(policy) if use_ema else None

        # Normalizer: used in _train_epoch / _validate, stored in every inference checkpoint
        self.normalizer = normalizer

        # ── Resume ──
        start_epoch = 0
        global_step = 0
        min_val_loss = float("inf")
        best_ckpt_info = None

        resume_path = config.get("resume", None)
        if resume_path and os.path.exists(resume_path):
            ckpt = torch.load(resume_path, map_location=self.device, weights_only=False)
            policy.load_state_dict(ckpt["policy_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if ema is not None and "ema_state_dict" in ckpt:
                ema.load_state_dict(ckpt["ema_state_dict"])
            # Epoch-level checkpoints are written after an epoch finished;
            # step-level ones are written mid-epoch, so that epoch is re-run.
            epoch_done = ckpt.get("epoch_complete", True)
            start_epoch = ckpt["epoch"] + 1 if epoch_done else ckpt["epoch"]
            global_step = ckpt["global_step"]
            min_val_loss = ckpt.get("best_val_loss", float("inf"))
            # Fast-forward scheduler to match global_step (no optimizer.step()
            # in between is intended here, so silence PyTorch's order warning)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                for _ in range(global_step):
                    lr_scheduler.step()
            print(f"Resumed from {resume_path} (epoch {start_epoch}, step {global_step})")

        # ── WandB ──
        if use_wandb:
            import wandb

            n_params = sum(p.numel() for p in policy.parameters())
            wandb.init(
                project=config.get("wandb_project", "ACT"),
                name=config.get("wandb_run_name", None),
                config={
                    **{k: v for k, v in config.items() if _json_serializable(v)},
                    "n_params": n_params,
                    "device": str(self.device),
                },
                resume="allow",
            )

        # ── Training history ──
        train_history = []
        validation_history = []

        for epoch in tqdm(range(start_epoch, num_epochs), desc="Training"):
            # ── Validation ──
            # Evaluate the EMA weights: they are what policy_best.pt stores
            # and what gets deployed, so model selection must be based on them.
            eval_policy = ema.averaged_model if ema is not None else policy
            val_summary = self._validate(eval_policy, val_dataloader)
            validation_history.append(val_summary)
            epoch_val_loss = val_summary["loss"].item()

            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                # Save EMA weights as best checkpoint (used for inference)
                best_weights = (
                    ema.averaged_model.state_dict() if ema is not None else policy.state_dict()
                )
                best_ckpt_info = (epoch, min_val_loss, deepcopy(best_weights))
                self._save_inference(best_ckpt_info[2], "policy_best.pt")

            # ── Training ──
            epoch_train_losses, global_step = self._train_epoch(
                policy,
                train_dataloader,
                optimizer,
                lr_scheduler,
                grad_clip,
                global_step,
                use_wandb,
                ema,
                save_every_steps,
                max_keep_ckpts,
                epoch,
                min_val_loss,
                max_steps,
            )
            train_history.extend(epoch_train_losses)
            epoch_train_summary = _compute_dict_mean(epoch_train_losses)

            # ── Console log (every epoch: one epoch covers every frame) ──
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(
                f"  Train loss: {epoch_train_summary['loss'].item():.5f} "
                f"(l1: {epoch_train_summary['l1'].item():.5f}, "
                f"kl: {epoch_train_summary['kl'].item():.5f})"
            )
            print(
                f"  Val loss:   {epoch_val_loss:.5f} "
                f"(l1: {val_summary['l1'].item():.5f}, "
                f"kl: {val_summary['kl'].item():.5f})"
            )

            # ── WandB epoch log ──
            if use_wandb:
                import wandb

                epoch_log = {
                    "val/loss": epoch_val_loss,
                    "val/l1": val_summary["l1"].item(),
                    "val/kl": val_summary["kl"].item(),
                    "epoch/epoch": epoch,
                    "epoch/ema_decay": ema.decay if ema is not None else 0.0,
                }
                if torch.cuda.is_available():
                    epoch_log["epoch/gpu_mem_gb"] = (
                        torch.cuda.max_memory_allocated(self.device) / 1e9
                    )
                    torch.cuda.reset_peak_memory_stats(self.device)
                wandb.log(epoch_log, step=global_step)

            # ── Periodic checkpoint ──
            if save_every > 0 and epoch % save_every == 0 and epoch > 0:
                ckpt_data = _full_state(
                    policy, optimizer, ema, epoch, global_step, min_val_loss, epoch_complete=True
                )
                # checkpoint_full.pt always holds the latest resumable state
                torch.save(ckpt_data, os.path.join(self.output_dir, "checkpoint_full.pt"))
                torch.save(ckpt_data, os.path.join(self.output_dir, f"checkpoint_epoch{epoch}.pt"))
                _prune_ckpts(self.output_dir, "checkpoint_epoch*.pt", max_keep_ckpts)
                self._plot_history(train_history, validation_history, epoch, num_epochs)

        # ── Final saves ──
        # Full checkpoint for potential resume — saved BEFORE copying EMA into
        # the live policy, so that resuming continues from the raw weights.
        last_epoch = max(start_epoch, num_epochs) - 1
        full_ckpt = _full_state(
            policy, optimizer, ema, last_epoch, global_step, min_val_loss, epoch_complete=True
        )
        torch.save(full_ckpt, os.path.join(self.output_dir, "checkpoint_full.pt"))

        # Copy EMA weights into policy before saving the final inference checkpoint
        if ema is not None:
            ema.copy_to(policy)
        self._save_inference(policy.state_dict(), "policy_final.pt")

        if best_ckpt_info is not None:
            best_epoch, best_loss, _ = best_ckpt_info
            print(f"\nBest checkpoint: epoch {best_epoch}, val loss {best_loss:.6f}")

        self._plot_history(train_history, validation_history, num_epochs, num_epochs)

        # ── WandB finish ──
        if use_wandb:
            import wandb

            artifact = wandb.Artifact("act-policy", type="model")
            artifact.add_file(os.path.join(self.output_dir, "policy_best.pt"))
            wandb.log_artifact(artifact)
            wandb.finish()

        return best_ckpt_info

    def _save_inference(self, state_dict, name):
        """Self-contained inference checkpoint: weights + config + normalizer stats."""
        payload = {
            "policy_state_dict": state_dict,
            "config": {k: v for k, v in self.config.items() if _json_serializable(v)},
            "normalizer_state_dict": (
                self.normalizer.state_dict() if self.normalizer is not None else None
            ),
        }
        torch.save(payload, os.path.join(self.output_dir, name))

    def _train_epoch(
        self,
        policy,
        dataloader,
        optimizer,
        lr_scheduler,
        grad_clip,
        global_step,
        use_wandb,
        ema=None,
        save_every_steps=0,
        max_keep_ckpts=5,
        epoch=0,
        best_val_loss=float("inf"),
        max_steps=None,
    ):
        """One training epoch (cut short at max_steps); returns (loss_dicts, global_step)."""
        policy.train()
        epoch_losses = []

        for batch in dataloader:
            if max_steps and global_step >= max_steps:
                break
            qpos = batch["qpos"].to(self.device)
            image = batch["image"].to(self.device)
            actions = batch["action"].to(self.device)
            is_pad = batch["is_pad"].to(self.device)

            if self.normalizer is not None:
                qpos = self.normalizer.normalize_qpos(qpos)
                actions = self.normalizer.normalize_action(actions)

            loss_dict = policy.compute_loss(qpos, image, actions, is_pad)
            loss = loss_dict["loss"]

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            optimizer.step()
            lr_scheduler.step()

            if ema is not None:
                ema.step(policy)

            detached = _detach_dict(loss_dict)
            epoch_losses.append(detached)
            global_step += 1

            if use_wandb:
                import wandb

                wandb.log(
                    {
                        "train/loss": detached["loss"].item(),
                        "train/l1": detached["l1"].item(),
                        "train/kl": detached["kl"].item(),
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/grad_norm": grad_norm.item(),
                        "train/ema_decay": ema.decay if ema is not None else 0.0,
                    },
                    step=global_step,
                )

            # ── Step-level checkpoint (crash recovery) ──
            if save_every_steps > 0 and global_step % save_every_steps == 0:
                step_path = os.path.join(self.output_dir, f"checkpoint_step{global_step}.pt")
                # Written mid-epoch: epoch_complete=False makes --resume re-run this epoch
                step_ckpt = _full_state(
                    policy, optimizer, ema, epoch, global_step, best_val_loss, epoch_complete=False
                )
                torch.save(step_ckpt, step_path)
                _prune_ckpts(self.output_dir, "checkpoint_step*.pt", max_keep_ckpts)

        return epoch_losses, global_step

    def _validate(self, policy, dataloader):
        policy.eval()
        epoch_dicts = []
        with torch.inference_mode():
            for batch in dataloader:
                qpos = batch["qpos"].to(self.device)
                image = batch["image"].to(self.device)
                actions = batch["action"].to(self.device)
                is_pad = batch["is_pad"].to(self.device)

                if self.normalizer is not None:
                    qpos = self.normalizer.normalize_qpos(qpos)
                    actions = self.normalizer.normalize_action(actions)

                loss_dict = policy.compute_loss(qpos, image, actions, is_pad)
                epoch_dicts.append(_detach_dict(loss_dict))
        return _compute_dict_mean(epoch_dicts)

    def _plot_history(self, train_history, validation_history, current_epoch, total_epochs):
        if not train_history:
            return
        for key in train_history[0]:
            plot_path = os.path.join(self.output_dir, f"train_val_{key}.png")
            train_values = [d[key].item() for d in train_history]
            val_values = [d[key].item() for d in validation_history]
            train_x = np.linspace(0, current_epoch, len(train_values))
            val_x = np.linspace(0, current_epoch, len(val_values))
            plt.figure()
            plt.plot(train_x, train_values, label="train", alpha=0.7)
            plt.plot(val_x, val_values, label="validation", alpha=0.7)
            plt.xlabel("Epoch")
            plt.ylabel(key)
            plt.title(key)
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()


def _cosine_schedule(optimizer, total_steps, min_lr_ratio=0.1):
    """
    Cosine decay from each group's base lr to min_lr_ratio * base lr, then flat.

    Unlike CosineAnnealingLR (which is periodic and climbs back up after
    T_max), the factor is clamped at total_steps — resuming from a mid-epoch
    checkpoint re-runs part of an epoch and may overshoot total_steps slightly.
    """

    def factor(step):
        progress = min(step, total_steps) / max(1, total_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _full_state(policy, optimizer, ema, epoch, global_step, best_val_loss, epoch_complete):
    """Everything --resume needs to continue training."""
    state = {
        "epoch": epoch,
        "epoch_complete": epoch_complete,
        "global_step": global_step,
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
    }
    if ema is not None:
        state["ema_state_dict"] = ema.state_dict()
    return state


def _prune_ckpts(ckpt_dir, pattern, max_keep):
    """Keep only the newest max_keep files matching pattern (0 = keep all)."""
    if max_keep <= 0:
        return
    # Sort numerically by the trailing number (checkpoint_step1000 > checkpoint_step500)
    paths = sorted(
        glob.glob(os.path.join(ckpt_dir, pattern)),
        key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit()) or 0),
    )
    for old in paths[:-max_keep]:
        os.remove(old)


def _compute_dict_mean(dicts):
    result = {}
    for k in dicts[0]:
        result[k] = sum(d[k] for d in dicts) / len(dicts)
    return result


def _detach_dict(d):
    return {k: v.detach() for k, v in d.items()}


def _json_serializable(v):
    return isinstance(v, (int, float, str, bool, list, type(None)))
