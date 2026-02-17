"""
Training workspace for ACT.

Training loop structure (validate-first, then train each epoch):
  for each epoch:
    1. Validation pass  → log val loss
    2. Training pass    → gradient update per batch + EMA step
    3. Log metrics, save checkpoints

Checkpointing strategy:
  - policy_best.ckpt    : EMA weights at lowest val loss — use for inference
  - policy_last.ckpt    : EMA weights at final epoch
  - checkpoint_full.pt  : full state (policy + EMA + optimizer + epoch/step)
                           written every save_every epochs, used for --resume
  - checkpoint_stepN.pt : step-level full checkpoint every save_every_steps steps
                           keeps last max_keep_ckpts, for crash recovery
  - dataset_stats.pkl   : normalizer stats for inference
  - config.json         : all hyperparameters for reproducibility

Optional wandb logging:
  Set use_wandb=True in config and optionally wandb_project / wandb_run_name.
  Per-step: train/loss, train/l1, train/kl, train/lr, train/grad_norm, train/ema_decay
  Per-epoch: val/loss, val/l1, val/kl, epoch/gpu_mem_gb
"""

import glob
import json
import os
import pickle
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
            Training keys: num_epochs, batch_size, ckpt_dir, seed,
                save_every (default 100), grad_clip (default 1.0),
                use_wandb (default False), wandb_project (default "ACT"),
                wandb_run_name (default None), resume (default None),
                max_keep_ckpts (default 5)
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        seed = config.get("seed", 42)
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.ckpt_dir = config["ckpt_dir"]
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def run(self, train_dataloader, val_dataloader, normalizer=None):
        """
        Run the full training loop.

        Returns:
            (best_epoch, best_val_loss, best_state_dict)
        """
        config = self.config
        num_epochs = config["num_epochs"]
        save_every = config.get("save_every", 100)
        save_every_steps = config.get("save_every_steps", 0)
        grad_clip = config.get("grad_clip", 1.0)
        use_wandb = config.get("use_wandb", False)
        max_keep_ckpts = config.get("max_keep_ckpts", 5)
        use_ema = config.get("use_ema", True)

        # ── Save config ──
        cfg_path = os.path.join(self.ckpt_dir, "config.json")
        with open(cfg_path, "w") as f:
            json.dump({k: v for k, v in config.items() if _json_serializable(v)}, f, indent=2)

        # ── Build model ──
        policy = ACTPolicy(config)
        policy.to(self.device)
        optimizer = policy.configure_optimizers()

        total_steps = len(train_dataloader) * num_epochs
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=config.get("lr", 1e-5) / 10
        )

        # ── EMA ──
        ema = EMAModel(policy) if use_ema else None

        # ── Save normalizer ──
        self.normalizer = normalizer  # used in _train_epoch and _validate
        if normalizer is not None:
            stats_path = os.path.join(self.ckpt_dir, "dataset_stats.pkl")
            with open(stats_path, "wb") as f:
                pickle.dump(normalizer.state_dict(), f)

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
            start_epoch = ckpt["epoch"] + 1
            global_step = ckpt["global_step"]
            min_val_loss = ckpt.get("best_val_loss", float("inf"))
            # Fast-forward scheduler to match global_step
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
            val_summary = self._validate(policy, val_dataloader)
            validation_history.append(val_summary)
            epoch_val_loss = val_summary["loss"].item()

            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                # Save EMA weights as best checkpoint (used for inference)
                best_weights = (
                    ema.averaged_model.state_dict() if ema is not None else policy.state_dict()
                )
                best_ckpt_info = (epoch, min_val_loss, deepcopy(best_weights))
                best_path = os.path.join(self.ckpt_dir, "policy_best.ckpt")
                torch.save(best_ckpt_info[2], best_path)

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
            )
            train_history.extend(epoch_train_losses)
            epoch_train_summary = _compute_dict_mean(epoch_train_losses)

            # ── Console log ──
            if epoch % 10 == 0 or epoch == num_epochs - 1:
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
                full_path = os.path.join(self.ckpt_dir, f"checkpoint_epoch{epoch}.pt")
                ckpt_data = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "policy_state_dict": policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": min_val_loss,
                }
                if ema is not None:
                    ckpt_data["ema_state_dict"] = ema.state_dict()
                torch.save(ckpt_data, full_path)
                # Keep only the last max_keep_ckpts periodic checkpoints
                if max_keep_ckpts > 0:
                    all_ckpts = sorted(
                        glob.glob(os.path.join(self.ckpt_dir, "checkpoint_epoch*.pt"))
                    )
                    for old in all_ckpts[:-max_keep_ckpts]:
                        os.remove(old)
                self._plot_history(train_history, validation_history, epoch, num_epochs)

        # ── Final saves ──
        # Copy EMA weights into policy before saving last checkpoint
        if ema is not None:
            ema.copy_to(policy)

        last_path = os.path.join(self.ckpt_dir, "policy_last.ckpt")
        torch.save(policy.state_dict(), last_path)

        # Full checkpoint for potential resume
        full_ckpt = {
            "epoch": num_epochs - 1,
            "global_step": global_step,
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": min_val_loss,
        }
        if ema is not None:
            full_ckpt["ema_state_dict"] = ema.state_dict()
        torch.save(full_ckpt, os.path.join(self.ckpt_dir, "checkpoint_full.pt"))

        if best_ckpt_info is not None:
            best_epoch, best_loss, _ = best_ckpt_info
            print(f"\nBest checkpoint: epoch {best_epoch}, val loss {best_loss:.6f}")

        self._plot_history(train_history, validation_history, num_epochs, num_epochs)

        # ── WandB finish ──
        if use_wandb:
            import wandb

            artifact = wandb.Artifact("act-policy", type="model")
            artifact.add_file(os.path.join(self.ckpt_dir, "policy_best.ckpt"))
            wandb.log_artifact(artifact)
            wandb.finish()

        return best_ckpt_info

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
    ):
        """One training epoch; returns (loss_dicts, updated_global_step)."""
        policy.train()
        epoch_losses = []

        for batch in dataloader:
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
                step_path = os.path.join(self.ckpt_dir, f"checkpoint_step{global_step}.pt")
                step_ckpt = {
                    "global_step": global_step,
                    "policy_state_dict": policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                if ema is not None:
                    step_ckpt["ema_state_dict"] = ema.state_dict()
                torch.save(step_ckpt, step_path)
                if max_keep_ckpts > 0:
                    step_ckpts = sorted(
                        glob.glob(os.path.join(self.ckpt_dir, "checkpoint_step*.pt"))
                    )
                    for old in step_ckpts[:-max_keep_ckpts]:
                        os.remove(old)

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
            plot_path = os.path.join(self.ckpt_dir, f"train_val_{key}.png")
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


def _compute_dict_mean(dicts):
    result = {}
    for k in dicts[0]:
        result[k] = sum(d[k] for d in dicts) / len(dicts)
    return result


def _detach_dict(d):
    return {k: v.detach() for k, v in d.items()}


def _json_serializable(v):
    return isinstance(v, (int, float, str, bool, list, type(None)))
