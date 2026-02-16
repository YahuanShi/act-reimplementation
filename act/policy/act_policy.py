"""
ACTPolicy — the policy wrapper for the ACT model.

This module wraps the DETRVAE model with:
1. Loss computation (L1 reconstruction + KL divergence)
2. Inference interface (predict_action)
3. Optimizer configuration (AdamW with separate backbone learning rate)
4. ImageNet normalization for input images

The loss function is:
  L = L1_loss + kl_weight * KL_divergence

Where:
  - L1_loss = mean(|actions - a_hat| * ~is_pad)
    Uses L1 (MAE) instead of L2 (MSE) for robustness to outliers.
    Masked by is_pad to ignore padding positions in the action chunk.

  - KL_divergence = KL(q(z|a,s) || N(0,I))
    Regularizes the posterior toward the standard normal prior.
    kl_weight controls the tradeoff — higher values make z less informative
    (more reliance on observations), lower values allow z to encode more
    action-specific information.

  - kl_weight=10 (default): tuned to balance action reconstruction quality
    with latent space regularization.

Optimizer notes:
  - AdamW with weight_decay=1e-4
  - Backbone parameters can have a separate (typically lower) learning rate
  - lr_backbone=0 freezes the backbone entirely
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.nn import functional as F

from act.model.detr_vae import build_act_model, kl_divergence


class ACTPolicy(nn.Module):
    """
    Policy wrapper for the ACT (Action Chunking with Transformers) model.

    Provides a clean interface for training (compute_loss) and inference
    (predict_action), handling image normalization and optimizer setup
    internally.

    Args:
        config: dict with model and training hyperparameters.
            Required keys:
                state_dim, action_dim, hidden_dim, dim_feedforward,
                backbone, lr_backbone, enc_layers, dec_layers, nheads,
                num_queries, camera_names, lr, kl_weight
            Optional keys:
                latent_dim (default 32), dropout (default 0.1),
                pre_norm (default False), weight_decay (default 1e-4)
    """

    def __init__(self, config):
        super().__init__()
        self.model = build_act_model(config)
        self.kl_weight = config["kl_weight"]
        self.config = config

        # ImageNet normalization — applied to images before feeding into ResNet.
        # The ResNet backbone was pretrained on ImageNet-normalized images, so
        # we must apply the same normalization at both training and inference.
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def compute_loss(self, qpos, image, actions, is_pad):
        """
        Compute the ACT training loss.

        Args:
            qpos:    (bs, state_dim) current joint positions
            image:   (bs, num_cam, 3, H, W) camera images, values in [0, 1]
            actions: (bs, seq_len, action_dim) ground-truth future actions
            is_pad:  (bs, seq_len) boolean mask, True for padded timesteps

        Returns:
            loss_dict: {'l1': L1 loss, 'kl': KL divergence, 'loss': total loss}
        """
        # Normalize images with ImageNet statistics
        image = self._normalize_images(image)

        # Truncate actions to the model's chunk size
        actions = actions[:, : self.model.num_queries]
        is_pad = is_pad[:, : self.model.num_queries]

        # Forward pass through DETRVAE
        a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, None, actions, is_pad)

        # L1 reconstruction loss, masked to ignore padded positions
        # all_l1 shape: (bs, chunk_size, action_dim)
        all_l1 = F.l1_loss(actions, a_hat, reduction="none")
        # Expand is_pad from (bs, chunk_size) to (bs, chunk_size, 1) for broadcasting
        l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()

        # KL divergence between posterior and prior
        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

        loss_dict = {
            "l1": l1,
            "kl": total_kld[0],
            "loss": l1 + self.kl_weight * total_kld[0],
        }
        return loss_dict

    @torch.inference_mode()
    def predict_action(self, qpos, image):
        """
        Predict an action chunk given current observations.

        At inference time, the CVAE uses the prior z=0 (no encoder needed),
        so the prediction is deterministic given the same observations.

        Args:
            qpos:  (bs, state_dim) current joint positions
            image: (bs, num_cam, 3, H, W) camera images, values in [0, 1]

        Returns:
            a_hat: (bs, num_queries, action_dim) predicted action chunk
        """
        image = self._normalize_images(image)
        a_hat, _, _ = self.model(qpos, image, None)
        return a_hat

    def configure_optimizers(self):
        """
        Build an AdamW optimizer with separate learning rates for backbone
        and non-backbone parameters.

        The backbone (pretrained ResNet) typically uses a lower learning rate
        to preserve pretrained features while allowing fine-tuning. Setting
        lr_backbone=0 freezes the backbone entirely.

        Returns:
            AdamW optimizer
        """
        lr = self.config["lr"]
        lr_backbone = self.config["lr_backbone"]
        weight_decay = self.config.get("weight_decay", 1e-4)

        # Separate backbone parameters from the rest
        backbone_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "backbones" in name:
                backbone_params.append(param)
            else:
                other_params.append(param)

        param_groups = [
            {"params": other_params},
            {"params": backbone_params, "lr": lr_backbone},
        ]

        optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
        return optimizer

    def _normalize_images(self, image):
        """
        Apply ImageNet normalization to a batch of multi-camera images.

        Args:
            image: (bs, num_cam, 3, H, W) with values in [0, 1]

        Returns:
            normalized image tensor with same shape
        """
        bs, num_cam, C, H, W = image.shape
        # Merge batch and camera dims, apply normalize once, restore shape.
        # transforms.Normalize operates channel-wise on (..., C, H, W) tensors.
        image = self.normalize(image.view(bs * num_cam, C, H, W))
        return image.view(bs, num_cam, C, H, W)
