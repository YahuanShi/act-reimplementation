"""
DETRVAE — the core model of ACT (Action Chunking with Transformers).

This module combines three key components into a single model:

1. **CVAE Encoder** (training only):
   Encodes the ground-truth action sequence into a latent variable z.
   Input:  [CLS, qpos_embed, action_embed_1, ..., action_embed_T]
   Output: mu, logvar -> reparameterize -> z (latent_dim-dimensional)

   At inference time, the encoder is not used. Instead, z is set to
   zeros (the mean of the prior distribution N(0, I)).

2. **Visual Backbone** (ResNet18):
   Extracts features from camera images. Multiple cameras are supported
   by concatenating their feature maps along the width dimension.

3. **DETR Transformer** (encoder-decoder):
   The encoder processes [latent_token, proprio_token, image_patches] to
   build a contextual memory. The decoder uses num_queries learnable query
   embeddings (one per future timestep) to cross-attend to this memory
   and predict a chunk of actions.

Data flow diagram:

  Training:
    actions ──> CVAE Encoder ──> mu, logvar ──> reparameterize ──> z
    images  ──> ResNet18 ──> visual features ─┐
    qpos    ──> Linear   ──> proprio          ├──> DETR Transformer ──> a_hat
    z       ──> Linear   ──> latent          ─┘

  Inference:
    z = zeros(latent_dim)                      (sample from prior mean)
    images  ──> ResNet18 ──> visual features ─┐
    qpos    ──> Linear   ──> proprio          ├──> DETR Transformer ──> a_hat
    z       ──> Linear   ──> latent          ─┘

Key dimensions:
  - state_dim:  robot joint position dimension (inferred from dataset)
  - action_dim: robot action dimension (inferred from dataset)
  - hidden_dim: transformer hidden dimension (default 512)
  - latent_dim: CVAE latent space dimension (default 32)
  - num_queries: action chunk size (e.g., 100 timesteps)
"""

import torch
from torch import nn
from torch.autograd import Variable

from .backbone import build_backbone
from .position_encoding import get_sinusoid_encoding_table
from .transformer import Transformer, TransformerEncoder, TransformerEncoderLayer


def reparametrize(mu, logvar):
    """
    Reparameterization trick for the CVAE.

    Instead of sampling z ~ N(mu, sigma^2) directly (which blocks gradient
    flow through the sampling operation), we sample eps ~ N(0, I) and compute:
      z = mu + sigma * eps

    This way gradients can flow through mu and sigma, while the randomness
    comes from eps which doesn't depend on learnable parameters.

    Args:
        mu:     (bs, latent_dim) mean of the posterior distribution
        logvar: (bs, latent_dim) log-variance of the posterior distribution

    Returns:
        z: (bs, latent_dim) sampled latent vector
    """
    # sigma = exp(logvar / 2) — more numerically stable than sqrt(exp(logvar))
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


class DETRVAE(nn.Module):
    """
    The complete ACT model combining CVAE, visual backbone, and DETR Transformer.

    Args:
        backbones: list of Joiner modules (backbone + position encoding).
                   All cameras share backbones[0]. Pass None for state-only mode.
        transformer: DETR Transformer (encoder-decoder) instance
        encoder: TransformerEncoder instance for the CVAE
        state_dim: dimension of the robot state (joint positions)
        action_dim: dimension of the robot action
        num_queries: number of action steps to predict (chunk size)
        camera_names: list of camera name strings
        latent_dim: dimension of the CVAE latent space (default 32)
    """

    def __init__(
        self,
        backbones,
        transformer,
        encoder,
        state_dim,
        action_dim,
        num_queries,
        camera_names,
        latent_dim=32,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        self.action_dim = action_dim

        hidden_dim = transformer.d_model

        # ── Output heads ──
        # Each of the num_queries decoder outputs is projected to an action
        self.action_head = nn.Linear(hidden_dim, action_dim)
        # Padding prediction head (optional, not used in loss but kept for
        # compatibility with the original architecture)
        self.is_pad_head = nn.Linear(hidden_dim, 1)

        # ── Decoder query embeddings ──
        # Each query corresponds to one future timestep in the action chunk.
        # These are learnable embeddings that the decoder uses as positional
        # anchors when cross-attending to the encoder memory.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # ── Visual backbone projection ──
        if backbones is not None:
            # Project backbone features (512 for ResNet18) to hidden_dim
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        else:
            # State-only mode (no images)
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = nn.Embedding(2, hidden_dim)
            self.backbones = None

        # ── CVAE encoder components ──
        self.latent_dim = latent_dim

        # CLS token: a learnable embedding prepended to the encoder input.
        # Its output representation is used to predict mu and logvar.
        self.cls_embed = nn.Embedding(1, hidden_dim)

        # Project actions and joint positions to the transformer hidden dimension
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim)
        self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)

        # Project the CLS output to mu and logvar (concatenated)
        # Output is 2 * latent_dim: first half = mu, second half = logvar
        self.latent_proj = nn.Linear(hidden_dim, latent_dim * 2)

        # Fixed sinusoidal position encoding for the CVAE encoder input
        # Sequence: [CLS(1), qpos(1), action_1, ..., action_T] = 2 + num_queries
        self.register_buffer(
            "pos_table", get_sinusoid_encoding_table(1 + 1 + num_queries, hidden_dim)
        )

        # ── Decoder extra components ──
        # Project the latent sample back to hidden_dim for the decoder
        self.latent_out_proj = nn.Linear(latent_dim, hidden_dim)

        # Learned position embeddings for the two extra tokens prepended to
        # the decoder input: [latent_token, proprio_token]
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)

    def forward(self, qpos, image, env_state, actions=None, is_pad=None):
        """
        Forward pass through the full DETRVAE model.

        Args:
            qpos:      (bs, state_dim) current robot joint positions
            image:     (bs, num_cam, C, H, W) camera images
            env_state: environment state (unused, pass None)
            actions:   (bs, seq, action_dim) ground-truth actions (training only)
            is_pad:    (bs, seq) boolean mask, True for padded positions

        Returns:
            a_hat:     (bs, num_queries, action_dim) predicted action chunk
            is_pad_hat: (bs, num_queries, 1) predicted padding (not used in loss)
            [mu, logvar]: latent distribution parameters (None at inference)
        """
        is_training = actions is not None
        bs, _ = qpos.shape

        # ════════════════════════════════════════════════════════════════
        # Step 1: Obtain latent z from the CVAE encoder (training only)
        # ════════════════════════════════════════════════════════════════
        if is_training:
            # Project each component to hidden_dim
            action_embed = self.encoder_action_proj(actions)  # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)

            # CLS token: (1, hidden_dim) -> (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1)

            # Concatenate: [CLS, qpos, action_1, ..., action_T]
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1)
            # Permute to (seq_len, bs, hidden_dim) for PyTorch Transformer
            encoder_input = encoder_input.permute(1, 0, 2)

            # Build padding mask: CLS and qpos are never padded
            cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device)
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)

            # Fixed sinusoidal position encoding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq_len, 1, hidden_dim)

            # Run the CVAE encoder
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)

            # Extract CLS token output (first position) for latent prediction
            encoder_output = encoder_output[0]  # (bs, hidden_dim)
            latent_info = self.latent_proj(encoder_output)  # (bs, latent_dim*2)
            mu = latent_info[:, : self.latent_dim]
            logvar = latent_info[:, self.latent_dim :]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)  # (bs, hidden_dim)
        else:
            # Inference: use the prior mean z = 0
            # Since KL regularization pushes the posterior toward N(0, I),
            # the prior mean is a reasonable single-sample estimate
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)

        # ════════════════════════════════════════════════════════════════
        # Step 2: Process observations and decode action chunk
        # ════════════════════════════════════════════════════════════════
        if self.backbones is not None:
            # Extract visual features from all cameras
            all_cam_features = []
            all_cam_pos = []
            for cam_id, cam_name in enumerate(self.camera_names):
                # All cameras share the same backbone (backbones[0])
                features, pos = self.backbones[0](image[:, cam_id])
                features = features[0]  # take last layer features
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)

            # Project robot state to hidden_dim
            proprio_input = self.input_proj_robot_state(qpos)

            # Concatenate all camera features along the width dimension
            # This effectively creates a wider "panoramic" feature map
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)

            # Run the DETR Transformer
            # The transformer prepends [latent, proprio] to the flattened
            # image features, then decodes num_queries action predictions
            hs = self.transformer(
                src,
                None,
                self.query_embed.weight,
                pos,
                latent_input,
                proprio_input,
                self.additional_pos_embed.weight,
            )[0]
        else:
            # State-only mode (no visual input)
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1)
            hs = self.transformer(
                transformer_input, None, self.query_embed.weight, self.pos.weight
            )[0]

        # ════════════════════════════════════════════════════════════════
        # Step 3: Project decoder output to actions
        # ════════════════════════════════════════════════════════════════
        a_hat = self.action_head(hs)  # (bs, num_queries, action_dim)
        is_pad_hat = self.is_pad_head(hs)  # (bs, num_queries, 1)

        return a_hat, is_pad_hat, [mu, logvar]


def kl_divergence(mu, logvar):
    """
    Compute KL divergence between the posterior q(z|a,s) = N(mu, sigma^2)
    and the prior p(z) = N(0, I).

    Analytical formula for KL(N(mu, sigma^2) || N(0, I)):
      KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
         = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))

    Returns three versions of the KL divergence:
      - total_kld: sum over latent dims, mean over batch -> scalar
      - dimension_wise_kld: mean over batch -> (latent_dim,)
      - mean_kld: mean over both dims -> scalar

    Args:
        mu:     (bs, latent_dim) posterior mean
        logvar: (bs, latent_dim) posterior log-variance

    Returns:
        total_kld:        (1,) scalar, used in the training loss
        dimension_wise_kld: (latent_dim,) per-dimension KL for monitoring
        mean_kld:         (1,) scalar, alternative aggregation
    """
    batch_size = mu.size(0)
    assert batch_size != 0

    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

    # total_kld: sum over latent_dim, mean over batch -> used in loss
    total_kld = klds.sum(1).mean(0, True)

    # dimension_wise_kld: mean over batch -> useful for monitoring which
    # latent dimensions are most utilized
    dimension_wise_kld = klds.mean(0)

    # mean_kld: mean over both dimensions -> alternative to total_kld
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


def build_encoder(hidden_dim, dropout, nhead, dim_feedforward, enc_layers, normalize_before):
    """
    Build the CVAE encoder (a standalone TransformerEncoder).

    This encoder is separate from the one inside the DETR Transformer.
    It processes [CLS, qpos, action_sequence] and produces the latent
    variable z through the CLS token output.

    Args:
        hidden_dim: transformer hidden dimension
        dropout: dropout rate
        nhead: number of attention heads
        dim_feedforward: FFN intermediate dimension
        enc_layers: number of encoder layers
        normalize_before: if True, use pre-norm

    Returns:
        TransformerEncoder instance
    """
    encoder_layer = TransformerEncoderLayer(
        hidden_dim, nhead, dim_feedforward, dropout, "relu", normalize_before
    )
    encoder_norm = nn.LayerNorm(hidden_dim) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, enc_layers, encoder_norm)
    return encoder


def build_act_model(config):
    """
    Build the complete DETRVAE model from a config dict.

    Expected config keys:
        state_dim, action_dim, hidden_dim, dim_feedforward,
        backbone, lr_backbone, enc_layers, dec_layers, nheads,
        num_queries, camera_names, latent_dim (optional, default 32),
        dropout (optional, default 0.1), pre_norm (optional, default False)

    Args:
        config: dict with model hyperparameters

    Returns:
        DETRVAE model instance
    """
    state_dim = config["state_dim"]
    action_dim = config["action_dim"]
    hidden_dim = config["hidden_dim"]
    dim_feedforward = config["dim_feedforward"]
    nhead = config["nheads"]
    enc_layers = config["enc_layers"]
    dec_layers = config["dec_layers"]
    num_queries = config["num_queries"]
    camera_names = config["camera_names"]
    latent_dim = config.get("latent_dim", 32)
    dropout = config.get("dropout", 0.1)
    pre_norm = config.get("pre_norm", False)

    # Build visual backbone
    backbones = []
    backbone = build_backbone(config["backbone"], hidden_dim, config["lr_backbone"])
    backbones.append(backbone)

    # Build DETR Transformer (encoder-decoder)
    transformer = Transformer(
        d_model=hidden_dim,
        dropout=dropout,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        num_encoder_layers=enc_layers,
        num_decoder_layers=dec_layers,
        normalize_before=pre_norm,
        return_intermediate_dec=True,
    )

    # Build CVAE encoder
    encoder = build_encoder(hidden_dim, dropout, nhead, dim_feedforward, enc_layers, pre_norm)

    model = DETRVAE(
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        action_dim=action_dim,
        num_queries=num_queries,
        camera_names=camera_names,
        latent_dim=latent_dim,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"DETRVAE parameters: {n_parameters / 1e6:.2f}M")

    return model
