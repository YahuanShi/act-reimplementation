"""
Checkpoint loading — single source of truth for rebuilding a trained ACT policy.

train.py, eval.py and inference.py all need the same ACTPolicy + normalizer. Training
stores the config and normalizer stats inside every inference checkpoint, so loading
never hardcodes dimensions and a single file is all a deployment needs.

Checkpoint formats:
  policy_best.pt / policy_final.pt  {"policy_state_dict", "config", "normalizer_state_dict"}
                                    (EMA weights; best = lowest val loss, final = last epoch)
  checkpoint_*.pt                   full training state for --resume (not loadable here)
  legacy policy_best.ckpt           bare state_dict next to config.json + dataset_stats.pkl
"""

import json
import os
import pickle

import torch

from act.dataset.normalizer import ACTNormalizer
from act.policy.act_policy import ACTPolicy
from act.policy.temporal_ensemble import InferencePipeline

# Looked up in this order when a directory is passed instead of a file
_DIR_CANDIDATES = ("policy_best.pt", "policy_final.pt", "policy_best.ckpt", "policy_last.ckpt")


def resolve_checkpoint(path):
    """A checkpoint file, or a training output directory → its best inference checkpoint."""
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        for name in _DIR_CANDIDATES:
            candidate = os.path.join(path, name)
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(f"No inference checkpoint ({', '.join(_DIR_CANDIDATES)}) in {path}")
    raise FileNotFoundError(f"Checkpoint not found: {path}")


def load_checkpoint(path, device="cpu"):
    """
    Read an inference checkpoint (new or legacy format).

    Returns: (state_dict, config, normalizer)
    """
    path = resolve_checkpoint(path)
    payload = torch.load(path, map_location=device, weights_only=False)

    normalizer = ACTNormalizer()
    if isinstance(payload, dict) and "policy_state_dict" in payload:
        if "config" not in payload:
            raise ValueError(f"{path} is a training checkpoint without config — use policy_*.pt")
        normalizer.load_state_dict(payload["normalizer_state_dict"])
        return payload["policy_state_dict"], payload["config"], normalizer

    # Legacy: bare state_dict, config and stats stored next to it
    ckpt_dir = os.path.dirname(path)
    stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Legacy checkpoint {path} needs {stats_path}")
    with open(stats_path, "rb") as f:
        normalizer.load_state_dict(pickle.load(f))
    config_path = os.path.join(ckpt_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    else:
        print(f"Warning: no config.json next to {path}, inferring from state_dict shapes.")
        config = infer_config_from_state_dict(payload)
    return payload, config, normalizer


def load_policy(checkpoint, device="cuda", temporal_agg=False, max_timesteps=400, k=0.01):
    """
    Load a trained ACT policy wrapped in an InferencePipeline (chunking / ensembling).

    Args:
        checkpoint: policy_*.pt file, or a training output directory (→ policy_best.pt)
        device: torch device
        temporal_agg: whether to use temporal ensembling
        max_timesteps: maximum episode length (size of the ensembling buffer)
        k: temporal ensembling decay constant

    Returns:
        (pipeline, config)
    """
    state_dict, config, normalizer = load_checkpoint(checkpoint, device)
    policy = ACTPolicy(config)
    policy.load_state_dict(state_dict)
    policy.eval()

    pipeline = InferencePipeline(
        policy,
        normalizer,
        chunk_size=config["num_queries"],
        temporal_agg=temporal_agg,
        max_timesteps=max_timesteps,
        k=k,
        device=device,
    )
    return pipeline, config


def infer_config_from_state_dict(state_dict):
    """
    Reconstruct model config by inspecting state_dict tensor shapes.

    Fallback for very old checkpoints without a config. nheads cannot be inferred
    from weights and is hardcoded to 8, latent_dim defaults to 32 — both match the
    standard training defaults.
    """
    # hidden_dim from query_embed: (num_queries, hidden_dim)
    query_embed = state_dict["model.query_embed.weight"]
    num_queries = query_embed.shape[0]
    hidden_dim = query_embed.shape[1]

    # action_dim from action_head: (action_dim, hidden_dim)
    action_dim = state_dict["model.action_head.weight"].shape[0]

    # state_dim from encoder_joint_proj: (hidden_dim, state_dim)
    state_dim = state_dict["model.encoder_joint_proj.weight"].shape[1]

    # latent_dim from latent_proj: (latent_dim*2, hidden_dim)
    latent_dim = state_dict["model.latent_proj.weight"].shape[0] // 2

    enc_layers = sum(
        1
        for k in state_dict
        if "transformer.encoder.layers" in k and "self_attn.in_proj_weight" in k
    )
    dec_layers = sum(
        1
        for k in state_dict
        if "transformer.decoder.layers" in k and "self_attn.in_proj_weight" in k
    )

    dim_feedforward = hidden_dim * 4
    for k, v in state_dict.items():
        if "transformer.encoder.layers.0.linear1.weight" in k:
            dim_feedforward = v.shape[0]
            break

    # Camera count from backbone keys; names are generic since originals aren't stored
    n_cams = len({k.split(".")[2] for k in state_dict if k.startswith("model.backbones.")})
    camera_names = [f"cam_{i}" for i in range(max(n_cams, 1))]

    return {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_dim": hidden_dim,
        "dim_feedforward": dim_feedforward,
        "backbone": "resnet18",
        "lr_backbone": 0,
        "enc_layers": enc_layers,
        "dec_layers": dec_layers,
        "nheads": 8,
        "latent_dim": latent_dim,
        "num_queries": num_queries,
        "camera_names": camera_names,
        "lr": 1e-5,
        "kl_weight": 10.0,
    }
