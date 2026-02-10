"""
Evaluation entry point for ACT.

Usage:
  python eval.py --ckpt_dir checkpoints --temporal_agg

This script:
1. Loads a trained ACT policy checkpoint and normalizer stats
2. Creates an InferencePipeline (with optional temporal ensembling)
3. Provides a function interface for integration with robot environments

For real robot evaluation:
  - Import load_policy() and create your own evaluation loop
  - The pipeline handles normalization/denormalization internally

For simulation evaluation:
  - Connect to the sim environment and call pipeline.get_action() per step
"""

import argparse
import os
import pickle

import torch

from act.policy.act_policy import ACTPolicy
from act.dataset.normalizer import ACTNormalizer
from inference.temporal_ensemble import InferencePipeline


def load_policy(ckpt_dir, device='cuda', temporal_agg=False,
                max_timesteps=400, k=0.01):
    """
    Load a trained ACT policy and normalizer for inference.

    Args:
        ckpt_dir: directory containing policy_best.ckpt and dataset_stats.pkl
        device: torch device
        temporal_agg: whether to use temporal ensembling
        max_timesteps: maximum episode length
        k: temporal ensembling decay constant

    Returns:
        pipeline: InferencePipeline ready for inference
        config: the model config dict
    """
    # Load config from checkpoint (we need to reconstruct the model)
    ckpt_path = os.path.join(ckpt_dir, 'policy_best.ckpt')
    stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')

    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    assert os.path.exists(stats_path), f"Stats not found: {stats_path}"

    # Load normalizer
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    normalizer = ACTNormalizer()
    normalizer.load_state_dict(stats)

    # Infer config from checkpoint
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)

    # Reconstruct config from state_dict shapes
    config = _infer_config_from_state_dict(state_dict)

    # Build policy and load weights
    policy = ACTPolicy(config)
    policy.load_state_dict(state_dict)
    policy.eval()

    # Build inference pipeline
    pipeline = InferencePipeline(
        policy, normalizer,
        chunk_size=config['num_queries'],
        temporal_agg=temporal_agg,
        max_timesteps=max_timesteps,
        k=k, device=device)

    return pipeline, config


def _infer_config_from_state_dict(state_dict):
    """
    Reconstruct model config by inspecting state_dict tensor shapes.

    This allows loading a checkpoint without needing a separate config file.
    """
    # hidden_dim from query_embed: (num_queries, hidden_dim)
    query_embed = state_dict['model.query_embed.weight']
    num_queries = query_embed.shape[0]
    hidden_dim = query_embed.shape[1]

    # action_dim from action_head: (action_dim, hidden_dim)
    action_dim = state_dict['model.action_head.weight'].shape[0]

    # state_dim from encoder_joint_proj: (hidden_dim, state_dim)
    state_dim = state_dict['model.encoder_joint_proj.weight'].shape[1]

    # Count encoder layers (transformer encoder, not CVAE encoder)
    enc_layers = sum(1 for k in state_dict
                     if 'transformer.encoder.layers' in k
                     and 'self_attn.in_proj_weight' in k)

    # Count decoder layers (in the transformer, not CVAE encoder)
    dec_layers = sum(1 for k in state_dict if 'transformer.decoder.layers' in k
                     and 'self_attn.in_proj_weight' in k)

    # dim_feedforward from linear1: (dim_feedforward, hidden_dim)
    for k, v in state_dict.items():
        if 'transformer.encoder.layers.0.linear1.weight' in k:
            dim_feedforward = v.shape[0]
            break
    else:
        dim_feedforward = hidden_dim * 4

    # nheads: infer from in_proj_weight shape (3*hidden_dim, hidden_dim)
    # Each head has hidden_dim // nheads features
    # We can't directly infer nheads, so use default
    nheads = 8

    # Count cameras from backbone keys
    camera_names = []
    for k in state_dict:
        if k.startswith('model.backbones.'):
            idx = int(k.split('.')[2])
            cam_name = f'cam_{idx}'
            if cam_name not in camera_names:
                camera_names.append(cam_name)
    if not camera_names:
        camera_names = ['cam_0']

    return {
        'state_dim': state_dim,
        'action_dim': action_dim,
        'hidden_dim': hidden_dim,
        'dim_feedforward': dim_feedforward,
        'backbone': 'resnet18',
        'lr_backbone': 0,
        'enc_layers': enc_layers,
        'dec_layers': dec_layers,
        'nheads': nheads,
        'num_queries': num_queries,
        'camera_names': camera_names,
        'lr': 1e-5,
        'kl_weight': 10.0,
    }


def main(args):
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA not available, using CPU")

    pipeline, config = load_policy(
        args.ckpt_dir, device=device,
        temporal_agg=args.temporal_agg,
        max_timesteps=args.max_timesteps,
        k=args.k)

    print(f"Loaded policy from {args.ckpt_dir}")
    print(f"  State dim: {config['state_dim']}")
    print(f"  Action dim: {config['action_dim']}")
    print(f"  Chunk size: {config['num_queries']}")
    print(f"  Temporal ensembling: {args.temporal_agg}")

    # Demonstration with random data
    print("\nRunning demo inference with random data...")
    pipeline.reset()
    for t in range(10):
        qpos = torch.randn(config['state_dim'])
        image = torch.rand(len(config['camera_names']), 3, 240, 320)
        action = pipeline.get_action(t, qpos, image)
        print(f"  t={t}: action shape={action.shape}, "
              f"values=[{action[0]:.3f}, {action[1]:.3f}, ...]")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate ACT policy')

    parser.add_argument('--ckpt_dir', type=str, required=True,
                        help='Directory with policy_best.ckpt and dataset_stats.pkl')
    parser.add_argument('--temporal_agg', action='store_true',
                        help='Use temporal ensembling')
    parser.add_argument('--max_timesteps', type=int, default=400)
    parser.add_argument('--k', type=float, default=0.01,
                        help='Temporal ensembling decay constant')
    parser.add_argument('--device', type=str, default='cuda')

    main(parser.parse_args())
