# ACT Reimplementation

A structured reimplementation of Action Chunking with Transformers (ACT) for robot imitation learning, focusing on chunked action prediction and transformer-based policy modeling.

## Objectives

- **Mechanism Understanding**: Analyze how action chunking reduces effective horizon and mitigates error accumulation, and how the CVAE + Transformer architecture models observation-to-action mappings.
- **Pipeline Reproduction**: Reproduce the end-to-end imitation learning pipeline with LeRobot data format for easy dataset integration.

## Project Structure

```
act/                        # core package
├── model/                  # DETRVAE model (transformer, backbone, CVAE, position encoding)
├── policy/                 # ACTPolicy (training loss + inference interface)
└── dataset/                # LeRobot dataset adapter + normalizer
training/                   # training workspace (loop, optimizer, checkpointing)
inference/                  # temporal ensembling and InferencePipeline
evaluation/                 # rollout runner and mock environment
scripts/                    # data utilities and benchmarking tools
tests/                      # pytest test suite (27 tests)
experiments/                # step-by-step validation scripts
train.py                    # training entry point
eval.py                     # evaluation entry point
inference.py                # real robot inference (UR + Weiss gripper + RealSense)
```

## Setup

```bash
# Install core + dev dependencies
uv venv && uv sync --extra dev

# (Optional) robot hardware dependencies
uv sync --extra robot

# Install pre-commit hooks
pre-commit install
```

Or use the provided script:

```bash
bash setup.sh
```

## Training

```bash
# Basic training
python train.py --dataset lerobot/aloha_sim_transfer_cube_human \
                --chunk_size 100 --epochs 2000 --batch_size 8

# With wandb logging
python train.py --dataset lerobot/aloha_sim_transfer_cube_human \
                --wandb --wandb_project ACT --wandb_run_name run1

# Resume from checkpoint
python train.py --dataset lerobot/aloha_sim_transfer_cube_human \
                --resume checkpoints/checkpoint_full.pt
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--dataset` | (required) | LeRobot dataset repo_id |
| `--chunk_size` | 100 | Action chunk length |
| `--epochs` | 2000 | Number of training epochs |
| `--batch_size` | 8 | Batch size |
| `--lr` | 1e-5 | Learning rate |
| `--kl_weight` | 10.0 | KL divergence weight |
| `--grad_clip` | 1.0 | Gradient clipping max norm |
| `--resume` | None | Path to `checkpoint_full.pt` |
| `--wandb` | False | Enable wandb logging |

Output files (saved to `--ckpt_dir`, default `checkpoints/`):

- `policy_best.ckpt` — best val loss, inference-ready state_dict
- `policy_last.ckpt` — final epoch state_dict
- `checkpoint_full.pt` — full state for `--resume`
- `dataset_stats.pkl` — normalizer statistics
- `config.json` — all hyperparameters
- `train_val_*.png` — loss curves

## Evaluation

```bash
python eval.py --ckpt_dir checkpoints --dataset lerobot/aloha_sim_transfer_cube_human
```

## Real Robot Inference

Requires hardware: UR robot arm, Weiss CRG 30-050 gripper, Intel RealSense cameras.

```bash
python inference.py \
    --ckpt_dir checkpoints \
    --robot_ip 192.168.1.10 \
    --cam_exterior <serial_number> \
    --cam_wrist <serial_number> \
    --temporal_agg          # enable temporal ensembling
```

Use `--dry_run` to test the inference loop without executing robot commands.

## Scripts

```bash
# Visualize dataset action distributions and trajectories
python scripts/visualize_dataset.py --dataset lerobot/aloha_sim_transfer_cube_human

# Benchmark inference latency
python scripts/benchmark_inference.py

# Convert HDF5 robot episodes to LeRobot format
python scripts/hdf5_to_lerobot.py --input episodes/ --output lerobot_dataset/
```

## Tests

```bash
pytest tests/
```

27 tests covering normalizer, policy, temporal ensemble, and end-to-end integration.

## Implementation Progress

- [x] Step 1: Project skeleton and position encodings
- [x] Step 2: Transformer encoder, decoder, and DETR-style Transformer
- [x] Step 3: ResNet18 backbone with FrozenBatchNorm2d
- [x] Step 4–5: CVAE encoder and full DETRVAE model
- [x] Step 6: ACTPolicy (compute_loss + predict_action + optimizer)
- [x] Step 7: LeRobot dataset adapter and ACTNormalizer
- [x] Step 8: Training workspace with checkpointing, wandb, and LR scheduling
- [x] Step 9: Temporal ensembling and InferencePipeline
- [x] Step 10: train.py / eval.py entry points and end-to-end integration
- [x] Step 11: Real robot inference, scripts, evaluation runner, and test suite

## References

- Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" (ACT, RSS 2023)
- LeRobot: [huggingface/lerobot](https://github.com/huggingface/lerobot)
