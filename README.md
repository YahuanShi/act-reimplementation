# ACT Reimplementation

A from-scratch reimplementation of [Action Chunking with Transformers (ACT)](https://tonyzhaozh.github.io/aloha/) for robot imitation learning, focusing on chunked action prediction, CVAE-based style encoding, and temporal ensembling for smooth real-robot deployment.

## Demo

<!-- TODO: Add demo GIF / video of real robot pick-and-place deployment -->

---

## TODO

- [ ] **Demo** — Add demo GIF / video of real robot pick-and-place deployment
- [ ] **Checkpoints** — Release trained checkpoints with download link
- [ ] **Sim results** — Train on ALOHA TransferCube and report success rate (paper: ~90% scripted, ~50% human data)

---

## 1. Motivation

This is a **from-scratch reimplementation** of ACT, not a fork. Every line of code is written by hand after studying the original paper and codebase. The primary goal is to deeply understand how action chunking and CVAE-based style encoding work in a visuomotor imitation learning setting; the secondary goal is to provide a clean, minimal codebase others can learn from.

### Differences from the Original

| Aspect | [Original (Zhao et al.)](https://github.com/tonyzhaozh/act) | This Project |
|--------|-------------------------------------------------------------|--------------|
| **Code origin** | Complete ALOHA framework (~5k LOC) | From-scratch rewrite (~2.5k LOC) |
| **Config system** | argparse + hardcoded constants | Plain argparse, single `train.py` |
| **Data format** | HDF5 / RoboMimic | LeRobot v3 (modern standard) |
| **EMA** | Not used | Buffer-aware EMA for stable inference |
| **Image augmentation** | Random crop only | `RandomCenterCrop` + `ColorJitter` (opt-in via `--crop_ratio`, `--color_jitter`) |
| **LR schedule** | Constant | Cosine annealing with warmdown |
| **Checkpointing** | Epoch only | Epoch + step-level (crash recovery) |
| **Comments** | Minimal | Learning-note style: explains the *why* behind each design |
| **Environment** | conda + pip | `bash setup.sh` one-click (uv) |
| **Scope** | ALOHA bimanual | Single-arm UR5 pick-and-place |

### ACT vs. Diffusion Policy

Both projects share the same robot hardware (UR5 + Weiss CRG gripper + dual RealSense). The key difference is in how they represent the action distribution:

| | ACT | Diffusion Policy |
|--|-----|-----------------|
| **Action representation** | Deterministic transformer (z=0 at inference) | Iterative denoising (100 DDPM / 16 DDIM steps) |
| **Multimodality** | CVAE latent at training time only | Full denoising trajectory at inference |
| **Inference speed** | ~1 forward pass | ~100 forward passes (DDPM) |
| **Temporal context** | Single frame (no obs history) | Obs history window (To frames) |
| **Smoothing** | Temporal ensembling | Action chunking (execute n steps) |
| **Strength** | Fast, smooth via ensembling | Better multimodal coverage |

---

## 2. ACT

ACT is a robot imitation learning method from [Zhao et al., 2023 (RSS)](https://tonyzhaozh.github.io/aloha/) that predicts a **chunk** of future actions at each timestep rather than a single action. The key insight is that chunked prediction reduces the **effective decision horizon** and suppresses compounding errors — the same mechanism that makes model-predictive control more robust than single-step reactive policies.

---

### 2.1 Core Concepts

#### 2.1.1 Action Chunking

Standard BC maps one observation to one action. Prediction errors accumulate over T steps:

```
Standard BC:
  obs_t → action_t
  obs_t+1 → action_t+1     (policy sees its own error from step t)
  ...
  (error compounds over T steps)
```

ACT predicts an entire chunk of k future actions in one call, then executes all k before re-querying:

```
ACT (chunk_size = k):

  Query at t:
      obs_t ──────────────► [a_t, a_t+1, ..., a_t+k-1]
                              ↑ single forward pass, k=100 by default

  Execute:
      step t:    send a_t
      step t+1:  send a_t+1   (no new inference, no error feedback)
      ...
      step t+k-1: send a_t+k-1

  Query at t+k:
      obs_t+k ────────────► [a_t+k, ..., a_t+2k-1]
```

Re-planning every k steps means the policy is insulated from single-step noise.
Each chunk is a smooth, consistent trajectory segment — the robot is never asked
to react to its own tiny prediction errors mid-chunk.

---

#### 2.1.2 CVAE Style Encoder (Training Only)

Human demonstrations contain **style variation** even for the same task: the same pick can be fast or slow, arcing or direct. A plain transformer conditioned only on images would average over these modes, producing blurry or hesitant predictions.

ACT introduces a CVAE encoder that captures this style variation as a latent variable `z` during training, then discards it at inference (setting `z = 0`):

```
TRAINING                              INFERENCE
─────────────────────────────         ──────────────────────────
action_seq ──┐                        z = 0  (prior mean)
             ├─► Encoder ─► μ, σ         │
qpos ────────┘         │                 │
                    z ~ N(μ, σ)          │
                        │                │
obs_features ───────────┼───────────►  Decoder ─► action_chunk
                        ↓
                    Decoder ─► action_chunk

Loss = L1(pred, target) + β · KL(N(μ,σ) ‖ N(0,I))
       ↑ reconstruction       ↑ force z toward standard normal
```

`β = kl_weight` (default 10.0) controls the regularization strength.
The latent dimension is `z_dim = 32` (set via `--latent_dim`).

The CVAE acts as a **training-time regularizer**: it forces the decoder to model
the full action distribution rather than collapse to a single trajectory.
At inference, `z = 0` gives a single deterministic prediction — the "average style".

---

#### 2.1.3 DETR-Style Transformer

The architecture follows [DETR (Carion et al., 2020)](https://arxiv.org/abs/2005.12872), originally designed for object detection. The key idea: instead of predicting one output at a time, a fixed set of learned **query tokens** (one per predicted timestep) all attend to the image features in parallel.

```
Visual branch (per camera):
  RGB image (H×W×3)
    └─► ResNet18 (FrozenBatchNorm2d)   ← ImageNet pretrained, frozen BN stats
          └─► feature map (C×h×w)
                └─► 2D sine pos encoding
                      └─► flatten → (h*w, C) tokens

All cameras concatenated → image_tokens (num_cam * h * w, C)

Decoder:
  k action query tokens   (learned, k = chunk_size)
  + z token               (latent style, zero at inference)
        │
        ├─► self-attention among queries
        │
        └─► cross-attention over image_tokens
              │
              └─► linear projection → k action predictions  (k, action_dim)
```

**Why FrozenBatchNorm2d?** Standard BatchNorm computes running stats from the
current batch. With small robot batches (8–16 samples), these estimates are noisy
and hurt training. FrozenBatchNorm uses the fixed ImageNet statistics, letting the
pretrained ResNet backbone work reliably even at small batch sizes.

---

#### 2.1.4 Temporal Ensembling

With standard action chunking, the robot executes all k actions before re-querying.
Temporal ensembling instead re-queries at *every step* and blends the overlapping
predictions using exponential weighting:

```
Step t:   query → [â_t,   â_t+1, â_t+2, ..., â_t+k-1]
Step t+1: query → [â_t+1, â_t+2, â_t+3, ..., â_t+k  ]
Step t+2: query → [â_t+2, â_t+3, â_t+4, ..., â_t+k+1]

At step t+2, three predictions cover timestep t+2:
  w_0 * â_t+2 (from step t+2, age=0, weight=1.0)
+ w_1 * â_t+2 (from step t+1, age=1, weight=e^-m)
+ w_2 * â_t+2 (from step t,   age=2, weight=e^-2m)
─────────────────────────────────────────────────
  blended_action_t+2  (normalized by sum of weights)
```

`m` is the ensemble decay parameter — larger `m` makes the policy more reactive
(recent predictions dominate); smaller `m` produces smoother motion.

**Cost:** temporal ensembling requires one policy query per step instead of one
per chunk, making it ~`chunk_size`× more expensive. See `scripts/benchmark_inference.py`
to measure latency on your hardware before enabling it on a real robot.

---

### 2.2 Network at a Glance

```
Input:
  qpos  (state_dim,)         ← current joint positions
  image (num_cam, C, H, W)   ← RGB from each camera

                    ┌─────────────────────────────────────┐
  image ───────────►│  ResNet18 × num_cam  (FrozenBN)     │
                    │  2D pos encoding + flatten           │──► image_tokens
                    └─────────────────────────────────────┘
                                                          │
  z (latent, 0 at inference) ──────────────────────────► │
  k learned action queries ────────────────────────────► │
                                                          │
                    ┌─────────────────────────────────────┤
                    │  DETR Transformer Decoder            │
                    │  self-attn → cross-attn(image_tokens)│
                    └─────────────────────────────────────┘
                                    │
                                    ▼
                    action_chunk  (chunk_size, action_dim)
```

| Module | Class | Parameters (default config) |
|--------|-------|---------------------------|
| Visual backbone | `ACTBackbone` | ResNet18, FrozenBatchNorm2d, ImageNet init |
| CVAE encoder | in `DETRVAE` | TransformerEncoder, z_dim=32 (training only) |
| Action decoder | `DETRTransformer` | enc_layers=4, dec_layers=7, nheads=8, hidden=512 |
| Policy wrapper | `ACTPolicy` | L1 + β·KL loss, cosine LR, EMA |

---

## 3. Project Architecture

```
.
├── act/
│   ├── model/
│   │   ├── transformer.py          # Transformer encoder, decoder, full enc-dec
│   │   ├── detr_vae.py             # DETRVAE: backbone + CVAE encoder + DETR decoder
│   │   ├── backbone.py             # ResNet18 with FrozenBatchNorm2d
│   │   ├── position_encoding.py    # 1D sinusoidal + 2D sine position encodings
│   │   ├── ema.py                  # Buffer-aware EMA (parameters + buffers)
│   │   └── augmentation.py         # RandomCenterCrop + ColorJitter (train/eval auto-switch)
│   ├── policy/
│   │   └── act_policy.py           # ACTPolicy: compute_loss, predict_action, optimizer
│   └── dataset/
│       ├── lerobot_wrapper.py      # LeRobot v3 dataset adapter (image + state + action)
│       └── normalizer.py           # ACTNormalizer: fit from dataset, normalize qpos/action
├── training/
│   └── workspace.py                # Training loop, EMA, checkpointing, wandb logging
├── inference/
│   └── temporal_ensemble.py        # InferencePipeline with temporal ensembling
├── evaluation/
│   ├── runner.py                   # Rollout runner (success via info['is_success']) + mock env
│   └── aloha_env.py                # gym-aloha (MuJoCo) adapter → ACT observation format
├── tests/
│   ├── test_policy.py              # ACTPolicy loss and inference shape tests
│   ├── test_normalizer.py          # Normalizer fit, normalize, denormalize
│   ├── test_temporal_ensemble.py   # TemporalEnsemble correctness
│   ├── test_ema.py                 # EMA decay schedule, step, copy_to, state_dict
│   ├── test_integration.py         # End-to-end pipeline smoke test
│   └── test_eval.py                # Rollout success logic, augmentation, ALOHA env adapter
├── scripts/
│   ├── hdf5_to_lerobot.py          # HDF5 robot episodes → LeRobot v3 format
│   ├── visualize_dataset.py        # Plot action distributions and trajectories
│   └── benchmark_inference.py      # Measure inference latency (chunk_size, cameras)
├── experiments/                    # Step-by-step validation scripts (run in order)
│   ├── 01_transformer_test.py      # Transformer encoder/decoder shapes and gradients
│   ├── 02_backbone_test.py         # ResNet18 backbone feature extraction
│   ├── 03_cvae_test.py             # CVAE latent space and KL divergence
│   ├── 04_model_test.py            # Full DETRVAE forward pass
│   ├── 05_policy_test.py           # ACTPolicy loss computation
│   ├── 06_dataset_test.py          # LeRobot dataset loading and batching
│   ├── 07_train_smoke.py           # Short training loop with synthetic data
│   ├── 08_inference_test.py        # InferencePipeline and temporal ensembling
│   └── 09_integration_test.py      # Full pipeline: dataset → train → checkpoint → infer
├── train.py                        # Training entry point
├── eval.py                         # Evaluation entry point (ALOHA sim or mock env)
├── inference.py                    # Real robot inference (UR + Weiss gripper + RealSense)
├── setup.sh                        # One-click environment setup (uv)
└── pyproject.toml                  # Dependencies and project metadata
```

---

## 4. Setup

```bash
# One-click install (creates virtual environment via uv)
bash setup.sh

# Or manually
uv venv && uv sync --extra dev

# (Optional) ALOHA MuJoCo simulation for evaluation (gym-aloha)
uv sync --extra dev --extra sim

# (Optional) robot hardware dependencies
uv sync --extra robot

# Install pre-commit hooks
pre-commit install
```

Requires Python 3.10+ and a CUDA-capable GPU for training.

---

## 5. Usage

### Training

> **Epochs here are not the paper's epochs.** One epoch is a pass over *every frame*;
> the original ACT samples one frame per episode per epoch. For ALOHA sim
> (50 episodes × 400 frames, batch 8) one epoch ≈ 2250 steps, so `--epochs 20`
> (≈ 45k steps) is in the range of the paper's training budget, while the paper's
> `2000` would be ~400× longer.

```bash
# ALOHA sim pick-and-place (scripted demos — start here)
python train.py \
    --dataset lerobot/aloha_sim_transfer_cube_scripted \
    --chunk_size 100 --epochs 20 --batch_size 8 \
    --ckpt_dir checkpoints/transfer_cube_scripted

# Local dataset (--data_root must point to the dataset directory, not its parent)
python train.py \
    --dataset local/ur5_pick_place --data_root ./data/ur5_pick_place \
    --chunk_size 50 --epochs 30 --batch_size 16

# With augmentation (random crop to 95% of H/W + color jitter; center crop at eval)
python train.py \
    --dataset lerobot/aloha_sim_transfer_cube_human \
    --crop_ratio 0.95 --color_jitter

# With wandb logging
python train.py \
    --dataset lerobot/aloha_sim_transfer_cube_human \
    --wandb --wandb_project ACT --wandb_run_name run1

# Step-level checkpointing for crash recovery (saves every 500 steps, keeps last 3)
python train.py \
    --dataset lerobot/aloha_sim_transfer_cube_human \
    --save_every_steps 500 --max_keep_ckpts 3
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | required | LeRobot dataset repo_id |
| `--data_root` | None | Local dataset root directory |
| `--chunk_size` | 100 | Action chunk length (`num_queries`) |
| `--epochs` | 20 | Passes over all frames (see note above) |
| `--val_ratio` | 0.1 | Fraction of **episodes** held out for validation |
| `--batch_size` | 8 | Batch size |
| `--lr` | 1e-5 | Learning rate |
| `--kl_weight` | 10.0 | CVAE KL divergence weight (β) |
| `--latent_dim` | 32 | CVAE latent dimension |
| `--grad_clip` | 1.0 | Gradient clipping max norm |
| `--save_every` | 5 | Save epoch checkpoint every N epochs |
| `--crop_ratio` | 1.0 | Random-crop fraction of H/W during training (1.0 = off) |
| `--color_jitter` | False | Color jitter during training |
| `--save_every_steps` | 0 | Save step checkpoint every N steps (0 = off) |
| `--no_ema` | — | Disable EMA (EMA is on by default) |
| `--wandb` | False | Enable wandb logging |
| `--resume` | None | Path to `checkpoint_full.pt` to resume |

Output files (saved to `--ckpt_dir`, default `checkpoints/`):

- `policy_best.ckpt` — EMA weights at lowest val loss, inference-ready
- `policy_last.ckpt` — EMA weights at final epoch
- `checkpoint_full.pt` — full state for `--resume` (policy + EMA + optimizer + epoch)
- `checkpoint_epochN.pt` — periodic full checkpoints (keeps last `--max_keep_ckpts`)
- `checkpoint_stepN.pt` — step-level checkpoints for crash recovery
- `dataset_stats.pkl` — normalizer statistics
- `config.json` — all hyperparameters
- `train_val_*.png` — loss curves (loss, l1, kl)

### Resume Training

```bash
python train.py \
    --dataset lerobot/aloha_sim_transfer_cube_human \
    --resume checkpoints/checkpoint_full.pt \
    --epochs 30
```

### Real Robot Inference (UR + Weiss CRG + RealSense)

Hardware: UR robot arm, Weiss CRG 30-050 gripper, Intel RealSense cameras.

```bash
# Install robot dependencies
uv sync --extra robot

# Dry run — verify full pipeline without sending robot commands
python inference.py \
    --ckpt_dir checkpoints \
    --robot_ip 192.168.1.10 --dry_run

# Real deployment with temporal ensembling (smoother motion)
python inference.py \
    --ckpt_dir checkpoints \
    --robot_ip 192.168.1.10 \
    --cam_exterior <serial> --cam_wrist <serial> \
    --temporal_agg

# Custom frequency and chunk size
python inference.py \
    --ckpt_dir checkpoints \
    --robot_ip 192.168.1.10 \
    --frequency 10 --chunk_size 50 --max_steps 500
```

### Evaluation (ALOHA Simulation)

Rolls out the policy in gym-aloha (MuJoCo, rendered headless via EGL) and reports
the success rate. Requires `uv sync --extra sim`.

```bash
# 50 episodes of TransferCube, chunked execution, 5 videos saved
python eval.py --ckpt_dir checkpoints/transfer_cube_scripted --env aloha

# Same with temporal ensembling
python eval.py --ckpt_dir checkpoints/transfer_cube_scripted --env aloha --temporal_agg

# Insertion task (train on lerobot/aloha_sim_insertion_scripted first)
python eval.py --ckpt_dir checkpoints/insertion --env aloha --task AlohaInsertion-v0

# Pipeline check without a simulator (random observations)
python eval.py --ckpt_dir checkpoints --env mock --n_episodes 2
```

An episode is a success when the env reports `info["is_success"]` (TransferCube:
reward 4 — cube held by the left gripper, off the table). Episode `i` resets with
`--seed + i`, so runs are reproducible. Results go to `<ckpt_dir>/eval_aloha_{chunk,tagg}/`:
`eval_results.json` plus `episode_XXX_{success,fail}.mp4`. Without EGL, set `MUJOCO_GL=osmesa`.

### Scripts

```bash
# Convert HDF5 robot episodes to LeRobot v3 format
python scripts/hdf5_to_lerobot.py \
    --input /path/to/episodes --repo_id local/my_dataset --root ./data/my_dataset

# Visualize action distributions and episode lengths (sanity check before training)
python scripts/visualize_dataset.py --repo_id lerobot/aloha_sim_transfer_cube_human

# Measure inference latency for different chunk sizes and modes
python scripts/benchmark_inference.py --device cuda --num_cam 2
```

### Tests

```bash
pytest tests/ -v
```

40 tests covering normalizer, policy, temporal ensembling, EMA, evaluation rollouts, augmentation, and end-to-end integration.

---

## Acknowledgement

Reference: Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" (ACT, RSS 2023)

Source repo: https://github.com/tonyzhaozh/act

LeRobot: https://github.com/huggingface/lerobot
