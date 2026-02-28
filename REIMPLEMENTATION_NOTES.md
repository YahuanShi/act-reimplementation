# Reimplementation Notes

Lessons learned from reimplementing ACT (Action Chunking with Transformers) from scratch,
after the Diffusion Policy reimplementation. Companion to the DP repo's
`REIMPLEMENTATION_NOTES.md`; the sections follow the same order.

---

## 1. Approach

**Build bottom-up, validate every block in isolation.** Each stage has a script in
`experiments/` that runs on its own: position encodings → transformer encoder/decoder →
ResNet18 backbone → CVAE encoder → full DETRVAE → policy → dataset → training → inference.
A shape or masking bug found in `03_cvae_test.py` costs minutes; the same bug found after
a 3-hour training run costs the run.

**Read the original code next to the paper.** Design decisions that only appear in the code:
- `FrozenBatchNorm2d` in the backbone (BatchNorm statistics are not updated)
- The CVAE encoder sees `[CLS, qpos, action sequence]`; only the CLS output is used for z
- At inference z = 0 (the prior mean), the encoder is not run at all
- Separate learning rate for the backbone (`lr_backbone`)

**Do not hardcode the robot.** `state_dim`, `action_dim` and camera names come from the
dataset, so the same code trains ALOHA (14-dim, bimanual), the Franka sim (10-dim abs eef)
and a UR arm (7-dim joints + gripper).

---

## 2. Architecture — What Actually Matters

### CVAE
- **KL weight 10** keeps z informative but small; with too little weight the decoder
  ignores z, with too much the posterior collapses to the prior and training L1 stalls.
- **The KL term is near 0 most of training** (≈ 1e-3–1e-4 here). That is expected:
  z carries only the "style" of a demo, and at inference z = 0 anyway.

### Transformer
- DETR-style: image features + 2D sine position encodings for every camera token, plus
  latent and proprioception tokens, go through the encoder; `num_queries` = chunk size
  learned queries go through the decoder.
- Cameras share one backbone; their feature maps are concatenated along the width.

### Normalization (bug found and fixed)
The normalizer was fitted and used at inference, but training fed **raw** qpos/actions
to the model. The loss still went down, so nothing looked wrong — the deployed policy just
saw a different input distribution. Normalize in the training *and* validation loops,
and test that `normalize → denormalize` is the identity.

### EMA
Reused from the DP repo: buffer-aware, power-law warmup. Validate and select
`policy_best` on the **EMA** weights, since those are what gets deployed.

### Data Augmentation
`RandomCenterCrop` + optional `ColorJitter` as `nn.Module`s that switch between random
(train) and center (eval) behaviour via `self.training` — no branching at call sites.
On the Franka sim, `--crop_ratio 0.9` was used for both tasks.

---

## 3. Action Chunking and Temporal Ensembling

- **Chunked execution:** query every `chunk_size` steps, execute the whole chunk open-loop.
- **Temporal ensembling:** query every step; the action for time t averages every
  prediction that covers t with weights `exp(-k·i)`, i = 0 for the *oldest* prediction.
  Older predictions get more weight — the paper's choice, counter-intuitive at first.
- **Bug found and fixed:** the ensemble buffer marked "no prediction yet" with zeros, so a
  legitimately zero action (e.g. a joint at 0 rad, a closed gripper) was dropped from the
  average. Track populated rows with a separate boolean mask.
- On robomimic Can, temporal ensembling improved success from 80% to 88% (chunk 20).
- **Chunk size matters more than the execution mode.** Can, chunk 10 / 20 / 50:
  chunked 74% / 80% / 38%, ensembling 74% / 88% / 64%. A 50-step chunk runs 2.5 s open
  loop and cannot react to a slipping grasp; ensembling re-queries every step and
  recovers most of the loss.

---

## 4. Data Pipeline

### LeRobot Format
Same as the DP repo. `action_is_pad` from LeRobot marks chunk steps beyond the episode end;
padded steps are excluded from the L1 loss (and from open-loop metrics).

### Train/Val Split by Episode (bug found and fixed)
A per-frame random split put neighbouring — nearly identical — frames on both sides, so
the validation loss was optimistic and `policy_best` was chosen on leaked data. Split by
whole episodes (`split_by_episode`, seeded and tested).

### What an "Epoch" Means
One epoch here visits every frame; the original ACT samples one random frame per episode
per epoch. The paper's 2000 epochs are ≈ 400× more gradient steps in this convention.
`--max_steps` gives a step budget comparable with the DP repo.

### Normalizer Fit Speed
Iterating the HF dataset row by row took ~60 s for 23k frames; column access
(`hf[key]`) takes ~5 s.

---

## 5. Inference Engineering

The real-robot loop (`inference.py`) shares its conventions and helpers with the DP repo:

- **Verify conventions against the data, not the code.** The dataset stores joint angles
  in **degrees** and the gripper as **1 = open, 0 = closed**. An earlier version of this
  script assumed radians and 0 = open — both wrong. `check_qpos_in_distribution` now warns
  before the first move when the robot state is far outside the training statistics.
- **Never block the control loop on sensors.** Cameras and the gripper are read by
  background threads; `get_obs()` returns cached values and raises if they are stale.
- **Absolute-time scheduling:** step i runs at `t_start + i·dt` (`precise_wait`), so one
  slow step delays only itself.
- **Gripper hysteresis + dwell time** turns the continuous gripper channel into a stable
  open/close command (`act/control.py`, same as DP).
- **Train/deploy consistency** comes from the checkpoint config: image resolution
  (aspect-aware crop), control rate (refuses a mismatch unless allowed), camera order.
- **Warm up** cameras and CUDA kernels before the timed loop.

---

## 6. Engineering Patterns

- **Self-contained checkpoints:** `policy_best.pt` / `policy_final.pt` hold weights +
  config + normalizer stats (`act/policy/loader.py`); one file is all a deployment needs.
  Old checkpoint directories (`.ckpt` + `config.json` + `dataset_stats.pkl`) still load.
- **Resumable training:** `checkpoint_full.pt` (latest) and step checkpoints store raw
  weights, EMA and optimizer separately; a mid-epoch checkpoint re-runs its epoch. The
  cosine schedule is clamped so resuming cannot push it past its end.
- **Same CLI and layout as the DP repo:** `--repo_id/--root/--output_dir`,
  `eval.py --checkpoint --sim/--mock`, `evaluation/{offline,sim_policy,runner,trial_recorder}.py`.
- **Tests run in CI without hardware or simulators:** simulator-dependent tests skip
  when gym-aloha / robot_sim are not installed.

---

## 7. Simulation Results

Shared Franka simulation (`../robot-sim`: robosuite + robomimic proficient-human demos),
50 episodes per entry, seeds 100000–100049, same seeds as the DP repo:

| Policy | Lift | Can | Square |
|---|---|---|---|
| Diffusion Policy (30k steps) | 100% | 96% | 70% |
| ACT chunked (chunk 20, 60 epochs) | 100% | 80% | 66% |
| ACT temporal ensembling | 100% | 88% | 70% |

Both methods with their papers' default hyperparameters; see the robot-sim README for
intervals and analysis.

ALOHA TransferCube (scripted demos, chunk 100, 20 epochs, seeds 1000–1049): 96% chunked,
94% temporal ensembling — the paper reports 86% (Table I, final transfer stage), so the
reimplementation reproduces the original task. On the human demos: 50% chunked, 74% temporal
ensembling (paper: 50%). Without the CVAE (`--no_cvae`): 56% / 66%, within noise of the CVAE
model — the paper's large drop (35.3% → 2%, averaged over two tasks) does not show up on
TransferCube with one training seed.

---

## 8. ACT vs Diffusion Policy

| Aspect | ACT | Diffusion Policy |
|---|---|---|
| Generation | CVAE, single forward pass | Iterative denoising (DDIM 16 steps) |
| Multimodality | Latent z (set to 0 at inference) | Sampling from noise |
| Inference cost | 1 forward pass | 16 U-Net passes |
| Observation | Current frame only | Last To = 2 frames |
| Action execution | Chunk or temporal ensemble | Receding horizon (execute Ta of H) |
| Backbone | DETR transformer | 1D temporal U-Net, FiLM conditioning |
| Loss | L1 + KL | Noise-prediction MSE |

---

## 9. Common Mistakes Checklist

- [ ] qpos and actions are normalized in training, validation **and** inference
- [ ] Train/val split is by episode, not by frame
- [ ] Padded chunk steps are excluded from the loss (`is_pad`)
- [ ] Temporal ensemble distinguishes "no prediction" from "zero action"
- [ ] `policy.eval()` before inference (center crop, no dropout, z = 0)
- [ ] Validation and `policy_best` use the EMA weights
- [ ] Joint units and gripper polarity match the dataset (check frame 0 of every episode)
- [ ] Camera stacking order at inference = `config["camera_names"]`
- [ ] Control rate = dataset fps
- [ ] `ruff check` passes before every commit
