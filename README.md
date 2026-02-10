# ACT Reimplementation

A structured reimplementation of Action Chunking with Transformers (ACT) for robot imitation learning, focusing on chunked action prediction and transformer-based policy modeling.

## Objectives

- **Mechanism Understanding**: Analyze how action chunking reduces effective horizon and mitigates error accumulation, and how the CVAE + Transformer architecture models observation-to-action mappings.
- **Pipeline Reproduction**: Reproduce the end-to-end imitation learning pipeline with LeRobot data format for easy dataset integration.

## Project Structure

```
act/                    # core package
├── model/              # DETRVAE model (transformer, backbone, CVAE, position encoding)
├── policy/             # ACTPolicy (training loss + inference interface)
└── dataset/            # LeRobot dataset adapter + normalizer
training/               # training workspace (loop, optimizer, checkpointing)
inference/              # temporal ensembling for smooth inference
experiments/            # step-by-step validation scripts
train.py                # training entry point
eval.py                 # evaluation entry point
```

## Setup

```bash
uv venv && uv pip install -e .
```

## Current Progress

- [x] Step 1.1: Project skeleton (pyproject.toml, package structure)
- [x] Step 1.2: Position encodings (1D sinusoidal + 2D sine for images)
- [x] Step 2.1-2.3: Transformer encoder, decoder, and full DETR-style Transformer
- [x] Step 2.4: Transformer validation experiment
- [x] Step 3.1: ResNet18 backbone with FrozenBatchNorm2d
- [x] Step 3.2: Backbone validation experiment
- [x] Step 4.1-5.1: CVAE encoder + DETRVAE full model
- [x] Step 4.2-5.2: CVAE and model validation experiments
- [x] Step 6.1: ACTPolicy (compute_loss + predict_action + optimizer)
- [x] Step 6.2: Policy validation experiment
- [x] Step 7.1: LeRobot dataset adapter
- [x] Step 7.2: Data normalization (ACTNormalizer)
- [x] Step 7.3: Dataset validation experiment
- [x] Step 8.1: Training workspace (training loop + checkpointing)
- [x] Step 8.2: Training smoke test
- [x] Step 9.1: Temporal ensembling implementation
- [x] Step 9.2: Inference pipeline validation
- [x] Step 10.1: train.py + eval.py entry scripts
- [x] Step 10.2: End-to-end integration test
