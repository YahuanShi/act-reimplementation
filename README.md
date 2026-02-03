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
