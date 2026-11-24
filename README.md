# `qqn-llm` — Gemma 4 Full-Weight Fine-Tuning with `qqn-jax`

Experiment harness integrating [`qqn-jax`](./qqn-jax) (Quadratic Quasi-Newton) with full-weight **Gemma 4 E2B** fine-tuning in JAX, supporting fast local smoke testing on an 8 GB GPU and multi-GPU FSDP runs on AWS EC2.

---

## Quick Start (Local Smoke Test)

```bash
# 1. Create venv and install dependencies
python3 -m venv .venv
.venv/bin/pip install -e ./qqn-jax -e ".[cuda,dev]"

# 2. Run local RTX 2080 smoke test (AdamW vs QQN-LBFGS vs QQN-Adam, <60s)
.venv/bin/python -m qqn_llm.train --config configs/smoke_local.yaml

# 3. Generate Markdown & CSV comparison report (Node.js)
npm run report -- runs/smoke_local

# 4. Run full test suite (qqn-jax + PyTree + 4-device CPU FSDP + E2E smoke)
.venv/bin/pytest qqn-jax/tests tests
```

---

## Running Gemma 4 E2B on AWS (Phase 1: Auto-Run & Efficacy Reporting)

All AWS orchestration and reporting uses zero-dependency Node.js (`>=20`) backed by AWS CLI v2.

### 1. Launch (with Multi-Region Capacity Fallback)

By default, `aws:launch`:
- Auto-resolves the latest official AWS Deep Learning GPU AMI (Ubuntu 24.04) in each region via SSM.
- Tries regions in order (`us-east-1 -> us-east-2 -> us-west-2 -> eu-west-1`) and automatically falls back on `InsufficientInstanceCapacity` / `Unsupported` / `SpotMaxPriceTooLow`.
- Syncs logs and results to S3 (`s3://qqn-llm-470240306861-us-east-1/qqn-llm/runs/<run-id>/`) every 30 seconds and on exit.
- Automatically terminates the instance when the run finishes or hits `--max-wall-minutes`.

```bash
# Estimate On-Demand vs Spot cost
npm run aws:cost -- --instance g6e.12xlarge --hours 2

# Preview launch command & user-data (--dry-run)
npm run aws:launch -- \
  --instance g6e.12xlarge \
  --config configs/gemma4_e2b_aws_g6e.yaml \
  --ref master \
  --dry-run

# Live launch with custom region fallback order and Spot pricing
npm run aws:launch -- \
  --instance g6e.12xlarge \
  --config configs/gemma4_e2b_aws_g6e.yaml \
  --regions us-east-1,us-east-2,us-west-2,eu-west-1 \
  --ref master \
  --spot
```

> **Note:** Make sure your local commits (including `qqn-jax` changes) are pushed to the remote branch passed via `--ref` (default: `master`). The preconfigured `qqn-llm-ec2-profile` IAM instance profile (least-privilege S3 access to `qqn-llm-470240306861-us-east-1` + SSM) is attached by default.

### 2. Monitoring a Running Job

```bash
# Check EC2 instance state & public IP across all tracked regions
npm run aws:status

# Tail live training stdout (synced to S3 every 30s; defaults to latest run)
npm run aws:tail
npm run aws:tail -- --run-id <run-id> --lines 100

# Tail boot/install user-data log if the instance is still setting up its venv
npm run aws:tail -- --user-data

# List all artifacts uploaded to S3 so far
npm run aws:logs -- --run-id <run-id>

# Manually terminate an instance early if needed
npm run aws:terminate -- --instance-id <id> --region <region>
```

### 3. Finding & Pulling Results

- **Local runs**: written to `runs/<experiment_name>/`
- **AWS runs**: synced to `s3://qqn-llm-470240306861-us-east-1/qqn-llm/runs/<run-id>/`

Download an AWS run locally and render the Markdown + CSV efficacy summary table in one command:

```bash
npm run aws:pull -- --run-id <run-id>
```

Each run directory contains:
- `<optimizer_name>.json` — full step-by-step telemetry (train/val loss, perplexity, cumulative `value_and_grad` evals, tokens seen, wall-clock seconds, step size `t`, line-search acceptance, gradient norm).
- `summary.md` & `summary.csv` — aggregated comparison leaderboard across all optimizers in the sweep.
- `train.log` & `user-data.log` — full training and bootstrap logs.

---

## Configurations

- `configs/smoke_local.yaml` — Miniaturized Gemma-4-structured model (RMSNorm + GQA + GeGLU + PLE) for fast local 8 GB GPU smoke tests.
- `configs/smoke_fsdp_cpu.yaml` — 4-virtual-device CPU FSDP sharding verification.
- `configs/gemma4_e2b_aws_g6e.yaml` — Full-weight Gemma 4 E2B bring-up on `g6e.12xlarge` (4× L40S, 192 GB).
- `configs/gemma4_e2b_aws_sweep.yaml` — Full-weight Gemma 4 E2B optimizer sweep on `p4d.24xlarge` / `p5.48xlarge` (8× A100/H100).

## Phase 2 Roadmap (Weight Persistence & Inference Serving)

Phase 1 AWS configs set `checkpoint_every: 0` to keep runs fast and focused on efficacy telemetry. The Orbax checkpointer (`src/qqn_llm/checkpoint.py`) is already wired into the training loop: setting `checkpoint_every: <N>` saves the full fine-tuned Gemma 4 weight PyTree to `runs/<run-id>/checkpoints/<opt>_step_<N>/` (automatically synced to S3), ready for Phase 2 inference serving via `gm.text.Gemma4Sampler` or safetensors export to vLLM.
