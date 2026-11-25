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
- `configs/gemma4_e2b_aws_refined.yaml` — **Phase 1 Refined Quantitative Benchmark** (`--mode refined`): full GSM8K (`7,473` train / `512` val, `eval_every: 2`), symmetric exact-`float32` L2-SP anchor controls (equal-step `38` & equal-eval `46` AdamW), and best-of-breed QQN finalists (`preconditioned`, `scaled`, `monotone_quadratic`, and pure QQN).
- `configs/gemma4_e2b_pokemon_pirate.yaml` — **Phase 2 Non-Quantitative End-Use** (`--mode pokemon-pirate`): runs only the Phase-1 best-of-breed winner (`qqn_two_stage_precond` with `max_t: 1.0`) on `data/pokemon_pirate.jsonl` (`1,216` unique 1:1 stratified examples: `1,024` train + `192` val) and persists the `bfloat16` Orbax checkpoint to S3.
- `configs/gemma4_e2b_aws_two_phase.yaml` — Earlier two-stage rough → polish GSM8K ablation (`--mode two-phase`).
- `configs/gemma4_e2b_aws_linesearch_suite.yaml` — Mode A (`N=192`, `eval_every: 1`) line-search & `tangent_mode` ablation (`--mode linesearch-suite`).
- `configs/gemma4_e2b_aws_macro_batch.yaml` — Single-stage Mode B macro-batch sweep (`--mode macro-batch`).
- `configs/gemma4_e2b_aws_sweep.yaml` — Initial Mode A oracle & anisotropy sweep (`--mode sweep`).

## Phase 2 End-Use (Best-of-Breed Fine-Tuning & Inference Serving)

Phase 2 applies the single best-of-breed optimizer identified in Phase 1 (`qqn_two_stage_precond` by default; update `configs/gemma4_e2b_pokemon_pirate.yaml` if another finalist wins `--mode refined`) to conditional behavioral fine-tuning and interactive serving:

```bash
# 1. (Optional) Regenerate or customize data/pokemon_pirate.jsonl (608 Pokemon-pirate + 608 normal/near-miss = 1,216)
npm run data:pokemon-pirate

# 2. Train Gemma 4 E2B with the Phase-1 winner and save `qqn_two_stage_precond_best` (~9.3 GB bf16) to S3
npm run aws:launch -- --instance p4d.24xlarge --mode pokemon-pirate --spot

# 3. Serve the fine-tuned checkpoint on a single-GPU g6e.xlarge (~$0.42/hr Spot) + auto-run trigger check
npm run aws:serve -- --run-id <train-run-id> --spot

# 4. Port-forward securely via SSM (printed by aws:serve) and chat interactively
npm run chat -- --url http://127.0.0.1:8000
```
