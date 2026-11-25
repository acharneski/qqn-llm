# `qqn-llm` — Gemma 4 Full-Weight Fine-Tuning with `qqn-jax`

Experiment harness integrating [`qqn-jax`](./qqn-jax) (Quadratic Quasi-Newton) with full-weight **Gemma 4 E2B** (`4,651,283,747` trainable parameters) fine-tuning in JAX, supporting fast local smoke testing on an 8 GB GPU and 8-way FSDP runs on AWS EC2 (`8× NVIDIA A100-40GB`).

- **Full Experimental Report:** [`RESULTS.md`](./RESULTS.md)
- **Research Paper Draft:** [`PAPER.md`](./PAPER.md)

---

## Headline Results (Gemma 4 E2B, 4.65B Full-Weight SFT on `8× A100-40GB`)

Across nine AWS `p4d.24xlarge` benchmark sweeps on **GSM8K** (`7,473` train, `seq_len=512`, `float32` master weights + `bfloat16` compute, per-layer `nn.remat`), **Two-Stage Metric-Preconditioned QQN** (`AdamW` or `QQN-Adam` small-batch rough acquisition → warm-started `lbfgs_adam` Strong Wolfe polish along $d(t) = t(1-t)(-\alpha_{\tan} H_0\nabla f) + t^2(-H\nabla f)$ with exact `float32` L2-SP anchoring) outperformed equal-step, objective-matched `AdamW` in every full-dataset sweep and matched compute-equalized `AdamW` at **`1.51` validation perplexity**:

| Benchmark Regime (Run ID) | Optimizer | Steps / Evals | Best Val Loss | Best PPL | Best ΔL1 / ΔL2 | Accept % | Outcome |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Full GSM8K Two-Stage, `256` Val** ([`20-51-24-770Z`](./runs/qqn-2026-11-24T20-51-24-770Z/summary.md)) | **`two_phase_adamw_to_lbfgs_adam`** (`precond`) | `37 / 46` *(at 37/46)* | **`0.4101`** | **`1.51`** | `159.6k / 4.43`\* | **`100%`** | **#1 overall**; **+58% larger Stage-2 val drop** (`-0.0209` vs `-0.0132` nats) than `adamw_two_phase` (`0.4183`) on identical 768-ex batches |
| **Full GSM8K Two-Stage, `256` Val** ([`19-55-33-780Z`](./runs/qqn-2026-11-24T19-55-33-780Z/summary.md)) | **`two_phase_qqn_adam_to_lbfgs_adam`** *(pure QQN)* | `37 / 74` *(at 37/74)* | **`0.4139`** | **`1.51`** | **`153.1k / 4.26`** | **`100%`** | End-to-end QQN beats `adamw_two_phase` (`0.4200`) & `adamw_small_only` (`0.4241`); Stage-1 runs **2.4× faster/step** (`1.22s` vs `2.96s`) |
| **Refined Symmetric-L2-SP, `512` Val — Equal Data** ([`01-52-33-891Z`](./runs/qqn-2026-11-25T01-52-33-891Z/summary.md)) | **`qqn_two_stage_precond`** *(7 polish batches)* | `38 / 54` *(at 36/50)* | **`0.4145`** | **`1.51`** | `147.6k / 4.13` | **`100%`** | Beats objective-matched `adamw_two_stage_l2sp` (`0.4200`), `adamw_small_only` (`0.4177`), and step-38 `adamw_equal_evals` (`0.4161`) |
| **Refined Symmetric-L2-SP, `512` Val — Equal Evals** ([`01-52-33-891Z`](./runs/qqn-2026-11-25T01-52-33-891Z/summary.md)) | **`adamw_two_stage_equal_evals`** *(11 polish batches)* | `46 / 46` *(at 46/46)* | **`0.4129`** | **`1.51`** | **`133.8k / 3.71`** | `100%` | `+4` fresh 768-ex macro-batches (`8,448` vs `5,376` polish ex) edge 7-batch QQN by `-0.0016` nats (virtual tie at **`1.51` PPL**, `455s` vs `591s`) |
| **Refined Symmetric-L2-SP, `512` Val — Tangent Control** ([`01-52-33-891Z`](./runs/qqn-2026-11-25T01-52-33-891Z/summary.md)) | `qqn_two_stage_scaled` ($-\tau\nabla f$) | `28 / 42` *(early stop)* | `0.4360` | `1.55` | `131.8k / 3.58` | `89%` | Stalls `3×` consecutively (`t = 0`) on interior zoom (`t ∈ {0.25, 0.5}`) due to unpreconditioned Euclidean tangent |
| **Deterministic `N=192` Subset, `eval_every=1`** ([`22-25-26-512Z`](./runs/qqn-2026-11-24T22-25-26-512Z/summary.md)) | `adamw_2e-5` / **`lbfgs_adam_sw_precond`** | `7/7` vs `6/7` | **`0.5349` / `0.5392`** | **`1.71`** | `97.3k` / `105.2k` | **`100%`** | Virtual tie in `7` gradient evals before subset memorization; Möbius `reparam` inflates $|\phi'(1)|$ by $\tau^{-2}$ (`0.5691`, `41` evals) |

*\*Precision-matched `float32` anchor ΔL1/ΔL2 from [`19-55-33-780Z`](./runs/qqn-2026-11-24T19-55-33-780Z/summary.md); see [`RESULTS.md` §7.1](./RESULTS.md) for the `+180k` `bfloat16`-anchor vocabulary rounding note.*

### Key Takeaways
1. **Curvature wins per batch; data diversity competes per gradient evaluation:** On the exact same `7` Stage-2 macro-batches (`5,376` examples), `qqn_two_stage_precond` beat `adamw_two_stage_l2sp` (`0.4145` vs `0.4200`) and already passed `AdamW`'s final 38-step loss by **Batch 4 (`0.4179` at step 32)**. Granting `AdamW` `4` additional fresh 768-example macro-batches (`8,448` examples, `46` evals) reached `0.4129` (`1.51` PPL), exposing a single-batch validation sawtooth when QQN extrapolated to `t = 2.0` on inner step 2 (motivating `max_t: 1.0` in Phase 2).
2. **Dual diagonal preconditioning (`lbfgs_adam` + `tangent_mode: preconditioned`) is mandatory at 4.65B scale:** Scalar-initial-Hessian L-BFGS ($H_0 = \gamma I$) stalled at `Mean t ≈ 0.001` (`0.7812` val loss), while norm-matched Euclidean tangents (`scaled`, $-\tau\nabla f$) succeeded in Mode A *only* when $t=1$ annihilated the $t(1-t)$ arch and stalled immediately when Stage-2 polish required interior Strong Wolfe zoom (`t ∈ {0.25, 0.5}`). Only `preconditioned` ($-\alpha_{\tan} H_0\nabla f$) achieved **100% step acceptance** across every run.
3. **Path reparameterization is not Wolfe-invariant:** Möbius interval reparameterization $t(s) = \tau s / ((1-s)+\tau s)$ preserves the geometric curve of canonical QQN while normalizing $dt/ds|_{s=0}=\tau$, but reciprocally inflates $dt/ds|_{s=1}=1/\tau \sim 10^4$, blowing up $|\phi'(1)|/|\phi'(0)|$ by $\Theta(\tau^{-2})$ and biasing Strong Wolfe toward $s=2.0$ oversteps (`192.8k` ΔL1).
4. **Full-weight 4.65B quasi-Newton polish fits in `8× 40 GB` FSDP:** Combining `float32` master weights, `bfloat16` sharded `(s_i, y_i)` curvature deques, per-layer `nn.remat`, `@jax.checkpoint` micro-batch `lax.scan`, `logsumexp` CE on 262k-vocab logits, buffer donation, and host offloading of `best_g` during multi-probe zoom eliminates OOMs on 768-example macro-batches.

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
