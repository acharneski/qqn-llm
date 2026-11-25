# `qqn-llm` — Gemma 4 Full-Weight Fine-Tuning with `qqn-jax`

Experiment harness integrating [`qqn-jax`](./qqn-jax) (Quadratic Quasi-Newton) with full-weight **Gemma 4 E2B** (`4,651,283,747` trainable parameters) fine-tuning in JAX, supporting fast local smoke testing on an 8 GB GPU and 8-way FSDP runs on AWS EC2 (`8× NVIDIA A100-40GB`).

- **Full Experimental Report:** [`RESULTS.md`](./RESULTS.md)
- **Research Paper Draft:** [`PAPER.md`](./PAPER.md)

---

## Headline Results: QQN vs. AdamW on Gemma 4 E2B (`4.65B` Full-Weight SFT)

Across nine AWS `p4d.24xlarge` sweeps (`8× A100-40GB`, FSDP) on **GSM8K** (`seq_len=512`, `float32` master weights + `bfloat16` compute), we benchmarked **QQN** (quadratic path interpolation between a preconditioned descent tangent and an Adam-preconditioned L-BFGS curvature oracle) as a family against **AdamW**, tracking validation NLL/perplexity, total gradient evaluations (`Evals`), steady-state wall-clock time, and exact `float64` parameter drift from pretrained weights ($\Delta L_1 = \|\theta - \theta_0\|_1$, $\Delta L_2 = \|\theta - \theta_0\|_2$):

| Regime | Method | Steps / Evals | Best Val (PPL) | Best ΔL1 | Best ΔL2 | Steady Time | Verdict vs. AdamW |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Two-Stage Full GSM8K — Equal Data** (`512` val, [`01-52-33-891Z`](./runs/qqn-2026-11-25T01-52-33-891Z/summary.md)) | **QQN** *(7 polish batches)*<br>`AdamW` *(7 polish batches, L2-SP)*<br>`AdamW` *(small-batch only)* | `38 / 54` *(@36/50)*<br>`38 / 38`<br>`38 / 38` | **`0.4145` (`1.51`)**<br>`0.4200` (`1.52`)<br>`0.4177` (`1.52`) | `147.6k`<br>**`130.3k`**<br>`165.0k` | `4.13`<br>**`3.59`**<br>`4.51` | `591s`<br>`312s`<br>**`126s`** | **Proven:** QQN wins on identical data & objective (passes 38-step `AdamW` by Batch 4: `0.4179`) |
| **Two-Stage Full GSM8K — Equal Evals** (`512` val, [`01-52-33-891Z`](./runs/qqn-2026-11-25T01-52-33-891Z/summary.md)) | `AdamW` *(11 polish batches, L2-SP)* | `46 / 46` | **`0.4129` (`1.51`)** | **`133.8k`** | **`3.71`** | **`455s`** | **Tied (`1.51` PPL):** `+4` fresh 768-ex batches (`8,448` vs `5,376` ex) edge 7-batch QQN by `-0.0016` nats |
| **Two-Stage Full GSM8K** (`256` val, [`20-51-24-770Z`](./runs/qqn-2026-11-24T20-51-24-770Z/summary.md) & [`19-55-33-780Z`](./runs/qqn-2026-11-24T19-55-33-780Z/summary.md)) | **QQN** *(Hybrid & Pure QQN)*<br>`AdamW` *(Two-Stage & Small-Only)* | `37 / 46–74`<br>`37 / 37` | **`0.4101–0.4139` (`1.51`)**<br>`0.4171–0.4241` (`1.52–1.53`) | `153.1k–159.6k`\*<br>`137.4k–178.6k` | `4.26–4.43`\*<br>`3.73–4.84` | `342–381s`<br>**`106–264s`** | **Proven:** QQN takes `#1` & `#2` in both sweeps (**+58%** larger Stage-2 val drop on identical batches) |
| **Single-Stage Stochastic GSM8K** (`128` val, [`RESULTS.md` §5](./RESULTS.md)) | **QQN** *(192-ex, `inner=3`)*<br>`AdamW` *(192-ex)* | `45 / 60`<br>`45 / 45` | `0.4396` (`1.55`)<br>**`0.4234` (`1.53`)** | **`202.4k`**<br>`215.5k` | `6.27`<br>**`5.92`** | `231s`<br>**`195s`** | **Not Proven:** QQN leads through step 20 (`-10%` ΔL1) but over-fits small batches late without two-stage polish |
| **Deterministic `N=192` Subset** (`64` val, `eval_every=1`, [`22-25-26-512Z`](./runs/qqn-2026-11-24T22-25-26-512Z/summary.md)) | **QQN** *(best at step 6 / eval 7)*<br>`AdamW` *(best at step 7 / eval 7)* | `16 / 23` *(@6/7)*<br>`17 / 17` *(@7/7)* | `0.5392` (**`1.71`**)<br>**`0.5349` (`1.71`)** | `105.2k`<br>**`97.3k`** | `2.91`<br>**`2.59`** | `162s`<br>**`78s`** | **Virtual Tie:** Both hit `1.71` PPL in `7` gradient evals before memorizing the 192-example subset |

*\*Precision-matched `float32`-anchor ΔL1/ΔL2 from [`19-55-33-780Z`](./runs/qqn-2026-11-24T19-55-33-780Z/summary.md) (see [`RESULTS.md` §7.1](./RESULTS.md) for the `+180k` `bfloat16`-anchor unseen-vocab rounding offset in `20-51-24-770Z`). Internal QQN oracle, line-search, and tangent ablations are detailed in [`RESULTS.md`](./RESULTS.md) and [`PAPER.md`](./PAPER.md).*

### Summary: Where QQN Wins, Weight Drift (`ΔL1` / `ΔL2`), and Tradeoffs
- **Where QQN is proven vs. not proven against `AdamW`:** QQN is **proven superior at matched steps and matched data exposure** in Two-Stage SFT—extracting a **58% larger Stage-2 validation gain** on identical 768-example macro-batches and beating 38-step `AdamW` across every full-dataset sweep. Conversely, QQN is **not proven superior at matched total gradient evaluations or wall-clock time** when fresh data is plentiful (`AdamW` given `+4` fresh 768-example batches ties at `1.51` PPL in 23% less time), **ties `AdamW` in 7 evals** on tiny fixed subsets before memorization, and **underperforms `AdamW` in single-stage noisy mini-batch training** (`0.4396` vs. `0.4234`) or cold-started large-batch L-BFGS (`0.4408`).
- **Pretrained weight preservation (`ΔL1` & `ΔL2`):** Compared to standard single-stage `AdamW` (`165.0k–178.6k` ΔL1, `4.51–4.84` ΔL2), Two-Stage QQN reaches lower validation loss with **11%–14% smaller L1/L2 deviation from $\theta_0$** (`147.6k–153.1k` ΔL1, `4.13–4.26` ΔL2; and **17% less ΔL1** at early Mode A convergence, `99.0k` vs. `119.5k`). When `AdamW` is also given identical Stage-2 L2-SP anchoring (`130.3k–133.8k` ΔL1, `3.59–3.71` ΔL2), QQN's curvature-scaled steps travel **~10%–13% farther in L1/L2** to achieve steeper per-batch loss reductions—and on unanchored memorized subsets (`train_loss < 1e-3`), aggressive quasi-Newton steps nearly double terminal drift (`308k` vs. `158k` ΔL1) unless paired with L2-SP regularization or early stopping.
- **Strengths of QQN:** **(1) Sample efficiency during polish**—extracts substantially more generalization signal per fixed macro-batch, suiting fixed-dataset, distillation, or rollout-limited post-training; **(2) Automatic step-size adaptation**—100% line-search acceptance across `t ∈ [0.25, 2.0]` without per-step schedule tuning; **(3) Compact parameter drift** relative to standard unanchored `AdamW` when paired with `float32` L2-SP anchoring.
- **Weaknesses of QQN:** **(1) Per-step probe & wall-clock overhead**—Strong Wolfe curvature verification costs `~1.4–1.8` gradient evaluations/step (`1.3×–1.9×` slower than `AdamW` in steady state), which `AdamW` can match in validation PPL if allowed to spend extra evaluations on *fresh* examples; **(2) Structural sensitivity**—requires a rough first-order Stage-1 warmup, consistent large Stage-2 macro-batches, and dual diagonal preconditioning (plain scalar L-BFGS stalls at `t ≈ 0.001`); **(3) Higher FSDP memory footprint**—retaining sharded curvature pairs `(s_i, y_i)` and multi-probe buffers adds `~4.6 GB/GPU` at `m = 4` alongside host gradient staging.

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
