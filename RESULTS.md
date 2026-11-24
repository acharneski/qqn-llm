# Phase 1 Experiment Results: Full-Weight Gemma 4 E2B Fine-Tuning with `qqn-jax`

## 1. Executive Summary

We integrated our Quadratic Quasi-Newton optimizer ([`qqn-jax`](./qqn-jax)) with full-weight fine-tuning of **Gemma 4 E2B** (`4,651,283,747` trainable parameters, 35 transformer layers + Per-Layer Embeddings) in JAX/Flax across **8× NVIDIA A100-40GB GPUs** (`p4d.24xlarge` Spot, FSDP sharded) on **GSM8K** (`seq_len=512`, `float32` master weights, `bfloat16` compute, per-layer `nn.remat`).

Across five AWS benchmark sweeps comparing **QQN** variants against **AdamW** baselines in **Mode A (deterministic fixed-subset SFT)**, **Mode B (full-dataset 7,473-example stochastic macro-batch SFT)**, and **Two-Phase (small-batch rough → intensive polish) SFT**, we establish six core results:

1. **Diagonal preconditioning is essential for quasi-Newton LLM fine-tuning.** Unpreconditioned `LBFGSOracle` (`H₀ = γI`) and `SecantOracle` stalled at `Mean t ≈ 0.001` (`Best Val = 0.7812`) even with `float32` master weights due to severe cross-layer scale anisotropy across Gemma 4's 262k-vocab embeddings, RMSNorm, and MLP layers. Promoting **`AdamPreconditionedLBFGSOracle` (`lbfgs_adam`)** into `qqn-jax` — replacing `H₀ = γI` with $H_0 = \operatorname{diag}(\eta / (\sqrt{\hat{v}} + \varepsilon))$ — immediately unlocked full convergence (`0.7812 → 0.5558` in Mode A; `0.4236–0.4243` in Two-Phase Mode B).
2. **QQN decisively beats AdamW in Mode A (deterministic fine-tuning), achieving lower validation loss with significantly less weight perturbation.** Every QQN variant beat `adamw_2e-5` (`0.6055` val loss, PPL `1.83`). The champion configuration — **`lbfgs_adam_strong_wolfe`** — reached **`Best Val = 0.5558` (PPL `1.74`)** with **37.7% less L1 weight change (`97.59k` vs `156.74k`)** and **33.2% less L2 weight change (`2.622` vs `3.926`)** than `qqn_adam_armijo_t1`, in only **25 total evaluations** (~1.39 evals/step).
3. **Line-search & path ablations match QQN theory.** Ranking all 6 strategies on `lbfgs_adam` produced the exact theoretical ordering: `Strong Wolfe (0.5558) > Armijo Quadratic (0.5599) > Bounded Spline (0.5613) > Armijo Extrap (0.5617) > Null Search (0.5623) > Linear Chord (0.5626)`.
4. **Bounded cubic Hermite splines are stable but not cost-effective on preconditioned LLMs.** Keeping splines strictly bounded (`rounds=2, max_cp=8`) eliminated instability and OOMs, but because Adam-preconditioned steps are already well-scaled around `t = 1.0`, spline probing cost 2.5–4× more evaluations without beating Strong Wolfe.
5. **In Mode B (stochastic full-GSM8K macro-batches), `qqn_lbfgs_adam` leads early (steps 5–20) while `adamw_2e-5` edges out late-stage generalization (`0.4234` vs `0.4367–0.4396`).** Adding cosine LR decay and clearing stale L-BFGS curvature across batch switches eliminated step-28 stalling (raising acceptance from `90% → 100%` and cutting final ΔL1 from `387k → 206.5k`), with `qqn_lbfgs_adam_m5` beating `adamw_2e-5` through step 20 (`0.4517` vs `0.4531`) at lower L1 drift (`160.1k` vs `176.3k`).
6. **Two-Phase Hypothesis Test confirms L-BFGS belongs in Phase 2, not Phase 1 — and closes the Mode B gap to `0.0023`.** Running intensive 768-example `lbfgs_adam` from step 1 stalled at `0.4451` (3.3× slower than small-batch AdamW), whereas preceding it with 25 small-batch rough steps dropped `lbfgs_adam` to **`0.4236`** (from AdamW rough) and **`0.4243`** (from QQN-Adam rough, a **`-0.0458`** polish drop from `0.4701`). `adamw_two_phase` (`0.4213`, ΔL1 `153.2k`) still narrowly led overall due to cold-start moment shock and lack of grad-clipping/weight-decay in Phase 2 `lbfgs_adam`.

---

## 2. Experimental Setup

| Component | Configuration |
| :--- | :--- |
| **Model** | `gm.nn.Gemma4_E2B` (`GEMMA4_E2B_IT`, `text_only=True`, **4,651,283,747** trainable params) |
| **Precision** | `float32` master weights + `bfloat16` forward/backward compute; `float32` dot products & scalars |
| **Parallelism** | 8-way FSDP (`NamedSharding`) on AWS `p4d.24xlarge` (8× A100-40GB, NVLink, Spot ~$9.50/hr) |
| **Memory Optimizations** | Per-layer `nn.remat`, `@jax.checkpoint` over `lax.scan` micro-batches, `P("fsdp", None, None)` on 262k-vocab logits, `logsumexp` CE, single shared `value_and_grad` JIT, individually FSDP-sharded L-BFGS `(s_i, y_i)` deque |
| **Dataset** | `openai/gsm8k` (`seq_len=512`, `micro_batch_size=8` = 1 seq/GPU, response-only loss mask) |
| **State-Change Metrics** | Exact `float64` $\|\theta_t - \theta_0\|_1$ (`ΔL1`) and $\|\theta_t - \theta_0\|_2$ (`ΔL2`) from initial pretrained weights, measured at both `Best Val` and `Final` |

---

## 3. Experiment 1 — Mode A: Oracle & Anisotropy Ablation
**Run ID:** [`qqn-2026-11-24T10-58-18-775Z`](./runs/qqn-2026-11-24T10-58-18-775Z/summary.md)  
**Regime:** Deterministic fixed subset (`192` train / `64` val GSM8K examples, `30` steps, `eval_every=5`).

| Optimizer | Steps | Evals | Init Train | Final Train | Best Val | Best PPL | Best ΔL1 | Best ΔL2 | Final ΔL1 | Final ΔL2 | Steady (s) | Mean `t` | Accept % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `qqn_adam_2e-5` | 30 | 61 | 2.0611 | 0.0006 | **0.5573** | 1.75 | 156.64k | 3.925 | 253.93k | 6.752 | 268.2 | 1.000 | 100% |
| `qqn_lbfgs_adam_m5` | 30 | 94 | 2.0611 | 0.0002 | **0.5608** | 1.75 | **99.04k** | **2.682** | 365.07k | 11.651 | 452.4 | 0.633 | 63% |
| `adamw_2e-5` | 30 | 30 | 2.0611 | 0.0000 | **0.5663** | 1.76 | 119.51k | 3.175 | 184.30k | 5.049 | 129.9 | 1.000 | 100% |
| `adamw_5e-5` | 30 | 30 | 2.0611 | 0.0000 | **0.5815** | 1.79 | 145.57k | 3.753 | 356.40k | 9.876 | 130.6 | 1.000 | 100% |
| `qqn_lbfgs_m5` | 30 | 75 | 2.0611 | 0.6816 | **0.7812** | 2.18 | 1.08k | 0.143 | 1.08k | 0.143 | 341.8 | 0.001 | 100% |

**Key Takeaways:**
- **`qqn_lbfgs_adam_m5` hit `Best Val = 0.5608` at step 5** (2× fewer steps than `adamw_2e-5`'s step-10 `0.5663`), with **17.1% less ΔL1 (`99.04k` vs `119.51k`)** and **15.5% less ΔL2 (`2.682` vs `3.175`)**.
- **Anisotropy proof:** Plain `qqn_lbfgs_m5` stalled at `t ≈ 0.001` even with `float32` master weights, confirming that scalar `H₀ = γI` cannot condition 4.65B heterogeneous transformer weights.
- **Operational finding:** `qqn_lbfgs_adam_m5`'s `63%` acceptance came exclusively from steps 20–30 after `train_loss` hit the `bfloat16` compute floor (`0.0002`), motivating the early-stop guard (`stop_on_train_loss: 1e-3`) used in Experiment 2.

---

## 4. Experiment 2 — Mode A: Expanded 9-Config Line-Search & Path Suite
**Run ID:** [`qqn-2026-11-24T13-49-57-514Z`](./runs/qqn-2026-11-24T13-49-57-514Z/summary.md)  
**Regime:** Deterministic fixed subset (`192` train / `64` val, `max_steps=25`, `stop_on_train_loss=1e-3`, `stop_on_consecutive_stalls=2`).

| Optimizer | Steps | Evals | Final Train | Best Val | Best PPL | Best ΔL1 | Best ΔL2 | Final ΔL1 | Final ΔL2 | Steady (s) | s/eval | Mean `t` | Accept % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`lbfgs_adam_strong_wolfe`** | 18 | 25 | 0.0007 | **0.5558** | **1.74** | **97.59k** | **2.622** | 328.65k | 10.378 | 127.8 | 5.55 | 0.944 | 94% |
| `qqn_adam_armijo_t1` | 25 | 26 | 0.0069 | **0.5561** | 1.74 | 156.74k | 3.926 | 239.59k | 6.305 | 115.4 | 4.81 | 1.000 | 100% |
| `qqn_adam_bounded_spline` | 25 | 100 | 0.0069 | **0.5571** | 1.75 | 156.67k | 3.925 | 239.50k | 6.302 | 435.3 | 4.58 | 1.000 | 100% |
| `lbfgs_adam_armijo_t1` | 18 | 23 | 0.0005 | **0.5599** | 1.75 | 99.66k | 2.698 | 347.77k | 11.057 | 130.2 | 6.20 | 0.944 | 94% |
| `lbfgs_adam_bounded_spline` | 16 | 65 | 0.0227 | **0.5613** | 1.75 | 99.79k | 2.703 | 261.75k | 8.149 | 288.8 | 4.81 | 0.625 | 88% |
| `lbfgs_adam_armijo_extrap` | 16 | 33 | 0.0008 | **0.5617** | 1.75 | 99.67k | 2.698 | 331.70k | 10.547 | 152.8 | 5.09 | 1.000 | 100% |
| `lbfgs_adam_null_search` | 16 | 17 | 0.0009 | **0.5623** | 1.75 | 100.55k | 2.724 | 307.55k | 9.670 | **85.9** | 5.73 | 1.000 | 100% |
| `lbfgs_adam_linear_chord` | 17 | 23 | 0.0007 | **0.5626** | 1.76 | 100.60k | 2.725 | 350.04k | 11.160 | 113.8 | 5.42 | 0.897 | 100% |
| `adamw_2e-5` | 19 | 19 | 0.0007 | **0.6055** | 1.83 | 75.58k | 1.926 | 164.13k | 4.486 | 80.7 | 4.49 | 1.000 | 100% |

**Key Takeaways:**
- **Every QQN configuration beat `adamw_2e-5` by 0.043–0.050 validation loss.**
- **Strong Wolfe is the clear line-search winner:** Adding the curvature condition (`c2=0.9`) cost only `+2` evals over Armijo `t=1` (`25` vs `23`) while yielding both the **best validation loss (`0.5558`)** and the **smallest weight drift among all top runs (`97.59k` ΔL1, `2.622` ΔL2)** — 37.7% less L1 and 33.2% less L2 than `qqn_adam_armijo_t1`.
- **Quadratic path beats linear chord:** `lbfgs_adam_linear_chord` (removing the `-∇f` tangent) ranked lowest among all `lbfgs_adam` variants (`0.5626`, `100.60k` ΔL1).
- **Bounded splines don't pay back their eval cost here:** `lbfgs_adam_bounded_spline` (`65` evals, `288.8s`) and `qqn_adam_bounded_spline` (`100` evals, `435.3s`) stayed cleanly bounded but did not outperform `strong_wolfe` (`25` evals, `127.8s`).

---

## 5. Experiment 3 — Mode B: Full-Dataset Stochastic Macro-Batch SFT
**Run ID:** [`qqn-2026-11-24T15-32-13-067Z`](./runs/qqn-2026-11-24T15-32-13-067Z/summary.md)  
**Regime:** Full GSM8K training set (`7,473` train / `128` val), `macro_batch_examples=192`, `inner_steps_per_batch=3`, `45` steps, cosine LR decay (`1.0 → 0.1`), `clear_lbfgs_on_refresh=true`.

| Optimizer | Steps | Evals | Init Train | Final Train | Best Val | Best PPL | Best ΔL1 | Best ΔL2 | Final ΔL1 | Final ΔL2 | Steady (s) | s/eval | Mean `t` | Accept % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`adamw_2e-5`** | 45 | 45 | 2.0678 | 0.3723 | **0.4234** | **1.53** | 215.51k | 5.917 | 215.51k | 5.917 | **195.0** | 4.43 | 1.000 | 100% |
| `qqn_lbfgs_adam_m4_spline` | 45 | 195 | 2.0678 | 0.2847 | **0.4367** | 1.55 | 205.62k | 6.363 | 205.62k | 6.363 | 817.5 | 4.30 | 1.000 | 100% |
| `qqn_lbfgs_adam_m5` | 45 | 60 | 2.0678 | 0.2639 | **0.4396** | 1.55 | **202.38k** | 6.267 | 206.53k | 6.400 | 231.1 | 3.98 | 1.000 | 100% |
| `qqn_adam_2e-5` | 27 | 54 | 2.0678 | 0.4201 | **0.4529** | 1.57 | 204.34k | 5.178 | 204.34k | 5.178 | 208.9 | 4.02 | 0.889 | 89% |

### Mode B Validation & Drift Trajectory

| Step | `adamw_2e-5` (val / ΔL1) | `qqn_lbfgs_adam_m5` (val / ΔL1) | `qqn_lbfgs_adam_m4_spline` (val / ΔL1) |
| :---: | :---: | :---: | :---: |
| **5** | `0.6094` / `56.2k` | **`0.5269`** / `78.4k` | **`0.5268`** / `78.5k` |
| **10** | `0.4840` / `112.6k` | **`0.4740`** / `114.2k` | **`0.4744`** / `113.4k` |
| **20** | `0.4531` / `176.3k` | **`0.4517`** / **`160.1k`** | **`0.4510`** / **`156.5k`** |
| **35** | **`0.4292`** / `211.7k` | `0.4418` / `197.6k` | `0.4381` / `196.0k` |
| **45** | **`0.4234`** / `215.5k` | `0.4445` / `206.5k` | `0.4367` / `205.6k` |

**Key Takeaways:**
- **Cosine LR decay + batch-boundary L-BFGS clearing fixed Mode B stability completely:** Compared to the earlier unscheduled Mode B run (`qqn-2026-11-24T12-23-04-734Z`), `qqn_lbfgs_adam_m5` went from stalling at step 30 (`val=0.4615`, `Final ΔL1=387k`) to **100% acceptance across all 45/45 steps (`Best Val=0.4396`, `Final ΔL1=206.5k` — 47% less drift)**.
- **Early-phase speed & lower L1 drift:** Both `qqn_lbfgs_adam` variants beat `adamw_2e-5` through step 20 (`0.4510–0.4517` vs `0.4531`) while using **9–11% less L1 weight change (`156.5k–160.1k` vs `176.3k`)**, and finished step 45 with lower total L1 (`205.6k–206.5k` vs `215.5k`).
- **Why `adamw_2e-5` edged out steps 25–45 (`0.4234` vs `0.4367`):** `qqn_lbfgs_adam` fits each 192-example macro-batch aggressively during the 3 inner steps (`train_loss ≈ 0.24–0.28` vs `adamw`'s `0.37–0.40`), and lacks `adamw`'s decoupled weight decay (`0.01`) and gradient clipping (`1.0`). Reducing `inner_steps_per_batch` from `3 → 2` (or adding decoupled weight decay to `lbfgs_adam`) is the natural next lever for stochastic Mode B.

---

## 6. Experiment 4 — Mode B Two-Phase Hypothesis Test (Small-Batch Rough → Intensive Polish)
**Run ID:** [`qqn-2026-11-24T16-47-07-351Z`](./runs/qqn-2026-11-24T16-47-07-351Z/summary.md)  
**Hypothesis:** L-BFGS curvature is antagonistic with noisy small micro-batches in Phase 1 (rough learning), which should use an Adam oracle + small batches (`128` ex/step, `inner=1`), whereas Phase 2 (polish) should use intensive batches (`768` ex/batch, `inner=4`) where `lbfgs_adam` + Strong Wolfe has clean curvature.  
**Regime:** Full GSM8K (`7,473` train / `256` val), Phase 1 = `25` steps (`128` ex/step, `lr=2e-5`), Phase 2 = `12` steps (`768` ex/batch, `inner=4`, `lr=1e-5`); compared against `adamw_small_only` (37×128), `adamw_two_phase` (isolating batch-size scaling), and `lbfgs_adam_intensive_only` (16×768 from step 1).

| Optimizer | Steps | Evals | Init Train | Final Train | Best Val | Best PPL | Best ΔL1 | Best ΔL2 | Final ΔL1 | Final ΔL2 | Steady (s) | s/eval | Mean `t` | Accept % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`adamw_two_phase`** | 37 | 37 | 2.1732 | 0.3692 | **0.4213** | **1.52** | **153.21k** | **4.179** | 153.21k | 4.179 | 264.8 | 7.36 | 1.000 | 100% |
| `adamw_small_only` | 37 | 37 | 2.1732 | 0.4295 | **0.4225** | 1.53 | 165.09k | 4.504 | 165.09k | 4.504 | **106.3** | 2.95 | 1.000 | 100% |
| `two_phase_adamw_to_lbfgs_adam` | 37 | 45 | 2.1732 | 0.2906 | **0.4236** | 1.53 | 227.74k | 6.191 | 231.06k | 6.286 | 369.4 | 8.40 | 1.000 | 100% |
| `two_phase_qqn_adam_to_lbfgs_adam` | 37 | 67 | 2.1732 | 0.2985 | **0.4243** | 1.53 | 229.22k | 6.137 | 233.28k | 6.251 | 319.9 | 4.92 | 1.000 | 100% |
| `lbfgs_adam_intensive_only` | 16 | 24 | 2.0779 | 0.3618 | **0.4451** | 1.56 | 132.14k | 4.029 | 136.29k | 4.189 | 346.4 | 15.75 | 1.000 | 100% |

### Two-Phase Validation Trajectory (256-example Val Split)

| Step (Phase) | `adamw_small_only` | `adamw_two_phase` | `two_phase_adamw_to_lbfgs_adam` | `two_phase_qqn_adam_to_lbfgs_adam` | `lbfgs_adam_intensive_only` |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **5** (P1) | `0.5900` | `0.5682` | `0.5677` | `0.5818` | **`0.5152`** |
| **15** (P1) | `0.4551` | `0.4519` | `0.4518` | `0.4848` | **`0.4451`** |
| **25** (end P1) | **`0.4310`** | `0.4369` | `0.4368` | `0.4701` | *(ended at 16: `0.4470`)* |
| **30** (P2) | **`0.4256`** | `0.4354` | `0.4843` *(cold-start spike)* | `0.4552` | — |
| **35** (P2) | `0.4231` | **`0.4224`** | `0.4236` | `0.4243` | — |
| **37** (final) | `0.4225` | **`0.4213`** | `0.4272` | `0.4267` | — |

**Key Takeaways:**
- **Phase-1 part of the hypothesis is strongly confirmed:** Running intensive 768-example `lbfgs_adam` from step 1 (`lbfgs_adam_intensive_only`) stalled at **`0.4451`** despite taking `346.4s` (3.3× slower than `adamw_small_only`). Moving `lbfgs_adam` to Phase 2 after 25 small-batch rough steps cut its val loss by **-0.0215 (`0.4451 → 0.4236`)** and closed the single-phase Mode B gap to AdamW from `0.0162` down to **`0.0023`**.
- **Dramatic Phase-2 polish recovery on `two_phase_qqn_adam_to_lbfgs_adam`:** After `rough_qqn_adam` plateaued at `0.4701` (step 25), switching to intensive `polish_lbfgs_adam_sw` drove a **`-0.0458` val drop (`0.4701 → 0.4243`)** in 10 steps — virtually matching `two_phase_adamw_to_lbfgs_adam` (`0.4236`).
- **Why `adamw_two_phase` (`0.4213`, ΔL1 `153.2k`) still narrowly won:**
  1. **Cold-start moment shock at step 26:** Resetting the Adam second moment `v=0` when entering `polish_lbfgs_adam_sw` (without grad clipping or warmup) caused a sharp transient spike at step 30 (`0.4368 → 0.4843`) and added ~`+60k` ΔL1 (`227.7k` vs `adamw_two_phase`'s `153.2k`). Passing warm-started `(m, v)` moments from Phase 1 into Phase 2 `lbfgs_adam` — or adding a 2-step warmup + grad clipping — would eliminate this shock.
  2. **Batch overfitting with `inner_steps_per_batch=4`:** Even on 768 examples, 4 inner L-BFGS steps pushed `train_loss` down to `0.29` (vs `adamw_two_phase`'s `0.37`), causing val to tick up slightly on steps 35 → 37 (`0.4236 → 0.4272`). `inner_steps_per_batch=2` + decoupled weight decay is the cleaner polish recipe.

---

## 7. Conclusions & Phase 2 Status

| Regime | Best Optimizer | Best Val (PPL) | Key Advantage |
| :--- | :--- | :---: | :--- |
| **Deterministic / Curated SFT (Mode A)** | **`lbfgs_adam` + `strong_wolfe`** | **0.5558 (1.74)** | Beats AdamW by `0.050` val loss; **38% less ΔL1** than QQN-Adam at 25 evals |
| **Fastest Wall-Clock (Mode A)** | **`lbfgs_adam` + `null` / `armijo_t1`** | **0.5599–0.5623 (1.75)** | ~86–130s steady-state; matches AdamW speed with much better val loss |
| **Full-Dataset Stochastic SFT (Mode B)** | **`adamw_two_phase`** / **`two_phase_*_to_lbfgs_adam`** | **0.4213 (1.52)** / **0.4236 (1.53)** | Two-phase intensive polish beats small-batch-only AdamW (`0.4225`) and rescues `lbfgs_adam` from `0.4451 → 0.4236` |

**Next Steps:**
1. **Warm-start Phase 2 `lbfgs_adam` moments + add optional decoupled weight decay / grad clipping** to eliminate the step-26 cold-start spike (`0.4368 → 0.4843`) and trim Phase-2 ΔL1 drift.
2. **Phase 2 Custom Behavior Fine-Tuning & Serving (implemented):** Train Gemma 4 E2B on [`data/pokemon_pirate.jsonl`](./data/pokemon_pirate.jsonl) via `--mode pokemon-pirate`, persist `bfloat16` Orbax checkpoints to S3, and serve interactively on `g6e.xlarge` via `npm run aws:serve` + `npm run chat`.
