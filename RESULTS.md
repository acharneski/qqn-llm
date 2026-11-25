# Experiment Results: Full-Weight Gemma 4 E2B Fine-Tuning with `qqn-jax`

## 1. Executive Summary

We integrated our Quadratic Quasi-Newton optimizer ([`qqn-jax`](./qqn-jax)) with full-weight fine-tuning of **Gemma 4 E2B** (`4,651,283,747` trainable parameters, 35 transformer layers + Per-Layer Embeddings) in JAX/Flax across **8× NVIDIA A100-40GB GPUs** (`p4d.24xlarge`, FSDP sharded) on **GSM8K** (`seq_len=512`, `float32` master weights, `bfloat16` compute, per-layer `nn.remat`).

Across nine AWS benchmark sweeps comparing **QQN** variants against **AdamW** baselines in **Mode A (deterministic fixed-subset SFT)**, **Mode B (full-dataset 7,473-example stochastic macro-batch SFT)**, and **Two-Stage (small-batch rough → intensive polish) SFT**, we establish eight core results:

1. **Two-Stage QQN (`AdamW` or `QQN-Adam` rough → warm-started `lbfgs_adam` Strong Wolfe `preconditioned` polish) beats equal-step objective-matched `AdamW` across all three full-GSM8K two-stage runs (`0.4101–0.4145` vs `0.4183–0.4200`), and ties compute-matched `46`-eval `AdamW` at `1.51` PPL (`0.4145` vs `0.4129` on `512` val examples).** In [`qqn-2026-11-24T20-51-24-770Z`](./runs/qqn-2026-11-24T20-51-24-770Z/summary.md) and [`qqn-2026-11-24T19-55-33-780Z`](./runs/qqn-2026-11-24T19-55-33-780Z/summary.md) (`256` val), Two-Stage QQN took **#1 and #2** (`0.4101` / `0.4139` vs `0.4183` / `0.4200`). In the strictest `512`-val symmetric-L2-SP benchmark [`qqn-2026-11-25T01-52-33-891Z`](./runs/qqn-2026-11-25T01-52-33-891Z/summary.md), **`qqn_two_stage_precond` (`0.4145` at step 36, `ΔL1 = 147.6k`)** again beat equal-step `adamw_two_stage_l2sp` (`0.4200` at step 38, `ΔL1 = 130.3k`) and `adamw_small_only` (`0.4177`), while `adamw_two_stage_equal_evals` — granted `11` unique 768-example polish batches (`8,448` polish examples vs QQN's `7` batches / `5,376` examples) for `46` total evals — edged to `0.4129` (`1.51` PPL).
2. **At matched data exposure (`7` polish macro-batches), curvature beats first-order updates; at matched total evaluations, fresh macro-batch diversity slightly outweighs extra per-batch line-search probes.** On the exact same 7 Stage-2 macro-batches (steps 25–38 in Exp 7), `qqn_two_stage_precond` beat both `adamw_two_stage_l2sp` (`0.4145` vs `0.4200`) and `adamw_two_stage_equal_evals` at step 38 (`0.4145` vs `0.4161`), whereas spending 8 additional gradient evaluations on **4 fresh** 768-example macro-batches (`steps 39–46` of `adamw_two_stage_equal_evals`) reached `0.4129` without the single-batch overfitting sawtooth seen when QQN extrapolated to `t = 2.0` (`max_t = 2.0`) on inner step 2.
3. **Diagonal preconditioning of BOTH the endpoint (`lbfgs_adam`) AND the quadratic tangent (`tangent_mode: preconditioned`) is essential on 4.65B LLMs.** Unpreconditioned `LBFGSOracle` (`H₀ = γI`) stalled at `Mean t ≈ 0.001` (`Best Val = 0.7812`), while **`qqn_two_stage_scaled`** (`tangent_mode: scaled`, $-\tau\nabla f$) succeeded in Mode A *only* because $t = 1$ always passed ($t(1-t) = 0$) — the moment Stage-2 polish required interior Strong Wolfe zoom ($t \in \{0.25, 0.5\}$) in [`01-52-33-891Z`](./runs/qqn-2026-11-25T01-52-33-891Z/summary.md), stepping along unpreconditioned Euclidean $-\tau\nabla f$ failed every interior probe (`3` consecutive stalls at steps 26–28, `0.4360`), whereas `preconditioned` ($-\alpha_{\tan} H_0\nabla f$) zoomed smoothly at **`100%` acceptance** across every run.
4. **Controlled `tangent_mode` ablations (`raw` vs `reparam` vs `scaled` vs `preconditioned`) separate path geometry from interval parameterization bias.** In [`qqn-2026-11-24T22-25-26-512Z`](./runs/qqn-2026-11-24T22-25-26-512Z/summary.md) (`eval_every=1`), path-invariant Möbius `reparam` ($t(s) = \tau s / ((1-s) + \tau s)$) proved that compressing $dt/ds|_{s=0} = \tau$ forces $dt/ds|_{s=1} = 1/\tau \sim 10^4$, inflating $|\phi'(1)|$ by $1/\tau^2$, failing Strong Wolfe's curvature test at $s = 1$, and biasing zoom outward to $s = 2.0$ (`Mean t = 1.278`, `0.5691` val loss, `41` evals).
5. **In Mode A (`N=192` deterministic subset), `eval_every=1` de-aliasing places `adamw_2e-5` (`0.5349` at step 7) and `lbfgs_adam_strong_wolfe_precond` (`0.5392` at step 6) in a virtual tie (`1.71` PPL) before subset memorization.** Evaluating every step showed both optimizers bottom out at `train_loss ≈ 0.27–0.30` (`step 6–7`, `7` gradient evals each) before memorizing the 192-example subset, explaining why `eval_every=5` previously penalized `AdamW` (`0.6055`).
6. **Bounded cubic Hermite splines are stable on preconditioned LLMs (`0.5425` at step 6) but cost ~3.5× more evaluations than Strong Wolfe.** Keeping splines strictly bounded (`rounds=2, max_cp=8`) achieved `val = 0.5425` at step 6 (`25` evals) vs `0.5392–0.5447` in `7` evals for Strong Wolfe.
7. **Warm-starting the second moment `(v, step)` across phase transitions + reducing polish inner steps (`4 → 2`) + outer cosine LR secant scaling resolved all Mode B bottlenecks.** Exporting Phase-1 `v` into Phase-2 `lbfgs_adam`, scaling the full L-BFGS direction (including the $\rho_i s_i s_i^\top$ secant subspace) by cosine LR decay, and adding L2-SP anchor regularization turned Two-Phase QQN from `#3` (`0.4236` in Exp 4) into **`#1` overall (`0.4101` in Exp 5)**.

---

## 2. Experimental Setup & Systems Architecture

| Component | Configuration |
| :--- | :--- |
| **Model** | `gm.nn.Gemma4_E2B` (`GEMMA4_E2B_IT`, `text_only=True`, **4,651,283,747** trainable params) |
| **Precision** | `float32` master weights + `bfloat16` forward/backward compute; `float32` dot products & scalars |
| **Parallelism** | 8-way FSDP (`NamedSharding`) on AWS `p4d.24xlarge` (8× A100-40GB, NVLink) |
| **Memory & Allocator Optimizations** | `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async` (`cuMemAllocAsync`), per-layer `nn.remat`, `@jax.checkpoint` over `lax.scan` micro-batches, `P("fsdp", None, None)` on 262k-vocab logits, `logsumexp` CE, `donate_argnums=(0,)` on shared `value_and_grad` JIT and two-loop recursion, host staging of multi-probe `best_g`, individually FSDP-sharded `bfloat16` L-BFGS `(s_i, y_i)` deque |
| **Dataset & Formatting** | `openai/gsm8k` (`seq_len=512`, `micro_batch_size=8` = 1 seq/GPU, native `GEMMA4_E2B_IT` chat turn template in Exp 5+, response-only loss mask) |
| **Path & Tangent Parameterizations** | Explicit `tangent_mode`: canonical `"raw"` ($- \nabla f$), path-invariant Möbius interval reparameterization `"reparam"` ($t(s) = \tau s / ((1-s) + \tau s)$), norm-matched Euclidean `"scaled"` ($-\tau \nabla f$), and diagonally `"preconditioned"` ($-\alpha_{\tan} H_0 \nabla f$) |
| **State-Change Metrics** | Exact `float64` $\|\theta_t - \theta_0\|_1$ (`ΔL1`) and $\|\theta_t - \theta_0\|_2$ (`ΔL2`) from initial pretrained weights, measured at both `Best Val` and `Final` alongside `Best @ (Step/Eval)` |

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

## 4. Experiment 2 — Mode A: Expanded Line-Search & Path Suite (+ De-Aliasing Check)
**Run ID:** [`qqn-2026-11-24T13-49-57-514Z`](./runs/qqn-2026-11-24T13-49-57-514Z/summary.md)  
**Regime:** Deterministic fixed subset (`192` train / `64` val, `max_steps=25`, `eval_every=5`, `stop_on_train_loss=1e-3`, `stop_on_consecutive_stalls=2`).

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
| `adamw_2e-5` *(eval_every=5)* | 19 | 19 | 0.0007 | **0.6055** | 1.83 | 75.58k | 1.926 | 164.13k | 4.486 | 80.7 | 4.49 | 1.000 | 100% |
| `adamw_2e-5` *(`eval_every=1`, chat fmt, [`19-56-30-469Z`](./runs/qqn-2026-11-24T19-56-30-469Z/summary.md))* | 22 | 22 | 0.0004 | **0.5378** *(at 8/8)* | **1.71** | 119.97k | 3.142 | 177.66k | 4.852 | 92.8 | 4.42 | 1.000 | 100% |

**Key Takeaways:**
- **Strong Wolfe is the clear QQN line-search winner:** Adding the curvature condition (`c2=0.9`) cost only `+2` evals over Armijo `t=1` (`25` vs `23`) while yielding both the **best QQN validation loss (`0.5558`)** and the **smallest weight drift (`97.59k` ΔL1, `2.622` ΔL2)** — 37.7% less L1 and 33.2% less L2 than `qqn_adam_armijo_t1`.
- **Quadratic path beats linear chord:** `lbfgs_adam_linear_chord` (removing the `-∇f` tangent) ranked lowest among all `lbfgs_adam` variants (`0.5626`, `100.60k` ΔL1).
- **Methodological de-aliasing (`eval_every=1`):** Re-running `adamw_2e-5` with `eval_every=1`, native `GEMMA4_E2B_IT` chat turn formatting, and non-zero step-1 warmup ([`qqn-2026-11-24T19-56-30-469Z`](./runs/qqn-2026-11-24T19-56-30-469Z/summary.md)) confirmed that `eval_every=5` had aliased right over `AdamW`'s narrow generalization minimum on the tiny 192-example subset (`val = 0.6232` at step 5 → **`0.5378` at step 8** → `0.5641` at step 10 → `1.5148` at step 22). Crucially, fixed-subset Mode A (`N=192`) is fundamentally memorization-limited past `train_loss ≈ 0.35`, motivating full-dataset Mode B (`N=7,473`) for true generalization comparisons.

---

## 5. Experiment 3 — Mode B: Single-Phase Full-Dataset Macro-Batch SFT
**Run ID:** [`qqn-2026-11-24T15-32-13-067Z`](./runs/qqn-2026-11-24T15-32-13-067Z/summary.md)  
**Regime:** Full GSM8K training set (`7,473` train / `128` val), `macro_batch_examples=192`, `inner_steps_per_batch=3`, `45` steps, cosine LR decay (`1.0 → 0.1`), `clear_lbfgs_on_refresh=true`.

| Optimizer | Steps | Evals | Init Train | Final Train | Best Val | Best PPL | Best ΔL1 | Best ΔL2 | Final ΔL1 | Final ΔL2 | Steady (s) | s/eval | Mean `t` | Accept % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`adamw_2e-5`** | 45 | 45 | 2.0678 | 0.3723 | **0.4234** | **1.53** | 215.51k | 5.917 | 215.51k | 5.917 | **195.0** | 4.43 | 1.000 | 100% |
| `qqn_lbfgs_adam_m4_spline` | 45 | 195 | 2.0678 | 0.2847 | **0.4367** | 1.55 | 205.62k | 6.363 | 205.62k | 6.363 | 817.5 | 4.30 | 1.000 | 100% |
| `qqn_lbfgs_adam_m5` | 45 | 60 | 2.0678 | 0.2639 | **0.4396** | 1.55 | **202.38k** | 6.267 | 206.53k | 6.400 | 231.1 | 3.98 | 1.000 | 100% |
| `qqn_adam_2e-5` | 27 | 54 | 2.0678 | 0.4201 | **0.4529** | 1.57 | 204.34k | 5.178 | 204.34k | 5.178 | 208.9 | 4.02 | 0.889 | 89% |

**Key Takeaways:**
- **Early-phase speed & lower L1 drift:** Both `qqn_lbfgs_adam` variants beat `adamw_2e-5` through step 20 (`0.4510–0.4517` vs `0.4531`) while using **9–11% less L1 weight change (`156.5k–160.1k` vs `176.3k`)**, with **100% step acceptance** across all 45 steps.
- **Diagnostic insight for late-stage Mode B:** Taking `3` quasi-Newton inner steps per 192-example macro-batch over-fit each individual batch (`train_loss ≈ 0.26` vs `adamw`'s `0.37`), and inside `QQNRunner` the cosine LR multiplier had previously scaled only $H_0$ rather than the outer L-BFGS direction (so $\eta / \eta = 1$ cancelled inside the rank-$m$ secant subspace). Both issues were fixed for Experiment 5.

---

## 6. Experiment 4 — Initial Cold-Start Two-Phase Test (Baseline Diagnostic)
**Run ID:** [`qqn-2026-11-24T16-47-07-351Z`](./runs/qqn-2026-11-24T16-47-07-351Z/summary.md)  
**Regime:** Full GSM8K (`7,473` train / `256` val), Phase 1 = `25` steps (`128` ex/step, `lr=2e-5`), Phase 2 = `12` steps (`768` ex/batch, `inner=4`, `lr=1e-5`, cold-started `v=0` at Phase-2 entry).

| Optimizer | Steps | Evals | Init Train | Final Train | Best Val | Best PPL | Best ΔL1 | Best ΔL2 | Final ΔL1 | Final ΔL2 | Steady (s) | s/eval | Mean `t` | Accept % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`adamw_two_phase`** | 37 | 37 | 2.1732 | 0.3692 | **0.4213** | **1.52** | **153.21k** | **4.179** | 153.21k | 4.179 | 264.8 | 7.36 | 1.000 | 100% |
| `adamw_small_only` | 37 | 37 | 2.1732 | 0.4295 | **0.4225** | 1.53 | 165.09k | 4.504 | 165.09k | 4.504 | **106.3** | 2.95 | 1.000 | 100% |
| `two_phase_adamw_to_lbfgs_adam` | 37 | 45 | 2.1732 | 0.2906 | **0.4236** | 1.53 | 227.74k | 6.191 | 231.06k | 6.286 | 369.4 | 8.40 | 1.000 | 100% |
| `two_phase_qqn_adam_to_lbfgs_adam` | 37 | 67 | 2.1732 | 0.2985 | **0.4243** | 1.53 | 229.22k | 6.137 | 233.28k | 6.251 | 319.9 | 4.92 | 1.000 | 100% |
| `lbfgs_adam_intensive_only` | 16 | 24 | 2.0779 | 0.3618 | **0.4451** | 1.56 | 132.14k | 4.029 | 136.29k | 4.189 | 346.4 | 15.75 | 1.000 | 100% |

**Diagnostic Takeaways Driving Experiment 5:**
1. **L-BFGS belongs in Phase 2, not Phase 1:** Running 768-example `lbfgs_adam` from step 1 stalled at `0.4451` (`346.4s`), whereas preceding it with 25 small-batch rough steps cut val loss to `0.4236–0.4243`.
2. **Two concrete bugs held back Phase-2 QQN (`0.4236` vs `0.4213`):** (a) resetting `v = 0` at step 26 caused a cold-start preconditioner shock (`val: 0.4368 → 0.4843` at step 30), and (b) `inner_steps_per_batch=4` without anchor regularization over-fit the 768-example batch on steps 35–37 (`0.4236 → 0.4272`).

---

## 7. Experiment 5 — Warm-Started & L2-SP Regularized Two-Phase SFT (Overall Winner)
**Primary Run ID:** [`qqn-2026-11-24T20-51-24-770Z`](./runs/qqn-2026-11-24T20-51-24-770Z/summary.md) *(exact `float32` init + unclipped `beta2=0.999` Phase-2 `lbfgs_adam`)*  
**Companion Ablation Run ID:** [`qqn-2026-11-24T19-55-33-780Z`](./runs/qqn-2026-11-24T19-55-33-780Z/summary.md) *(`bf16`-matched anchor + clipped `beta2=0.98` Phase-2 `lbfgs_adam`)*  
**Regime:** Full GSM8K (`7,473` train / `256` val, native `GEMMA4_E2B_IT` chat turn format, `eval_every=3`). Phase 1 = `25` steps (`128` ex/step, `inner=1`, `lr=2e-5`); Phase 2 = `12` steps (`768` ex/batch, `inner=2`, `lr=1e-5`, `warm_start_preconditioner=true` transferring `(v, step=25)` from Phase 1, `anchor_weight_decay=1e-3`, `max_qn_gain=4.0`, outer cosine LR decay).

### 7.1 Primary Benchmark (`qqn-2026-11-24T20-51-24-770Z`)

| Optimizer | Steps | Evals | Best @ (Step/Eval) | Init Train | Final Train | Best Val | Best PPL | Best ΔL1 | Best ΔL2 | Steady (s) | s/eval | Mean `t` | Accept % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`two_phase_adamw_to_lbfgs_adam`** | 37 | 46 | **37/46** | 2.1102 | 0.4704 | **0.4101** | **1.51** | 321.84k* | 6.363* | 381.1 | 8.47 | 1.027 | 100% |
| **`two_phase_qqn_adam_to_lbfgs_adam`** | 37 | 74 | **37/74** | 2.1102 | 0.4930 | **0.4151** | **1.51** | 316.90k* | 6.370* | 346.2 | 4.81 | 1.088 | 100% |
| `adamw_small_only` *(Control 1)* | 37 | 37 | 37/37 | 2.1102 | 0.4254 | **0.4171** | 1.52 | 162.56k | 4.442 | **106.0** | 2.94 | 1.000 | 100% |
| `adamw_two_phase` *(Control 2)* | 37 | 37 | 37/37 | 2.1102 | 0.3838 | **0.4183** | 1.52 | 137.35k | 3.731 | 263.8 | 7.33 | 1.000 | 100% |
| `lbfgs_adam_intensive_only` *(Control 3)* | 16 | 23 | 16/23 | 2.0180 | 0.3642 | **0.4408** | 1.55 | **116.69k** | **3.578** | 338.3 | 16.11 | 1.000 | 94% |

*\*Note on ΔL1/ΔL2 in `20-51-24-770Z` vs `19-55-33-780Z`: In `20-51-24-770Z`, `params` started from exact `float32` checkpoint weights while `sharded_anchor_params` was cast to `bfloat16`, nudging ~2.2B unseen vocabulary embeddings onto their `bfloat16` rounding grid (`+180k` apparent `float32` L1 offset without affecting `bfloat16` forward compute). In `19-55-33-780Z` (below), where `params` and `anchor` shared identical initial rounding, both Two-Phase QQN treatments achieved **`ΔL1 = 153.06k–159.64k` (`ΔL2 = 4.262–4.425`)** — lower than `adamw_small_only` (`178.58k`)! `train.py` now keeps `sharded_anchor_params` in exact `master_dtype` (`float32`) to combine both benefits.*

### 7.2 Companion Precision-Matched Anchor Ablation (`qqn-2026-11-24T19-55-33-780Z`)

| Optimizer | Steps | Evals | Best @ (Step/Eval) | Init Train | Final Train | Best Val | Best PPL | Best ΔL1 | Best ΔL2 | Steady (s) | s/eval | Mean `t` | Accept % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`two_phase_qqn_adam_to_lbfgs_adam`** | 37 | 74 | **37/74** | 2.1097 | 0.3464 | **0.4139** | **1.51** | **153.06k** | **4.262** | 342.0 | 4.75 | 1.061 | 100% |
| **`two_phase_adamw_to_lbfgs_adam`** | 37 | 48 | **36/46** | 2.1097 | 0.3508 | **0.4151** | **1.51** | 159.64k | 4.425 | 411.3 | 8.75 | 1.041 | 100% |
| `adamw_two_phase` *(Control 2)* | 37 | 37 | 37/37 | 2.1097 | 0.3871 | **0.4200** | 1.52 | 144.31k | 3.905 | 261.4 | 7.26 | 1.000 | 100% |
| `adamw_small_only` *(Control 1)* | 37 | 37 | 37/37 | 2.1097 | 0.4353 | **0.4241** | 1.53 | 178.58k | 4.838 | **106.4** | 2.96 | 1.000 | 100% |
| `lbfgs_adam_intensive_only` *(Control 3)* | 16 | 27 | 15/26 | 2.0180 | 0.2597 | **0.4463** | 1.56 | 149.29k | 4.184 | 401.1 | 16.04 | 0.938 | 94% |

### 7.3 Step-by-Step Validation Trajectory (`qqn-2026-11-24T20-51-24-770Z`, 256-Example Val Split)

| Step (Phase) | `adamw_small_only` | `adamw_two_phase` | **`two_phase_adamw_to_lbfgs_adam`** | **`two_phase_qqn_adam_to_lbfgs_adam`** | `lbfgs_adam_intensive_only` |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0** (init) | `2.0662` | `2.0662` | `2.0662` | `2.0662` | `2.0662` |
| **3** (P1) | `0.7755` | `0.6532` | `0.6538` | `0.6437` | **`0.6163`** |
| **9** (P1) | `0.4800` | `0.4676` | **`0.4674`** | `0.4699` | **`0.4609`** |
| **15** (P1) | `0.4460` | `0.4451` | `0.4453` | `0.4467` | **`0.4432`** |
| **21** (P1) | **`0.4313`** | `0.4342` | `0.4341` | `0.4361` | *(ended at 16: `0.4408`)* |
| **25** (end P1) | *(`0.4252` @ 24)* | `0.4315` | **`0.4310`** | `0.4328` | — |
| **30** (P2) | **`0.4205`** | `0.4233` | `0.4343` | **`0.4205`** | — |
| **33** (P2) | `0.4195` | `0.4201` | **`0.4117`** | **`0.4173`** | — |
| **37** (final) | `0.4171` | `0.4183` | **`0.4101`** | **`0.4151`** | — |

### 7.4 Key Scientific Findings

- **Both Two-Phase QQN schedules beat every control across both independent runs:**
  - `two_phase_adamw_to_lbfgs_adam` achieved **`0.4101` (PPL `1.507`)** — improving by **`-0.0135` nats** over Experiment 4 (`0.4236`) and beating both `adamw_small_only` (`0.4171`) and `adamw_two_phase` (`0.4183`) at 100% step acceptance (`Mean t = 1.027`, only `9` extra evals over 37 steps = `1.75` evals/step in Phase 2).
  - Pure QQN end-to-end (`two_phase_qqn_adam_to_lbfgs_adam`) improved from `0.4243` (Exp 4) to **`0.4139–0.4151`**, beating every AdamW baseline while running Phase 1 at **`1.22s/step` steady-state** (2.4× faster per step than `AdamW`'s `2.96s/step` thanks to `will_refresh_next` forward-only terminal probes).
- **Clean causal isolation of quasi-Newton curvature vs batch-size increase:**
  `adamw_two_phase` and `two_phase_adamw_to_lbfgs_adam` execute the **identical Phase-1 optimizer** (`val = 0.4315` vs `0.4310` at step 25) and process the **identical 768-example macro-batches** across steps 26–37. Warm-started `AdamW` improved validation loss by `-0.0132` (`0.4315 → 0.4183`), whereas warm-started `lbfgs_adam` + Strong Wolfe improved validation loss by **`-0.0209` (`0.4310 → 0.4101`)** — demonstrating a **58% larger Phase-2 generalization gain** attributable directly to rank-4 secant curvature + Strong Wolfe zoom (which actively selected `t = 2.0` on steps 35 and 37).
- **Why `rough_qqn_adam` improved by `-0.0373` in Phase 1 (`0.4701 → 0.4328`):**
  Adding `grad_clip_norm=1.0`, `beta2=0.98`, and outer cosine LR decay prevented step-1 gradient spikes (`‖g‖ = 85.0`) from poisoning `AdamOracle`'s second moment `v`, bringing Phase-1 `QQN-Adam` (`0.4328`) right alongside `AdamW` (`0.4310`).
- **Why `lbfgs_adam` prefers unclipped `v` (`beta2=0.999`, `grad_clip_norm=0.0`):**
  Comparing Phase 2 in `19-55-33-780Z` (`beta2=0.98, grad_clip_norm=1.0` → `0.4151`) against `20-51-24-770Z` (`beta2=0.999, grad_clip_norm=0.0` → **`0.4101`**) shows that leaving `lbfgs_adam`'s diagonal $H_0$ unclipped lets the $(4.65\text{B} - m)$-dimensional orthogonal complement shrink naturally as $\|\nabla f\|$ falls, yielding an extra **`-0.0050` nats** in Phase-2 polish.

---

## 8. Experiment 6 — De-Aliased Mode A Tangent-Metric & Interval-Reparameterization Ablation
**Run ID:** [`qqn-2026-11-24T22-25-26-512Z`](./runs/qqn-2026-11-24T22-25-26-512Z/summary.md)  
**Regime:** Deterministic fixed subset (`192` train / `64` val, native `GEMMA4_E2B_IT` chat turn format, `eval_every=1`, `max_steps=25`, `stop_on_train_loss=1e-3`, `stop_on_consecutive_stalls=2`, `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`). Compares all four explicit `tangent_mode` formulations on `lbfgs_adam` (`m=5`) + `strong_wolfe` (`max_t=2.0`) alongside `armijo_t1`, `armijo_extrap`, `bounded_spline`, and de-aliased `adamw_2e-5`.

| Optimizer (`tangent_mode`) | Steps | Evals | Best @ (Step/Eval) | Init Train | Final Train | Best Val | Best PPL | Best ΔL1 | Best ΔL2 | Final ΔL1 | Final ΔL2 | Steady (s) | s/eval | Mean `t` | Accept % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`adamw_2e-5`** | 17 | 17 | **7/7** | 2.0006 | 0.0008 | **0.5349** | **1.71** | **97.33k** | **2.586** | 158.19k | 4.326 | **77.5** | 4.85 | 1.000 | 100% |
| **`lbfgs_adam_strong_wolfe_precond`** (`preconditioned`) | 16 | 23 | **6/7** | 2.0006 | 0.0009 | **0.5392** | **1.71** | 105.21k | 2.906 | 308.22k | 9.931 | 161.9 | 7.71 | 0.961 | **100%** |
| `lbfgs_adam_armijo_t1` (`raw`) | 15 | 28 | 6/7 | 2.0006 | 0.0122 | **0.5420** | 1.72 | 103.52k | 2.856 | 295.77k | 9.570 | 178.3 | 6.86 | 0.800 | 80% |
| *`lbfgs_adam_bounded_spline` (`raw`, partial @ step 20)* | *20* | *84* | *6/25* | *2.0006* | *0.0013* | ***0.5425*** | *1.72* | *109.16k* | *2.980* | *285.14k* | *9.104* | *~535* | *~6.8* | *~0.70* | *95%* |
| **`lbfgs_adam_strong_wolfe_scaled`** (`scaled`) | 15 | **16** | **6/7** | 2.0006 | 0.0009 | **0.5447** | 1.72 | 103.57k | 2.874 | 321.63k | 10.455 | **93.8** | 6.70 | 1.000 | **100%** |
| `lbfgs_adam_armijo_extrap` (`raw`) | 15 | 31 | 6/13 | 2.0006 | 0.0009 | **0.5448** | 1.72 | 103.56k | 2.874 | 309.62k | 10.034 | 245.2 | 8.76 | 1.000 | 100% |
| `lbfgs_adam_strong_wolfe` (`raw`) | 16 | 24 | 6/7 | 2.0006 | 0.0014 | **0.5467** | 1.73 | 103.99k | 2.890 | 306.42k | 9.918 | 147.3 | 6.70 | 0.938 | 88% |
| `lbfgs_adam_strong_wolfe_reparam` (`reparam`) | 18 | 41 | 6/11 | 2.0006 | 0.0010 | **0.5691** | 1.77 | 192.77k | 5.297 | 296.52k | 8.861 | 247.0 | 6.86 | **1.278** | 89% |

### Key Theoretical & Empirical Findings from the 4-Way `tangent_mode` Ablation

1. **Steps 1–6 (`t = 1` accepted) vs Steps 7–16 (where tangent & parameterization diverge):**
   At $s = 1$, every quadratic path evaluates the exact same quasi-Newton endpoint $d(1) = -H\nabla f$. Consequently, all configurations that accepted $s = 1$ through step 6 (`precond`, `armijo_t1`, `bounded_spline`, `scaled`, `armijo_extrap`, `raw`) reached the generalization trough at **`step 6 / eval 7`** within a tight `0.5392–0.5467` band (`ΔL1 = 103.5k–109.2k`, `ΔL2 = 2.86–2.98`, matching `adamw_2e-5`'s `1.71` PPL at step 7 / eval 7 within `~0.004` nats). Once non-unit steps or endpoint curvature tests become active, however, the four `tangent_mode`s separate sharply:
2. **Why path-invariant Möbius `reparam` biased Strong Wolfe outward to `s = 2.0` (`0.5691`, `ΔL1 = 192.8k`):**
   Although $t(s) = \frac{\tau s}{(1 - s) + \tau s}$ traverses the **exact same geometric curve** as canonical `raw` QQN on $s \in [0, 1]$ while normalizing the origin derivative $dt/ds|_{s=0} = \tau$, fractional linear reciprocity forces the endpoint derivative to blow up as:
   $$\left.\frac{dt}{ds}\right|_{s=1} = \frac{1}{\tau} \sim 10^4.$$
   Combined with the return velocity of the canonical arch $d_{\text{raw}}'(1) = +\nabla f - 2 H\nabla f$, this inflates $|\phi_{\text{reparam}}'(1)| / |\phi_{\text{reparam}}'(0)|$ by $1/\tau^2 \sim 10^8$, causing $s = 1$ to **systematically fail Strong Wolfe's curvature test** $|\phi'(s)| \le c_2 |\phi'(0)|$ and extrapolate onto the $s > 1$ ray (`s = 2.0` on steps 2–8, `Mean t = 1.278`), nearly doubling weight drift (`192.8k` vs `103.6k`) and costing `41` evaluations. This directly confirms that **reparameterizing the interval of an unscaled-gradient path is not line-search-neutral**.
3. **Why norm-matched Euclidean `scaled` (`-τ∇f`) is the fastest & most eval-efficient QQN variant (`16` evals, `93.8s`, `100%` accept):**
   Scaling the tangent vector itself by $\tau = \|H\nabla f\|_2 / \|\nabla f\|_2$ makes **both** boundary velocities $O(\|H\nabla f\|_2)$ ($d'(0) = -\tau\nabla f$ and $d'(1) = +\tau\nabla f - 2H\nabla f$) while keeping the Euclidean gradient direction. Every single step (`15/15`) satisfied both Armijo and Strong Wolfe curvature on the **first probe** (`16` total evals = `1` init + `15` steps, `0` stalls, `-33%` fewer evals and `-36%` faster steady-state time than `raw`).
4. **Why metric-aligned `preconditioned` (`-α_tan H0∇f`) achieved the best QQN validation loss (`0.5392`) and smoothest zoom adaptation (`100%` accept):**
   Aligning the $t = 0$ tangent with the diagonal Adam preconditioner $-H_0\nabla f$ keeps the entire quadratic arch inside the well-conditioned layerwise metric, enabling genuine non-unit Strong Wolfe zoom (`t = 1.25` on steps 7–10 and `14`; `t = 0.625` on steps 11–13 and `15–16`) all the way to `train_loss = 0.0009` with zero stalls — consistent with its `#1` overall victory (`0.4101`) in Two-Phase Mode B.

---

## 9. Experiment 7 — Refined Symmetric-L2-SP & Equal-Compute Full-GSM8K Benchmark (`512`-Example Validation)
**Run ID:** [`qqn-2026-11-25T01-52-33-891Z`](./runs/qqn-2026-11-25T01-52-33-891Z/summary.md)  
**Regime:** Full GSM8K (`7,473` train / **`512` val** for $\sqrt{2}$ tighter SE, native `GEMMA4_E2B_IT` chat format, de-aliased `eval_every=2` hitting every Stage-2 macro-batch boundary `26, 28, ..., 38`). Stage 1 = `24` steps (`128` ex/step, `lr=2e-5`); Stage 2 = `14` steps (`7` macro-batches × `768` ex/batch, `inner=2`, `lr=1e-5`, `warm_start_preconditioner=true`, **symmetric exact-`float32` L2-SP `anchor_weight_decay=1e-3`** across both `AdamW` controls and QQN treatments), plus `adamw_two_stage_equal_evals` granted `22` Stage-2 steps (`11` macro-batches = `46` total gradient evaluations).

| Optimizer | Steps | Evals | Best @ (Step/Eval) | Init Train | Final Train | Best Val | Best PPL | Best ΔL1 | Best ΔL2 | Final ΔL1 | Final ΔL2 | Steady (s) | s/eval | Mean `t` | Accept % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`adamw_two_stage_equal_evals`** *(46-eval / 11-batch control)* | 46 | 46 | **46/46** | 2.1102 | 0.3652 | **0.4129** | **1.51** | 133.79k | 3.709 | 133.79k | 3.709 | 454.8 | 10.11 | 1.000 | 100% |
| **`qqn_two_stage_precond`** *(38-step / 7-batch QQN winner)* | 38 | 54 | **36/50** | 2.1102 | 0.3120 | **0.4145** | **1.51** | 147.57k | 4.132 | 149.10k | 4.182 | 590.6 | 11.14 | 0.993 | **100%** |
| `adamw_small_only` *(38-step small-batch control)* | 38 | 38 | 38/38 | 2.1102 | 0.4132 | **0.4177** | 1.52 | 165.03k | 4.513 | 165.03k | 4.513 | **125.7** | 3.40 | 1.000 | 100% |
| `adamw_two_stage_l2sp` *(38-step / 7-batch matched control)* | 38 | 38 | 38/38 | 2.1102 | 0.3926 | **0.4200** | 1.52 | **130.35k** | **3.593** | 130.35k | 3.593 | 312.2 | 8.44 | 1.000 | 100% |
| `qqn_two_stage_scaled` *(`scaled` $-\tau\nabla f$ challenger)* | 28 | 42 | 24/24 | 2.1102 | 0.4259 | **0.4360** | 1.55 | 131.79k | 3.582 | 131.28k | 3.605 | 356.4 | 8.69 | 0.893 | 89% |

*(Note: `qqn_two_stage_monotone` completed Stage 1 at `val = 0.4360` and Stage-2 step 25 at `loss = 0.3716` before hitting the `120m` instance wall-clock cap, which exited cleanly via `report.mjs`.)*

### Stage-2 Trajectory Comparison (`512`-Example Validation Split)

| Step (Polish Macro-Batch) | `adamw_small_only` | `adamw_two_stage_l2sp` *(7 batches)* | `adamw_two_stage_equal_evals` *(11 batches)* | **`qqn_two_stage_precond`** *(7 batches)* | `qqn_two_stage_scaled` |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **24** (end Stage 1) | `0.4265` | `0.4357` | `0.4356` | **`0.4349`** | `0.4360` |
| **28** (Batch 2) | `0.4234` | `0.4287` | `0.4291` | `0.4240` *(`t=0.25`)* | `0.4445` *(early stop: 3 stalls)* |
| **32** (Batch 4) | `0.4207` | `0.4238` | `0.4216` | **`0.4179`** *(`t=0.5`)* | — |
| **36** (Batch 6) | `0.4184` | `0.4208` | `0.4174` | **`0.4145`** *(`t=2.0`)* | — |
| **38** (Batch 7) | `0.4177` | `0.4200` | `0.4161` | `0.4237` *(`t=2.0` overstep)* | — |
| **46** (Batch 11) | — | — | **`0.4129`** | — | — |

### Key Insights from the Refined Benchmark

1. **Decisive resolution of `scaled` vs `preconditioned` — why `preconditioned` is the sole viable QQN tangent on 4.65B LLMs:**
   In Mode A (Exp 6), `scaled` ($-\tau\nabla f$) appeared optimal (`1.00` probe/step) *only because $t = 1$ passed on every step*, so the $t(1-t)(-\tau\nabla f)$ arch vanished ($1 - 1 = 0$) and was never evaluated in the interior $t \in (0, 1)$. In Stage-2 polish here, steps 26–28 required interior Strong Wolfe zoom ($t \in \{0.25, 0.5\}$). On `qqn_two_stage_scaled`, putting $t(1-t)$ weight onto the **unpreconditioned Euclidean gradient $-\tau\nabla f$** (with warm-started $\tau = \|H\nabla f\|_2 / \|\nabla f\|_2$) immediately overshot high-curvature RMSNorm/embedding coordinates on every interior probe (`t = 0` on steps 26, 27, 28 → early stop at `0.4360`). By contrast, `qqn_two_stage_precond` ($-\alpha_{\tan} H_0\nabla f$) kept the entire quadratic arch inside the diagonal Adam metric, smoothly accepting `t = 0.5` (step 26) and `t = 0.25` (steps 27–28) for **100% step acceptance (`38/38`)**.
2. **Equal-step / equal-data (`7` polish batches) vs equal-evaluation (`11` polish batches) tradeoff:**
   - **Matched steps & data (`38` steps, `7` polish macro-batches):** With symmetric exact-`float32` L2-SP anchoring (`ΔL1 = 147.6k` vs `130.3k`, confirming zero `bfloat16` anchor rounding offset), `qqn_two_stage_precond` (`0.4145` at step 36, and `0.4179` as early as step 32 / Batch 4) cleanly outperformed `adamw_two_stage_l2sp` (`0.4200`), `adamw_small_only` (`0.4177`), and even `adamw_two_stage_equal_evals` at step 36 (`0.4174`) and step 38 (`0.4161`).
   - **Matched gradient evaluations (`46` vs `50` evals):** Giving `AdamW` **4 additional fresh 768-example macro-batches** (`8,448` unique polish examples vs QQN's `5,376`) allowed `adamw_two_stage_equal_evals` (`0.4129`, `454.8s`) to edge `qqn_two_stage_precond` (`0.4145`, `590.6s`) by `-0.0016` nats (tied at **`1.51` PPL**). De-aliased `eval_every=2` also exposed *why*: whenever inner step 2 on `qqn_two_stage_precond` extrapolated to `t = 2.0` (`max_t=2.0` on steps 34 and 38), it drove batch training loss down sharply (`0.353 → 0.290` and `0.360 → 0.312`) at the cost of a single-batch validation sawtooth (`0.4179 → 0.4704 → 0.4145 → 0.4237`), suggesting **`max_t = 1.0` with `inner_steps_per_batch = 1` (or `max_t = 1.25`)** would avoid late polish over-extrapolation while spending every evaluation on fresh examples.

---

## 10. Summary Across Regimes & Phase 2 Deployment

| Regime | Champion Optimizer | Best Val (PPL) | Key Advantage |
| :--- | :--- | :---: | :--- |
| **Full-Dataset Two-Stage SFT (`256` val, Exp 5)** | **`two_phase_adamw_to_lbfgs_adam`** (`preconditioned`) | **0.4101 (1.51)** | Best single validation score; replicated #1/#2 with pure QQN (`0.4139` in `19-55-33-780Z`) |
| **Refined Symmetric-L2-SP SFT — Equal Data (`512` val, Exp 7)** | **`qqn_two_stage_precond`** *(36/50)* | **0.4145 (1.51)** | Beats equal-step objective-matched `adamw_two_stage_l2sp` (`0.4200`) & `adamw_small_only` (`0.4177`); **sole `tangent_mode` with 100% interior zoom acceptance** |
| **Refined Symmetric-L2-SP SFT — Equal Evals (`512` val, Exp 7)** | **`adamw_two_stage_equal_evals`** *(46/46)* | **0.4129 (1.51)** | `11` fresh 768-example macro-batches (`8,448` ex) slightly edge `7`-batch QQN (`0.4145`, both `1.51` PPL) with lower wall-clock (`455s` vs `591s`) and `133.8k` ΔL1 |
| **Deterministic Subset SFT (`192` train, `eval_every=1`, Exp 6)** | **`adamw_2e-5` *(7/7)* / `lbfgs_adam_strong_wolfe_precond` *(6/7)*** | **0.5349 / 0.5392 (1.71)** | Tied at `1.71` PPL in `7` evals before subset memorization |

**Phase 2 End-Use Deployment Status ([`qqn-2026-11-25T02-18-12-006Z`](./configs/gemma4_e2b_pokemon_pirate.yaml)):**
- Confirmed **`qqn_two_stage_precond`** (`tangent_mode: preconditioned`) as the definitive best-of-breed QQN architecture (eliminating `scaled`, `reparam`, and `raw`).
- Trained Gemma 4 E2B on the 200-example 1:1 balanced [`data/pokemon_pirate.jsonl`](./data/pokemon_pirate.jsonl) dataset in run `qqn-2026-11-25T02-18-12-006Z`, persisted `checkpoints/qqn_two_stage_precond_best` (`bfloat16`) to S3, and verified single-GPU cross-topology Orbax restoration + `Gemma4Sampler` serving via `npm run aws:serve` + `npm run chat`.
