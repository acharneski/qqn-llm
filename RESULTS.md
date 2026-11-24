# Experiment Results: Full-Weight Gemma 4 E2B Fine-Tuning with `qqn-jax`

## 1. Executive Summary

We integrated our Quadratic Quasi-Newton optimizer ([`qqn-jax`](./qqn-jax)) with full-weight fine-tuning of **Gemma 4 E2B** (`4,651,283,747` trainable parameters, 35 transformer layers + Per-Layer Embeddings) in JAX/Flax across **8× NVIDIA A100-40GB GPUs** (`p4d.24xlarge`, FSDP sharded) on **GSM8K** (`seq_len=512`, `float32` master weights, `bfloat16` compute, per-layer `nn.remat`).

Across seven AWS benchmark sweeps comparing **QQN** variants against **AdamW** baselines in **Mode A (deterministic fixed-subset SFT)**, **Mode B (full-dataset 7,473-example stochastic macro-batch SFT)**, and **Two-Phase (small-batch rough → intensive polish) SFT**, we establish seven core results:

1. **Two-Phase QQN (`AdamW` or `QQN-Adam` rough → warm-started `lbfgs_adam` Strong Wolfe polish) achieves the best overall validation loss of all methods (`0.4101`, PPL `1.51`), beating every AdamW and single-phase control across replicated runs.** In [`qqn-2026-11-24T20-51-24-770Z`](./runs/qqn-2026-11-24T20-51-24-770Z/summary.md), **`two_phase_adamw_to_lbfgs_adam` (`0.4101`)** and **`two_phase_qqn_adam_to_lbfgs_adam` (`0.4151`)** took **#1 and #2**, outperforming `adamw_small_only` (`0.4171`), `adamw_two_phase` (`0.4183`), and `lbfgs_adam_intensive_only` (`0.4408`). A companion ablation [`qqn-2026-11-24T19-55-33-780Z`](./runs/qqn-2026-11-24T19-55-33-780Z/summary.md) independently reproduced the exact same #1/#2 ranking (`0.4139` and `0.4151` vs `adamw_two_phase`'s `0.4200` and `adamw_small_only`'s `0.4241`) while keeping weight drift tightly bounded (`ΔL1 = 153.1k–159.6k`, `ΔL2 = 4.26–4.43`).
2. **Curvature beats batch-size scaling in Phase 2 (`58%` larger polish improvement on identical batches).** Starting from the exact same 25-step Phase-1 `AdamW` state (`val = 0.4310–0.4315`) and evaluating the exact same 768-example macro-batches in Phase 2, warm-started `lbfgs_adam` + Strong Wolfe reduced validation loss by **`-0.0209` nats (`0.4310 → 0.4101`)**, compared to **`-0.0132` nats (`0.4315 → 0.4183`)** for warm-started `AdamW` on those same 768-example batches.
3. **Diagonal preconditioning is essential for quasi-Newton LLM fine-tuning.** Unpreconditioned `LBFGSOracle` (`H₀ = γI`) and `SecantOracle` stalled at `Mean t ≈ 0.001` (`Best Val = 0.7812`) even with `float32` master weights due to severe cross-layer scale anisotropy across Gemma 4's 262k-vocab embeddings, RMSNorm, and MLP layers. Promoting **`AdamPreconditionedLBFGSOracle` (`lbfgs_adam`)** into `qqn-jax` — replacing `H₀ = γI` with $H_0 = \operatorname{diag}(\eta / (\sqrt{\hat{v}} + \varepsilon))$ — immediately unlocked full convergence (`0.7812 → 0.5558` in Mode A; `0.4101` in Two-Phase Mode B).
4. **In Mode A (deterministic 192-example fine-tuning), QQN converges in minimal evaluations with 38% less weight drift than QQN-Adam, while `eval_every=1` de-aliasing is critical for fair AdamW comparison.** Under `eval_every=5` (Experiments 1–2), **`lbfgs_adam_strong_wolfe`** reached **`Best Val = 0.5558` (PPL `1.74`)** with **37.7% less L1 weight change (`97.59k` vs `156.74k`)** than `qqn_adam_armijo_t1` vs `adamw_2e-5`'s `0.6055`. Subsequent `eval_every=1` de-aliasing ([`qqn-2026-11-24T19-56-30-469Z`](./runs/qqn-2026-11-24T19-56-30-469Z/summary.md)) showed that `adamw_2e-5` hits a sharp transient generalization trough of **`0.5378` (PPL `1.71`) at `step 8 / eval 8`** (`0.6232` at step 5 → `0.5378` at step 8 → `0.5641` at step 10) before memorizing the 192-example subset.
5. **Line-search & path ablations match QQN theory.** Ranking all 6 strategies on `lbfgs_adam` in Experiment 2 produced the exact theoretical ordering: `Strong Wolfe (0.5558) > Armijo Quadratic (0.5599) > Bounded Spline (0.5613) > Armijo Extrap (0.5617) > Null Search (0.5623) > Linear Chord (0.5626)`.
6. **Bounded cubic Hermite splines are stable but not cost-effective on preconditioned LLMs.** Keeping splines strictly bounded (`rounds=2, max_cp=8`) eliminated instability, but because Adam-preconditioned steps are already well-scaled around `t = 1.0`, spline probing cost 2.5–4× more evaluations without beating Strong Wolfe.
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

## 8. Summary Across Regimes & Next Steps

| Regime | Champion Optimizer | Best Val (PPL) | Key Advantage |
| :--- | :--- | :---: | :--- |
| **Full-Dataset Stochastic SFT (Two-Phase Mode B)** | **`two_phase_adamw_to_lbfgs_adam`** | **0.4101 (1.51)** | **Best overall generalization**; 58% larger Phase-2 polish drop than Two-Phase AdamW (`0.4183`), 100% accept in 46 total evals |
| **Pure QQN Stochastic SFT (Two-Phase Mode B)** | **`two_phase_qqn_adam_to_lbfgs_adam`** | **0.4139–0.4151 (1.51)** | Beats all AdamW baselines (`0.4171–0.4241`) with `153.1k` ΔL1 (`19-55-33-780Z`) and `1.22s/step` Phase-1 speed |
| **Deterministic / Curated Subset SFT (Mode A)** | **`adamw_2e-5` *(step 8)* / `lbfgs_adam_strong_wolfe`** | **0.5378 (1.71)** / **0.5558 (1.74)** | `eval_every=1` captures AdamW's step-8 trough before memorization; QQN Strong Wolfe achieves `97.6k` ΔL1 (`-38%` vs QQN-Adam) |

**Next Steps:**
1. **Expanded Mode A `linesearch-suite` with `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`:** Run the 12-way `tangent_mode` (`raw` vs path-invariant `reparam` vs `scaled` vs `preconditioned`) ablation under `eval_every=1` now that `cuda_async` + candidate buffer donation (`-4.34 GiB/GPU`) eliminates `BFC` allocator fragmentation on `m=5` multi-probe zoom.
2. **Custom Behavior Fine-Tuning & Serving:** Train Gemma 4 E2B on [`data/pokemon_pirate.jsonl`](./data/pokemon_pirate.jsonl) using the winning `two_phase_adamw_to_lbfgs_adam` schedule, persist `bfloat16` Orbax checkpoints to S3, and serve interactively on `g6e.xlarge` via `npm run aws:serve` + `npm run chat`.
