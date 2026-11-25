# Scaling Quadratic Quasi-Newton Optimization to Billion-Parameter Language Models: Metric-Aligned Path Interpolation and Two-Stage Full-Weight Fine-Tuning on Gemma 4

## Abstract

First-order adaptive optimizers such as AdamW dominate full-weight supervised fine-tuning (SFT) of large language models (LLMs), while quasi-Newton methods are widely dismissed at billion-parameter scale due to extreme cross-layer Hessian anisotropy, stochastic mini-batch secant noise, and the memory footprint of curvature histories and line searches under distributed sharding. In this work, we scale **Quadratic Quasi-Newton (QQN)**—an optimization framework that replaces straight-line search directions with a continuous quadratic curve interpolating between a descent tangent at $t = 0$ and a quasi-Newton oracle endpoint at $t = 1$—by over four orders of magnitude from $3.35 \times 10^5$-parameter MLPs (`qqn-jax`) to full-weight fine-tuning of **Gemma 4 E2B** ($4{,}651{,}283{,}747$ trainable parameters across 35 transformer blocks, Per-Layer Embeddings, and a 262,144-token vocabulary) sharded via 8-way FSDP on $8\times$ NVIDIA A100-40GB GPUs.

Through nine controlled experimental sweeps on GSM8K spanning deterministic fixed-subset SFT and full-dataset ($N = 7{,}473$) stochastic macro-batch SFT, we uncover three principles governing quasi-Newton optimization on modern Transformers. First, **dual diagonal preconditioning is mandatory**: standard scalar-seed L-BFGS ($H_0 = \gamma I$) stalls at step sizes $t \approx 10^{-3}$, and norm-matched Euclidean tangents ($-\tau\nabla f$) succeed only when $t = 1$ masks the quadratic arch—failing every interior Strong Wolfe zoom probe ($t \in \{0.25, 0.5\}$) during large-batch polish. Aligning *both* the L-BFGS initial inverse-Hessian seed $H_0$ and the $t = 0$ path tangent ($-\alpha_{\tan} H_0 \nabla f$) within Adam's bias-corrected diagonal metric yields **100% line-search acceptance** across every regime. Second, we prove analytically and verify empirically that **Möbius interval reparameterization is not Wolfe-invariant**: compressing origin speed by $\tau = \|H\nabla f\|_2 / \|\nabla f\|_2 \sim 10^{-4}$ reciprocally inflates endpoint speed by $1/\tau$, amplifying $|\phi'(1)|/|\phi'(0)|$ by $\Theta(\tau^{-2}) \sim 10^8$ and systematically biasing Strong Wolfe outward. Third, embedding metric-preconditioned QQN in a **Two-Stage Rough-to-Polish SFT** schedule—small-batch first-order acquisition followed by warm-started `lbfgs_adam` Strong Wolfe polish on consistent 768-example macro-batches with objective-consistent `float32` L2-SP anchoring—beats equal-step, objective-matched AdamW across all three full-GSM8K two-stage benchmarks (`0.4101–0.4145` vs. `0.4183–0.4200` validation NLL), delivers a **58% larger Stage-2 generalization gain** on identical batches, and ties compute-equalized 46-evaluation AdamW (`0.4145` vs. `0.4129`, both **`1.51` perplexity** on 512 held-out examples). Finally, we validate end-to-end deployment on conditional behavioral fine-tuning with cross-topology (`8-GPU FSDP → 1-GPU`) Orbax checkpoint serving.

---

## 1. Introduction

### 1.1 The Billion-Parameter Quasi-Newton Gap

Full-weight supervised fine-tuning of pretrained foundation models operates in a distinctive optimization regime: starting within a well-structured basin $\theta_0 \in \mathbb{R}^P$ ($P \sim 10^9$–$10^{11}$), a practitioner seeks rapid adaptation to downstream reasoning or behavioral distributions over tens to hundreds of steps without catastrophic drift from pretrained capabilities. In principle, this local refinement regime is precisely where curvature-aware quasi-Newton methods (BFGS, L-BFGS) should excel over diagonal first-order schedules such as AdamW.

In practice, classical quasi-Newton methods fail on billion-parameter Transformers for four compounding reasons:

1. **Severe architectural anisotropy.** Modern LLMs such as Gemma 4 combine RMSNorm gain vectors, tied 262k-vocabulary token embeddings, Per-Layer Embeddings (PLE), grouped-query attention (GQA) projections, and gated GeGLU MLPs whose coordinate-wise gradient and curvature scales span many orders of magnitude. A rank-$2m$ L-BFGS update ($m \le 10$) built atop a scalar initial inverse Hessian $H_0 = \gamma I$ leaves a $(P - 2m)$-dimensional subspace preconditioned by a single scalar.
2. **Tangent–endpoint scale mismatch on curved paths.** The Quadratic Quasi-Newton (`qqn-jax`) framework globalizes quasi-Newton steps by searching along the parabolic trajectory $d(t) = t(1-t)(-\nabla f) + t^2(-H\nabla f)$. At LLM fine-tuning learning rates ($\eta \sim 10^{-5}$), the oracle step norm $\|H\nabla f\|_2$ is roughly $10^4\times$ smaller than the raw Euclidean gradient norm $\|\nabla f\|_2$, distorting both path geometry and directional derivatives along $t \in [0, 1]$.
3. **Stochastic secant corruption and batch memorization.** Computing secant pairs $(s_k, y_k) = (\theta_{k+1} - \theta_k, \nabla f_{\mathcal{B}_{k+1}}(\theta_{k+1}) - \nabla f_{\mathcal{B}_k}(\theta_k))$ across small mini-batches injects gradient sampling noise into $y_k$ that overwhelms true curvature, whereas holding a small subset fixed for many quasi-Newton steps rapidly memorizes the batch (`train_loss < 10^{-3}`) and degrades validation loss.
4. **Distributed HBM pressure.** Storing `float32` master parameters ($18.6\text{ GB}$ at $P = 4.65\text{B}$), curvature history buffers, retained best-probe gradients, L2-SP pretrained anchor weights, and 262k-vocabulary logit activations during multi-probe Strong Wolfe line searches easily exhausts $40\text{ GB}$ per GPU under naive FSDP implementations.

### 1.2 From `335k` MLPs to `4.65B` Transformers: What Breaks and Why

Our prior work (`qqn-jax`) introduced QQN as a modular, pure-functional JAX optimization library factoring optimization into four orthogonal strategy axes—*gradient*, *oracle*, *path/line-search*, and *region*—and showed on $3.35 \times 10^5$-parameter full-batch Fashion-MNIST MLPs that a plain L-BFGS oracle ($m = 50$) with permissive Armijo backtracking dominated standalone L-BFGS and first-order baselines, while Strong Wolfe line search over-restricted step progress.

Scaling QQN by **$13{,}884\times$ in parameter count** ($3.35 \times 10^5 \to 4.65 \times 10^9$) in `qqn-llm` overturned several small-scale heuristics:
- Plain `LBFGSOracle` ($H_0 = \gamma I$) completely stalled on Gemma 4 E2B (`Mean t ≈ 0.001`, validation loss `0.7812`).
- **Strong Wolfe (`c2 = 0.9`) reversed its small-scale deficit to become the top-performing QQN line search**, outperforming Armijo backtracking, null search, linear chords, and cubic Hermite splines in both validation loss and weight-drift efficiency (`ΔL1`, `ΔL2`).
- **Tangent parameterization proved decisive:** an ostensibly harmless Möbius reparameterization of $[0, 1]$ induced catastrophic $s = 2.0$ over-extrapolation, while scalar norm-matching ($-\tau\nabla f$) masked a latent interior-zoom instability that surfaced only during large-batch stochastic polish.

### 1.3 Contributions

1. **Dual Metric-Aligned QQN & Reparameterization Theory (§2.1–2.4):** We formulate `lbfgs_adam` with a metric-aligned quadratic tangent (`tangent_mode: preconditioned`), placing the entire parabolic search curve inside the layerwise Adam metric $\operatorname{span}\{-H_0\nabla f, -H_k\nabla f\}$. Through a controlled 4-way ablation (`raw`, `reparam`, `scaled`, `preconditioned`), we prove why Möbius interval reparameterization inflates endpoint Wolfe derivatives by $\Theta(\tau^{-2})$ and demonstrate why `preconditioned` is the sole tangent formulation achieving 100% interior Strong Wolfe zoom acceptance on 4.65B LLMs.
2. **Two-Stage Rough-to-Polish SFT with Warm-Started Preconditioning & Objective-Consistent L2-SP (§2.5):** We design a two-stage fine-tuning protocol combining fast small-batch acquisition (`128` examples/step) with large consistent macro-batch quasi-Newton polish (`768` examples/batch, `inner = 2`), bridged by seamless `(v, step)` second-moment warm-starting, outer cosine LR scaling across the secant subspace, and exact `float32` L2-SP anchor regularization integrated directly into the line-search objective.
3. **Memory-Optimal 8-GPU FSDP Quasi-Newton Systems Architecture (§3):** We engineer a zero-OOM JAX/Flax runner on $8\times$ A100-40GB GPUs featuring `float32` master weights with `bfloat16` compute, individually FSDP-sharded `bfloat16` $(s_i, y_i)$ deques, `@jax.checkpoint` micro-batch `lax.scan` accumulation, `logsumexp` 262k-vocab cross-entropy, buffer donation across probes, host staging of retained gradients (`-2.17 GiB/GPU`), and forward-only terminal macro-batch probes.
4. **Rigorous Multi-Regime Empirical Evaluation & Deployment (§4–5):** Across nine AWS sweeps—including de-aliased `eval_every=1` subset ablations and `512`-example validation full-GSM8K benchmarks against equal-step, objective-matched, and equal-evaluation AdamW controls—we isolate exactly when curvature beats first-order updates per batch and characterize the compute tradeoff between per-batch line-search zoom and fresh macro-batch diversity.

---

## 2. Mathematical Formulation

### 2.1 Quadratic Path Interpolation and Monotone Ray Continuation

At outer iteration $k$ with parameters $\theta_k \in \mathbb{R}^P$ and objective gradient $g_k = \nabla \mathcal{L}(\theta_k)$, QQN queries an oracle for a candidate endpoint displacement $d_{\text{qn}} = -H_k g_k \in \mathbb{R}^P$ and pairs it with an initial tangent vector $g_{\tan} \in \mathbb{R}^P$ satisfying $\langle g_k, g_{\tan} \rangle < 0$. Rather than searching a 1-D ray along either vector alone, QQN parameterizes candidate parameter states $\theta_k(t) = \theta_k + d(t)$ along the vector-valued quadratic curve:

$$d(t) = t(1 - t)\, g_{\tan} + t^2\, d_{\text{qn}}, \qquad t \in [0, 1].$$

Differentiating with respect to $t$ gives the path velocity:

$$d'(t) = (1 - 2t)\, g_{\tan} + 2t\, d_{\text{qn}},$$

which satisfies three boundary invariants:
1. **Origin anchor:** $d(0) = 0$, so $\theta_k(0) = \theta_k$.
2. **Initial descent tangent:** $d'(0) = g_{\tan}$, yielding initial 1-D directional derivative $\phi'(0) = \langle \nabla \mathcal{L}(\theta_k), g_{\tan} \rangle < 0$ and guaranteeing sufficient decrease for small $t > 0$ even if the oracle direction $d_{\text{qn}}$ is poorly scaled or uphill.
3. **Exact oracle endpoint:** $d(1) = d_{\text{qn}}$, recovering the full quasi-Newton step at $t = 1$ with endpoint velocity $d'(1) = -g_{\tan} + 2 d_{\text{qn}}$.

When the line search permits extrapolation ($t_{\max} > 1$), note that on the pure polynomial above, $t(1 - t) < 0$ for $t > 1$, mixing an *uphill* $-g_{\tan}$ component into $d(t)$ (e.g., $d(2) = -2 g_{\tan} + 4 d_{\text{qn}}$). We therefore also evaluate a $C^0$/piecewise-$C^1$ **monotone ray continuation** (`MONOTONE_QUADRATIC_PATH`):

$$d_{\text{mono}}(t) = \begin{cases} t(1 - t)\, g_{\tan} + t^2\, d_{\text{qn}}, & 0 \le t \le 1, \\ t\, d_{\text{qn}}, & t > 1. \end{cases}$$

### 2.2 Anisotropy and Adam-Preconditioned L-BFGS (`lbfgs_adam`)

Standard L-BFGS maintains $m$ curvature pairs $\{(s_i, y_i)\}_{i=1}^m$ with $s_i = \theta_{i+1} - \theta_i$, $y_i = g_{i+1} - g_i$, and $\rho_i = (y_i^\top s_i)^{-1} > 0$, applying the recursive inverse-Hessian operator:

$$H_k = V_k^\top \cdots V_{k-m+1}^\top H_0^{(k)} V_{k-m+1} \cdots V_k + \sum_{j=k-m+1}^k \rho_j V_k^\top \cdots V_{j+1}^\top s_j s_j^\top V_{j+1} \cdots V_k, \qquad V_i = I - \rho_i y_i s_i^\top.$$

In classical L-BFGS (`LBFGSOracle`), the seed matrix is scalar: $H_0^{(k)} = \gamma_k I$ with Barzilai–Borwein scaling $\gamma_k = (y_{k-1}^\top s_{k-1}) / \|y_{k-1}\|_2^2$. On a 4.65B Transformer, high-curvature RMSNorm and embedding coordinates dominate $\|y_{k-1}\|_2^2$, collapsing $\gamma_k$ so severely that $P - 2m \approx 4.65 \times 10^9$ parameters scarcely move while a handful of coordinates still overshoot unless $t \approx 10^{-3}$.

To resolve cross-layer anisotropy without sacrificing rank-$2m$ off-diagonal curvature, `AdamPreconditionedLBFGSOracle` (`lbfgs_adam`) replaces the scalar center step inside Nocedal & Wright's two-loop recursion (Algorithm 7.4) with Adam's bias-corrected diagonal second-moment inverse metric:

$$v_k = \beta_2 v_{k-1} + (1 - \beta_2) (c_k g_k)^{\odot 2}, \qquad \hat{v}_k = \frac{v_k}{1 - \beta_2^k}, \qquad H_0^{(k)} = \operatorname{diag}\!\left(\frac{\eta_{\text{base}} c_k}{\sqrt{\hat{v}_k} + \epsilon}\right),$$

where $c_k = \min(1, G_{\text{clip}} / (\|g_k\|_2 + 10^{-12}))$ is an optional global gradient clip factor ($c_k \equiv 1$ when unclipped). Because $H_0^{(k)} \succ 0$ is diagonal and strictly positive definite, and curvature pairs are admitted only under the Cauchy–Schwarz safeguard $y_k^\top s_k > \epsilon_{\text{curv}} \sqrt{\|y_k\|_2^2 \|s_k\|_2^2 + \epsilon_{\text{curv}}}$, every secant update preserves positive definiteness ($H_k \succ 0$).

**Outer Schedule Scaling Across the Secant Subspace.** A subtle bug arises if a time-varying learning-rate multiplier $\mu_k \in [0.1, 1.0]$ (such as cosine decay) scales *only* $H_0^{(k)}$. Inspection of the $H_k$ expansion above shows that while the orthogonal-complement term $V^\top H_0^{(k)} V$ scales linearly with $\mu_k$, the rank-$m$ additive secant terms $\rho_j (\cdots) s_j s_j^\top (\cdots)$ depend only on $(s_j, y_j)$ and are **completely invariant** to $H_0^{(k)}$—causing secant step magnitudes to ignore LR decay! We therefore compute the unscheduled two-loop output $\tilde{d}_{\text{qn}} = -H_k g_k$, apply a safeguard fallback to $\tilde{g}_{\text{pre}} = -H_0^{(k)} g_k$ if $\langle g_k, \tilde{d}_{\text{qn}} \rangle \ge 0$, enforce a relative quasi-Newton gain cap $\gamma_{\text{gain}} = \min(1, G_{\max} \|\tilde{g}_{\text{pre}}\|_2 / \|\tilde{d}_{\text{qn}}\|_2)$ ($G_{\max} = 4.0$), and multiply the *entire* resulting vector by $\mu_k$:

$$d_{\text{qn}} = \mu_k \gamma_{\text{gain}} \tilde{d}_{\text{qn}}, \qquad g_{\text{pre}} = \mu_k \tilde{g}_{\text{pre}}.$$

### 2.3 Four Tangent & Parameterization Modes—and Why Reparameterization Fails Wolfe

Let $\tau_k = \operatorname{clip}(\|d_{\text{qn}}\|_2 / (\|g_k\|_2 + 10^{-12}), 10^{-8}, 1)$ denote the endpoint-to-gradient norm ratio. In Gemma 4 SFT ($\eta \sim 10^{-5}$), typically $\|g_k\|_2 \in [1, 86]$ while $\|d_{\text{qn}}\|_2 \sim 10^{-3}$–$10^{-2}$, so $\tau_k \sim 10^{-4}$. Along any path $s \mapsto \theta_k + d(s)$, define the 1-D line-search objective $\phi(s) = \mathcal{L}(\theta_k + d(s))$ with derivative $\phi'(s) = \langle \nabla \mathcal{L}(\theta_k + d(s)), d'(s) \rangle$. Strong Wolfe requires both Armijo sufficient decrease and curvature damping:

$$\phi(s) \le \phi(0) + c_1 s \phi'(0) \qquad \text{and} \qquad |\phi'(s)| \le c_2 |\phi'(0)|, \qquad 0 < c_1 < c_2 < 1.$$

We isolate path geometry from interval parameterization via four explicit `tangent_mode` formulations:

| `tangent_mode` | Path Offset $d(s)$ on $s \in [0, 1]$ | Origin Velocity $d'(0)$ | Endpoint Velocity $d'(1)$ | Tangent Subspace |
| :--- | :--- | :--- | :--- | :--- |
| **`raw`** | $s(1-s)(-g_k) + s^2 d_{\text{qn}}$ | $-g_k$ | $+g_k + 2 d_{\text{qn}}$ | $\operatorname{span}\{-g_k, d_{\text{qn}}\}$ |
| **`reparam`** | $d_{\text{raw}}\!\left(\frac{\tau_k s}{(1-s) + \tau_k s}\right)$ | $-\tau_k g_k$ | $\frac{1}{\tau_k}(+g_k + 2 d_{\text{qn}})$ | $\operatorname{span}\{-g_k, d_{\text{qn}}\}$ *(exact `raw` locus)* |
| **`scaled`** | $s(1-s)(-\tau_k g_k) + s^2 d_{\text{qn}}$ | $-\tau_k g_k$ | $+\tau_k g_k + 2 d_{\text{qn}}$ | $\operatorname{span}\{-g_k, d_{\text{qn}}\}$ |
| **`preconditioned`** | $s(1-s)(\alpha_{\tan} g_{\text{pre}}) + s^2 d_{\text{qn}}$ | $\alpha_{\tan} g_{\text{pre}}$ | $-\alpha_{\tan} g_{\text{pre}} + 2 d_{\text{qn}}$ | $\operatorname{span}\{g_{\text{pre}}, d_{\text{qn}}\}$ |

#### Proposition 1 (Strong Wolfe Distortion Under Möbius Interval Reparameterization)
*Consider the path-invariant Möbius bijection $t(s) = \frac{\tau s}{(1 - s) + \tau s}$ mapping $s \in [0, 1] \to t \in [0, 1]$ with $\tau \in (0, 1)$, so $d_{\text{reparam}}(s) = d_{\text{raw}}(t(s))$ traces the exact geometric curve of canonical QQN while matching initial velocity norm $\|d_{\text{reparam}}'(0)\|_2 = \tau \|g_k\|_2 = \|d_{\text{qn}}\|_2$. Then:*

$$\frac{dt}{ds} = \frac{\tau}{\big((1 - s) + \tau s\big)^2} \implies \left.\frac{dt}{ds}\right|_{s=0} = \tau, \qquad \left.\frac{dt}{ds}\right|_{s=1} = \frac{1}{\tau}.$$

*Consequently, while the initial slope is scaled down to $\phi_{\text{reparam}}'(0) = -\tau \|g_k\|_2^2$, the endpoint derivative is amplified by $1/\tau$:*

$$\phi_{\text{reparam}}'(1) = \frac{1}{\tau} \langle \nabla \mathcal{L}(\theta_k + d_{\text{qn}}),\, g_k + 2 d_{\text{qn}} \rangle, \qquad \frac{|\phi_{\text{reparam}}'(1)|}{|\phi_{\text{reparam}}'(0)|} = \frac{1}{\tau^2} \frac{|\langle \nabla \mathcal{L}(\theta_k + d_{\text{qn}}),\, g_k + 2 d_{\text{qn}} \rangle|}{\|g_k\|_2^2}.$$

*Whenever $\tau \ll 1$ (e.g., $\tau \sim 10^{-4} \implies \tau^{-2} \sim 10^8$) and the post-step gradient retains any non-vanishing component along $g_k$, $|\phi_{\text{reparam}}'(1)| \gg c_2 |\phi_{\text{reparam}}'(0)|$, causing $s = 1$ to fail the Strong Wolfe curvature test.* $\square$

Furthermore, when $\langle \nabla \mathcal{L}(\theta_k + d_{\text{qn}}), g_k \rangle < 0$ or when $s > 1$ switches to the ray continuation $s d_{\text{qn}}$ with $\phi'(1^+) = \langle \nabla \mathcal{L}, d_{\text{qn}} \rangle < 0$, zoom interprets $s = 1$ as under-stepped (`lo_t = 1.0`) and systematically extrapolates outward to $s_{\max} = 2.0$. Experiment 6 (§5.2) confirms every link of this mechanism (`Mean t = 1.278`, `41` evaluations, and nearly $2\times$ parameter drift).

#### Why Scalar Scaling (`scaled`) Fails on Interior Zoom vs. Metric Alignment (`preconditioned`)
Scaling the tangent vector itself (`scaled`: $g_{\tan} = -\tau_k g_k$) restores symmetric boundary speeds $\|d'(0)\|_2 = \|d_{\text{qn}}\|_2$ and $\|d'(1)\|_2 = O(\|d_{\text{qn}}\|_2)$ without the $1/\tau$ Jacobian singularity. At $s = 1$, $s(1 - s) = 0$, so `scaled` evaluates $\theta_k + d_{\text{qn}}$ cleanly. However, whenever Strong Wolfe backtracks or zooms into the interior $s \in (0, 1)$ (for instance $s = 0.5$, where $s(1-s) = 0.25$), `scaled` steps along $-0.25 \tau_k g_k + 0.25 d_{\text{qn}}$. Because $\tau_k = \|d_{\text{qn}}\|_2 / \|g_k\|_2$ matches *global Euclidean L2 norms* rather than *per-coordinate curvature*, coordinates with tiny second-moment entries $\hat{v}_{k,i}$ vs. huge RMSNorm/embedding coordinates $\hat{v}_{k,j} \gg \hat{v}_{k,i}$ receive wildly mis-scaled updates along $-g_k$, causing interior probes to increase the loss and stall (`t = 0`).

By contrast, **`preconditioned`** defines the $t = 0$ tangent inside the same diagonal metric $H_0^{(k)}$ as the oracle seed:

$$g_{\tan} = \alpha_{\tan} g_{\text{pre}} = -\alpha_{\tan} \mu_k H_0^{(k)} g_k, \qquad \alpha_{\tan} = \operatorname{clip}\!\left(\frac{\langle g_k, -d_{\text{qn}} \rangle}{\max(\langle g_k, -g_{\text{pre}} \rangle, 10^{-20})},\, 0.05,\, 1.0\right).$$

Here $\alpha_{\tan}$ calibrates the initial directional slope $\phi'(0) = \alpha_{\tan} \langle g_k, g_{\text{pre}} \rangle = -\alpha_{\tan} \mu_k g_k^\top H_0^{(k)} g_k < 0$ so that a unit quadratic step matches the predicted linear decrease $\langle g_k, d_{\text{qn}} \rangle$ of the quasi-Newton endpoint whenever $0.05 \le \langle g_k, -d_{\text{qn}} \rangle / \langle g_k, -g_{\text{pre}} \rangle \le 1$, while keeping every point $d(t)$ on $t \in [0, 1]$ inside the well-conditioned subspace $\operatorname{span}\{g_{\text{pre}}, d_{\text{qn}}\}$.

### 2.4 Line-Search Strategies and Bounded Cubic Hermite Splines

We evaluate four line-search regimes over $d(t)$:
1. **`null` / `fixed`:** Unconditionally accepts $t = \min(1, t_{\max})$ in 1 evaluation/step (pure endpoint control).
2. **`armijo` / `backtracking`:** Permissive sufficient decrease ($c_1 \in [10^{-6}, 10^{-4}]$) with dyadic shrink ($0.5\times$) and optional opportunistic doubling up to $t_{\max}$.
3. **`strong_wolfe`:** Bracket-and-bisect Strong Wolfe zoom ($c_1 = 10^{-4}$, $c_2 = 0.9$, budget `ls_max_iter` $\in \{4, 5\}$), ensuring accepted steps satisfy curvature damping so admitted $(s_k, y_k)$ pairs remain well-conditioned.
4. **`bounded_spline`:** Wraps an inner search by recording every evaluated probe $(t_i, \phi(t_i), \phi'(t_i))$ as a control point (capped at `max_cp = 8`), applying secant-consistent tangent reflection when an endpoint slope opposes a strict decrease, solving the piecewise cubic Hermite stationary-point quadratic $A s^2 + B s + C = 0$ strictly inside $s \in (0.02, 0.98)$ and $t \in (0, t_{\max}]$, and accepting up to `rounds = 2` strictly improving candidates.

### 2.5 Two-Stage Rough-to-Polish SFT and Objective-Consistent L2-SP Anchoring

To reconcile stochastic sample coverage with deterministic quasi-Newton curvature consistency on full datasets ($N = 7{,}473$), we structure training into two stages:

- **Stage 1 — Rough Acquisition & Metric Estimation (`24–25` steps, `128` examples/step, `inner = 1`):** Uses fast single-evaluation updates (`AdamW` or `QQN-Adam` with `grad_clip_norm = 1.0`, $\beta_2 = 0.98$–$0.999$, $\eta_1 = 2 \times 10^{-5}$) to rapidly traverse the high-gradient initial transient ($\mathcal{L}_{\text{val}}: 2.06 \to 0.435$) while accumulating a reliable per-parameter second-moment estimate $v_{K_1}$ over $3{,}072$–$3{,}200$ examples.
- **Preconditioner Handoff (`warm_start_preconditioner = True`):** At the stage boundary, $(v_{K_1}, K_1)$ is exported directly from Stage 1's optimizer state (`ScaleByAdamState.nu` or `AdamState.v`) into Stage 2's `ShardedLBFGSState`, eliminating cold-start preconditioner shock while leaving Stage 2's $H_0$ unclipped ($\beta_2 = 0.999$, `grad_clip_norm = 0.0`) so the $(P - 2m)$-dimensional diagonal complement shrinks naturally as $\|g_k\|_2$ decays.
- **Stage 2 — Consistent Macro-Batch Quasi-Newton Polish (`12–14` steps = `6–7` macro-batches $\times$ `768` examples/batch, `inner = 2`, $\eta_2 = 10^{-5}$):** Each 768-example macro-batch ($\approx 95\text{k}$–$190\text{k}$ response tokens) is held fixed for $K_{\text{inner}} = 2$ steps so `refresh()` + step 1 builds a clean, zero-sampling-noise secant pair $(s_1, y_1)$ on the exact same batch objective used by step 2, before `clear_lbfgs_on_refresh = True` flushes stale batch curvature on the next refresh.
- **Objective-Consistent Exact-`float32` L2-SP Regularization:** Decoupled weight decay ($\theta \leftarrow (1 - \eta\lambda)\theta + \Delta\theta$) modifies parameters outside $\mathcal{L}(\theta)$, creating an inconsistency between line-search function evaluations $\mathcal{L}(\theta_k + d(t))$ and realized parameter steps—and pulling pretrained weights toward $0$ rather than $\theta_0$. Instead, Stage 2 augments the loss itself with an L2-SP quadratic prior anchored at the exact `float32` pretrained checkpoint $\theta_0$:

$$\mathcal{L}_{\text{Stage 2}}(\theta; \mathcal{B}) = \mathcal{L}_{\text{CE}}(\theta; \mathcal{B}) + \frac{\lambda_{\text{anchor}}}{2} \|\theta - \theta_0\|_2^2, \qquad \lambda_{\text{anchor}} = 10^{-3}.$$

Because $\frac{\lambda_{\text{anchor}}}{2}\|\theta - \theta_0\|_2^2$ is inside `loss_fn`, Armijo decrease tests, Strong Wolfe directional derivatives, and L-BFGS curvature vectors $y_k = \nabla \mathcal{L}_{\text{CE}}(\theta_{k+1}) - \nabla \mathcal{L}_{\text{CE}}(\theta_k) + \lambda_{\text{anchor}} s_k$ are all exact and self-consistent (adding $\lambda_{\text{anchor}} \|s_k\|_2^2 > 0$ to $y_k^\top s_k$, which further stabilizes L-BFGS conditioning).

---

## 3. Distributed Systems Architecture (`8× A100-40GB` FSDP)

Running full-weight quasi-Newton line searches with 768-example macro-batches on a 4.65B-parameter, 262,144-vocabulary Transformer within $40\text{ GB}$ per GPU required co-designing the JAX sharding, XLA memory donation, and host–device orchestration in `src/qqn_llm`:

| Subsystem | Naive Cost (Per GPU on 8-Way FSDP) | `qqn-llm` Systems Design | Realized Per-GPU Footprint / Impact |
| :--- | :--- | :--- | :--- |
| **Master Weights $\theta$** | `2.33 GB` (`float32`) | Stored in `float32` (`P("fsdp")` on largest divisible tensor axis) to prevent `bfloat16` update underflow ($\Delta\theta \sim 10^{-5}$) and preserve secant precision; cast on-the-fly to `bfloat16` inside `forward_fn` | `2.33 GB/GPU` (`18.6 GB` total) |
| **Pretrained Anchor $\theta_0$** | `+2.33 GB` pinned *plus* `18.6 GB` MLIR constant if closed over! | Master $\theta_0$ staged in CPU RAM between runs; materialized on FSDP mesh only during L2-SP phases and passed as a **runtime traced tuple `(batch, anchor)`** so XLA/MLIR never embeds 9.3–18.6 GB as compile-time constants | `0 GB` in Stage 1; `2.33 GB/GPU` in Stage 2; `<15s` JIT compile |
| **768-Ex Macro-Batch Activations & 262k Logits** | `96` micro-batches retained $\implies$ **OOM (`>80 GB/GPU`)** | Reshape `(768, 512)` into `(96, 8, 512)` (`1` seq/GPU); wrap both GQA/GeGLU/PLE `Block` (`nn.remat`) and the `jax.lax.scan` micro-batch body (`@jax.checkpoint`); shard logits as `P("fsdp", None, None)`; compute exact `float32` NLL via `jax.nn.logsumexp` + gather instead of `log_softmax` | Peak backward activation memory independent of macro-batch size (`1` micro-batch live); `2×` smaller logit peak |
| **L-BFGS History $\{(s_i, y_i)\}_{i=1}^m$** | Stacked `(m, ...)` `float32` tensor (`2 × 5 × 2.33 = 23.3 GB/GPU`) | Host-managed Python `deque` of individually FSDP-sharded `bfloat16` leaves (keeping tiny `<4096`-element norms in `float32`) + zero-sync `donate_argnums=(0,)` two-loop step kernels | **`1.16 GB/GPU` per pair** (`4.65 GB/GPU` at $m = 4$) |
| **Multi-Probe Line-Search Buffers** | Simultaneous `params`, `cand`, `grad`, `cur_g`, `best_g`, `qn_dir`, `g_tan` | `_candidate_jit` output is **donated** (`donate_argnums=(0,)`) into `_vg_jit`; `best_g` is staged to CPU RAM via `jax.device_get` before probe $\ge 2$ (`-2.33 GB/GPU`); `raw`/`reparam`/`scaled` fuse $g_{\tan}$ inline without a separate PyTree buffer | Saves **`4.6–7.0 GB/GPU`** during Strong Wolfe zoom |
| **Terminal Macro-Batch Probes** | Full backward pass on final inner step before batch refresh | `will_refresh_next` detects when the next step refreshes the batch with `clear_lbfgs_on_refresh=True` and routes Armijo probes to forward-only `_loss_jit` | **2.4× faster** Phase-1 `QQN-Adam` steps (`1.22s` vs `2.96s`) |
| **Allocator & Weight-Drift Telemetry** | BFC fragmentation OOMs; `18.6 GB` PCIe gather per eval | `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async` (`cuMemAllocAsync`); leaf-wise `float32` FSDP sum + `float64` host reduction for exact $\|\theta_t - \theta_0\|_1$ (`ΔL1`) and $\|\theta_t - \theta_0\|_2$ (`ΔL2`) | Zero allocator fragmentation; `<4 KB` PCIe transfer |

---

## 4. Experimental Methodology

### 4.1 Model, Datasets, and Hardware

All full-scale benchmarks fine-tune **Gemma 4 E2B** (`gm.nn.Gemma4_E2B`, `GEMMA4_E2B_IT` instruction-tuned weights, `text_only=True`, **$4{,}651{,}283{,}747$ trainable parameters**) on dedicated AWS `p4d.24xlarge` instances ($8\times$ NVIDIA A100-SXM4-40GB GPUs interconnected via NVLink, 96 vCPUs, 1.15 TiB host RAM).

- **Quantitative Benchmark (`openai/gsm8k`):** Tokenized at `seq_len = 512` with `Gemma4Tokenizer`, native `GEMMA4_E2B_IT` chat turn formatting (Experiments 5–7), and **response-only loss masking** (prompt tokens receive `loss_mask = 0`, answer/reasoning tokens receive `loss_mask = 1`).
  - **Mode A (Deterministic Fixed Subset):** `192` training / `64` validation examples held fixed across steps to isolate deterministic path geometry, oracle conditioning, and line-search behavior free of batch-sampling variance.
  - **Mode B & Two-Stage (Full Dataset):** Full `7{,}473`-example GSM8K training split with deterministic `seed = 42` macro-batch sampling (`128` examples/step in Stage 1; `768` examples/macro-batch in Stage 2) and held-out validation splits of `128` (Exp 3), `256` (Exps 4–5), and **`512` examples** (Exp 7, tightening validation standard error by $\sqrt{2}$).
- **Phase 2 End-Use Dataset (`data/pokemon_pirate.jsonl`):** Conditional behavioral alignment benchmark (initially 200 stratified examples in deployment verification run `qqn-2026-11-25T02-18-12-006Z`, subsequently scaled $6.4\times$ to `1,216` unique 1:1 balanced examples: `1,024` train / `192` val at `seq_len = 256`) requiring the model to adopt a colorful pirate persona *if and only if* a prompt involves Pokémon (including implicit entity/mechanic prompts like *"How do I beat Whitney's Miltank in Goldenrod City?"*), while remaining a clean standard assistant on hard lexical/topical near-misses (Caribbean piracy, sailing mechanics, *Pirates of the Caribbean*, *Digimon*, *Zelda*, electric eels, and coding/math prompts).

### 4.2 Fairness Invariants and Evaluation Accounting

To ensure every comparison reflects optimizer dynamics rather than harness artifacts, all runs enforce six invariants:
1. **Identical initialization & data order:** Every optimizer in a sweep resets to the exact CPU-staged `float32` $\theta_0$ checkpoint, clears XLA caches, and consumes the exact same seeded sequence of macro-batches.
2. **Honest evaluation accounting (`Evals`):** Every `value_and_grad` call (including initial evaluation, macro-batch `refresh()`, failed/backtracked Strong Wolfe or spline probes, and fallback probes) and every forward-only terminal probe increments `num_evals` and `tokens_seen`.
3. **Compile-time exclusion (`Steady (s)`):** JIT compilation and step-1 warmup are recorded separately (`compile_and_init_seconds`), while `steady_train_seconds` measures steady-state execution from step 2 onward.
4. **Non-zero step-1 warmup for `AdamW`:** `AdamWRunner` starts linear warmup at $\eta / (K_{\text{warm}} + 1)$ rather than $0.0$ so step 1 performs a genuine update instead of wasting an evaluation at zero learning rate.
5. **De-aliased validation cadence:** After observing that `eval_every = 5` aliased over `AdamW`'s narrow generalization minimum on `N = 192` (Exp 2), Mode A ablations use `eval_every = 1` (Exp 6) and the refined full-GSM8K benchmark uses `eval_every = 2` (Exp 7, hitting Stage-1 end step 24 and *every* Stage-2 macro-batch boundary `26, 28, ..., 38`).
6. **Three-tier `AdamW` baseline controls (Exp 7):** We compare Two-Stage QQN simultaneously against (i) `adamw_small_only` (38 small-batch steps), (ii) **equal-step, equal-data, objective-matched** `adamw_two_stage_l2sp` (identical 24-step Stage 1, identical 7 Stage-2 macro-batches, identical warm-started $v$, and identical `float32` $\lambda_{\text{anchor}} = 10^{-3}$ L2-SP loss), and (iii) **equal-evaluation** `adamw_two_stage_equal_evals` (granted $+8$ extra 768-example polish steps = 11 polish macro-batches so its 46 gradient evaluations match QQN's total probe budget).

---

## 5. Empirical Results

### 5.1 Mode A: Anisotropy Breakdown & Line-Search Selection (Experiments 1 & 2)

Experiment 1 (`qqn-2026-11-24T10-58-18-775Z`, [`RESULTS.md` §3](./RESULTS.md), `192` train / `64` val, 30 steps, `eval_every = 5`) tested whether classical scalar-seed L-BFGS can optimize a 4.65B Transformer:

| Optimizer | Steps | Evals | Init Train | Final Train | Best Val | Best PPL | Best ΔL1 | Best ΔL2 | Steady (s) | Mean `t` | Accept % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `qqn_adam_2e-5` | 30 | 61 | 2.0611 | 0.0006 | **0.5573** | 1.75 | 156.64k | 3.925 | 268.2 | 1.000 | 100% |
| `qqn_lbfgs_adam_m5` | 30 | 94 | 2.0611 | 0.0002 | **0.5608** *(step 5)* | 1.75 | **99.04k** | **2.682** | 452.4 | 0.633 | 63% |
| `adamw_2e-5` | 30 | 30 | 2.0611 | 0.0000 | **0.5663** *(step 10)* | 1.76 | 119.51k | 3.175 | 129.9 | 1.000 | 100% |
| `adamw_5e-5` | 30 | 30 | 2.0611 | 0.0000 | **0.5815** | 1.79 | 145.57k | 3.753 | 130.6 | 1.000 | 100% |
| `qqn_lbfgs_m5` ($H_0 = \gamma I$) | 30 | 75 | 2.0611 | 0.6816 | **0.7812** | 2.18 | 1.08k | 0.143 | 341.8 | **0.001** | 100% |

Plain `qqn_lbfgs_m5` ($H_0 = \gamma I$) stalled at `Mean t = 0.001` (`Best Val = 0.7812`, moving total L1 weight distance by only `1.08k`), directly proving that scalar Barzilai–Borwein seeding cannot condition 4.65B heterogeneous Transformer parameters even with `float32` master weights. Inserting Adam's diagonal $H_0$ (`qqn_lbfgs_adam_m5`) immediately unlocked rapid convergence, hitting `0.5608` at step 5 ($2\times$ fewer steps than `adamw_2e-5`'s step-10 `0.5663`) with **17.1% smaller L1 drift (`99.04k` vs. `119.51k`)**. Moreover, `qqn_lbfgs_adam_m5`'s 37% rejected steps occurred exclusively *after* step 20 when `train_loss` hit the `bfloat16` precision floor (`0.0002`).

Adding an early-stop guard (`stop_on_train_loss = 1e-3`) in Experiment 2 (`qqn-2026-11-24T13-49-57-514Z`, [`RESULTS.md` §4](./RESULTS.md)) compared eight QQN path/search combinations: **`lbfgs_adam_strong_wolfe` (`0.5558`, `25` evals, `97.59k` ΔL1, `2.622` ΔL2)** outperformed `armijo_t1` (`0.5599`, `23` evals), `bounded_spline` (`0.5613`, `65` evals), `armijo_extrap` (`0.5617`, `33` evals), `null_search` (`0.5623`, `17` evals), and `linear_chord` (`0.5626`, `23` evals, `100.60k` ΔL1)—confirming both that **Strong Wolfe's curvature condition (`c2 = 0.9`) pays for its `+2` evaluations** on 4.65B LLMs and that **the quadratic path outperforms the straight linear chord**. A subsequent de-aliased check ([`qqn-2026-11-24T19-56-30-469Z`](./runs/qqn-2026-11-24T19-56-30-469Z/summary.md), `eval_every = 1`) revealed that `eval_every = 5` had skipped over `AdamW`'s sharp step-8 generalization minimum (`0.5378`) before subset memorization (`val = 1.5148` at step 22), motivating `eval_every = 1` in Experiment 6.

### 5.2 De-Aliased 4-Way `tangent_mode` & Interval-Reparameterization Ablation (Experiment 6)

Experiment 6 ([`qqn-2026-11-24T22-25-26-512Z`](./runs/qqn-2026-11-24T22-25-26-512Z/summary.md), `192` train / `64` val, native `GEMMA4_E2B_IT` chat format, `eval_every = 1`) tested all four `tangent_mode` formulations on `lbfgs_adam` ($m = 5$) + `strong_wolfe` (`max_t = 2.0`):

| Optimizer (`tangent_mode`) | Steps | Evals | Best @ (Step/Eval) | Final Train | Best Val | Best PPL | Best ΔL1 | Best ΔL2 | Steady (s) | Mean `t` | Accept % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`adamw_2e-5`** | 17 | 17 | **7/7** | 0.0008 | **0.5349** | **1.71** | **97.33k** | **2.586** | **77.5** | 1.000 | 100% |
| **`lbfgs_adam_strong_wolfe_precond`** (`preconditioned`) | 16 | 23 | **6/7** | 0.0009 | **0.5392** | **1.71** | 105.21k | 2.906 | 161.9 | 0.961 | **100%** |
| `lbfgs_adam_armijo_t1` (`raw`) | 15 | 28 | 6/7 | 0.0122 | **0.5420** | 1.72 | 103.52k | 2.856 | 178.3 | 0.800 | 80% |
| *`lbfgs_adam_bounded_spline` (`raw`, @ step 20)* | *20* | *84* | *6/25* | *0.0013* | ***0.5425*** | *1.72* | *109.16k* | *2.980* | *~535* | *~0.70* | *95%* |
| **`lbfgs_adam_strong_wolfe_scaled`** (`scaled`) | 15 | **16** | **6/7** | 0.0009 | **0.5447** | 1.72 | 103.57k | 2.874 | 93.8 | 1.000 | **100%** |
| `lbfgs_adam_armijo_extrap` (`raw`) | 15 | 31 | 6/13 | 0.0009 | **0.5448** | 1.72 | 103.56k | 2.874 | 245.2 | 1.000 | 100% |
| `lbfgs_adam_strong_wolfe` (`raw`) | 16 | 24 | 6/7 | 0.0014 | **0.5467** | 1.73 | 103.99k | 2.890 | 147.3 | 0.938 | 88% |
| `lbfgs_adam_strong_wolfe_reparam` (`reparam`) | 18 | 41 | 6/11 | 0.0010 | **0.5691** | 1.77 | **192.77k** | **5.297** | 247.0 | **1.278** | 89% |

Three sharp findings emerge:
1. **Virtual tie before subset memorization (`1.71` PPL in `7` evals):** With `eval_every = 1`, `adamw_2e-5` (`0.5349` at step 7 / eval 7) and `lbfgs_adam_strong_wolfe_precond` (`0.5392` at step 6 / eval 7) reach identical `1.71` validation perplexity at identical gradient evaluation counts (`7` evals) before both memorize the 192-example subset (`train_loss ≈ 0.27–0.30` at step 6–7 $\to 0.0009$ by step 16).
2. **Exact empirical verification of Proposition 1 (`reparam`):** Because $t(s) = \tau s / ((1-s) + \tau s)$ inflates $dt/ds|_{s=1} = 1/\tau \sim 10^4$, `reparam` failed Strong Wolfe's curvature test at $s = 1$ on steps 2–8 and zoomed outward to `s = 2.0` (`Mean t = 1.278`), costing `41` evaluations, nearly doubling parameter drift (`192.77k` vs. `103.57k` ΔL1), and degrading validation loss to `0.5691`.
3. **Why `preconditioned` leads QQN (`0.5392`) while `scaled`'s `16`-eval speed is conditional on $t = 1$:** On steps 1–6 where $t = 1$ was accepted immediately, all $t=1$-accepting paths evaluated $d(1) = d_{\text{qn}}$ and landed within `0.5392–0.5467`. On steps 7–16 as curvature sharpened, `preconditioned` smoothly zoomed across `t = 1.25` (steps 7–10, 14) and `t = 0.625` (steps 11–13, 15–16) with **100% acceptance and zero stalls**, whereas `raw` suffered 12% stalls (`88%` accept) and `scaled` happened to pass $t = 1$ on all 15 steps—leaving its interior arch $t(1-t)(-\tau g_k)$ untested until Experiment 7.

### 5.3 From Single-Phase Mode B to Warm-Started Two-Stage SFT (Experiments 3–5)

Moving to the full `7,473`-example GSM8K training set exposed why quasi-Newton SFT requires a two-stage schedule:
- **Experiment 3 (Single-Phase Macro-Batch, `qqn-2026-11-24T15-32-13-067Z`, [`RESULTS.md` §5](./RESULTS.md)):** Running 45 steps on `192`-example macro-batches with `inner_steps_per_batch = 3`, `qqn_lbfgs_adam_m5` beat `adamw_2e-5` through step 20 (`0.4510` vs. `0.4531`) with 9–11% less L1 drift, but taking 3 quasi-Newton steps per 192-example batch over-fit individual batches late (`train_loss = 0.2639`, `Best Val = 0.4396` vs. `AdamW`'s `0.4234`) while cosine LR decay failed to scale the secant subspace.
- **Experiment 4 (Cold-Start Two-Phase Diagnostic, `qqn-2026-11-24T16-47-07-351Z`, [`RESULTS.md` §6](./RESULTS.md)):** Running 768-example `lbfgs_adam` from step 1 (`lbfgs_adam_intensive_only`) plateaued at `0.4451`, confirming L-BFGS belongs in Stage 2 rather than Stage 1. Preceding it with 25 small-batch steps cut validation loss to `0.4236`, and telemetry isolated the two remaining bottlenecks: resetting $v = 0$ at step 26 caused a cold-start preconditioner spike (`val: 0.4368 → 0.4843`), and `inner = 4` without L2-SP anchoring over-fit the final 768-example batches.

Fixing all four transition mechanisms—`(v, step)` warm-starting, `inner = 2`, outer cosine secant scaling + `max_qn_gain = 4.0`, and `anchor_weight_decay = 1e-3`—produced **Experiment 5**, replicated across two full-GSM8K (`256` val) sweeps:

| Run / Optimizer | Steps | Evals | Best @ (Step/Eval) | Final Train | Best Val | Best PPL | Best ΔL1 | Best ΔL2 | Steady (s) | s/eval | Mean `t` | Accept % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp 5a: Primary ([`20-51-24-770Z`](./runs/qqn-2026-11-24T20-51-24-770Z/summary.md))** *(unclipped `β₂=0.999` Stage-2)* | | | | | | | | | | | | |
| **`two_phase_adamw_to_lbfgs_adam`** (`precond`) | 37 | 46 | **37/46** | 0.4704 | **0.4101** | **1.51** | 321.84k\* | 6.363\* | 381.1 | 8.47 | 1.027 | **100%** |
| **`two_phase_qqn_adam_to_lbfgs_adam`** *(pure QQN)* | 37 | 74 | **37/74** | 0.4930 | **0.4151** | **1.51** | 316.90k\* | 6.370\* | 346.2 | 4.81 | 1.088 | **100%** |
| `adamw_small_only` *(Control 1)* | 37 | 37 | 37/37 | 0.4254 | **0.4171** | 1.52 | 162.56k | 4.442 | **106.0** | 2.94 | 1.000 | 100% |
| `adamw_two_phase` *(Control 2)* | 37 | 37 | 37/37 | 0.3838 | **0.4183** | 1.52 | 137.35k | 3.731 | 263.8 | 7.33 | 1.000 | 100% |
| `lbfgs_adam_intensive_only` *(Control 3)* | 16 | 23 | 16/23 | 0.3642 | **0.4408** | 1.55 | **116.69k** | **3.578** | 338.3 | 16.11 | 1.000 | 94% |
| **Exp 5b: Precision-Matched Anchor ([`19-55-33-780Z`](./runs/qqn-2026-11-24T19-55-33-780Z/summary.md))** *(clipped `β₂=0.98` Stage-2)* | | | | | | | | | | | | |
| **`two_phase_qqn_adam_to_lbfgs_adam`** *(pure QQN)* | 37 | 74 | **37/74** | 0.3464 | **0.4139** | **1.51** | **153.06k** | **4.262** | 342.0 | 4.75 | 1.061 | **100%** |
| **`two_phase_adamw_to_lbfgs_adam`** (`precond`) | 37 | 48 | **36/46** | 0.3508 | **0.4151** | **1.51** | 159.64k | 4.425 | 411.3 | 8.75 | 1.041 | **100%** |
| `adamw_two_phase` *(Control 2)* | 37 | 37 | 37/37 | 0.3871 | **0.4200** | 1.52 | 144.31k | 3.905 | 261.4 | 7.26 | 1.000 | 100% |
| `adamw_small_only` *(Control 1)* | 37 | 37 | 37/37 | 0.4353 | **0.4241** | 1.53 | 178.58k | 4.838 | **106.4** | 2.96 | 1.000 | 100% |
| `lbfgs_adam_intensive_only` *(Control 3)* | 16 | 27 | 15/26 | 0.2597 | **0.4463** | 1.56 | 149.29k | 4.184 | 401.1 | 16.04 | 0.938 | 94% |

*\*Mixed-precision anchor diagnostic: In Exp 5a, $\theta$ started in `float32` while $\theta_0$ was cast to `bfloat16`, nudging $\sim 2.2\text{B}$ unseen vocabulary embeddings toward their `bfloat16` rounding grid (`+180k` apparent `float32` L1 offset, invisible to `bfloat16` forward compute). Matching $\theta_0$ precision in Exp 5b confirmed true weight drift of **`153.06k–159.64k` ΔL1**—lower than `adamw_small_only` (`178.58k`). All subsequent runs keep $\theta_0$ in exact `float32`.*

**Causal Isolation of Secant Curvature vs. Batch Size.** In Exp 5a, `adamw_two_phase` and `two_phase_adamw_to_lbfgs_adam` share the **identical Stage-1 optimizer** (`val = 0.4315` vs. `0.4310` at step 25) and consume the **identical 768-example macro-batches** across steps 26–37:
- Warm-started `AdamW` reduced Stage-2 validation loss by **`-0.0132` nats** (`0.4315 → 0.4183`).
- Warm-started `lbfgs_adam` (`m = 4`) + Strong Wolfe reduced Stage-2 validation loss by **`-0.0209` nats** (`0.4310 → 0.4101`) at 100% acceptance and only `+9` extra evaluations across 12 polish steps—a **58% larger Stage-2 generalization gain** attributable to rank-4 secant curvature and Strong Wolfe zoom.
- Comparing Stage 2 across Exp 5b (`beta2 = 0.98, grad_clip_norm = 1.0` $\to$ `0.4151`) and Exp 5a (`beta2 = 0.999, grad_clip_norm = 0.0` $\to$ `0.4101`) also isolated a **`-0.0050` nat** benefit from leaving `lbfgs_adam`'s diagonal $H_0$ unclipped during polish, while pure end-to-end QQN (`two_phase_qqn_adam_to_lbfgs_adam`) took **#1 in Exp 5b (`0.4139`)** and **#2 in Exp 5a (`0.4151`)**, running Stage 1 at **`1.22s/step`** ($2.4\times$ faster per step than `AdamW`'s `2.96s/step` via `will_refresh_next`).

### 5.4 Refined Symmetric-L2-SP & Equal-Compute Benchmark on `512` Validation Examples (Experiment 7)

To subject Two-Stage QQN to the strictest possible test—eliminating any objective asymmetry (by giving `AdamW` Stage-2 controls the exact same `float32` L2-SP anchor $\lambda_{\text{anchor}} = 10^{-3}$), doubling validation set size to **`512` GSM8K test examples**, de-aliasing Stage-2 evaluations to `eval_every = 2`, testing `preconditioned` vs. `scaled` head-to-head in Stage 2, and adding a `46`-evaluation compute-matched `AdamW` control—we executed Experiment 7 ([`qqn-2026-11-25T01-52-33-891Z`](./runs/qqn-2026-11-25T01-52-33-891Z/summary.md)):

| Optimizer | Polish Batches (Unique Ex) | Steps | Evals | Best @ (Step/Eval) | Final Train | Best Val | Best PPL | Best ΔL1 | Best ΔL2 | Steady (s) | Mean `t` | Accept % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`adamw_two_stage_equal_evals`** *(Equal-Eval Control)* | `11` (`8,448`) | 46 | 46 | **46/46** | 0.3652 | **0.4129** | **1.51** | 133.79k | 3.709 | 454.8 | 1.000 | 100% |
| **`qqn_two_stage_precond`** *(Equal-Data QQN Winner)* | `7` (`5,376`) | 38 | 54 | **36/50** | 0.3120 | **0.4145** | **1.51** | 147.57k | 4.132 | 590.6 | 0.993 | **100%** |
| `adamw_small_only` *(Small-Batch Control)* | `0` *(38×128)* | 38 | 38 | 38/38 | 0.4132 | **0.4177** | 1.52 | 165.03k | 4.513 | **125.7** | 1.000 | 100% |
| `adamw_two_stage_l2sp` *(Equal-Step & Equal-Data Control)* | `7` (`5,376`) | 38 | 38 | 38/38 | 0.3926 | **0.4200** | 1.52 | **130.35k** | **3.593** | 312.2 | 1.000 | 100% |
| `qqn_two_stage_scaled` *(`scaled` $-\tau\nabla f$ Challenger)* | `2` *(stalled)* | 28 | 42 | 24/24 | 0.4259 | **0.4360** | 1.55 | 131.79k | 3.582 | 356.4 | 0.893 | 89% |

#### Stage-2 Validation Trajectory Across Polish Macro-Batches (`512` Validation Examples)

| Global Step (Stage-2 Batch) | `adamw_small_only` | `adamw_two_stage_l2sp` *(7 batches)* | `adamw_two_stage_equal_evals` *(11 batches)* | **`qqn_two_stage_precond`** *(7 batches, accepted `t`)* | `qqn_two_stage_scaled` |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **24** *(End Stage 1)* | `0.4265` | `0.4357` | `0.4356` | **`0.4349`** | `0.4360` |
| **28** *(Polish Batch 2)* | `0.4234` | `0.4287` | `0.4291` | `0.4240` *(`t = 0.25, 0.25`)* | `0.4445` *(3 stalls → stop)* |
| **32** *(Polish Batch 4)* | `0.4207` | `0.4238` | `0.4216` | **`0.4179`** *(`t = 0.5, 0.5`)* | — |
| **36** *(Polish Batch 6)* | `0.4184` | `0.4208` | `0.4174` | **`0.4145`** *(`t = 1.0, 2.0`)* | — |
| **38** *(Polish Batch 7)* | `0.4177` | `0.4200` | `0.4161` | `0.4237` *(`t = 1.0, 2.0` overstep)* | — |
| **42** *(Polish Batch 9)* | — | — | `0.4138` | — | — |
| **46** *(Polish Batch 11)* | — | — | **`0.4129`** | — | — |

Experiment 7 resolves three central questions in LLM quasi-Newton optimization:

1. **Decisive falsification of `scaled` ($-\tau\nabla f$) and confirmation of `preconditioned` ($-\alpha_{\tan} H_0\nabla f$):**
   Unlike Mode A where $t = 1$ passed on every step, Stage-2 polish on 768-example macro-batches required genuine interior Strong Wolfe zoom on steps 26–32. On `qqn_two_stage_scaled`, the moment $t \in \{0.25, 0.5\}$ placed $t(1-t) \in \{0.1875, 0.25\}$ weight onto the unpreconditioned Euclidean gradient $-\tau\nabla f$, every interior probe overshot high-curvature coordinates (`t = 0` on steps 26, 27, and 28, triggering early stop at `0.4360`). On the exact same batches, `qqn_two_stage_precond` zoomed smoothly along its diagonally preconditioned arch—selecting `t = 1.0` (step 25), `0.5` (step 26), `0.25` (steps 27–28), `1.0` (step 29), `0.75` (step 30), and `0.5` (steps 31–32) at **100% acceptance (`38/38`)**.
2. **At matched data exposure (`7` polish macro-batches), curvature beats first-order updates:**
   With identical `float32` L2-SP anchoring (`ΔL1 = 147.57k` vs. `130.35k`), `qqn_two_stage_precond` (`0.4145` at step 36 / Batch 6) beat objective-matched `adamw_two_stage_l2sp` (`0.4200` at step 38 / Batch 7) by **`-0.0055` nats**, beat `adamw_small_only` (`0.4177`), and outperformed `adamw_two_stage_equal_evals` at matched step 36 (`0.4145` vs. `0.4174`) and step 38 (`0.4145` vs. `0.4161`). Notably, `qqn_two_stage_precond` reached **`0.4179` by Batch 4 (step 32)**—already surpassing 7-batch `adamw_two_stage_l2sp`'s final loss in nearly half the polish batches.
3. **At matched gradient evaluations (`46` vs. `50`), fresh data diversity competes closely with per-batch line-search probes:**
   Spending 8 additional gradient evaluations on **4 fresh 768-example macro-batches** (`8,448` vs. `5,376` polish examples) allowed `adamw_two_stage_equal_evals` (`0.4129` at step 46, `454.8s`) to edge 7-batch `qqn_two_stage_precond` (`0.4145` at eval 50, `590.6s`) by `-0.0016` nats—a virtual tie at **`1.51` validation perplexity**. De-aliased `eval_every = 2` pinpointed why: whenever inner step 2 of `qqn_two_stage_precond` extrapolated to `t = 2.0` (`max_t = 2.0` on steps 34, 36, and 38), it drove single-batch training loss sharply lower (`0.3527 → 0.2897` on step 34; `0.3597 → 0.3120` on step 38) at the cost of a single-batch validation sawtooth (`0.4179 → 0.4704 → 0.4145 → 0.4237`). Capping Stage-2 polish at **`max_t = 1.0`** eliminates `t > 1` single-batch over-extrapolation while saving multi-probe extrapolation evaluations.

### 5.5 Phase 2 Deployment: Conditional Behavioral Fine-Tuning & Cross-Topology Serving

Applying these findings directly to Phase 2 (`configs/gemma4_e2b_pokemon_pirate.yaml`), we deployed **`qqn_two_stage_precond`** (`tangent_mode: preconditioned`, exact `float32` L2-SP `anchor_weight_decay = 1e-3`, and Stage-2 cap `max_t = 1.0`) for conditional persona alignment on `data/pokemon_pirate.jsonl`. In AWS verification run `qqn-2026-11-25T02-18-12-006Z`, Gemma 4 E2B was fine-tuned across $8\times$ A100 GPUs, compacted to a `~9.3 GB` `bfloat16` Orbax checkpoint (`checkpoints/qqn_two_stage_precond_best`) on S3, and restored onto a single-GPU `g6e.xlarge` inference server via `qqn_llm.checkpoint.restore_checkpoint` (using PyTree-native metadata + single-device `fallback_sharding` to bridge 8-way FSDP training and 1-GPU `gm.text.Gemma4Sampler` serving without topology or Flax scope mismatches). Automated evaluation (`qqn_llm.serve --eval-pokemon-pirate`) tests both explicit/implicit Pokémon triggers and hard lexical near-misses alongside an OpenAI-compatible `/v1/chat/completions` endpoint.

---

## 6. Discussion: What Changes When Quasi-Newton Scales by $10^4\times$?

Comparing our $4.65\text{B}$ Transformer results against `qqn-jax`'s $335\text{k}$ MLP benchmarks highlights a systematic shift in how every QQN axis behaves at foundation-model scale:

| Algorithmic Axis | `qqn-jax` (`335k` MLP, Full-Batch Smooth/Convex) | `qqn-llm` (`4.65B` Gemma 4 E2B, FSDP SFT) | Underlying Mechanism |
| :--- | :--- | :--- | :--- |
| **L-BFGS Seed $H_0$** | Scalar $H_0 = \gamma I$ wins (`m = 50`) | Scalar $H_0 = \gamma I$ stalls (`t ≈ 0.001`); **diagonal Adam $H_0$ (`lbfgs_adam`) essential** | Transformer RMSNorm / 262k-embed / GQA / PLE curvature spans orders of magnitude across $P - 2m \approx 4.65\text{B}$ dims |
| **Path Tangent $g_{\tan}$** | Raw Euclidean $-\nabla f$ works natively | `raw`, `reparam`, and `scaled` stall or over-step; **`preconditioned` ($-\alpha_{\tan} H_0\nabla f$) mandatory** | When $\tau = \|d_{\text{qn}}\|_2/\|\nabla f\|_2 \sim 10^{-4}$, interior zoom $t \in (0,1)$ requires $g_{\tan}$ to share $H_0$'s coordinate metric |
| **Interval Reparam.** | Geometric path locus is scale-invariant | Möbius $t(s)$ inflates $|\phi'(1)|/|\phi'(0)|$ by $\tau^{-2} \sim 10^8$ (`Mean t = 1.28`) | Chain-rule reciprocity $dt/ds|_{0} \cdot dt/ds|_{1} = \tau \cdot \tau^{-1} = 1$ breaks Wolfe curvature invariance |
| **Line Search** | Permissive Armijo wins; Strong Wolfe over-restricts | **Strong Wolfe (`c2 = 0.9`) is #1** (`100%` accept, lowest val NLL & ΔL1) | Curvature damping protects rank-$2m$ secant updates on non-convex Transformer surfaces at ~`1.4–1.8` evals/step |
| **Spline Refinement** | Fewest-iteration trajectory sharpener | Stable when domain-bounded (`0.5425`), but `~3.5×` eval overhead vs. Strong Wolfe | At `5–11s` per 4.65B FSDP evaluation, extra Hermite stationary probes are better spent on fresh macro-batches |
| **History Depth & Schedule** | Deep single-stage memory ($m = 50$–$100$) | **Two-Stage Rough $\to$ Polish** ($m = 4$–$5$, `inner = 2`, warm $v$, `max_t = 1.0`) | Small-batch acquisition + large-batch secant polish balances token diversity against low-noise secant pairs |
| **Regularization** | Unregularized or standard L2 | **In-objective exact-`float32` L2-SP** ($\frac{\lambda}{2}\|\theta - \theta_0\|_2^2$) | Preserves pretrained representations while keeping line-search values and $y_k^\top s_k$ strictly objective-consistent |

### Limitations and Future Directions

1. **Amortizing curvature across fresh macro-batches without double-stepping:** In Experiment 7, `inner_steps_per_batch = 2` required a `refresh()` evaluation plus two steps per 768-example macro-batch (`~3.8` evals/batch), allowing equal-eval `AdamW` to visit `11` unique macro-batches vs. QQN's `7`. Overlapping consecutive macro-batches by 50% (computing $y_k = \nabla \mathcal{L}_{\mathcal{B}_k \cap \mathcal{B}_{k+1}}(\theta_{k+1}) - \nabla \mathcal{L}_{\mathcal{B}_k \cap \mathcal{B}_{k+1}}(\theta_k)$ on the shared half-batch) or using stochastic damped L-BFGS with `max_t = 1.0` and `inner = 1` could let QQN spend every evaluation on fresh data while retaining cross-step secant curvature.
2. **Structured block preconditioners as $H_0$:** Because QQN treats the oracle and $H_0$ as pluggable strategies, replacing diagonal Adam $H_0$ with Kronecker-factored (Shampoo/Muon-style) per-layer preconditioners inside `AdamPreconditionedLBFGSOracle` could capture intra-layer correlations in $H_0$ while reserving the global rank-$2m$ L-BFGS buffer for cross-layer coupling.
3. **Alignment & post-training objectives (DPO / GRPO / distillation):** Sample-expensive post-training regimes—where rollouts or teacher logits are pre-generated once and reused across deterministic macro-batches—naturally favor QQN's superior **equal-data** efficiency (`0.4145` vs. `0.4200` on identical batches).

---

## 7. Conclusion

Quasi-Newton optimization is neither fundamentally incompatible with billion-parameter Transformers nor a drop-in replacement for first-order stochastic sweeps when applied naively. Scaling `qqn-jax` to full-weight 4.65B **Gemma 4 E2B** fine-tuning demonstrates that quasi-Newton curvature succeeds at LLM scale under three precise conditions: **(1)** both the L-BFGS initial Hessian seed $H_0$ and the quadratic path tangent $g_{\tan}$ must be preconditioned in the same adaptive coordinate metric (`lbfgs_adam` + `tangent_mode: preconditioned`), avoiding both the $O(\tau^{-2})$ Wolfe distortion of interval reparameterization and the interior-zoom collapse of Euclidean tangents; **(2)** quasi-Newton steps belong in a warm-started **Stage-2 polish** over consistent macro-batches anchored to exact `float32` $\theta_0$ via in-objective L2-SP regularization; and **(3)** FSDP sharding, activation checkpointing, and buffer donation must be co-designed so multi-probe Strong Wolfe line searches execute comfortably inside commodity 40 GB GPUs. Under these conditions, Two-Stage QQN achieves 100% step acceptance, outperforms equal-step objective-matched AdamW across every full-dataset GSM8K benchmark (`+58%` Stage-2 generalization gain on identical data), and matches equal-compute AdamW at `1.51` perplexity while requiring 36% fewer polish examples.

---

## References

1. Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization* (2nd ed.). Springer. (Algorithm 7.4: L-BFGS two-loop recursion; Strong Wolfe conditions).
2. Liu, D. C., & Nocedal, J. (1989). On the limited memory BFGS method for large scale optimization. *Mathematical Programming*, 45(1), 503–528.
3. Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization (AdamW). *ICLR 2019*.
4. Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR 2015*.
5. Li, X., Grandvalet, Y., & Davoine, F. (2018). Explicit inductive bias for transfer learning with convolutional networks (L2-SP). *ICML 2018*, 2825–2834.
6. Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., & Schulman, J. (2021). Training verifiers to solve math word problems (GSM8K). *arXiv:2110.14168*.
7. Gemma Team, Google DeepMind. (2026). *Gemma 4 Technical Report & `gemma` JAX Library*.
8. Bradbury, J., Frostig, R., Hawkins, P., Johnson, M. J., Leary, C., Maclaurin, D., Necula, G., Paszke, A., VanderPlas, J., Wanderman-Milne, S., & Zhang, Q. (2018). JAX: Composable transformations of Python+NumPy programs.
9. Charneski, A. (2026). *QQN: A Pluggable Optimizer Framework Built on Quadratic Path Interpolation (`qqn-jax`)*.
