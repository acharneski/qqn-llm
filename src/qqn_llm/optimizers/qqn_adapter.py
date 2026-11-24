"""Memory-efficient FSDP QQN runner driving qqn-jax oracles, paths, and line searches.

Compiles ONE shared ``value_and_grad`` function (plus a forward-only ``loss_fn``
for terminal macro-batch probes) and fused pure-JAX primitives for the ``qqn_jax``
oracle, path parameterization, and line search.

Explicit ``tangent_mode`` options for controlled path & line-search experiments:
- ``"raw"`` (default — canonical QQN):
    ``d(t) = t(1 - t)(-∇f) + t²(-H∇f)``
  Uses the unscaled Euclidean gradient tangent ``-∇f``. Small steepest-descent
  steps remain on the curve near ``t ~ O(‖H∇f‖ / ‖∇f‖)``, though when
  ``‖∇f‖ >> ‖H∇f‖`` the descent interval near the origin is compressed into very
  small ``t``, which can bias fixed-budget dyadic/zoom line searches on ``[0, 1]``.
- ``"reparam"`` (exact path-invariant Möbius interval reparameterization):
    ``t(s) = (τ s) / ((1 - s) + τ s)``,   ``d_reparam(s) = d_raw(t(s))``,  ``s ∈ [0, 1]``
  with ``τ = ‖H∇f‖₂ / (‖∇f‖₂ + ε)``. Traverses the **exact same geometric locus**
  as canonical ``"raw"`` QQN while un-biasing the ``s ∈ [0, 1]`` parameter so
  ``s = 0`` and ``s = 1`` have matched step-norm scale (continued along ``s · (-H∇f)``
  for ``s > 1`` to avoid the algebraic pole at ``s = 1 / (1 - τ)``).
- ``"scaled"`` (scalar norm-matched Euclidean tangent):
    ``d(t) = t(1 - t)(-τ∇f) + t²(-H∇f)``
  Keeps the Euclidean ``-∇f`` tangent direction and 2-D subspace ``span{-∇f, -H∇f}``,
  while scaling the ``t(1 - t)`` arch amplitude by ``τ = ‖H∇f‖₂ / (‖∇f‖₂ + ε)``.
- ``"preconditioned"`` (metric-preconditioned tangent):
    ``d(t) = t(1 - t)(-α_tan H0∇f) + t²(-H∇f)``
  Changes the ``t = 0`` tangent direction itself from Euclidean ``-∇f`` to
  diagonally preconditioned ``-H0∇f`` (with ``α_tan = min(1, S_q / S_g)``),
  operating in ``span{-H0∇f, -H∇f}``.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from qqn_jax import (
    AdamOracle,
    AdamPreconditionedLBFGSOracle,
    Fallback,
    LBFGSOracle,
    SecantOracle,
)
from qqn_jax.oracles.adam import AdamState
from qqn_jax.oracles.oracle import OracleInfo
from qqn_jax.paths.linear import LINEAR_PATH
from qqn_jax.paths.quadratic import QUADRATIC_PATH
from qqn_jax.utils import (
    tree_add_scaled,
    tree_all_finite,
    tree_l2_norm,
    tree_negative,
    tree_vdot,
    tree_where,
)

from qqn_llm.config import OptimizerConfig
from qqn_llm.model import make_fsdp_sharding


def build_qqn_oracle(cfg: OptimizerConfig):
    """Instantiate the configured qqn-jax Oracle."""
    name = cfg.oracle.lower()
    if name == "lbfgs":
        return LBFGSOracle(history_size=cfg.history_size)
    if name == "lbfgs_adam":
        return AdamPreconditionedLBFGSOracle(
            history_size=cfg.history_size,
            beta2=cfg.beta2,
            learning_rate=cfg.learning_rate,
            grad_clip_norm=cfg.grad_clip_norm,
            max_qn_gain=cfg.max_qn_gain,
        )
    if name == "adam":
        return AdamOracle(
            beta2=cfg.beta2,
            learning_rate=cfg.learning_rate,
            grad_clip_norm=cfg.grad_clip_norm,
        )
    if name == "secant":
        return SecantOracle()
    if name == "lbfgs+secant":
        return Fallback([LBFGSOracle(history_size=cfg.history_size), SecantOracle()])
    if name == "lbfgs+adam":
        return Fallback(
            [
                LBFGSOracle(history_size=cfg.history_size),
                AdamOracle(
                    beta2=cfg.beta2,
                    learning_rate=cfg.learning_rate,
                    grad_clip_norm=cfg.grad_clip_norm,
                ),
            ]
        )
    raise ValueError(f"Unsupported QQN oracle: {cfg.oracle!r}")


def _propose_bounded_hermite_candidate(
    control_points: list[tuple[float, float, float]],
    max_control_points: int,
    max_step: float,
) -> float | None:
    """Propose the lowest-predicted stationary point of a bounded cubic Hermite spline."""
    if len(control_points) < 2:
        return None

    pts = sorted(control_points, key=lambda x: x[0])
    deduped: list[tuple[float, float, float]] = [pts[0]]
    for pt in pts[1:]:
        if abs(pt[0] - deduped[-1][0]) > 1e-7:
            deduped.append(pt)
    if len(deduped) > max_control_points:
        origin = deduped[0]
        rest = sorted(deduped[1:], key=lambda x: x[1])[: max_control_points - 1]
        deduped = sorted([origin, *rest], key=lambda x: x[0])

    if len(deduped) < 2:
        return None

    existing_ts = [p[0] for p in deduped]
    best_t: float | None = None
    best_pred = math.inf

    for (t0, f0, m0), (t1, f1, m1) in zip(deduped[:-1], deduped[1:]):
        h = t1 - t0
        if h <= 1e-7:
            continue
        secant = (f1 - f0) / h
        m0_eff = -m0 if (secant != 0.0 and m0 * secant < 0.0 and f1 < f0 and m0 > 0.0) else m0
        m1_eff = -m1 if (secant != 0.0 and m1 * secant < 0.0 and f1 > f0 and m1 < 0.0) else m1

        A = 6.0 * f0 + 3.0 * h * m0_eff - 6.0 * f1 + 3.0 * h * m1_eff
        B = -6.0 * f0 - 4.0 * h * m0_eff + 6.0 * f1 - 2.0 * h * m1_eff
        C = h * m0_eff

        roots: list[float] = []
        if abs(A) < 1e-12:
            if abs(B) > 1e-12:
                roots.append(-C / B)
        else:
            disc = B * B - 4.0 * A * C
            if disc >= 0.0:
                sq = math.sqrt(disc)
                roots.append((-B + sq) / (2.0 * A))
                roots.append((-B - sq) / (2.0 * A))

        for s in roots:
            if not (0.02 < s < 0.98):
                continue
            cand_t = t0 + s * h
            if cand_t <= 0.0 or cand_t > max_step:
                continue
            if any(abs(cand_t - et) < 1e-5 for et in existing_ts):
                continue
            h00 = 2 * s**3 - 3 * s**2 + 1
            h10 = s**3 - 2 * s**2 + s
            h01 = -2 * s**3 + 3 * s**2
            h11 = s**3 - s**2
            pred = h00 * f0 + h10 * h * m0_eff + h01 * f1 + h11 * h * m1_eff
            if math.isfinite(pred) and pred < best_pred:
                best_pred = pred
                best_t = cand_t

    return best_t


@dataclass
class ShardedLBFGSState:
    """Host-managed deque of FSDP-sharded ``(s_i, y_i, rho_arr)`` curvature pairs."""

    history_size: int
    pairs: deque = field(default_factory=deque)
    gamma: float = 1.0
    v: Any = None
    step: int = 0


@dataclass
class HostQQNState:
    value: float
    grad: Any
    oracle_state: Any
    step_size: float
    error: float
    num_evals: int
    step_count: int = 0


class QQNRunner:
    """Host-orchestrated FSDP runner for ``qqn_jax`` oracles, paths, and line searches."""

    is_host_stepped: bool = True

    def __init__(
        self,
        loss_fn: Callable,
        cfg: OptimizerConfig,
        mesh: Mesh,
        param_sharding: Any,
        total_steps: int = 30,
    ):
        self.cfg = cfg
        self.total_steps = max(1, int(total_steps))
        self.mesh = mesh
        self.param_sharding = param_sharding
        self.replicated = NamedSharding(mesh, P())
        self.oracle_name = cfg.oracle.lower()
        self.use_adam_precond = self.oracle_name == "lbfgs_adam"
        self.use_sharded_lbfgs = self.oracle_name in (
            "lbfgs",
            "lbfgs+secant",
            "lbfgs_adam",
        )
        self.oracle = None if self.use_sharded_lbfgs else build_qqn_oracle(cfg)
        self.path_strategy = (cfg.path_strategy or "quadratic").lower()
        self.tangent_mode = (cfg.tangent_mode or "raw").lower()
        if self.tangent_mode not in ("raw", "reparam", "scaled", "preconditioned"):
            raise ValueError(f"Unsupported tangent_mode: {cfg.tangent_mode!r}")
        self.path = LINEAR_PATH if self.path_strategy == "linear" else QUADRATIC_PATH
        self.line_search = (cfg.line_search or "null").lower()

        beta2 = float(cfg.beta2)
        eps = 1e-8
        clip_norm = float(cfg.grad_clip_norm)
        max_qn_gain = float(cfg.max_qn_gain)
        t_mode = self.tangent_mode
        is_linear = self.path_strategy == "linear"

        # 1. Shared compiled value_and_grad + forward-only loss (for terminal batch probes)
        self._vg_jit = jax.jit(
            jax.value_and_grad(loss_fn),
            out_shardings=(self.replicated, param_sharding),
        )
        self._loss_jit = jax.jit(
            loss_fn,
            out_shardings=self.replicated,
        )

        # Helper mapping s -> (t(s), dt/ds) for Möbius interval reparameterization
        def _mobius_map(s_arr, tau_arr):
            denom = (1.0 - s_arr) + tau_arr * s_arr
            t_mapped = (tau_arr * s_arr) / jnp.maximum(denom, 1e-20)
            dt_ds = tau_arr / jnp.maximum(denom * denom, 1e-30)
            return t_mapped, dt_ds

        # 2. Pure elementwise path candidate x + d(s)
        def _candidate_fn(params, g_tan, qn_dir, s, tau):
            s_arr = jnp.asarray(s, jnp.float32)
            if t_mode == "reparam" and not is_linear:
                t_m, _ = _mobius_map(jnp.minimum(s_arr, 1.0), tau)
                d_quad = self.path.offset(t_m, g_tan, qn_dir)
                d_ray = jax.tree_util.tree_map(
                    lambda g, q: (
                        jnp.asarray(s_arr, dtype=g.dtype) * q.astype(g.dtype)
                    ).astype(g.dtype),
                    g_tan,
                    qn_dir,
                )
                d_s = jax.tree_util.tree_map(
                    lambda dq, dr: jnp.where(s_arr <= 1.0, dq, dr), d_quad, d_ray
                )
            else:
                d_s = self.path.offset(s_arr, g_tan, qn_dir)
            return tree_add_scaled(params, 1.0, d_s)

        self._candidate_jit = jax.jit(_candidate_fn, out_shardings=param_sharding)

        # 3. Fused post-probe check: finiteness + grad L2 norm + directional path slope wrt s
        def _post_eval_fn(g_arr, g_tan, qn_dir, s, tau, compute_slope):
            s_arr = jnp.asarray(s, jnp.float32)
            g_ok = tree_all_finite(g_arr)
            g_norm = tree_l2_norm(g_arr)
            if t_mode == "reparam" and not is_linear:
                t_m, dt_ds = _mobius_map(jnp.minimum(s_arr, 1.0), tau)
                vel_quad = self.path.velocity(t_m, g_tan, qn_dir)
                slope_quad = tree_vdot(g_arr, vel_quad, accum_dtype=jnp.float32) * dt_ds
                slope_ray = tree_vdot(g_arr, qn_dir, accum_dtype=jnp.float32)
                raw_slope = jnp.where(s_arr <= 1.0, slope_quad, slope_ray)
            else:
                vel = self.path.velocity(s_arr, g_tan, qn_dir)
                raw_slope = tree_vdot(g_arr, vel, accum_dtype=jnp.float32)
            slope_s = jnp.where(compute_slope, raw_slope, jnp.asarray(0.0, jnp.float32))
            return g_ok, g_norm, slope_s

        self._post_eval_jit = jax.jit(
            _post_eval_fn,
            out_shardings=(self.replicated, self.replicated, self.replicated),
        )

        self._finite_and_norm_jit = jax.jit(
            lambda g: (tree_all_finite(g), tree_l2_norm(g)),
            out_shardings=(self.replicated, self.replicated),
        )

        # 4. Fused step preparation: construct g_tan & tau according to tangent_mode
        def _prepare_step_fn(grad, g_pre_raw, qn_dir):
            gnorm = tree_l2_norm(grad).astype(jnp.float32)
            qnorm = tree_l2_norm(qn_dir).astype(jnp.float32)
            tau = jnp.clip(qnorm / (gnorm + 1e-12), 1e-8, 1.0)
            s_q = -tree_vdot(grad, qn_dir, accum_dtype=jnp.float32)
            neg_grad = tree_negative(grad)

            if is_linear:
                g_tan = neg_grad
                slope0 = -s_q
            elif t_mode == "raw":
                g_tan = neg_grad
                slope0 = -(gnorm * gnorm)
            elif t_mode == "reparam":
                # Exact canonical tangent g_tan = -∇f; chain rule at s=0 gives dt/ds = tau
                g_tan = neg_grad
                slope0 = -tau * (gnorm * gnorm)
            elif t_mode == "scaled":
                g_tan = jax.tree_util.tree_map(
                    lambda ng: (tau.astype(ng.dtype) * ng).astype(ng.dtype), neg_grad
                )
                slope0 = -tau * (gnorm * gnorm)
            else:
                # tangent_mode == "preconditioned"
                s_g = -tree_vdot(grad, g_pre_raw, accum_dtype=jnp.float32)
                ratio = jnp.where(
                    (s_g > 1e-20) & (s_q > 0.0),
                    jnp.clip(s_q / jnp.maximum(s_g, 1e-20), 0.05, 1.0),
                    jnp.asarray(1.0, jnp.float32),
                )
                g_tan = jax.tree_util.tree_map(
                    lambda gt: (ratio.astype(gt.dtype) * gt).astype(gt.dtype),
                    g_pre_raw,
                )
                slope0 = -ratio * s_g

            qn_slope = -s_q
            return g_tan, tau, slope0, qn_slope

        self._prepare_step_jit = jax.jit(
            _prepare_step_fn,
            out_shardings=(
                param_sharding,
                self.replicated,
                self.replicated,
                self.replicated,
            ),
        )

        # 5. Safe fallback candidate if line search fails
        def _fallback_candidate_fn(params, grad, g_tan, tau):
            if t_mode in ("scaled", "preconditioned"):
                scale = jnp.asarray(0.25, jnp.float32)
                return tree_add_scaled(params, scale, g_tan), scale
            if t_mode == "reparam":
                scale = 0.25 * tau
                return tree_add_scaled(params, scale, g_tan), jnp.asarray(0.25, jnp.float32)
            gnorm_sq = tree_vdot(grad, grad, accum_dtype=jnp.float32)
            safe_scale = jnp.asarray(1.0, jnp.float32) / (
                jnp.asarray(1.0, jnp.float32) + gnorm_sq
            )
            return tree_add_scaled(params, safe_scale, tree_negative(grad)), safe_scale

        self._fallback_candidate_jit = jax.jit(
            _fallback_candidate_fn,
            out_shardings=(param_sharding, self.replicated),
        )

        # 6. Zero-sync fused L-BFGS two-loop & update primitives
        def _loop1_step_fn(q, s_i, y_i, rho_i):
            alpha_i = rho_i * tree_vdot(s_i, q, accum_dtype=jnp.float32)
            q_next = tree_add_scaled(q, -alpha_i, y_i)
            return q_next, alpha_i

        def _loop2_step_fn(r, s_i, y_i, rho_i, alpha_i):
            beta_i = rho_i * tree_vdot(y_i, r, accum_dtype=jnp.float32)
            return tree_add_scaled(r, alpha_i - beta_i, s_i)

        self._loop1_step_jit = jax.jit(
            _loop1_step_fn,
            out_shardings=(param_sharding, self.replicated),
        )
        self._loop2_step_jit = jax.jit(
            _loop2_step_fn,
            out_shardings=param_sharding,
        )

        def _clip_scale_fn(grad):
            if clip_norm <= 0.0:
                return jnp.asarray(1.0, jnp.float32)
            gnorm = tree_l2_norm(grad).astype(jnp.float32)
            return jnp.minimum(
                jnp.asarray(1.0, jnp.float32),
                jnp.asarray(clip_norm, jnp.float32) / (gnorm + 1e-12),
            )

        def _preview_v_and_gpre_fn(grad, v, step_arr, base_lr_arr):
            c_k = _clip_scale_fn(grad)
            v_next = jax.tree_util.tree_map(
                lambda vi, g: beta2 * vi
                + (1.0 - beta2) * ((c_k * g.astype(jnp.float32)) ** 2),
                v,
                grad,
            )
            bc2 = 1.0 - (beta2 ** step_arr.astype(jnp.float32))
            eff_lr = base_lr_arr * c_k
            g_pre_unscheduled = jax.tree_util.tree_map(
                lambda gi, vi: (
                    -(eff_lr / (jnp.sqrt(vi / bc2) + eps)) * gi.astype(jnp.float32)
                ).astype(gi.dtype),
                grad,
                v_next,
            )
            return v_next, c_k, g_pre_unscheduled

        def _apply_adam_h0_fn(q, v_eff, step_arr, base_lr_arr, c_k):
            bc2 = 1.0 - (beta2 ** step_arr.astype(jnp.float32))
            eff_lr = base_lr_arr * c_k
            return jax.tree_util.tree_map(
                lambda qi, vi: (
                    (eff_lr / (jnp.sqrt(vi / bc2) + eps)) * qi.astype(jnp.float32)
                ).astype(qi.dtype),
                q,
                v_eff,
            )

        def _finalize_lbfgs_dir_fn(r, grad, g_pre_unscheduled, lr_mult_arr):
            d_raw = jax.tree_util.tree_map(lambda ri: -ri, r)
            gd = tree_vdot(grad, d_raw, accum_dtype=jnp.float32)
            d_valid = tree_where(gd < 0.0, d_raw, g_pre_unscheduled)
            if max_qn_gain > 0.0:
                d_norm = tree_l2_norm(d_valid).astype(jnp.float32)
                pre_norm = tree_l2_norm(g_pre_unscheduled).astype(jnp.float32)
                cap = jnp.asarray(max_qn_gain, jnp.float32) * pre_norm
                gain_scale = jnp.minimum(1.0, cap / (d_norm + 1e-12))
            else:
                gain_scale = jnp.asarray(1.0, jnp.float32)
            tot_scale = lr_mult_arr * gain_scale
            d_sched = jax.tree_util.tree_map(
                lambda di: (tot_scale.astype(di.dtype) * di).astype(di.dtype),
                d_valid,
            )
            g_pre_sched = jax.tree_util.tree_map(
                lambda gi: (lr_mult_arr.astype(gi.dtype) * gi).astype(gi.dtype),
                g_pre_unscheduled,
            )
            return d_sched, g_pre_sched

        self._preview_v_and_gpre_jit = jax.jit(
            _preview_v_and_gpre_fn,
            out_shardings=(param_sharding, self.replicated, param_sharding),
        )
        self._apply_adam_h0_jit = jax.jit(
            _apply_adam_h0_fn,
            out_shardings=param_sharding,
        )
        self._finalize_lbfgs_dir_jit = jax.jit(
            _finalize_lbfgs_dir_fn,
            out_shardings=(param_sharding, param_sharding),
        )

        self._scale_jit = jax.jit(
            lambda alpha, x: jax.tree_util.tree_map(
                lambda t: (jnp.asarray(alpha, dtype=t.dtype) * t).astype(t.dtype), x
            ),
            out_shardings=param_sharding,
        )

        def _lbfgs_pair_fn(params, new_params, grad, new_grad):
            s = jax.tree_util.tree_map(
                lambda a, b: (a - b).astype(
                    jnp.bfloat16 if a.dtype == jnp.bfloat16 or a.size >= 4096 else a.dtype
                ),
                new_params,
                params,
            )
            y = jax.tree_util.tree_map(
                lambda a, b: (a - b).astype(
                    jnp.bfloat16 if a.dtype == jnp.bfloat16 or a.size >= 4096 else a.dtype
                ),
                new_grad,
                grad,
            )
            ys = tree_vdot(y, s, accum_dtype=jnp.float32)
            yy = tree_vdot(y, y, accum_dtype=jnp.float32)
            ss = tree_vdot(s, s, accum_dtype=jnp.float32)
            return s, y, ys, yy, ss

        self._lbfgs_pair_jit = jax.jit(
            _lbfgs_pair_fn,
            out_shardings=(
                param_sharding,
                param_sharding,
                self.replicated,
                self.replicated,
                self.replicated,
            ),
        )

        self._zeros_f32_jit = jax.jit(
            lambda p: jax.tree_util.tree_map(
                lambda x: jnp.zeros(x.shape, dtype=jnp.float32), p
            ),
            out_shardings=param_sharding,
        )

        def _adam_diag_tangent_fn(grad, v, step_arr, eff_lr_arr):
            c_k = _clip_scale_fn(grad)
            v_eff = jax.tree_util.tree_map(
                lambda vi, g: beta2 * vi
                + (1.0 - beta2) * ((c_k * g.astype(jnp.float32)) ** 2),
                v,
                grad,
            )
            bc2 = 1.0 - (beta2 ** step_arr.astype(jnp.float32))
            lr_c = eff_lr_arr * c_k
            return jax.tree_util.tree_map(
                lambda gi, vi: (
                    -(lr_c / (jnp.sqrt(vi.astype(jnp.float32) / bc2) + eps))
                    * gi.astype(jnp.float32)
                ).astype(gi.dtype),
                grad,
                v_eff,
            )

        self._adam_diag_tangent_jit = jax.jit(
            _adam_diag_tangent_fn, out_shardings=param_sharding
        )

        self._oracle_init_jit = None
        self._oracle_dir_jit = None
        self._oracle_update_jit = None

    def _ensure_oracle_jits(self, params):
        if self.use_sharded_lbfgs or self._oracle_init_jit is not None:
            return
        assert self.oracle is not None
        oracle_shapes = jax.eval_shape(self.oracle.init, params)
        oracle_sharding = make_fsdp_sharding(oracle_shapes, self.mesh)
        self._oracle_init_jit = jax.jit(
            self.oracle.init, out_shardings=oracle_sharding
        )
        self._oracle_dir_jit = jax.jit(
            self.oracle.direction,
            out_shardings=(self.param_sharding, oracle_sharding),
        )
        self._oracle_update_jit = jax.jit(
            self.oracle.update,
            out_shardings=oracle_sharding,
            donate_argnums=(0,),
        )

    def _lr_multiplier(self, step_count: int) -> float:
        """Cosine LR multiplier (1.0 -> 0.1) when ``cfg.use_lr_schedule`` is enabled."""
        if not self.cfg.use_lr_schedule:
            return 1.0
        progress = min(1.0, max(0.0, step_count / float(self.total_steps)))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    def _sharded_lbfgs_direction_and_tangent(
        self, grad, state: ShardedLBFGSState, lr_mult: float = 1.0
    ):
        """Zero-sync Nocedal & Wright Algorithm 7.4 returning ``(qn_dir, g_pre, v_cand)``."""
        lr_mult_arr = jnp.asarray(lr_mult, jnp.float32)
        if self.use_adam_precond:
            step_t = jnp.asarray(state.step + 1, jnp.int32)
            base_lr_arr = jnp.asarray(float(self.cfg.learning_rate), jnp.float32)
            v_cand, c_k, g_pre_unsched = self._preview_v_and_gpre_jit(
                grad, state.v, step_t, base_lr_arr
            )
            if not state.pairs:
                g_pre_sched = self._scale_jit(lr_mult_arr, g_pre_unsched)
                return g_pre_sched, g_pre_sched, v_cand

            q = grad
            alphas = []
            for s_i, y_i, rho_arr in state.pairs:
                q, alpha_i = self._loop1_step_jit(q, s_i, y_i, rho_arr)
                alphas.append(alpha_i)

            r = self._apply_adam_h0_jit(q, v_cand, step_t, base_lr_arr, c_k)

            for (s_i, y_i, rho_arr), alpha_i in zip(
                reversed(state.pairs), reversed(alphas)
            ):
                r = self._loop2_step_jit(r, s_i, y_i, rho_arr, alpha_i)

            qn_dir, g_pre_sched = self._finalize_lbfgs_dir_jit(
                r, grad, g_pre_unsched, lr_mult_arr
            )
            return qn_dir, g_pre_sched, v_cand

        # Plain scalar-H0 L-BFGS (H0 = gamma * I)
        gamma_eff = float(state.gamma) * lr_mult
        g_pre = self._scale_jit(jnp.asarray(-gamma_eff, jnp.float32), grad)
        if not state.pairs:
            return g_pre, g_pre, None

        q = grad
        alphas = []
        for s_i, y_i, rho_arr in state.pairs:
            q, alpha_i = self._loop1_step_jit(q, s_i, y_i, rho_arr)
            alphas.append(alpha_i)

        r = self._scale_jit(jnp.asarray(state.gamma, jnp.float32), q)
        for (s_i, y_i, rho_arr), alpha_i in zip(
            reversed(state.pairs), reversed(alphas)
        ):
            r = self._loop2_step_jit(r, s_i, y_i, rho_arr, alpha_i)

        qn_dir, g_pre_sched = self._finalize_lbfgs_dir_jit(
            r,
            grad,
            self._scale_jit(jnp.asarray(-float(state.gamma), jnp.float32), grad),
            lr_mult_arr,
        )
        return qn_dir, g_pre_sched, None

    def _sharded_lbfgs_update(
        self,
        state: ShardedLBFGSState,
        params,
        new_params,
        grad,
        new_grad,
        v_cand,
        accepted: bool,
    ) -> ShardedLBFGSState:
        if self.use_adam_precond and accepted and v_cand is not None:
            state.v = v_cand
            state.step += 1
        if not accepted:
            return state
        s, y, ys_arr, yy_arr, ss_arr = self._lbfgs_pair_jit(
            params, new_params, grad, new_grad
        )
        ys, yy, ss = float(ys_arr), float(yy_arr), float(ss_arr)
        eps = 1e-10
        if ys > eps * ((yy * ss + eps) ** 0.5):
            rho_arr = jax.device_put(
                jnp.asarray(1.0 / ys, jnp.float32), self.replicated
            )
            state.pairs.appendleft((s, y, rho_arr))
            while len(state.pairs) > state.history_size:
                old = state.pairs.pop()
                del old
            if yy > 0.0:
                state.gamma = ys / yy
        else:
            del s, y
        return state

    def init(
        self,
        params,
        macro_batch,
        *,
        warm_precond: tuple[Any, int] | None = None,
    ) -> HostQQNState:
        self._ensure_oracle_jits(params)
        val, grad = self._vg_jit(params, macro_batch)
        if self.use_sharded_lbfgs:
            if self.use_adam_precond:
                if warm_precond is not None and warm_precond[0] is not None:
                    v_init = jax.device_put(
                        jax.tree_util.tree_map(
                            lambda x: x.astype(jnp.float32), warm_precond[0]
                        ),
                        self.param_sharding,
                    )
                    init_step = max(0, int(warm_precond[1]))
                else:
                    v_init = self._zeros_f32_jit(params)
                    init_step = 0
            else:
                v_init = None
                init_step = 0
            oracle_state = ShardedLBFGSState(
                history_size=self.cfg.history_size, v=v_init, step=init_step
            )
        else:
            oracle_state = self._oracle_init_jit(params)
            if (
                warm_precond is not None
                and warm_precond[0] is not None
                and isinstance(oracle_state, AdamState)
            ):
                v_warm = jax.tree_util.tree_map(
                    lambda x, ref: x.astype(ref.dtype), warm_precond[0], oracle_state.v
                )
                oracle_state = AdamState(
                    m=oracle_state.m,
                    v=v_warm,
                    step=jnp.asarray(max(0, int(warm_precond[1])), jnp.int32),
                )
        _, gnorm = self._finite_and_norm_jit(grad)
        return HostQQNState(
            value=float(val),
            grad=grad,
            oracle_state=oracle_state,
            step_size=1.0,
            error=float(gnorm),
            num_evals=1,
        )

    def export_preconditioner(self, state: HostQQNState) -> tuple[Any, int] | None:
        """Export diagonal second-moment ``(v, step)`` for warm-starting a subsequent phase."""
        ost = state.oracle_state
        if isinstance(ost, ShardedLBFGSState) and ost.v is not None:
            return ost.v, int(ost.step)
        if isinstance(ost, AdamState):
            return ost.v, int(ost.step)
        return None

    def refresh(self, params, state: HostQQNState, macro_batch) -> HostQQNState:
        val, grad = self._vg_jit(params, macro_batch)
        _, gnorm = self._finite_and_norm_jit(grad)
        if self.use_sharded_lbfgs and self.cfg.clear_lbfgs_on_refresh:
            state.oracle_state.pairs.clear()
        return HostQQNState(
            value=float(val),
            grad=grad,
            oracle_state=state.oracle_state,
            step_size=1.0,
            error=float(gnorm),
            num_evals=state.num_evals + 1,
            step_count=state.step_count,
        )

    def step(
        self,
        params,
        state: HostQQNState,
        macro_batch,
        *,
        will_refresh_next: bool = False,
    ):
        grad = state.grad
        lr_mult = self._lr_multiplier(state.step_count)
        v_cand = None

        if self.use_sharded_lbfgs:
            qn_dir, g_pre_raw, v_cand = self._sharded_lbfgs_direction_and_tangent(
                grad, state.oracle_state, lr_mult=lr_mult
            )
        else:
            qn_dir, _ = self._oracle_dir_jit(params, grad, state.oracle_state)
            if lr_mult != 1.0 and self.oracle_name in ("adam", "lbfgs+adam"):
                qn_dir = self._scale_jit(jnp.asarray(lr_mult, jnp.float32), qn_dir)
            if self.tangent_mode == "preconditioned" and isinstance(
                state.oracle_state, AdamState
            ):
                eff_lr = jnp.asarray(
                    float(self.cfg.learning_rate) * lr_mult, jnp.float32
                )
                g_pre_raw = self._adam_diag_tangent_jit(
                    grad,
                    state.oracle_state.v,
                    state.oracle_state.step + jnp.asarray(1, jnp.int32),
                    eff_lr,
                )
            else:
                g_pre_raw = qn_dir

        g_tan, tau_arr, slope0_arr, qn_slope_arr = self._prepare_step_jit(
            grad, g_pre_raw, qn_dir
        )
        del g_pre_raw
        slope0 = float(slope0_arr)
        qn_slope = float(qn_slope_arr)

        c1 = float(self.cfg.c1)
        c2 = float(self.cfg.c2)
        shrink = 0.5
        grow = 2.0
        max_iter = int(self.cfg.ls_max_iter)
        max_step = float(self.cfg.max_t)
        init_step = float(state.step_size) if self.cfg.remember_step_size else 1.0
        init_step = min(max(init_step, 1e-6), max_step)
        need_slope = (
            self.line_search == "strong_wolfe" or self.path_strategy == "spline"
        )

        grad_unused_after_step = will_refresh_next and (
            (not self.use_sharded_lbfgs) or self.cfg.clear_lbfgs_on_refresh
        )

        evals = 0
        control_points: list[tuple[float, float, float]] = [(0.0, state.value, slope0)]

        def eval_t(t_val: float, *, value_only: bool = False):
            nonlocal evals
            t_arr = jnp.asarray(t_val, jnp.float32)
            cand = self._candidate_jit(params, g_tan, qn_dir, t_arr, tau_arr)
            if value_only:
                v_arr = self._loss_jit(cand, macro_batch)
                evals += 1
                v = float(v_arr)
                ok = math.isfinite(v)
                return cand, v, grad, ok, state.error, 0.0
            v_arr, g_arr = self._vg_jit(cand, macro_batch)
            evals += 1
            v = float(v_arr)
            g_ok, g_norm, slope_arr = self._post_eval_jit(
                g_arr, g_tan, qn_dir, t_arr, tau_arr, jnp.asarray(need_slope)
            )
            ok = math.isfinite(v) and bool(g_ok)
            slope_t = float(slope_arr) if (need_slope and ok) else 0.0
            if ok and need_slope:
                control_points.append((t_val, v, slope_t))
            return cand, v, g_arr, ok, float(g_norm), slope_t

        # --- Inner line search ---
        if self.line_search in ("null", "fixed"):
            best_t = min(1.0, max_step)
            if grad_unused_after_step:
                t_arr = jnp.asarray(best_t, jnp.float32)
                best_p = self._candidate_jit(params, g_tan, qn_dir, t_arr, tau_arr)
                best_v, best_g, best_ok, best_gnorm = (
                    state.value,
                    grad,
                    True,
                    state.error,
                )
                accepted = True
            else:
                best_p, best_v, best_g, best_ok, best_gnorm, _ = eval_t(best_t)
                accepted = best_ok
        elif self.line_search == "strong_wolfe":
            lo_t, lo_v = 0.0, state.value
            hi_t: float | None = None
            cur_t = init_step
            cur_p, cur_v, cur_g, cur_ok, cur_gnorm, cur_m = eval_t(cur_t)
            best_t, best_p, best_v, best_g, best_ok, best_gnorm = (
                cur_t,
                cur_p,
                cur_v,
                cur_g,
                cur_ok,
                cur_gnorm,
            )
            accepted = False
            for ls_i in range(1, max_iter + 1):
                armijo = cur_ok and (cur_v <= state.value + c1 * cur_t * slope0)
                curv = cur_ok and (abs(cur_m) <= c2 * abs(slope0))
                if armijo and curv:
                    if cur_p is not best_p:
                        del best_p, best_g
                        best_t, best_p, best_v, best_g, best_ok, best_gnorm = (
                            cur_t,
                            cur_p,
                            cur_v,
                            cur_g,
                            cur_ok,
                            cur_gnorm,
                        )
                    accepted = True
                    break
                if (not cur_ok) or (not armijo) or (ls_i > 1 and cur_v >= lo_v):
                    hi_t = cur_t
                else:
                    if cur_m >= 0.0:
                        hi_t = cur_t
                    else:
                        lo_t, lo_v = cur_t, cur_v
                if ls_i == max_iter:
                    if cur_p is not best_p:
                        del cur_p, cur_g
                    accepted = best_ok and (
                        best_v <= state.value + c1 * best_t * slope0
                    )
                    break
                if hi_t is not None:
                    t_next = 0.5 * (lo_t + hi_t)
                else:
                    t_next = min(cur_t * grow, max_step)
                    if abs(t_next - cur_t) < 1e-7:
                        if cur_p is not best_p:
                            del cur_p, cur_g
                        accepted = best_ok and (
                            best_v <= state.value + c1 * best_t * slope0
                        )
                        break
                if cur_p is not best_p:
                    del cur_p, cur_g
                p2, v2, g2, ok2, gn2, m2 = eval_t(t_next)
                cur_t, cur_p, cur_v, cur_g, cur_ok, cur_gnorm, cur_m = (
                    t_next,
                    p2,
                    v2,
                    g2,
                    ok2,
                    gn2,
                    m2,
                )
                if ok2 and (not best_ok or v2 < best_v):
                    del best_p, best_g
                    best_t, best_p, best_v, best_g, best_ok, best_gnorm = (
                        t_next,
                        p2,
                        v2,
                        g2,
                        ok2,
                        gn2,
                    )
        else:
            val_only = grad_unused_after_step and self.path_strategy != "spline"
            best_t = init_step
            best_p, best_v, best_g, best_ok, best_gnorm, _ = eval_t(
                best_t, value_only=val_only
            )
            accepted = best_ok and (best_v <= state.value + c1 * best_t * slope0)
            ls_iters = 1

            if accepted and max_step > best_t:
                while best_t * grow <= max_step and ls_iters < max_iter:
                    next_t = best_t * grow
                    p2, v2, g2, ok2, gn2, _ = eval_t(next_t, value_only=val_only)
                    ls_iters += 1
                    if (
                        ok2
                        and (v2 <= state.value + c1 * next_t * slope0)
                        and (v2 < best_v)
                    ):
                        del best_p, best_g
                        best_t, best_p, best_v, best_g, best_ok, best_gnorm = (
                            next_t,
                            p2,
                            v2,
                            g2,
                            ok2,
                            gn2,
                        )
                    else:
                        del p2, g2
                        break

            while not accepted and ls_iters < max_iter:
                best_t *= shrink
                del best_p, best_g
                best_p, best_v, best_g, best_ok, best_gnorm, _ = eval_t(
                    best_t, value_only=val_only
                )
                ls_iters += 1
                accepted = best_ok and (best_v <= state.value + c1 * best_t * slope0)

        # --- Bounded Cubic Hermite Spline Refinement ---
        if self.path_strategy == "spline" and self.cfg.spline_refine_rounds > 0:
            max_cp = max(2, int(self.cfg.spline_max_control_points))
            for _ in range(int(self.cfg.spline_refine_rounds)):
                prop_t = _propose_bounded_hermite_candidate(
                    control_points, max_control_points=max_cp, max_step=max_step
                )
                if prop_t is None:
                    break
                p_sp, v_sp, g_sp, ok_sp, gn_sp, _ = eval_t(prop_t)
                if ok_sp and (not best_ok or v_sp < best_v):
                    del best_p, best_g
                    best_t, best_p, best_v, best_g, best_ok, best_gnorm = (
                        prop_t,
                        p_sp,
                        v_sp,
                        g_sp,
                        ok_sp,
                        gn_sp,
                    )
                    if best_v <= state.value + c1 * best_t * slope0:
                        accepted = True
                else:
                    del p_sp, g_sp

        # Fallback safe step ONLY if line search failed
        if self.line_search not in ("null", "fixed") and not (
            best_ok and (best_v <= state.value or accepted)
        ):
            fb_p, fb_t_arr = self._fallback_candidate_jit(params, grad, g_tan, tau_arr)
            if grad_unused_after_step and self.path_strategy != "spline":
                fb_v_arr = self._loss_jit(fb_p, macro_batch)
                fb_g = grad
                fb_ok_g, fb_gnorm = True, state.error
            else:
                fb_v_arr, fb_g = self._vg_jit(fb_p, macro_batch)
                fb_ok_g, fb_gnorm = self._finite_and_norm_jit(fb_g)
            evals += 1
            fb_v = float(fb_v_arr)
            if math.isfinite(fb_v) and bool(fb_ok_g) and fb_v < state.value:
                del best_p, best_g
                best_p, best_v, best_g, best_t, best_gnorm = (
                    fb_p,
                    fb_v,
                    fb_g,
                    float(fb_t_arr),
                    float(fb_gnorm),
                )
                accepted = True
            else:
                del fb_p, fb_g, best_p, best_g
                best_p, best_v, best_g, best_t, best_gnorm = (
                    params,
                    state.value,
                    grad,
                    0.0,
                    state.error,
                )
                accepted = False

        del qn_dir, g_tan

        if self.use_sharded_lbfgs:
            if grad_unused_after_step:
                if self.use_adam_precond and accepted and v_cand is not None:
                    state.oracle_state.v = v_cand
                    state.oracle_state.step += 1
                new_oracle_state = state.oracle_state
            else:
                new_oracle_state = self._sharded_lbfgs_update(
                    state.oracle_state,
                    params,
                    best_p,
                    grad,
                    best_g,
                    v_cand,
                    accepted,
                )
        else:
            if accepted:
                oracle_info = OracleInfo(
                    params=params,
                    new_params=best_p,
                    grad=grad,
                    new_grad=best_g,
                    t=jnp.asarray(best_t, jnp.float32),
                    step_size=jnp.asarray(best_t, jnp.float32),
                )
                new_oracle_state = self._oracle_update_jit(
                    state.oracle_state, oracle_info
                )
            else:
                new_oracle_state = state.oracle_state

        reduction = state.value - best_v
        new_num_evals = state.num_evals + evals
        new_state = HostQQNState(
            value=best_v,
            grad=best_g,
            oracle_state=new_oracle_state,
            step_size=best_t if accepted and best_t > 0.0 else 1.0,
            error=best_gnorm,
            num_evals=new_num_evals,
            step_count=state.step_count + 1,
        )
        metrics = {
            "loss": best_v,
            "grad_norm": best_gnorm,
            "num_evals": new_num_evals,
            "step_size": best_t,
            "ls_success": 1.0 if accepted else 0.0,
            "qn_slope": qn_slope,
            "last_reduction": reduction,
        }
        return best_p, new_state, metrics
