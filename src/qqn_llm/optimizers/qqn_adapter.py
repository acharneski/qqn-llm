"""Memory-efficient FSDP QQN runner driving qqn-jax oracles, paths, and line searches.

Compiles ONE shared ``value_and_grad`` function and small pure-JAX primitives
for the ``qqn_jax`` oracle, path strategy (``quadratic``, ``linear``, or
``spline``), and line search (``backtracking``, ``strong_wolfe``, ``null``,
``fixed``), orchestrating the 1-D search over scalar losses/slopes on the host.

Supports:
- Oracles: ``lbfgs``, ``lbfgs_adam`` (AdamPreconditionedLBFGSOracle), ``adam``,
  ``secant``, ``lbfgs+secant``, ``lbfgs+adam``.
- Paths:
  - ``quadratic``: ``d(t) = t(1-t)(-∇f) + t²(-H∇f)``
  - ``linear``: ``d(t) = t(-H∇f)`` (straight-chord control)
  - ``spline``: quadratic path + bounded piecewise cubic Hermite spline
    refinement (reusing measured ``(t_i, f_i, m_i)`` control points, strictly
    capped by ``spline_refine_rounds`` and ``spline_max_control_points``).
- Line searches: ``backtracking`` (Armijo + optional extrapolation),
  ``strong_wolfe`` (Armijo + curvature zoom), ``null`` / ``fixed`` (t = 1).
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
from qqn_jax.oracles.oracle import OracleInfo
from qqn_jax.paths.linear import LINEAR_PATH
from qqn_jax.paths.quadratic import QUADRATIC_PATH
from qqn_jax.utils import (
    tree_add_scaled,
    tree_all_finite,
    tree_l2_norm,
    tree_negative,
    tree_vdot,
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
            history_size=cfg.history_size, learning_rate=cfg.learning_rate
        )
    if name == "adam":
        return AdamOracle(learning_rate=cfg.learning_rate)
    if name == "secant":
        return SecantOracle()
    if name == "lbfgs+secant":
        return Fallback([LBFGSOracle(history_size=cfg.history_size), SecantOracle()])
    if name == "lbfgs+adam":
        return Fallback(
            [
                LBFGSOracle(history_size=cfg.history_size),
                AdamOracle(learning_rate=cfg.learning_rate),
            ]
        )
    raise ValueError(f"Unsupported QQN oracle: {cfg.oracle!r}")


def _propose_bounded_hermite_candidate(
    control_points: list[tuple[float, float, float]],
    max_control_points: int,
    max_step: float,
) -> float | None:
    """Propose the lowest-predicted stationary point of a bounded cubic Hermite spline.

    Each control point is ``(t_i, f_i, m_i)`` where ``m_i = ⟨∇f(x+d(t_i)), d'(t_i)⟩``.
    Applies QQN's upstream/downstream tangent symmetry reflection (algorithm.md §4.3)
    and trims control points to the best ``max_control_points`` to prevent fractal
    subdivision.
    """
    if len(control_points) < 2:
        return None

    # Deduplicate near-identical t values and cap control point count
    pts = sorted(control_points, key=lambda x: x[0])
    deduped: list[tuple[float, float, float]] = [pts[0]]
    for pt in pts[1:]:
        if abs(pt[0] - deduped[-1][0]) > 1e-7:
            deduped.append(pt)
    if len(deduped) > max_control_points:
        # Always keep origin (index 0) and lowest-value points
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
        # Upstream/downstream tangent symmetry safeguard
        m0_eff = -m0 if (secant != 0.0 and m0 * secant < 0.0 and f1 < f0 and m0 > 0.0) else m0
        m1_eff = -m1 if (secant != 0.0 and m1 * secant < 0.0 and f1 > f0 and m1 < 0.0) else m1

        # Cubic Hermite derivative f'(s) = A s^2 + B s + C = 0 on s in [0, 1]
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
            # Evaluate cubic Hermite value at s
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
    """Host-managed deque of FSDP-sharded ``(s_i, y_i, rho_i)`` curvature pairs."""

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
        self.path = LINEAR_PATH if self.path_strategy == "linear" else QUADRATIC_PATH
        self.line_search = (cfg.line_search or "null").lower()

        # 1. Single compiled value_and_grad (shared across init, line search, fallback)
        self._vg_jit = jax.jit(
            jax.value_and_grad(loss_fn),
            out_shardings=(self.replicated, param_sharding),
        )

        # 2. Pure elementwise path candidate x + d(t)
        def _candidate_fn(params, grad, qn_dir, t):
            grad_dir = tree_negative(grad)
            d_t = self.path.offset(t, grad_dir, qn_dir)
            return tree_add_scaled(params, 1.0, d_t)

        self._candidate_jit = jax.jit(
            _candidate_fn, out_shardings=param_sharding
        )

        # 3. Directional derivative along path at arbitrary t: <g, d'(t)>
        def _path_slope_fn(grad_at_t, grad0, qn_dir, t):
            grad_dir = tree_negative(grad0)
            vel = self.path.velocity(t, grad_dir, qn_dir)
            return tree_vdot(grad_at_t, vel, accum_dtype=jnp.float32)

        self._path_slope_jit = jax.jit(
            _path_slope_fn, out_shardings=self.replicated
        )

        # 4. Initial directional derivative slope0 = <grad, d'(0)> and qn_slope = <grad, qn_dir>
        def _slopes_fn(grad, qn_dir):
            grad_dir = tree_negative(grad)
            v0 = self.path.velocity(jnp.asarray(0.0, jnp.float32), grad_dir, qn_dir)
            slope0 = tree_vdot(grad, v0, accum_dtype=jnp.float32)
            qn_slope = tree_vdot(grad, qn_dir, accum_dtype=jnp.float32)
            gnorm = tree_l2_norm(grad)
            return slope0, qn_slope, gnorm

        self._slopes_jit = jax.jit(
            _slopes_fn,
            out_shardings=(self.replicated, self.replicated, self.replicated),
        )

        self._finite_and_norm_jit = jax.jit(
            lambda g: (tree_all_finite(g), tree_l2_norm(g)),
            out_shardings=(self.replicated, self.replicated),
        )

        def _fallback_candidate_fn(params, grad):
            gnorm_sq = tree_vdot(grad, grad, accum_dtype=jnp.float32)
            safe_scale = jnp.asarray(1.0, jnp.float32) / (
                jnp.asarray(1.0, jnp.float32) + gnorm_sq
            )
            return tree_add_scaled(params, safe_scale, tree_negative(grad)), safe_scale

        self._fallback_candidate_jit = jax.jit(
            _fallback_candidate_fn,
            out_shardings=(param_sharding, self.replicated),
        )

        # Sharded vector primitives for FSDP-safe L-BFGS & Adam-preconditioned L-BFGS
        self._vdot_jit = jax.jit(
            lambda a, b: tree_vdot(a, b, accum_dtype=jnp.float32),
            out_shardings=self.replicated,
        )
        self._axpy_jit = jax.jit(
            lambda x, alpha, y: tree_add_scaled(x, alpha, y),
            out_shardings=param_sharding,
        )
        self._scale_jit = jax.jit(
            lambda alpha, x: jax.tree_util.tree_map(
                lambda t: (jnp.asarray(alpha, dtype=t.dtype) * t).astype(t.dtype), x
            ),
            out_shardings=param_sharding,
        )
        self._sub_compact_jit = jax.jit(
            lambda a, b: jax.tree_util.tree_map(
                lambda x, y: (x - y).astype(
                    jnp.bfloat16 if x.dtype == jnp.bfloat16 or x.size >= 4096 else x.dtype
                ),
                a,
                b,
            ),
            out_shardings=param_sharding,
        )
        self._zeros_f32_jit = jax.jit(
            lambda p: jax.tree_util.tree_map(
                lambda x: jnp.zeros(x.shape, dtype=jnp.float32), p
            ),
            out_shardings=param_sharding,
        )

        beta2 = 0.999
        eps = 1e-8
        lr = float(cfg.learning_rate)

        def _update_v_fn(v, grad):
            return jax.tree_util.tree_map(
                lambda vi, g: beta2 * vi + (1.0 - beta2) * (g.astype(jnp.float32) ** 2),
                v,
                grad,
            )

        def _apply_adam_h0_fn(q, v, step_arr, lr_arr):
            bc2 = 1.0 - (beta2 ** step_arr.astype(jnp.float32))
            return jax.tree_util.tree_map(
                lambda qi, vi: (
                    (lr_arr / (jnp.sqrt(vi / bc2) + eps)) * qi.astype(jnp.float32)
                ).astype(qi.dtype),
                q,
                v,
            )

        self._update_v_jit = jax.jit(
            _update_v_fn, out_shardings=param_sharding, donate_argnums=(0,)
        )
        self._apply_adam_h0_jit = jax.jit(
            _apply_adam_h0_fn, out_shardings=param_sharding
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

    def _apply_h0(self, q, state: ShardedLBFGSState, lr_mult: float = 1.0):
        if self.use_adam_precond:
            step_t = max(1, state.step)
            eff_lr = jnp.asarray(float(self.cfg.learning_rate) * lr_mult, jnp.float32)
            return self._apply_adam_h0_jit(q, state.v, jnp.asarray(step_t, jnp.int32), eff_lr)
        return self._scale_jit(jnp.asarray(state.gamma, jnp.float32), q)

    def _sharded_lbfgs_direction(self, grad, state: ShardedLBFGSState, lr_mult: float = 1.0):
        """Nocedal & Wright Algorithm 7.4 over individually FSDP-sharded (s_i, y_i) trees."""
        if self.use_adam_precond:
            state.v = self._update_v_jit(state.v, grad)
            state.step += 1

        if not state.pairs:
            r = self._apply_h0(grad, state, lr_mult=lr_mult)
            return self._scale_jit(jnp.asarray(-1.0, jnp.float32), r)

        q = grad
        alphas: list[float] = []
        for s_i, y_i, rho_i in state.pairs:
            alpha_i = rho_i * float(self._vdot_jit(s_i, q))
            alphas.append(alpha_i)
            q = self._axpy_jit(q, jnp.asarray(-alpha_i, jnp.float32), y_i)

        r = self._apply_h0(q, state, lr_mult=lr_mult)

        for (s_i, y_i, rho_i), alpha_i in zip(reversed(state.pairs), reversed(alphas)):
            beta_i = rho_i * float(self._vdot_jit(y_i, r))
            r = self._axpy_jit(r, jnp.asarray(alpha_i - beta_i, jnp.float32), s_i)

        d = self._scale_jit(jnp.asarray(-1.0, jnp.float32), r)
        gd = float(self._vdot_jit(grad, d))
        if not (gd < 0.0):
            r_fb = self._apply_h0(grad, state, lr_mult=lr_mult)
            return self._scale_jit(jnp.asarray(-1.0, jnp.float32), r_fb)
        return d

    def _sharded_lbfgs_update(
        self, state: ShardedLBFGSState, params, new_params, grad, new_grad
    ) -> ShardedLBFGSState:
        s = self._sub_compact_jit(new_params, params)
        y = self._sub_compact_jit(new_grad, grad)
        ys = float(self._vdot_jit(y, s))
        yy = float(self._vdot_jit(y, y))
        ss = float(self._vdot_jit(s, s))
        eps = 1e-10
        if ys > eps * ((yy * ss + eps) ** 0.5):
            rho = 1.0 / ys
            state.pairs.appendleft((s, y, rho))
            while len(state.pairs) > state.history_size:
                old = state.pairs.pop()
                del old
            if yy > 0.0:
                state.gamma = ys / yy
        else:
            del s, y
        return state

    def init(self, params, macro_batch) -> HostQQNState:
        self._ensure_oracle_jits(params)
        val, grad = self._vg_jit(params, macro_batch)
        if self.use_sharded_lbfgs:
            v_init = self._zeros_f32_jit(params) if self.use_adam_precond else None
            oracle_state = ShardedLBFGSState(
                history_size=self.cfg.history_size, v=v_init, step=0
            )
        else:
            oracle_state = self._oracle_init_jit(params)
        _, gnorm = self._finite_and_norm_jit(grad)
        return HostQQNState(
            value=float(val),
            grad=grad,
            oracle_state=oracle_state,
            step_size=1.0,
            error=float(gnorm),
            num_evals=1,
        )

    def refresh(self, params, state: HostQQNState, macro_batch) -> HostQQNState:
        val, grad = self._vg_jit(params, macro_batch)
        _, gnorm = self._finite_and_norm_jit(grad)
        if self.use_sharded_lbfgs and self.cfg.clear_lbfgs_on_refresh:
            # Drop stale curvature pairs from the previous macro-batch while
            # retaining the cross-batch diagonal Adam second moment `v`.
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

    def step(self, params, state: HostQQNState, macro_batch):
        grad = state.grad
        lr_mult = self._lr_multiplier(state.step_count)
        if self.use_sharded_lbfgs:
            qn_dir = self._sharded_lbfgs_direction(grad, state.oracle_state, lr_mult=lr_mult)
        else:
            qn_dir, _ = self._oracle_dir_jit(params, grad, state.oracle_state)
            if lr_mult != 1.0 and self.oracle_name in ("adam", "lbfgs+adam"):
                qn_dir = self._scale_jit(jnp.asarray(lr_mult, jnp.float32), qn_dir)

        slope0_arr, qn_slope_arr, _ = self._slopes_jit(grad, qn_dir)
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
        need_slope = self.line_search == "strong_wolfe" or self.path_strategy == "spline"

        evals = 0
        control_points: list[tuple[float, float, float]] = [(0.0, state.value, slope0)]

        def eval_t(t_val: float):
            nonlocal evals
            t_arr = jnp.asarray(t_val, jnp.float32)
            cand = self._candidate_jit(params, grad, qn_dir, t_arr)
            v_arr, g_arr = self._vg_jit(cand, macro_batch)
            evals += 1
            v = float(v_arr)
            g_ok, g_norm = self._finite_and_norm_jit(g_arr)
            ok = bool(jnp.isfinite(v_arr)) and bool(g_ok)
            slope_t = (
                float(self._path_slope_jit(g_arr, grad, qn_dir, t_arr))
                if (need_slope and ok)
                else 0.0
            )
            if ok and need_slope:
                control_points.append((t_val, v, slope_t))
            return cand, v, g_arr, ok, float(g_norm), slope_t

        # --- Inner line search ---
        if self.line_search in ("null", "fixed"):
            best_t = min(1.0, max_step)
            best_p, best_v, best_g, best_ok, best_gnorm, _ = eval_t(best_t)
            accepted = best_ok
        elif self.line_search == "strong_wolfe":
            lo_t, lo_v, lo_m = 0.0, state.value, slope0
            hi_t: float | None = None
            t_cur = init_step
            best_t = t_cur
            best_p, best_v, best_g, best_ok, best_gnorm, cur_m = eval_t(t_cur)
            accepted = False
            for ls_i in range(1, max_iter + 1):
                armijo = best_ok and (best_v <= state.value + c1 * t_cur * slope0)
                curv = best_ok and (abs(cur_m) <= c2 * abs(slope0))
                if armijo and curv:
                    accepted = True
                    break
                if (not best_ok) or (not armijo) or (ls_i > 1 and best_v >= lo_v):
                    hi_t = t_cur
                else:
                    if cur_m >= 0.0:
                        hi_t = t_cur
                    else:
                        lo_t, lo_v, lo_m = t_cur, best_v, cur_m
                if ls_i == max_iter:
                    accepted = armijo
                    break
                if hi_t is not None:
                    t_next = 0.5 * (lo_t + hi_t)
                else:
                    t_next = min(t_cur * grow, max_step)
                    if abs(t_next - t_cur) < 1e-7:
                        accepted = armijo
                        break
                p2, v2, g2, ok2, gn2, m2 = eval_t(t_next)
                # Keep best feasible point seen so far
                if ok2 and (not best_ok or v2 < best_v):
                    del best_p, best_g
                    best_t, best_p, best_v, best_g, best_ok, best_gnorm = (
                        t_next, p2, v2, g2, ok2, gn2
                    )
                else:
                    del p2, g2
                t_cur, cur_m = t_next, m2
        else:
            # Default: permissive Armijo backtracking + optional extrapolation
            best_t = init_step
            best_p, best_v, best_g, best_ok, best_gnorm, _ = eval_t(best_t)
            accepted = best_ok and (best_v <= state.value + c1 * best_t * slope0)
            ls_iters = 1

            if accepted and max_step > best_t:
                while best_t * grow <= max_step and ls_iters < max_iter:
                    next_t = best_t * grow
                    p2, v2, g2, ok2, gn2, _ = eval_t(next_t)
                    ls_iters += 1
                    if ok2 and (v2 <= state.value + c1 * next_t * slope0) and (v2 < best_v):
                        del best_p, best_g
                        best_t, best_p, best_v, best_g, best_ok, best_gnorm = (
                            next_t, p2, v2, g2, ok2, gn2
                        )
                    else:
                        del p2, g2
                        break

            while not accepted and ls_iters < max_iter:
                best_t *= shrink
                del best_p, best_g
                best_p, best_v, best_g, best_ok, best_gnorm, _ = eval_t(best_t)
                ls_iters += 1
                accepted = best_ok and (best_v <= state.value + c1 * best_t * slope0)

        # --- Bounded Cubic Hermite Spline Refinement (when path_strategy == "spline") ---
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
                        prop_t, p_sp, v_sp, g_sp, ok_sp, gn_sp
                    )
                    if best_v <= state.value + c1 * best_t * slope0:
                        accepted = True
                else:
                    del p_sp, g_sp

        # Fallback safe gradient step ONLY if line search failed (skip for null/fixed)
        if self.line_search not in ("null", "fixed") and not (
            best_ok and (best_v <= state.value or accepted)
        ):
            fb_p, fb_t_arr = self._fallback_candidate_jit(params, grad)
            fb_v_arr, fb_g = self._vg_jit(fb_p, macro_batch)
            evals += 1
            fb_v = float(fb_v_arr)
            fb_ok_g, fb_gnorm = self._finite_and_norm_jit(fb_g)
            if bool(jnp.isfinite(fb_v_arr)) and bool(fb_ok_g) and fb_v < state.value:
                del best_p, best_g
                best_p, best_v, best_g, best_t, best_gnorm = (
                    fb_p, fb_v, fb_g, float(fb_t_arr), float(fb_gnorm)
                )
                accepted = True
            else:
                del fb_p, fb_g, best_p, best_g
                best_p, best_v, best_g, best_t, best_gnorm = (
                    params, state.value, grad, 0.0, state.error
                )
                accepted = False

        del qn_dir

        if self.use_sharded_lbfgs:
            new_oracle_state = self._sharded_lbfgs_update(
                state.oracle_state, params, best_p, grad, best_g
            )
        else:
            oracle_info = OracleInfo(
                params=params,
                new_params=best_p,
                grad=grad,
                new_grad=best_g,
                t=jnp.asarray(best_t, jnp.float32),
                step_size=jnp.asarray(best_t, jnp.float32),
            )
            new_oracle_state = self._oracle_update_jit(state.oracle_state, oracle_info)

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
