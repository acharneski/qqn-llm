"""OWL-like Half-Quantization QQN runner and exact PTQ baselines for FSDP LLMs.

Implements the constrained quantization optimization regime:
  minimize_θ   ‖θ‖₁
  subject to   L_train(θ) ≤ L_train(θ_ref) + ε_tol     (no-worse data constraint)
               θ_i ∈ C_16(θ_{k,i})                     (half-quantization region)

Key algorithmic components:
1. **Half-Quantization Constraint Regions** (:func:`HalfQuantizationRegion`):
   Replaces OWL-QN's sign orthants with 16-bit lattice intervals ``[q_lo, q_hi]``
   whose endpoints are exact target-precision values (``bfloat16``, ``float16``,
   or symmetric ``uniform`` ``b``-bit). Because ``0`` is on the grid, ``‖θ‖₁`` is
   strictly linear inside every open cell, and clamping at any boundary yields
   zero rounding error to 16-bit.
2. **Cell-Width-Scaled Curvature Metric & Exact KKT No-Decrease Projection**:
   Accumulates ``lbfgs_adam`` curvature pairs ``(s_j, y_j)`` on ``∇L_train`` with
   dimensionally consistent diagonal seed ``H_{0,ii} = η · w_i / (√v̂_i + δ)``,
   and solves the Riemannian KKT half-space projection in closed form:
       d* = d_L1 + λ* d_data,   λ* = relu(⟨g_k, d_L1⟩ − b_k) / (−⟨g_k, d_data⟩ + ε)
   for both the ``t = 0`` diagonal tangent and the ``t = 1`` L-BFGS endpoint.
3. **Post-Clip Convex Half-Space Reconciliation & Feasible QQN Line Search**:
   Projects ``θ_k + d(t)`` through ``NoDecrease`` then ``HalfQuantizationRegion``,
   convexly reconciles any clipping-induced uphill drift while keeping all
   loss-favorable boundary snaps exact, and verifies ``L_train(θ_cand) ≤ L_cap``
   plus ``‖θ_cand‖₁`` / boundary progress along the quadratic curve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from qqn_jax.paths.linear import LINEAR_PATH
from qqn_jax.paths.quadratic import MONOTONE_QUADRATIC_PATH, QUADRATIC_PATH
from qqn_jax.regions.half_quantization import (
    HalfQuantizationRegion,
    _float_grid_neighbors,
    half_quant_bounds,
    half_quant_cell_width,
    is_on_half_quant_boundary,
    round_to_half_quant,
    round_toward_zero_half_quant,
)
from qqn_jax.regions.identity import _identity_init, _identity_update
from qqn_jax.regions.orthant import OrthantRegion
from qqn_jax.regions.quantization import QuantizationRegion
from qqn_jax.regions.strategy import Region
from qqn_jax.utils import (
    tree_add_scaled,
    tree_all_finite,
    tree_l2_norm,
    tree_vdot,
    tree_where,
)

from qqn_llm.config import OptimizerConfig
from qqn_llm.optimizers.qqn_adapter import ShardedLBFGSState


def make_device_quant_telemetry_fn(
    mesh: Mesh,
    num_params: int,
    *,
    qtype: str = "bfloat16",
    bits: int = 16,
    step: float | None = None,
    lo: float = -1.0,
    hi: float = 1.0,
):
    """Build an FSDP-resident reducer computing exact L1 & quantization boundary metrics."""
    replicated = NamedSharding(mesh, P())
    n_safe = max(1, int(num_params))

    @jax.jit
    def _leaf_stats(params):
        l1_leaves = []
        q_err_l1_leaves = []
        q_err_max_leaves = []
        on_b_leaves = []
        zero_leaves = []
        for p in jax.tree_util.tree_leaves(params):
            pf = p.astype(jnp.float32)
            qf = round_to_half_quant(
                pf, qtype=qtype, bits=bits, step=step, lo=lo, hi=hi
            )
            err = jnp.abs(pf - qf)
            l1_leaves.append(
                jax.lax.with_sharding_constraint(
                    jnp.sum(jnp.abs(pf), dtype=jnp.float32), replicated
                )
            )
            q_err_l1_leaves.append(
                jax.lax.with_sharding_constraint(
                    jnp.sum(err, dtype=jnp.float32), replicated
                )
            )
            q_err_max_leaves.append(
                jax.lax.with_sharding_constraint(
                    jnp.max(err) if pf.size > 0 else jnp.asarray(0.0, jnp.float32),
                    replicated,
                )
            )
            on_b_leaves.append(
                jax.lax.with_sharding_constraint(
                    jnp.sum((err == 0.0).astype(jnp.float32), dtype=jnp.float32),
                    replicated,
                )
            )
            zero_leaves.append(
                jax.lax.with_sharding_constraint(
                    jnp.sum((pf == 0.0).astype(jnp.float32), dtype=jnp.float32),
                    replicated,
                )
            )
        return (
            jnp.stack(l1_leaves),
            jnp.stack(q_err_l1_leaves),
            jnp.stack(q_err_max_leaves),
            jnp.stack(on_b_leaves),
            jnp.stack(zero_leaves),
        )

    def compute(params) -> dict[str, float]:
        l1_v, qerr_v, qmax_v, onb_v, zero_v = _leaf_stats(params)
        l1_total = float(np.sum(np.asarray(l1_v, dtype=np.float64), dtype=np.float64))
        qerr_total = float(
            np.sum(np.asarray(qerr_v, dtype=np.float64), dtype=np.float64)
        )
        qmax_total = float(np.max(np.asarray(qmax_v, dtype=np.float64)))
        onb_total = float(np.sum(np.asarray(onb_v, dtype=np.float64), dtype=np.float64))
        zero_total = float(
            np.sum(np.asarray(zero_v, dtype=np.float64), dtype=np.float64)
        )
        return {
            "param_l1_norm": l1_total,
            "param_l1_mean": l1_total / n_safe,
            "quant_l1_error": qerr_total,
            "quant_l1_error_mean": qerr_total / n_safe,
            "quant_max_error": qmax_total,
            "quant_boundary_frac": min(1.0, max(0.0, onb_total / n_safe)),
            "zero_weight_frac": min(1.0, max(0.0, zero_total / n_safe)),
        }

    return compute


def make_round_params_jit(
    param_sharding: Any,
    *,
    qtype: str = "bfloat16",
    bits: int = 16,
    step: float | None = None,
    lo: float = -1.0,
    hi: float = 1.0,
):
    """Build an FSDP-sharded nearest-grid quantizer for evaluation."""
    return jax.jit(
        lambda p: round_to_half_quant(
            p, qtype=qtype, bits=bits, step=step, lo=lo, hi=hi
        ),
        out_shardings=param_sharding,
    )


@dataclass
class HostOWLQuantState:
    value: float
    grad: Any
    oracle_state: ShardedLBFGSState
    ref_train_loss: float
    loss_cap: float
    l1_norm: float
    boundary_frac: float
    step_size: float
    error: float
    num_evals: int
    step_count: int = 0


class OWLQuantRunner:
    """FSDP-sharded OWL-like Half-Quantization + L1 Goal + No-Worse Training Runner."""

    is_host_stepped: bool = True

    def __init__(
        self,
        loss_fn: Callable,
        cfg: OptimizerConfig,
        mesh: Mesh,
        param_sharding: Any,
        num_params: int,
        total_steps: int = 20,
    ):
        self.cfg = cfg
        self.mesh = mesh
        self.param_sharding = param_sharding
        self.replicated = NamedSharding(mesh, P())
        self.num_params = max(1, int(num_params))
        self.total_steps = max(1, int(total_steps))

        self.qtype = cfg.quant_qtype
        self.qbits = int(cfg.quant_bits)
        self.qstep = cfg.quant_step
        self.qlo = float(cfg.quant_lo)
        self.qhi = float(cfg.quant_hi)
        self.region_mode = cfg.quant_region_mode
        self.boundary_mode = cfg.quant_boundary_mode
        self.no_worse_tol = float(cfg.no_worse_tol)
        self.enforce_no_worse = bool(cfg.enforce_no_worse)
        self.cell_scaled_metric = bool(cfg.cell_scaled_metric)
        self.final_snap = bool(cfg.final_snap)

        path_strategy = (cfg.path_strategy or "quadratic").lower()
        if path_strategy == "linear":
            self.path = LINEAR_PATH
        elif path_strategy == "monotone_quadratic":
            self.path = MONOTONE_QUADRATIC_PATH
        else:
            self.path = QUADRATIC_PATH

        if self.region_mode == "half_quant":
            self.region = HalfQuantizationRegion(
                qtype=self.qtype,
                bits=self.qbits,
                step=self.qstep,
                lo=self.qlo,
                hi=self.qhi,
                boundary_mode=self.boundary_mode,
            )
        elif self.region_mode == "midpoint_cell":
            if self.qtype in ("bfloat16", "float16"):
                tgt_dt = jnp.bfloat16 if self.qtype == "bfloat16" else jnp.float16

                def _proj_float_mid(p_tree, c_tree, _st):
                    def _leaf(x, c):
                        dt = c.dtype
                        xf = x.astype(jnp.float32)
                        _, _, prev_q, next_q = _float_grid_neighbors(xf, tgt_dt)
                        r_near = round_to_half_quant(xf, qtype=self.qtype)
                        mid_lo = 0.5 * (prev_q + r_near)
                        mid_hi = 0.5 * (r_near + next_q)
                        return jnp.clip(c.astype(jnp.float32), mid_lo, mid_hi).astype(dt)

                    return jax.tree_util.tree_map(_leaf, p_tree, c_tree)

                self.region = Region(
                    init=_identity_init,
                    project=_proj_float_mid,
                    update=_identity_update,
                )
            else:
                self.region = QuantizationRegion(
                    bits=self.qbits, step=self.qstep, lo=self.qlo, hi=self.qhi
                )
        elif self.region_mode == "orthant":
            self.region = OrthantRegion()
        elif self.region_mode == "none":
            self.region = None
        else:
            raise ValueError(f"Unsupported quant_region_mode: {self.region_mode!r}")

        beta2 = float(cfg.beta2)
        clip_norm = float(cfg.grad_clip_norm)
        max_qn_gain = float(cfg.max_qn_gain)
        qtype = self.qtype
        qbits = self.qbits
        qstep = self.qstep
        qlo = self.qlo
        qhi = self.qhi
        region_mode = self.region_mode
        boundary_mode = self.boundary_mode
        cell_scaled = self.cell_scaled_metric
        n_params_f32 = float(self.num_params)

        self._vg_jit = jax.jit(
            jax.value_and_grad(loss_fn),
            out_shardings=(self.replicated, param_sharding),
            donate_argnums=(0,),
        )
        self._loss_jit = jax.jit(loss_fn, out_shardings=self.replicated)
        self._clone_params_jit = jax.jit(
            lambda p: jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), p),
            out_shardings=param_sharding,
        )
        self._zeros_f32_jit = jax.jit(
            lambda p: jax.tree_util.tree_map(
                lambda x: jnp.zeros(x.shape, dtype=jnp.float32), p
            ),
            out_shardings=param_sharding,
        )
        self._finite_and_norm_jit = jax.jit(
            lambda g: (tree_all_finite(g), tree_l2_norm(g)),
            out_shardings=(self.replicated, self.replicated),
        )
        self._telemetry_fn = make_device_quant_telemetry_fn(
            mesh,
            self.num_params,
            qtype=qtype,
            bits=qbits,
            step=qstep,
            lo=qlo,
            hi=qhi,
        )

        # Fast fused L1 + boundary fraction check for inner line-search probes
        def _l1_and_bfrac_fn(p):
            l1_parts = []
            onb_parts = []
            for leaf in jax.tree_util.tree_leaves(p):
                lf = leaf.astype(jnp.float32)
                l1_parts.append(
                    jax.lax.with_sharding_constraint(
                        jnp.sum(jnp.abs(lf), dtype=jnp.float32), self.replicated
                    )
                )
                onb = is_on_half_quant_boundary(
                    lf, qtype=qtype, bits=qbits, step=qstep, lo=qlo, hi=qhi
                )
                onb_parts.append(
                    jax.lax.with_sharding_constraint(
                        jnp.sum(onb.astype(jnp.float32), dtype=jnp.float32),
                        self.replicated,
                    )
                )
            return jnp.stack(l1_parts), jnp.stack(onb_parts)

        self._l1_and_bfrac_jit = jax.jit(_l1_and_bfrac_fn)

        def _free_mask_leaf(x):
            xf = x.astype(jnp.float32)
            if region_mode == "half_quant" and boundary_mode == "sticky":
                q_lo, q_hi = half_quant_bounds(
                    xf, qtype=qtype, bits=qbits, step=qstep, lo=qlo, hi=qhi
                )
                return (q_lo < q_hi).astype(jnp.float32)
            if region_mode in ("half_quant", "orthant"):
                return (xf != 0.0).astype(jnp.float32)
            return jnp.ones_like(xf)

        def _cell_width_leaf(x):
            xf = x.astype(jnp.float32)
            if not cell_scaled:
                return jnp.ones_like(xf)
            return half_quant_cell_width(
                xf, qtype=qtype, bits=qbits, step=qstep, lo=qlo, hi=qhi
            )

        def _update_v_and_diag_fn(params, grad, v, step_arr, base_lr_arr):
            if clip_norm > 0.0:
                gnorm = tree_l2_norm(grad).astype(jnp.float32)
                c_k = jnp.minimum(
                    jnp.asarray(1.0, jnp.float32),
                    jnp.asarray(clip_norm, jnp.float32) / (gnorm + 1e-12),
                )
            else:
                c_k = jnp.asarray(1.0, jnp.float32)

            v_next = jax.tree_util.tree_map(
                lambda vi, g: beta2 * vi
                + (1.0 - beta2) * ((c_k * g.astype(jnp.float32)) ** 2),
                v,
                grad,
            )
            bc2 = 1.0 - (beta2 ** step_arr.astype(jnp.float32))
            vhat_sum_leaves = [
                jax.lax.with_sharding_constraint(
                    jnp.sum(vi / bc2, dtype=jnp.float32), self.replicated
                )
                for vi in jax.tree_util.tree_leaves(v_next)
            ]
            rms_v = jnp.sqrt(
                jnp.maximum(jnp.sum(jnp.stack(vhat_sum_leaves)) / n_params_f32, 0.0)
            )
            sens_floor = 0.1 * rms_v + 1e-12
            denom_scale = rms_v + 1e-12

            free_count_leaves = []

            def _h0_diag_leaf(p_leaf, v_leaf):
                m_free = _free_mask_leaf(p_leaf)
                free_count_leaves.append(
                    jax.lax.with_sharding_constraint(
                        jnp.sum(m_free, dtype=jnp.float32), self.replicated
                    )
                )
                w_leaf = _cell_width_leaf(p_leaf)
                sqrt_vh = jnp.sqrt(jnp.maximum(v_leaf / bc2, 0.0))
                inv_sens = jnp.clip(
                    (rms_v + sens_floor) / (sqrt_vh + sens_floor), 0.05, 8.0
                )
                # Units: [params] / [grad] so two-loop L-BFGS secant terms match!
                return (base_lr_arr * w_leaf * inv_sens / denom_scale) * m_free

            h0_diag = jax.tree_util.tree_map(_h0_diag_leaf, params, v_next)
            free_total = jnp.sum(jnp.stack(free_count_leaves))

            # Scaled L1 pseudo-gradient in [grad] units: rms_v * sign(params) * m_free
            xi_grad = jax.tree_util.tree_map(
                lambda p_leaf: (
                    denom_scale
                    * jnp.sign(p_leaf.astype(jnp.float32))
                    * _free_mask_leaf(p_leaf)
                ).astype(p_leaf.dtype),
                params,
            )
            grad_free = jax.tree_util.tree_map(
                lambda p_leaf, g_leaf: (
                    g_leaf.astype(jnp.float32) * _free_mask_leaf(p_leaf)
                ).astype(g_leaf.dtype),
                params,
                grad,
            )
            gtan_l1 = jax.tree_util.tree_map(
                lambda h_i, xi_i: (-h_i * xi_i.astype(jnp.float32)).astype(xi_i.dtype),
                h0_diag,
                xi_grad,
            )
            gtan_data = jax.tree_util.tree_map(
                lambda h_i, g_i: (-h_i * g_i.astype(jnp.float32)).astype(g_i.dtype),
                h0_diag,
                grad_free,
            )
            return v_next, h0_diag, xi_grad, grad_free, gtan_l1, gtan_data, free_total

        self._update_v_and_diag_jit = jax.jit(
            _update_v_and_diag_fn,
            out_shardings=(
                param_sharding,
                param_sharding,
                param_sharding,
                param_sharding,
                param_sharding,
                param_sharding,
                self.replicated,
            ),
            donate_argnums=(2,),
        )

        # Two-loop L-BFGS kernels sharing h0_diag
        def _loop1_step_fn(q, s_i, y_i, rho_i):
            alpha_i = rho_i * tree_vdot(s_i, q, accum_dtype=jnp.float32)
            q_next = tree_add_scaled(q, -alpha_i, y_i)
            return q_next, alpha_i

        def _loop2_step_fn(r, s_i, y_i, rho_i, alpha_i):
            beta_i = rho_i * tree_vdot(y_i, r, accum_dtype=jnp.float32)
            return tree_add_scaled(r, alpha_i - beta_i, s_i)

        self._loop1_first_jit = jax.jit(
            _loop1_step_fn, out_shardings=(param_sharding, self.replicated)
        )
        self._loop1_donate_jit = jax.jit(
            _loop1_step_fn,
            out_shardings=(param_sharding, self.replicated),
            donate_argnums=(0,),
        )
        self._loop2_step_jit = jax.jit(
            _loop2_step_fn, out_shardings=param_sharding, donate_argnums=(0,)
        )
        self._apply_h0_jit = jax.jit(
            lambda q, h0: jax.tree_util.tree_map(
                lambda qi, hi: (hi * qi.astype(jnp.float32)).astype(qi.dtype), q, h0
            ),
            out_shardings=param_sharding,
            donate_argnums=(0,),
        )

        def _finalize_dir_fn(r, ref_grad, diag_fallback, params, lr_mult_arr):
            # Mask out locked boundary coordinates and negate r -> -H * ref_grad
            d_raw = jax.tree_util.tree_map(
                lambda ri, pi: (-ri.astype(jnp.float32) * _free_mask_leaf(pi)).astype(
                    pi.dtype
                ),
                r,
                params,
            )
            slope = tree_vdot(ref_grad, d_raw, accum_dtype=jnp.float32)
            d_valid = tree_where(slope < 0.0, d_raw, diag_fallback)
            if max_qn_gain > 0.0:
                d_norm = tree_l2_norm(d_valid).astype(jnp.float32)
                fb_norm = tree_l2_norm(diag_fallback).astype(jnp.float32)
                cap = jnp.asarray(max_qn_gain, jnp.float32) * fb_norm
                gain = jnp.minimum(1.0, cap / (d_norm + 1e-12))
            else:
                gain = jnp.asarray(1.0, jnp.float32)
            tot = lr_mult_arr * gain
            return jax.tree_util.tree_map(
                lambda di: (tot.astype(di.dtype) * di).astype(di.dtype), d_valid
            )

        self._finalize_dir_jit = jax.jit(
            _finalize_dir_fn, out_shardings=param_sharding, donate_argnums=(0,)
        )
        self._scale_jit = jax.jit(
            lambda alpha, x: jax.tree_util.tree_map(
                lambda t: (jnp.asarray(alpha, dtype=t.dtype) * t).astype(t.dtype), x
            ),
            out_shardings=param_sharding,
        )

        # Closed-form Riemannian KKT projection: d* = d_l1 + lambda* d_data
        def _kkt_combine_fn(
            params,
            grad,
            gtan_l1,
            gtan_data,
            dqn_l1,
            dqn_data,
            headroom_arr,
            enforce_arr,
        ):
            s_data_qn = -tree_vdot(grad, dqn_data, accum_dtype=jnp.float32)
            s_data_tan = -tree_vdot(grad, gtan_data, accum_dtype=jnp.float32)
            c_l1_qn = tree_vdot(grad, dqn_l1, accum_dtype=jnp.float32)
            c_l1_tan = tree_vdot(grad, gtan_l1, accum_dtype=jnp.float32)

            # First-order training-loss budget b_k: spend at most 35% of positive
            # headroom, and add a small inward cushion when hugging the ceiling so
            # second-order curvature never blocks all t > 0 steps.
            pos_h = jnp.maximum(headroom_arr, 0.0)
            neg_h = jnp.minimum(headroom_arr, 0.0)
            inward_qn = jnp.where(
                pos_h < 0.2 * jnp.maximum(s_data_qn, 1e-12),
                0.08 * jnp.maximum(s_data_qn, 0.0),
                0.0,
            )
            inward_tan = jnp.where(
                pos_h < 0.2 * jnp.maximum(s_data_tan, 1e-12),
                0.08 * jnp.maximum(s_data_tan, 0.0),
                0.0,
            )
            b_qn = 0.35 * pos_h + neg_h - inward_qn
            b_tan = 0.35 * pos_h + neg_h - inward_tan

            lam_qn = jnp.where(
                enforce_arr & (s_data_qn > 1e-20),
                jnp.maximum(0.0, c_l1_qn - b_qn) / (s_data_qn + 1e-20),
                0.0,
            )
            lam_tan = jnp.where(
                enforce_arr & (s_data_tan > 1e-20),
                jnp.maximum(0.0, c_l1_tan - b_tan) / (s_data_tan + 1e-20),
                0.0,
            )
            d_qn = tree_add_scaled(dqn_l1, lam_qn, dqn_data)
            g_tan = tree_add_scaled(gtan_l1, lam_tan, gtan_data)

            # L1 directional slope <sign(params), d_qn>
            xi_sign = jax.tree_util.tree_map(
                lambda p: jnp.sign(p.astype(jnp.float32)), params
            )
            l1_slope_qn = tree_vdot(xi_sign, d_qn, accum_dtype=jnp.float32)
            return g_tan, d_qn, b_qn, lam_qn, l1_slope_qn

        self._kkt_combine_jit = jax.jit(
            _kkt_combine_fn,
            out_shardings=(
                param_sharding,
                param_sharding,
                self.replicated,
                self.replicated,
                self.replicated,
            ),
        )

        # Projected QQN curved-path candidate:
        #   raw_cand = params + path.offset(t, g_tan, d_qn)
        #   nd_cand  = NoDecrease(params, raw_cand, budget = t * b_qn)
        #   reg_cand = Region.project(params, nd_cand)
        #   safe_cand = convex post-clip half-space reconciliation (keeps all
        #               loss-favorable 16-bit boundary snaps exact!)
        def _build_candidate_fn(
            params, grad, g_tan, d_qn, t_arr, b_budget_arr, enforce_arr
        ):
            d_t = self.path.offset(t_arr, g_tan, d_qn)
            budget_t = t_arr * b_budget_arr

            # 1. Pre-clip Euclidean half-space guard on free coordinates
            grad_free = jax.tree_util.tree_map(
                lambda p, g: g.astype(jnp.float32) * _free_mask_leaf(p), params, grad
            )
            c_raw = tree_vdot(grad_free, d_t, accum_dtype=jnp.float32)
            gg_free = tree_vdot(grad_free, grad_free, accum_dtype=jnp.float32)
            nd_coeff = jnp.where(
                enforce_arr & (gg_free > 1e-20),
                jnp.maximum(0.0, c_raw - budget_t) / (gg_free + 1e-20),
                0.0,
            )
            cand_nd = jax.tree_util.tree_map(
                lambda p, dt, gf: (
                    p.astype(jnp.float32) + dt.astype(jnp.float32) - nd_coeff * gf
                ).astype(p.dtype),
                params,
                d_t,
                grad_free,
            )

            # 2. Half-Quantization (or control) region projection
            if self.region is not None:
                cand_reg = self.region.project(params, cand_nd, ())
            else:
                cand_reg = cand_nd

            # 3. Post-clip convex half-space reconciliation:
            #    Decompose <grad, cand_reg - params> = P_pos - P_neg.
            #    Keep 100% of loss-decreasing coordinates (g_i * s_i <= 0) at their
            #    exact 16-bit boundary, and convexly damp only loss-increasing ones
            #    if box clipping broke <grad, s> <= budget_t.
            pos_parts = []
            neg_parts = []
            for p_l, c_l, g_l in zip(
                jax.tree_util.tree_leaves(params),
                jax.tree_util.tree_leaves(cand_reg),
                jax.tree_util.tree_leaves(grad),
            ):
                prod = g_l.astype(jnp.float32) * (
                    c_l.astype(jnp.float32) - p_l.astype(jnp.float32)
                )
                pos_parts.append(
                    jax.lax.with_sharding_constraint(
                        jnp.sum(jnp.maximum(prod, 0.0), dtype=jnp.float32),
                        self.replicated,
                    )
                )
                neg_parts.append(
                    jax.lax.with_sharding_constraint(
                        jnp.sum(jnp.maximum(-prod, 0.0), dtype=jnp.float32),
                        self.replicated,
                    )
                )
            p_pos = jnp.sum(jnp.stack(pos_parts))
            p_neg = jnp.sum(jnp.stack(neg_parts))
            allowed_pos = jnp.maximum(0.0, p_neg + budget_t)
            gamma_pos = jnp.where(
                enforce_arr & (p_pos > allowed_pos),
                jnp.clip(allowed_pos / (p_pos + 1e-20), 0.0, 1.0),
                1.0,
            )

            def _reconcile_leaf(p_l, c_l, g_l):
                pf = p_l.astype(jnp.float32)
                cf = c_l.astype(jnp.float32)
                sf = cf - pf
                uphill = (g_l.astype(jnp.float32) * sf) > 0.0
                out = jnp.where(uphill, pf + gamma_pos * sf, cf)
                return out.astype(p_l.dtype)

            return jax.tree_util.tree_map(_reconcile_leaf, params, cand_reg, grad)

        self._build_candidate_jit = jax.jit(
            _build_candidate_fn, out_shardings=param_sharding
        )

        # Headroom-restoration fallback inside the half-quantization region
        def _restoration_candidate_fn(params, d_data, scale_arr):
            raw = tree_add_scaled(params, scale_arr, d_data)
            if self.region is not None:
                return self.region.project(params, raw, ())
            return raw

        self._restoration_candidate_jit = jax.jit(
            _restoration_candidate_fn, out_shardings=param_sharding
        )

        # Terminal gradient-guided exact 16-bit boundary snap (zero global sort!)
        def _final_snap_candidate_fn(params, grad, mult_arr):
            gnorm_sq = tree_vdot(grad, grad, accum_dtype=jnp.float32)
            rms_g = jnp.sqrt(jnp.maximum(gnorm_sq / n_params_f32, 0.0))
            thresh = mult_arr * (rms_g + 1e-20)

            def _snap_leaf(p_l, g_l):
                pf = p_l.astype(jnp.float32)
                gf = g_l.astype(jnp.float32)
                q_lo, q_hi = half_quant_bounds(
                    pf, qtype=qtype, bits=qbits, step=qstep, lo=qlo, hi=qhi
                )
                on_b = q_lo == q_hi
                q_inner = jnp.where(pf >= 0.0, q_lo, q_hi)
                q_outer = jnp.where(pf >= 0.0, q_hi, q_lo)
                # First-order training-loss change if moved to q_inner
                d_loss_inner = gf * (q_inner - pf)
                # Always pick q_inner if it also helps/preserves L_train, OR if |g_i| <= thresh
                pick_inner = (d_loss_inner <= 0.0) | (jnp.abs(gf) <= thresh)
                snapped = jnp.where(pick_inner, q_inner, q_outer)
                return jnp.where(on_b, q_lo, snapped).astype(p_l.dtype)

            return jax.tree_util.tree_map(_snap_leaf, params, grad)

        self._final_snap_candidate_jit = jax.jit(
            _final_snap_candidate_fn, out_shardings=param_sharding
        )

        # L-BFGS curvature pair extraction on training-loss gradients
        def _lbfgs_pair_fn(params, new_params, grad, new_grad):
            s = jax.tree_util.tree_map(
                lambda a, b: (a - b).astype(
                    jnp.bfloat16
                    if a.dtype == jnp.bfloat16 or a.size >= 4096
                    else a.dtype
                ),
                new_params,
                params,
            )
            y = jax.tree_util.tree_map(
                lambda a, b: (a - b).astype(
                    jnp.bfloat16
                    if a.dtype == jnp.bfloat16 or a.size >= 4096
                    else a.dtype
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

    def _lr_multiplier(self, step_count: int) -> float:
        if not self.cfg.use_lr_schedule:
            return 1.0
        progress = min(1.0, max(0.0, step_count / float(self.total_steps)))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    def _eval_l1_and_bfrac(self, p) -> tuple[float, float]:
        l1_vec, onb_vec = self._l1_and_bfrac_jit(p)
        l1_tot = float(np.sum(np.asarray(l1_vec, dtype=np.float64), dtype=np.float64))
        onb_tot = float(np.sum(np.asarray(onb_vec, dtype=np.float64), dtype=np.float64))
        return l1_tot, min(1.0, max(0.0, onb_tot / float(self.num_params)))

    def _two_loop_apply(
        self, vec, ref_grad, diag_fallback, h0_diag, params, pairs, lr_mult_arr
    ):
        if not pairs:
            return self._scale_jit(lr_mult_arr, diag_fallback)
        q = vec
        alphas = []
        for idx, (s_i, y_i, rho_arr) in enumerate(pairs):
            step_l1 = self._loop1_first_jit if idx == 0 else self._loop1_donate_jit
            q, alpha_i = step_l1(q, s_i, y_i, rho_arr)
            alphas.append(alpha_i)
        r = self._apply_h0_jit(q, h0_diag)
        for (s_i, y_i, rho_arr), alpha_i in zip(reversed(pairs), reversed(alphas)):
            r = self._loop2_step_jit(r, s_i, y_i, rho_arr, alpha_i)
        return self._finalize_dir_jit(r, ref_grad, diag_fallback, params, lr_mult_arr)

    def _update_lbfgs_history(
        self,
        ost: ShardedLBFGSState,
        params,
        new_params,
        grad,
        new_grad,
        accepted: bool,
    ) -> ShardedLBFGSState:
        if not accepted or ost.history_size <= 0:
            return ost
        s, y, ys_arr, yy_arr, ss_arr = self._lbfgs_pair_jit(
            params, new_params, grad, new_grad
        )
        ys, yy, ss = float(ys_arr), float(yy_arr), float(ss_arr)
        eps = 1e-10
        if ys > eps * ((yy * ss + eps) ** 0.5):
            rho_arr = jax.device_put(
                jnp.asarray(1.0 / ys, jnp.float32), self.replicated
            )
            ost.pairs.appendleft((s, y, rho_arr))
            while len(ost.pairs) > ost.history_size:
                old = ost.pairs.pop()
                del old
            if yy > 0.0:
                ost.gamma = ys / yy
        else:
            del s, y
        return ost

    def init(
        self,
        params,
        macro_batch,
        *,
        warm_precond: tuple[Any, int] | None = None,
    ) -> HostOWLQuantState:
        val, grad = self._vg_jit(self._clone_params_jit(params), macro_batch)
        if warm_precond is not None and warm_precond[0] is not None:
            v_init = jax.device_put(
                jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), warm_precond[0]),
                self.param_sharding,
            )
            init_step = max(0, int(warm_precond[1]))
        else:
            v_init = self._zeros_f32_jit(params)
            init_step = 0
        ost = ShardedLBFGSState(
            history_size=int(self.cfg.history_size), v=v_init, step=init_step
        )
        _, gnorm = self._finite_and_norm_jit(grad)
        l1_norm, bfrac = self._eval_l1_and_bfrac(params)
        ref_loss = float(val)
        return HostOWLQuantState(
            value=ref_loss,
            grad=grad,
            oracle_state=ost,
            ref_train_loss=ref_loss,
            loss_cap=ref_loss + self.no_worse_tol,
            l1_norm=l1_norm,
            boundary_frac=bfrac,
            step_size=1.0,
            error=float(gnorm),
            num_evals=1,
            step_count=0,
        )

    def export_preconditioner(self, state: HostOWLQuantState) -> tuple[Any, int] | None:
        if state.oracle_state.v is not None:
            return state.oracle_state.v, int(state.oracle_state.step)
        return None

    def refresh(self, params, state: HostOWLQuantState, macro_batch) -> HostOWLQuantState:
        val, grad = self._vg_jit(self._clone_params_jit(params), macro_batch)
        _, gnorm = self._finite_and_norm_jit(grad)
        if self.cfg.clear_lbfgs_on_refresh:
            state.oracle_state.pairs.clear()
        l1_norm, bfrac = self._eval_l1_and_bfrac(params)
        ref_loss = float(val)
        return HostOWLQuantState(
            value=ref_loss,
            grad=grad,
            oracle_state=state.oracle_state,
            ref_train_loss=ref_loss,
            loss_cap=ref_loss + self.no_worse_tol,
            l1_norm=l1_norm,
            boundary_frac=bfrac,
            step_size=1.0,
            error=float(gnorm),
            num_evals=state.num_evals + 1,
            step_count=state.step_count,
        )

    def step(
        self,
        params,
        state: HostOWLQuantState,
        macro_batch,
        *,
        will_refresh_next: bool = False,
        is_last_phase_step: bool = False,
    ):
        del will_refresh_next
        ost = state.oracle_state
        grad = state.grad
        lr_mult = self._lr_multiplier(state.step_count)
        lr_mult_arr = jnp.asarray(lr_mult, jnp.float32)
        step_t = jnp.asarray(ost.step + 1, jnp.int32)
        base_lr_arr = jnp.asarray(float(self.cfg.learning_rate), jnp.float32)

        (
            v_next,
            h0_diag,
            xi_grad,
            grad_free,
            gtan_l1_unsched,
            gtan_data_unsched,
            free_total_arr,
        ) = self._update_v_and_diag_jit(params, grad, ost.v, step_t, base_lr_arr)
        ost.v = v_next
        ost.step += 1
        free_total = float(free_total_arr)

        # Fast-path when 100% of sticky coordinates have already locked onto 16-bit boundaries
        if free_total <= 0.0:
            del h0_diag, xi_grad, grad_free, gtan_l1_unsched, gtan_data_unsched
            new_state = HostOWLQuantState(
                value=state.value,
                grad=grad,
                oracle_state=ost,
                ref_train_loss=state.ref_train_loss,
                loss_cap=state.loss_cap,
                l1_norm=state.l1_norm,
                boundary_frac=state.boundary_frac,
                step_size=0.0,
                error=state.error,
                num_evals=state.num_evals,
                step_count=state.step_count + 1,
            )
            metrics = {
                "loss": state.value,
                "grad_norm": state.error,
                "num_evals": state.num_evals,
                "step_size": 0.0,
                "ls_success": 1.0,
                "qn_slope": 0.0,
                "last_reduction": 0.0,
                "l1_reduction": 0.0,
                "kkt_lambda": 0.0,
                "loss_cap": state.loss_cap,
            }
            return params, new_state, metrics

        dqn_l1 = self._two_loop_apply(
            xi_grad,
            xi_grad,
            gtan_l1_unsched,
            h0_diag,
            params,
            ost.pairs,
            lr_mult_arr,
        )
        dqn_data = self._two_loop_apply(
            grad_free,
            grad_free,
            gtan_data_unsched,
            h0_diag,
            params,
            ost.pairs,
            lr_mult_arr,
        )
        gtan_l1 = self._scale_jit(lr_mult_arr, gtan_l1_unsched)
        gtan_data = self._scale_jit(lr_mult_arr, gtan_data_unsched)
        del h0_diag, xi_grad, grad_free, gtan_l1_unsched, gtan_data_unsched

        headroom = state.loss_cap - state.value
        enforce_arr = jnp.asarray(self.enforce_no_worse)
        g_tan, d_qn, b_qn_arr, lam_qn_arr, l1_slope_arr = self._kkt_combine_jit(
            params,
            grad,
            gtan_l1,
            gtan_data,
            dqn_l1,
            dqn_data,
            jnp.asarray(headroom, jnp.float32),
            enforce_arr,
        )
        del gtan_l1, gtan_data, dqn_l1
        kkt_lambda = float(lam_qn_arr)
        l1_slope = float(l1_slope_arr)

        max_iter = max(1, int(self.cfg.ls_max_iter))
        max_step = float(self.cfg.max_t)
        cur_t = float(state.step_size) if self.cfg.remember_step_size else 1.0
        cur_t = min(max(cur_t, 0.25), max_step)
        evals = 0

        best_t = 0.0
        best_v = state.value
        best_g = grad
        best_gnorm = state.error
        best_l1 = state.l1_norm
        best_bfrac = state.boundary_frac
        accepted = False
        used_restoration = False
        restoration_scale = 0.25

        def _offload_grad(g_tree):
            if g_tree is None or g_tree is grad:
                return g_tree
            leaves = jax.tree_util.tree_leaves(g_tree)
            if leaves and isinstance(leaves[0], jax.Array):
                return jax.device_get(g_tree)
            return g_tree

        def _restore_grad(g_tree):
            if g_tree is None or g_tree is grad:
                return g_tree
            leaves = jax.tree_util.tree_leaves(g_tree)
            if leaves and not isinstance(leaves[0], jax.Array):
                return jax.device_put(g_tree, self.param_sharding)
            return g_tree

        for _ in range(max_iter):
            t_arr = jnp.asarray(cur_t, jnp.float32)
            cand = self._build_candidate_jit(
                params, grad, g_tan, d_qn, t_arr, b_qn_arr, enforce_arr
            )
            cand_l1, cand_bfrac = self._eval_l1_and_bfrac(cand)
            v_arr, g_arr = self._vg_jit(cand, macro_batch)
            del cand
            evals += 1
            cand_v = float(v_arr)
            g_ok, gnorm_arr = self._finite_and_norm_jit(g_arr)
            ok = math.isfinite(cand_v) and bool(g_ok)

            if self.enforce_no_worse:
                #Feasible if within no-worse cap (or strictly improving if started above cap)
                eff_cap = max(state.loss_cap, state.value) if state.value > state.loss_cap else state.loss_cap
                feasible = ok and (cand_v <= eff_cap + 1e-6)
            else:
                feasible = ok

            l1_drop = state.l1_norm - cand_l1
            bfrac_gain = cand_bfrac - state.boundary_frac
            goal_ok = (l1_drop > 1e-9 * max(1.0, state.l1_norm)) or (bfrac_gain > 1e-7)

            if feasible and goal_ok:
                if best_g is not grad:
                    del best_g
                best_t = cur_t
                best_v = cand_v
                best_g = g_arr
                best_gnorm = float(gnorm_arr)
                best_l1 = cand_l1
                best_bfrac = cand_bfrac
                accepted = True
                break

            del g_arr
            cur_t *= 0.5

        # If curvature right at the loss ceiling prevented L1 progress, take a small
        # region-projected data-descent step to replenish headroom for the next step.
        if not accepted and self.enforce_no_worse:
            fb_cand = self._restoration_candidate_jit(
                params, dqn_data, jnp.asarray(restoration_scale, jnp.float32)
            )
            fb_l1, fb_bfrac = self._eval_l1_and_bfrac(fb_cand)
            fb_v_arr, fb_g = self._vg_jit(fb_cand, macro_batch)
            del fb_cand
            evals += 1
            fb_v = float(fb_v_arr)
            fb_g_ok, fb_gnorm_arr = self._finite_and_norm_jit(fb_g)
            if (
                math.isfinite(fb_v)
                and bool(fb_g_ok)
                and fb_v <= max(state.loss_cap, state.value)
                and (fb_v < state.value - 1e-6 or fb_bfrac > state.boundary_frac + 1e-7)
            ):
                best_t = restoration_scale
                best_v = fb_v
                best_g = fb_g
                best_gnorm = float(fb_gnorm_arr)
                best_l1 = fb_l1
                best_bfrac = fb_bfrac
                accepted = True
                used_restoration = True
            else:
                del fb_g

        if accepted:
            if used_restoration:
                best_p = self._restoration_candidate_jit(
                    params, dqn_data, jnp.asarray(restoration_scale, jnp.float32)
                )
            else:
                best_p = self._build_candidate_jit(
                    params,
                    grad,
                    g_tan,
                    d_qn,
                    jnp.asarray(best_t, jnp.float32),
                    b_qn_arr,
                    enforce_arr,
                )
        else:
            best_p = params

        del g_tan, d_qn, dqn_data
        best_g = _restore_grad(best_g)

        # Optional terminal exact 16-bit boundary hardening (`final_snap: true`)
        if (
            is_last_phase_step
            and self.final_snap
            and self.region_mode == "half_quant"
            and best_bfrac < 0.999999
        ):
            snap_mults = (
                (1e9, 2.0, 0.75, 0.25, 0.0) if self.enforce_no_worse else (1e9,)
            )
            best_g = _offload_grad(best_g)
            for mult in snap_mults:
                s_cand = self._final_snap_candidate_jit(
                    best_p,
                    _restore_grad(best_g),
                    jnp.asarray(mult, jnp.float32),
                )
                s_v = float(self._loss_jit(s_cand, macro_batch))
                evals += 1
                if math.isfinite(s_v) and (
                    (not self.enforce_no_worse) or s_v <= state.loss_cap + 1e-5
                ):
                    s_l1, s_bfrac = self._eval_l1_and_bfrac(s_cand)
                    s_v_arr, s_g = self._vg_jit(
                        self._clone_params_jit(s_cand), macro_batch
                    )
                    evals += 1
                    _, s_gnorm_arr = self._finite_and_norm_jit(s_g)
                    if best_p is not params:
                        del best_p
                    if best_g is not grad:
                        del best_g
                    best_p = s_cand
                    best_v = float(s_v_arr)
                    best_g = s_g
                    best_gnorm = float(s_gnorm_arr)
                    best_l1 = s_l1
                    best_bfrac = s_bfrac
                    accepted = True
                    break
                del s_cand
            best_g = _restore_grad(best_g)

        ost = self._update_lbfgs_history(
            ost, params, best_p, grad, best_g, accepted and (best_p is not params)
        )

        l1_reduction = state.l1_norm - best_l1
        loss_reduction = state.value - best_v
        new_num_evals = state.num_evals + evals
        new_state = HostOWLQuantState(
            value=best_v,
            grad=best_g,
            oracle_state=ost,
            ref_train_loss=state.ref_train_loss,
            loss_cap=state.loss_cap,
            l1_norm=best_l1,
            boundary_frac=best_bfrac,
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
            "qn_slope": l1_slope,
            "last_reduction": loss_reduction,
            "l1_reduction": l1_reduction,
            "kkt_lambda": kkt_lambda,
            "loss_cap": state.loss_cap,
        }
        return best_p, new_state, metrics


@dataclass
class HostPTQState:
    value: float
    ref_train_loss: float
    loss_cap: float
    error: float
    num_evals: int
    step_count: int = 0


class PTQRunner:
    """Direct Post-Training Quantization baselines: Round-to-Nearest (RTN) & Round-Toward-Zero (RTZ)."""

    is_host_stepped: bool = True

    def __init__(
        self,
        loss_fn: Callable,
        cfg: OptimizerConfig,
        mesh: Mesh,
        param_sharding: Any,
    ):
        self.cfg = cfg
        self.mesh = mesh
        self.param_sharding = param_sharding
        self.replicated = NamedSharding(mesh, P())
        self._loss_jit = jax.jit(loss_fn, out_shardings=self.replicated)

        qtype = cfg.quant_qtype
        qbits = int(cfg.quant_bits)
        qstep = cfg.quant_step
        qlo = float(cfg.quant_lo)
        qhi = float(cfg.quant_hi)
        mode = (cfg.ptq_mode or "rtn").lower()
        if mode not in ("rtn", "rtz"):
            raise ValueError(f"Unsupported ptq_mode {cfg.ptq_mode!r}; choose 'rtn' or 'rtz'.")
        self.mode = mode

        if mode == "rtz":
            self._quant_jit = jax.jit(
                lambda p: round_toward_zero_half_quant(
                    p, qtype=qtype, bits=qbits, step=qstep, lo=qlo, hi=qhi
                ),
                out_shardings=param_sharding,
            )
        else:
            self._quant_jit = jax.jit(
                lambda p: round_to_half_quant(
                    p, qtype=qtype, bits=qbits, step=qstep, lo=qlo, hi=qhi
                ),
                out_shardings=param_sharding,
            )

    def init(
        self,
        params,
        macro_batch,
        *,
        warm_precond: tuple[Any, int] | None = None,
    ) -> HostPTQState:
        del warm_precond
        val = float(self._loss_jit(params, macro_batch))
        return HostPTQState(
            value=val,
            ref_train_loss=val,
            loss_cap=val + float(self.cfg.no_worse_tol),
            error=0.0,
            num_evals=1,
            step_count=0,
        )

    def refresh(self, params, state: HostPTQState, macro_batch) -> HostPTQState:
        val = float(self._loss_jit(params, macro_batch))
        return HostPTQState(
            value=val,
            ref_train_loss=val,
            loss_cap=val + float(self.cfg.no_worse_tol),
            error=0.0,
            num_evals=state.num_evals + 1,
            step_count=state.step_count,
        )

    def step(
        self,
        params,
        state: HostPTQState,
        macro_batch,
        *,
        will_refresh_next: bool = False,
        is_last_phase_step: bool = False,
    ):
        del will_refresh_next, is_last_phase_step
        if state.step_count == 0:
            q_params = self._quant_jit(params)
            new_val = float(self._loss_jit(q_params, macro_batch))
            new_evals = state.num_evals + 1
        else:
            q_params = params
            new_val = state.value
            new_evals = state.num_evals
        feasible = new_val <= state.loss_cap + 1e-6
        new_state = HostPTQState(
            value=new_val,
            ref_train_loss=state.ref_train_loss,
            loss_cap=state.loss_cap,
            error=0.0,
            num_evals=new_evals,
            step_count=state.step_count + 1,
        )
        metrics = {
            "loss": new_val,
            "grad_norm": 0.0,
            "num_evals": new_evals,
            "step_size": 1.0,
            "ls_success": 1.0 if feasible else 0.0,
            "qn_slope": 0.0,
            "last_reduction": state.value - new_val,
            "loss_cap": state.loss_cap,
        }
        return q_params, new_state, metrics
