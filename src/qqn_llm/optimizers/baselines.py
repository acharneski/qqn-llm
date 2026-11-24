"""Baseline optimizer runners (Optax AdamW and standalone L-BFGS)."""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import optax
from qqn_jax.utils import tree_l2_norm

from qqn_llm.config import OptimizerConfig


class BaselineState(NamedTuple):
    opt_state: Any
    step: jnp.ndarray
    num_evals: jnp.ndarray
    value: jnp.ndarray
    grad_norm: jnp.ndarray


class AdamWRunner:
    """Standard full-weight LLM fine-tuning baseline: warmup + cosine AdamW."""

    def __init__(self, loss_fn: Callable, cfg: OptimizerConfig, total_steps: int):
        self.loss_fn = loss_fn
        self.cfg = cfg
        warmup_steps = max(1, int(cfg.warmup_ratio * total_steps))
        decay_steps = max(1, total_steps - warmup_steps)
        schedule = optax.join_schedules(
            schedules=[
                optax.linear_schedule(
                    init_value=0.0,
                    end_value=cfg.learning_rate,
                    transition_steps=warmup_steps,
                ),
                optax.cosine_decay_schedule(
                    init_value=cfg.learning_rate,
                    decay_steps=decay_steps,
                    alpha=0.1,
                ),
            ],
            boundaries=[warmup_steps],
        )
        self.tx = optax.chain(
            optax.clip_by_global_norm(cfg.grad_clip_norm),
            optax.adamw(
                learning_rate=schedule,
                weight_decay=cfg.weight_decay,
                mu_dtype=jnp.float32,
            ),
        )
        self._value_and_grad = jax.value_and_grad(loss_fn)

    def init(self, params, macro_batch):
        opt_state = self.tx.init(params)
        val = self.loss_fn(params, macro_batch)
        return BaselineState(
            opt_state=opt_state,
            step=jnp.asarray(0, jnp.int32),
            num_evals=jnp.asarray(0, jnp.int32),
            value=val,
            grad_norm=jnp.asarray(0.0, jnp.float32),
        )

    def refresh(self, params, state, macro_batch):
        return state

    def step(self, params, state: BaselineState, macro_batch):
        value, grads = self._value_and_grad(params, macro_batch)
        grad_norm = tree_l2_norm(grads)
        updates, new_opt_state = self.tx.update(grads, state.opt_state, params)
        new_params = optax.apply_updates(params, updates)
        new_params = jax.tree_util.tree_map(
            lambda new_p, old_p: new_p.astype(old_p.dtype), new_params, params
        )
        new_evals = state.num_evals + jnp.asarray(1, jnp.int32)
        new_state = BaselineState(
            opt_state=new_opt_state,
            step=state.step + jnp.asarray(1, jnp.int32),
            num_evals=new_evals,
            value=value,
            grad_norm=grad_norm,
        )
        metrics = {
            "loss": value,
            "grad_norm": grad_norm,
            "num_evals": new_evals,
            "step_size": jnp.asarray(1.0, jnp.float32),
            "ls_success": jnp.asarray(1.0, jnp.float32),
            "qn_slope": jnp.asarray(0.0, jnp.float32),
            "last_reduction": state.value - value,
        }
        return new_params, new_state, metrics


class OptaxLBFGSRunner:
    """Standalone Optax L-BFGS baseline with zoom line search."""

    def __init__(self, loss_fn: Callable, cfg: OptimizerConfig):
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.tx = optax.lbfgs(
            memory_size=cfg.history_size,
            linesearch=optax.scale_by_zoom_linesearch(
                max_linesearch_steps=cfg.ls_max_iter,
                initial_guess_strategy="one",
            ),
        )
        self._value_and_grad = optax.value_and_grad_from_state(loss_fn)

    def init(self, params, macro_batch):
        opt_state = self.tx.init(params)
        val = self.loss_fn(params, macro_batch)
        return BaselineState(
            opt_state=opt_state,
            step=jnp.asarray(0, jnp.int32),
            num_evals=jnp.asarray(0, jnp.int32),
            value=val,
            grad_norm=jnp.asarray(0.0, jnp.float32),
        )

    def refresh(self, params, state, macro_batch):
        return state

    def step(self, params, state: BaselineState, macro_batch):
        def fn(p):
            return self.loss_fn(p, macro_batch)

        value, grads = self._value_and_grad(params, state=state.opt_state)
        grad_norm = tree_l2_norm(grads)
        updates, new_opt_state = self.tx.update(
            grads, state.opt_state, params, value=value, grad=grads, value_fn=fn
        )
        new_params = optax.apply_updates(params, updates)
        new_params = jax.tree_util.tree_map(
            lambda new_p, old_p: new_p.astype(old_p.dtype), new_params, params
        )
        # Optax zoom line search does not expose exact eval count in public API;
        # count 1 base eval + 1 line search eval minimum (conservatively tracked).
        new_evals = state.num_evals + jnp.asarray(2, jnp.int32)
        new_state = BaselineState(
            opt_state=new_opt_state,
            step=state.step + jnp.asarray(1, jnp.int32),
            num_evals=new_evals,
            value=value,
            grad_norm=grad_norm,
        )
        metrics = {
            "loss": value,
            "grad_norm": grad_norm,
            "num_evals": new_evals,
            "step_size": jnp.asarray(1.0, jnp.float32),
            "ls_success": jnp.asarray(1.0, jnp.float32),
            "qn_slope": jnp.asarray(0.0, jnp.float32),
            "last_reduction": state.value - value,
        }
        return new_params, new_state, metrics
