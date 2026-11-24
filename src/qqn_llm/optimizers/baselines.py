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


def _extract_adam_nu_and_count(tree: Any) -> tuple[Any, int] | None:
    """Recursively locate ``ScaleByAdamState(count, mu, nu)`` inside an Optax state tuple."""
    if hasattr(tree, "nu") and hasattr(tree, "count") and hasattr(tree, "mu"):
        return tree.nu, int(tree.count)
    if isinstance(tree, tuple):
        for elem in tree:
            found = _extract_adam_nu_and_count(elem)
            if found is not None:
                return found
    return None


def _inject_adam_nu_and_count(tree: Any, nu_warm: Any, step_warm: int) -> Any:
    """Return a copy of ``tree`` with ``ScaleByAdamState.(nu, count)`` warm-started."""
    if hasattr(tree, "nu") and hasattr(tree, "count") and hasattr(tree, "mu"):
        nu_cast = jax.tree_util.tree_map(
            lambda w, ref: w.astype(ref.dtype), nu_warm, tree.nu
        )
        cnt = jnp.asarray(max(0, int(step_warm)), dtype=tree.count.dtype)
        return tree._replace(nu=nu_cast, count=cnt)
    if isinstance(tree, tuple):
        items = [_inject_adam_nu_and_count(elem, nu_warm, step_warm) for elem in tree]
        if hasattr(tree, "_fields"):
            return type(tree)(*items)
        return tuple(items)
    return tree


class AdamWRunner:
    """Standard full-weight LLM fine-tuning baseline: warmup + cosine AdamW."""

    def __init__(self, loss_fn: Callable, cfg: OptimizerConfig, total_steps: int):
        self.loss_fn = loss_fn
        self.cfg = cfg
        warmup_steps = max(0, int(round(cfg.warmup_ratio * total_steps)))
        if warmup_steps > 0:
            decay_steps = max(1, total_steps - warmup_steps)
            # Start at lr / (warmup_steps + 1) rather than 0.0 so step 1 (count=0)
            # performs a genuine warmup update instead of a wasted zero-LR pass.
            init_lr = float(cfg.learning_rate) / float(warmup_steps + 1)
            schedule = optax.join_schedules(
                schedules=[
                    optax.linear_schedule(
                        init_value=init_lr,
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
        else:
            schedule = optax.cosine_decay_schedule(
                init_value=cfg.learning_rate,
                decay_steps=max(1, total_steps),
                alpha=0.1,
            )

        transforms = []
        if cfg.grad_clip_norm > 0.0:
            transforms.append(optax.clip_by_global_norm(cfg.grad_clip_norm))
        transforms.append(
            optax.adamw(
                learning_rate=schedule,
                b2=cfg.beta2,
                weight_decay=cfg.weight_decay,
                mu_dtype=jnp.float32,
            )
        )
        self.tx = optax.chain(*transforms)
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

    def apply_warm_preconditioner(
        self, state: BaselineState, warm_precond: tuple[Any, int] | None
    ) -> BaselineState:
        if warm_precond is None or warm_precond[0] is None:
            return state
        new_opt_state = _inject_adam_nu_and_count(
            state.opt_state, warm_precond[0], warm_precond[1]
        )
        return state._replace(opt_state=new_opt_state)

    def export_preconditioner(self, state: BaselineState) -> tuple[Any, int] | None:
        return _extract_adam_nu_and_count(state.opt_state)

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

    def apply_warm_preconditioner(self, state: BaselineState, warm_precond) -> BaselineState:
        del warm_precond
        return state

    def export_preconditioner(self, state: BaselineState) -> tuple[Any, int] | None:
        del state
        return None

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
