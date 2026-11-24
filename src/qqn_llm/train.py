"""Unified JAX training & evaluation loop for Gemma 4 + qqn-jax experiments."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from qqn_llm.checkpoint import save_checkpoint
from qqn_llm.config import ExperimentConfig, OptimizerConfig, load_config
from qqn_llm.data import DatasetBundle, load_dataset_bundle
from qqn_llm.loss import make_accumulated_loss_fn
from qqn_llm.model import (
    ModelBundle,
    build_model_and_params,
    make_batch_sharding,
    make_fsdp_sharding,
)
from qqn_llm.optimizers import AdamWRunner, OptaxLBFGSRunner, QQNRunner


def _make_runner(loss_fn, opt_cfg: OptimizerConfig, total_steps: int, bundle: ModelBundle):
    if opt_cfg.kind == "qqn":
        return QQNRunner(
            loss_fn, opt_cfg, mesh=bundle.mesh, param_sharding=bundle.param_sharding
        )
    if opt_cfg.kind == "adamw":
        return AdamWRunner(loss_fn, opt_cfg, total_steps=total_steps)
    if opt_cfg.kind == "lbfgs_optax":
        return OptaxLBFGSRunner(loss_fn, opt_cfg)
    raise ValueError(f"Unknown optimizer kind: {opt_cfg.kind!r}")


def _shard_batch(batch: dict[str, jnp.ndarray], bundle: ModelBundle) -> dict[str, jnp.ndarray]:
    sharding = make_batch_sharding(batch, bundle.mesh)
    return jax.device_put(batch, sharding)


def _compute_weight_deviation(params, init_params) -> dict[str, float]:
    """Compute exact L1 and L2 distance from baseline weights leaf-by-leaf on CPU.

    Operates one leaf at a time on the host in float64 so it uses zero extra
    GPU VRAM even on 4.65B-parameter FSDP models.
    """
    l1_total = 0.0
    l2_sq_total = 0.0
    n_total = 0
    for leaf, init_leaf in zip(
        jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(init_params)
    ):
        diff = np.asarray(leaf, dtype=np.float32) - np.asarray(init_leaf, dtype=np.float32)
        l1_total += float(np.sum(np.abs(diff), dtype=np.float64))
        l2_sq_total += float(np.sum(diff.astype(np.float64) ** 2, dtype=np.float64))
        n_total += diff.size
    l2_total = math.sqrt(l2_sq_total)
    n_safe = max(1, n_total)
    return {
        "weight_l1_dist": l1_total,
        "weight_l2_dist": l2_total,
        "weight_l1_mean": l1_total / n_safe,
        "weight_l2_rms": l2_total / math.sqrt(n_safe),
    }


def run_single_optimizer(
    exp_cfg: ExperimentConfig,
    opt_cfg: OptimizerConfig,
    bundle: ModelBundle,
    data: DatasetBundle,
    output_dir: Path,
    loss_fn,
    eval_loss_jit,
    sharded_val_batch: dict[str, jnp.ndarray],
) -> dict[str, Any]:
    """Run one optimizer configuration end-to-end and write JSON telemetry."""
    runner = _make_runner(
        loss_fn, opt_cfg, total_steps=exp_cfg.training.steps, bundle=bundle
    )
    host_stepped = bool(getattr(runner, "is_host_stepped", False))

    print(f"[{opt_cfg.name}] Placing initial parameters onto FSDP mesh...", flush=True)
    params = jax.device_put(
        jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), bundle.params),
        bundle.param_sharding,
    )

    if exp_cfg.data.batch_mode == "fixed_subset":
        first_batch = _shard_batch(data.fixed_train_batch, bundle)
    else:
        batch_iter = data.macro_batch_iterator(exp_cfg.training.steps + 1)
        first_batch = _shard_batch(next(batch_iter), bundle)

    if host_stepped:
        init_fn = runner.init
        refresh_fn = runner.refresh
        step_fn = runner.step
    else:
        # Infer exact FSDP out_shardings for state & metrics so input sharding == output
        # sharding across every step (prevents step-2 recompile & enables true buffer donation).
        state_shapes = jax.eval_shape(runner.init, params, first_batch)
        state_sharding = make_fsdp_sharding(state_shapes, bundle.mesh)
        replicated = NamedSharding(bundle.mesh, P())
        _, _, metrics_shapes = jax.eval_shape(runner.step, params, state_shapes, first_batch)
        metrics_sharding = jax.tree_util.tree_map(lambda _: replicated, metrics_shapes)

        init_fn = jax.jit(runner.init, out_shardings=state_sharding)
        refresh_fn = jax.jit(
            runner.refresh, out_shardings=state_sharding, donate_argnums=(1,)
        )
        step_fn = jax.jit(
            runner.step,
            out_shardings=(bundle.param_sharding, state_sharding, metrics_sharding),
            donate_argnums=(0, 1),
        )

    t_compile_start = time.perf_counter()
    print(
        f"[{opt_cfg.name}] Compiling & running init (batch shape={first_batch['input_ids'].shape})...",
        flush=True,
    )
    state = init_fn(params, first_batch)
    if not host_stepped:
        jax.block_until_ready(state)
    init_train_loss = float(state.value)
    print(
        f"[{opt_cfg.name}] init finished in {time.perf_counter() - t_compile_start:.1f}s "
        f"(init_train_loss={init_train_loss:.4f}); running initial val eval...",
        flush=True,
    )
    init_val = float(eval_loss_jit(params, sharded_val_batch))
    compile_and_init_sec = time.perf_counter() - t_compile_start
    print(
        f"[{opt_cfg.name}] Initial val_loss={init_val:.4f} (total init={compile_and_init_sec:.1f}s); starting steps...",
        flush=True,
    )

    history: list[dict[str, Any]] = []
    tokens_per_batch = int(jnp.sum(first_batch["loss_mask"]))
    current_batch = first_batch
    total_train_sec = 0.0
    steady_train_sec = 0.0
    best_val_loss = init_val
    best_val_weight_l1 = 0.0
    best_val_weight_l2 = 0.0
    final_dev = {"weight_l1_dist": 0.0, "weight_l2_dist": 0.0, "weight_l1_mean": 0.0, "weight_l2_rms": 0.0}
    consecutive_stalls = 0
    stopped_early_reason: str | None = None

    for step_idx in range(1, exp_cfg.training.steps + 1):
        if (
            exp_cfg.data.batch_mode == "macro_batch"
            and step_idx > 1
            and (step_idx - 1) % max(1, exp_cfg.data.inner_steps_per_batch) == 0
        ):
            current_batch = _shard_batch(next(batch_iter), bundle)
            state = refresh_fn(params, state, current_batch)
            if not host_stepped:
                jax.block_until_ready(state)

        t0 = time.perf_counter()
        params, state, metrics = step_fn(params, state, current_batch)
        if not host_stepped:
            jax.block_until_ready(metrics["loss"])
        dt = time.perf_counter() - t0
        total_train_sec += dt
        if step_idx > 1:
            steady_train_sec += dt

        rec: dict[str, Any] = {
            "step": step_idx,
            "train_loss": float(metrics["loss"]),
            "grad_norm": float(metrics["grad_norm"]),
            "num_evals": int(metrics["num_evals"]),
            "tokens_seen": int(metrics["num_evals"]) * tokens_per_batch,
            "step_size": float(metrics["step_size"]),
            "ls_success": float(metrics["ls_success"]),
            "qn_slope": float(metrics["qn_slope"]),
            "last_reduction": float(metrics["last_reduction"]),
            "step_seconds": dt,
            "cum_seconds": total_train_sec,
        }

        if rec["ls_success"] < 0.5:
            consecutive_stalls += 1
        else:
            consecutive_stalls = 0

        # Check early-stop convergence / stall guard (fixed_subset mode only for train_loss floor)
        should_stop = False
        if (
            exp_cfg.data.batch_mode == "fixed_subset"
            and exp_cfg.training.stop_on_train_loss > 0.0
            and rec["train_loss"] <= exp_cfg.training.stop_on_train_loss
        ):
            should_stop = True
            stopped_early_reason = f"train_loss <= {exp_cfg.training.stop_on_train_loss}"
        elif (
            exp_cfg.training.stop_on_consecutive_stalls > 0
            and consecutive_stalls >= exp_cfg.training.stop_on_consecutive_stalls
        ):
            should_stop = True
            stopped_early_reason = f"{consecutive_stalls} consecutive line-search stalls"

        if (
            exp_cfg.training.eval_every > 0
            and (
                step_idx % exp_cfg.training.eval_every == 0
                or step_idx == exp_cfg.training.steps
                or should_stop
            )
        ):
            val_loss = float(eval_loss_jit(params, sharded_val_batch))
            dev = _compute_weight_deviation(params, bundle.params)
            final_dev = dev
            rec["val_loss"] = val_loss
            rec["val_ppl"] = float(math.exp(min(val_loss, 20.0)))
            rec.update(dev)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_weight_l1 = dev["weight_l1_dist"]
                best_val_weight_l2 = dev["weight_l2_dist"]

        history.append(rec)
        val_str = (
            f" | val={rec['val_loss']:.4f} | ΔL1={rec['weight_l1_dist']:.1f} ΔL2={rec['weight_l2_dist']:.3f}"
            if "val_loss" in rec
            else ""
        )
        print(
            f"[{opt_cfg.name}] step {step_idx:3d}/{exp_cfg.training.steps} "
            f"| loss={rec['train_loss']:.4f}{val_str} "
            f"| evals={rec['num_evals']:3d} | t={rec['step_size']:.3g} "
            f"| {dt:.2f}s",
            flush=True,
        )

        if (
            exp_cfg.training.checkpoint_every > 0
            and step_idx % exp_cfg.training.checkpoint_every == 0
        ):
            save_checkpoint(params, output_dir / "checkpoints" / f"{opt_cfg.name}_step_{step_idx}")

        if should_stop:
            print(f"[{opt_cfg.name}] Early stop at step {step_idx}: {stopped_early_reason}", flush=True)
            break

    result = {
        "experiment_name": exp_cfg.experiment_name,
        "optimizer_name": opt_cfg.name,
        "optimizer_kind": opt_cfg.kind,
        "model_variant": exp_cfg.model.variant,
        "master_dtype": exp_cfg.model.master_dtype,
        "compute_dtype": exp_cfg.model.dtype,
        "num_params": bundle.num_params,
        "devices": [str(d) for d in jax.devices()],
        "init_train_loss": init_train_loss,
        "init_val_loss": init_val,
        "best_val_loss": best_val_loss,
        "best_val_weight_l1_dist": best_val_weight_l1,
        "best_val_weight_l2_dist": best_val_weight_l2,
        "final_weight_l1_dist": final_dev["weight_l1_dist"],
        "final_weight_l2_dist": final_dev["weight_l2_dist"],
        "final_weight_l1_mean": final_dev["weight_l1_mean"],
        "final_weight_l2_rms": final_dev["weight_l2_rms"],
        "compile_and_init_seconds": compile_and_init_sec,
        "total_train_seconds": total_train_sec,
        "steady_train_seconds": steady_train_sec,
        "history": history,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{opt_cfg.name}.json"
    out_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    # Explicitly release device arrays before returning so the next optimizer starts clean
    del params, state, runner
    gc.collect()
    return result


def run_experiment(
    exp_cfg: ExperimentConfig, output_dir: str | Path | None = None
) -> list[dict[str, Any]]:
    """Run all optimizers configured in ``exp_cfg``."""
    out_path = Path(output_dir or exp_cfg.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(
        f"[qqn-llm] Experiment={exp_cfg.experiment_name} | "
        f"model={exp_cfg.model.variant} | devices={jax.devices()}",
        flush=True,
    )
    bundle = build_model_and_params(
        exp_cfg.model,
        seed=exp_cfg.seed,
        fsdp_axis_size=exp_cfg.training.fsdp_axis_size,
    )
    print(f"[qqn-llm] Loaded {bundle.num_params:,} trainable parameters.", flush=True)
    data = load_dataset_bundle(exp_cfg.data, bundle.vocab_size, exp_cfg.seed)

    loss_fn = make_accumulated_loss_fn(
        bundle.forward_fn, l2_reg=exp_cfg.training.l2_reg, mesh=bundle.mesh
    )
    eval_loss_jit = jax.jit(loss_fn)
    sharded_val_batch = _shard_batch(data.val_batch, bundle)

    results = []
    for opt_cfg in exp_cfg.optimizers:
        try:
            res = run_single_optimizer(
                exp_cfg,
                opt_cfg,
                bundle,
                data,
                out_path,
                loss_fn=loss_fn,
                eval_loss_jit=eval_loss_jit,
                sharded_val_batch=sharded_val_batch,
            )
            results.append(res)
        except Exception as exc:
            print(f"[{opt_cfg.name}] FAILED: {exc}", flush=True)
            traceback.print_exc()
        finally:
            gc.collect()
            jax.clear_caches()
    return results


def main():
    parser = argparse.ArgumentParser(description="Run Gemma 4 + qqn-jax fine-tuning experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML experiment config")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_experiment(cfg, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
