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
from qqn_llm.config import ExperimentConfig, OptimizerConfig, PhaseConfig, load_config
from qqn_llm.data import DatasetBundle, load_dataset_bundle
from qqn_llm.loss import make_accumulated_loss_fn
from qqn_llm.model import (
    ModelBundle,
    build_model_and_params,
    make_batch_sharding,
    make_fsdp_sharding,
)
from qqn_llm.optimizers import AdamWRunner, OptaxLBFGSRunner, QQNRunner


def _phase_to_opt_cfg(name: str, phase: PhaseConfig) -> OptimizerConfig:
    return OptimizerConfig(
        name=name,
        kind=phase.kind,
        oracle=phase.oracle,
        history_size=phase.history_size,
        line_search=phase.line_search,
        path_strategy=phase.path_strategy,
        c1=phase.c1,
        c2=phase.c2,
        ls_max_iter=phase.ls_max_iter,
        max_t=phase.max_t,
        remember_step_size=phase.remember_step_size,
        spline_refine_rounds=phase.spline_refine_rounds,
        spline_max_control_points=phase.spline_max_control_points,
        use_lr_schedule=phase.use_lr_schedule,
        clear_lbfgs_on_refresh=phase.clear_lbfgs_on_refresh,
        tangent_mode=phase.tangent_mode,
        beta2=phase.beta2,
        max_qn_gain=phase.max_qn_gain,
        anchor_weight_decay=phase.anchor_weight_decay,
        learning_rate=phase.learning_rate,
        warmup_ratio=phase.warmup_ratio,
        weight_decay=phase.weight_decay,
        grad_clip_norm=phase.grad_clip_norm,
    )


def _resolve_phases(exp_cfg: ExperimentConfig, opt_cfg: OptimizerConfig) -> list[PhaseConfig]:
    if opt_cfg.phases:
        return opt_cfg.phases
    return [
        PhaseConfig(
            name="phase_1",
            steps=exp_cfg.training.steps,
            macro_batch_examples=exp_cfg.data.macro_batch_examples,
            inner_steps_per_batch=exp_cfg.data.inner_steps_per_batch,
            kind=opt_cfg.kind,
            oracle=opt_cfg.oracle,
            history_size=opt_cfg.history_size,
            line_search=opt_cfg.line_search,
            path_strategy=opt_cfg.path_strategy,
            c1=opt_cfg.c1,
            c2=opt_cfg.c2,
            ls_max_iter=opt_cfg.ls_max_iter,
            max_t=opt_cfg.max_t,
            remember_step_size=opt_cfg.remember_step_size,
            spline_refine_rounds=opt_cfg.spline_refine_rounds,
            spline_max_control_points=opt_cfg.spline_max_control_points,
            use_lr_schedule=opt_cfg.use_lr_schedule,
            clear_lbfgs_on_refresh=opt_cfg.clear_lbfgs_on_refresh,
            tangent_mode=opt_cfg.tangent_mode,
            warm_start_preconditioner=True,
            beta2=opt_cfg.beta2,
            max_qn_gain=opt_cfg.max_qn_gain,
            anchor_weight_decay=opt_cfg.anchor_weight_decay,
            learning_rate=opt_cfg.learning_rate,
            warmup_ratio=opt_cfg.warmup_ratio,
            weight_decay=opt_cfg.weight_decay,
            grad_clip_norm=opt_cfg.grad_clip_norm,
        )
    ]


def _make_runner(loss_fn, opt_cfg: OptimizerConfig, total_steps: int, bundle: ModelBundle):
    if opt_cfg.kind == "qqn":
        return QQNRunner(
            loss_fn,
            opt_cfg,
            mesh=bundle.mesh,
            param_sharding=bundle.param_sharding,
            total_steps=total_steps,
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
    """Host fallback computing exact L1/L2 distance leaf-by-leaf in float64."""
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


def _make_device_weight_deviation_fn(sharded_anchor_params, mesh, num_params: int):
    """Build an FSDP-resident L1/L2 deviation reducer returning <4 KB of per-leaf scalars.

    Reduces each FSDP-sharded tensor on-device in float32 and accumulates the
    per-leaf partial sums on host in float64 — avoiding 18.6 GB PCIe transfers
    every evaluation step.
    """
    replicated = NamedSharding(mesh, P())

    @jax.jit
    def _leaf_stats(params, anchor):
        l1_leaves = [
            jax.lax.with_sharding_constraint(
                jnp.sum(
                    jnp.abs(p.astype(jnp.float32) - p0.astype(jnp.float32)),
                    dtype=jnp.float32,
                ),
                replicated,
            )
            for p, p0 in zip(
                jax.tree_util.tree_leaves(params),
                jax.tree_util.tree_leaves(anchor),
            )
        ]
        l2_sq_leaves = [
            jax.lax.with_sharding_constraint(
                jnp.sum(
                    (p.astype(jnp.float32) - p0.astype(jnp.float32)) ** 2,
                    dtype=jnp.float32,
                ),
                replicated,
            )
            for p, p0 in zip(
                jax.tree_util.tree_leaves(params),
                jax.tree_util.tree_leaves(anchor),
            )
        ]
        return jnp.stack(l1_leaves), jnp.stack(l2_sq_leaves)

    n_safe = max(1, int(num_params))

    def compute(params) -> dict[str, float]:
        l1_vec, l2_sq_vec = _leaf_stats(params, sharded_anchor_params)
        l1_total = float(np.sum(np.asarray(l1_vec, dtype=np.float64), dtype=np.float64))
        l2_sq_total = float(
            np.sum(np.asarray(l2_sq_vec, dtype=np.float64), dtype=np.float64)
        )
        l2_total = math.sqrt(max(0.0, l2_sq_total))
        return {
            "weight_l1_dist": l1_total,
            "weight_l2_dist": l2_total,
            "weight_l1_mean": l1_total / n_safe,
            "weight_l2_rms": l2_total / math.sqrt(n_safe),
        }

    return compute


def run_single_optimizer(
    exp_cfg: ExperimentConfig,
    opt_cfg: OptimizerConfig,
    bundle: ModelBundle,
    data: DatasetBundle,
    output_dir: Path,
    base_train_loss_fn,
    eval_loss_jit,
    sharded_val_batch: dict[str, jnp.ndarray],
) -> dict[str, Any]:
    """Run one optimizer configuration (single- or multi-phase) and write JSON telemetry."""
    phases = _resolve_phases(exp_cfg, opt_cfg)
    total_planned_steps = sum(p.steps for p in phases)

    print(f"[{opt_cfg.name}] Placing initial parameters onto FSDP mesh...", flush=True)
    params = jax.device_put(
        jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), bundle.params),
        bundle.param_sharding,
    )

    init_val = float(eval_loss_jit(params, sharded_val_batch))
    init_train_loss: float | None = None
    compile_and_init_sec = 0.0
    history: list[dict[str, Any]] = []
    total_train_sec = 0.0
    steady_train_sec = 0.0
    best_val_loss = init_val
    best_val_step = 0
    best_val_evals = 0
    best_val_weight_l1 = 0.0
    best_val_weight_l2 = 0.0
    final_dev = {
        "weight_l1_dist": 0.0,
        "weight_l2_dist": 0.0,
        "weight_l1_mean": 0.0,
        "weight_l2_rms": 0.0,
    }
    global_step = 0
    cum_evals = 0
    cum_tokens = 0
    carry_precond: tuple[Any, int] | None = None

    for phase_idx, phase in enumerate(phases):
        phase_opt_cfg = _phase_to_opt_cfg(opt_cfg.name, phase)
        # Only materialize a GPU copy of initial weights when L2-SP anchor regularization
        # is explicitly requested for this phase (keeps 1.14 GiB/GPU free otherwise).
        # Pass `sharded_anchor_params` as an explicit traced tuple input `(batch, anchor)`
        # rather than closing over it inside `loss_fn` so MLIR never embeds 9.3 GB of
        # weights as an inlined compile-time constant!
        sharded_anchor_params = None
        if phase.anchor_weight_decay > 0.0:
            sharded_anchor_params = jax.device_put(
                jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), bundle.params),
                bundle.param_sharding,
            )
            phase_loss_fn = make_accumulated_loss_fn(
                bundle.forward_fn,
                l2_reg=exp_cfg.training.l2_reg,
                mesh=bundle.mesh,
                anchor_params=None,
                anchor_l2_reg=phase.anchor_weight_decay,
            )
        else:
            phase_loss_fn = base_train_loss_fn

        def _wrap_phase_batch(sharded_b):
            return (
                (sharded_b, sharded_anchor_params)
                if sharded_anchor_params is not None
                else sharded_b
            )

        runner = _make_runner(
            phase_loss_fn, phase_opt_cfg, total_steps=phase.steps, bundle=bundle
        )
        host_stepped = bool(getattr(runner, "is_host_stepped", False))

        if exp_cfg.data.batch_mode == "fixed_subset" and not opt_cfg.phases:
            raw_first_batch = _shard_batch(data.fixed_train_batch, bundle)
            batch_iter = None
        else:
            batch_iter = data.macro_batch_iterator(
                phase.steps + 1,
                macro_batch_examples=phase.macro_batch_examples,
                seed_offset=phase_idx * 1000,
            )
            raw_first_batch = _shard_batch(next(batch_iter), bundle)
        first_batch = _wrap_phase_batch(raw_first_batch)

        if host_stepped:
            init_fn = runner.init
            refresh_fn = runner.refresh
            step_fn = runner.step
        else:
            state_shapes = jax.eval_shape(runner.init, params, first_batch)
            state_sharding = make_fsdp_sharding(state_shapes, bundle.mesh)
            replicated = NamedSharding(bundle.mesh, P())
            _, _, metrics_shapes = jax.eval_shape(
                runner.step, params, state_shapes, first_batch
            )
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

        t_phase_init = time.perf_counter()
        phase_tag = f"{opt_cfg.name}:{phase.name}" if len(phases) > 1 else opt_cfg.name
        warm_to_use = carry_precond if phase.warm_start_preconditioner else None
        carry_precond = None
        warm_note = (
            f", warm_v_step={warm_to_use[1]}"
            if (warm_to_use is not None and warm_to_use[0] is not None)
            else ""
        )
        print(
            f"[{phase_tag}] Compiling & running init (kind={phase.kind}, "
            f"oracle={phase.oracle if phase.kind == 'qqn' else '-'}, "
            f"batch shape={raw_first_batch['input_ids'].shape}, inner={phase.inner_steps_per_batch}{warm_note})...",
            flush=True,
        )
        if host_stepped:
            state = init_fn(params, first_batch, warm_precond=warm_to_use)
        else:
            state = init_fn(params, first_batch)
            if warm_to_use is not None and hasattr(runner, "apply_warm_preconditioner"):
                state = runner.apply_warm_preconditioner(state, warm_to_use)
                state = jax.device_put(state, state_sharding)
            jax.block_until_ready(state)
        del warm_to_use

        phase_init_sec = time.perf_counter() - t_phase_init
        compile_and_init_sec += phase_init_sec
        if init_train_loss is None:
            init_train_loss = float(state.value)
        print(
            f"[{phase_tag}] init finished in {phase_init_sec:.1f}s "
            f"(train_loss={float(state.value):.4f}, val_loss={init_val if phase_idx == 0 else best_val_loss:.4f}); starting {phase.steps} steps...",
            flush=True,
        )

        tokens_per_batch = int(jnp.sum(raw_first_batch["loss_mask"]))
        current_batch = first_batch
        del raw_first_batch, first_batch
        prev_phase_evals = int(state.num_evals)
        cum_evals += prev_phase_evals
        cum_tokens += prev_phase_evals * tokens_per_batch
        consecutive_stalls = 0
        stopped_early_reason: str | None = None
        inner_k = max(1, phase.inner_steps_per_batch)

        for local_step in range(1, phase.steps + 1):
            global_step += 1
            if (
                batch_iter is not None
                and local_step > 1
                and (local_step - 1) % inner_k == 0
            ):
                raw_next_batch = _shard_batch(next(batch_iter), bundle)
                tokens_per_batch = int(jnp.sum(raw_next_batch["loss_mask"]))
                current_batch = _wrap_phase_batch(raw_next_batch)
                del raw_next_batch
                state = refresh_fn(params, state, current_batch)
                if not host_stepped:
                    jax.block_until_ready(state)

            will_refresh_next = (
                batch_iter is not None
                and local_step < phase.steps
                and (local_step % inner_k == 0)
            )

            t0 = time.perf_counter()
            if host_stepped:
                params, state, metrics = step_fn(
                    params, state, current_batch, will_refresh_next=will_refresh_next
                )
            else:
                params, state, metrics = step_fn(params, state, current_batch)
                jax.block_until_ready(metrics["loss"])
            dt = time.perf_counter() - t0
            total_train_sec += dt
            if local_step > 1:
                steady_train_sec += dt

            cur_phase_evals = int(metrics["num_evals"])
            delta_evals = max(0, cur_phase_evals - prev_phase_evals)
            prev_phase_evals = cur_phase_evals
            cum_evals += delta_evals
            cum_tokens += delta_evals * tokens_per_batch

            rec: dict[str, Any] = {
                "step": global_step,
                "phase": phase.name,
                "phase_step": local_step,
                "train_loss": float(metrics["loss"]),
                "grad_norm": float(metrics["grad_norm"]),
                "num_evals": cum_evals,
                "tokens_seen": cum_tokens,
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

            should_stop = False
            if (
                exp_cfg.data.batch_mode == "fixed_subset"
                and not opt_cfg.phases
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

            if exp_cfg.training.eval_every > 0 and (
                global_step % exp_cfg.training.eval_every == 0
                or local_step == phase.steps
                or global_step == total_planned_steps
                or should_stop
            ):
                val_loss = float(eval_loss_jit(params, sharded_val_batch))
                dev = _compute_weight_deviation(params, bundle.params)
                final_dev = dev
                rec["val_loss"] = val_loss
                rec["val_ppl"] = float(math.exp(min(val_loss, 20.0)))
                rec.update(dev)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_step = global_step
                    best_val_evals = cum_evals
                    best_val_weight_l1 = dev["weight_l1_dist"]
                    best_val_weight_l2 = dev["weight_l2_dist"]
                    if exp_cfg.training.save_best_checkpoint:
                        ckpt_path = output_dir / "checkpoints" / f"{opt_cfg.name}_best"
                        print(
                            f"[{phase_tag}] Saving new best checkpoint -> {ckpt_path} "
                            f"(val={val_loss:.4f}, dtype={exp_cfg.training.checkpoint_dtype})...",
                            flush=True,
                        )
                        save_checkpoint(
                            params,
                            ckpt_path,
                            dtype=exp_cfg.training.checkpoint_dtype,
                            metadata={
                                "experiment_name": exp_cfg.experiment_name,
                                "optimizer_name": opt_cfg.name,
                                "model_variant": exp_cfg.model.variant,
                                "checkpoint_dtype": exp_cfg.training.checkpoint_dtype,
                                "step": global_step,
                                "val_loss": val_loss,
                                "train_loss": rec["train_loss"],
                                "weight_l1_dist": dev["weight_l1_dist"],
                                "weight_l2_dist": dev["weight_l2_dist"],
                                "data_source": exp_cfg.data.source,
                                "jsonl_path": exp_cfg.data.jsonl_path,
                            },
                        )

            history.append(rec)
            val_str = (
                f" | val={rec['val_loss']:.4f} | ΔL1={rec['weight_l1_dist']:.1f} ΔL2={rec['weight_l2_dist']:.3f}"
                if "val_loss" in rec
                else ""
            )
            print(
                f"[{phase_tag}] step {global_step:3d}/{total_planned_steps} "
                f"| loss={rec['train_loss']:.4f}{val_str} "
                f"| evals={rec['num_evals']:3d} | t={rec['step_size']:.3g} "
                f"| {dt:.2f}s",
                flush=True,
            )

            if (
                exp_cfg.training.checkpoint_every > 0
                and global_step % exp_cfg.training.checkpoint_every == 0
            ):
                save_checkpoint(
                    params,
                    output_dir / "checkpoints" / f"{opt_cfg.name}_step_{global_step}",
                    dtype=exp_cfg.training.checkpoint_dtype,
                )

            if should_stop:
                print(
                    f"[{phase_tag}] Early stop at step {global_step}: {stopped_early_reason}",
                    flush=True,
                )
                break

        # Export diagonal preconditioner (v, step) for seamless Phase-2 handoff before freeing state
        if (
            phase_idx + 1 < len(phases)
            and phases[phase_idx + 1].warm_start_preconditioner
            and hasattr(runner, "export_preconditioner")
        ):
            carry_precond = runner.export_preconditioner(state)
        del state, runner, sharded_anchor_params
        gc.collect()

    del carry_precond
    if exp_cfg.training.save_final_checkpoint:
        final_ckpt_path = output_dir / "checkpoints" / f"{opt_cfg.name}_final"
        print(
            f"[{opt_cfg.name}] Saving final checkpoint -> {final_ckpt_path} "
            f"(dtype={exp_cfg.training.checkpoint_dtype})...",
            flush=True,
        )
        save_checkpoint(
            params,
            final_ckpt_path,
            dtype=exp_cfg.training.checkpoint_dtype,
            metadata={
                "experiment_name": exp_cfg.experiment_name,
                "optimizer_name": opt_cfg.name,
                "model_variant": exp_cfg.model.variant,
                "checkpoint_dtype": exp_cfg.training.checkpoint_dtype,
                "step": global_step,
                "best_val_loss": best_val_loss,
                "weight_l1_dist": final_dev["weight_l1_dist"],
                "weight_l2_dist": final_dev["weight_l2_dist"],
                "data_source": exp_cfg.data.source,
                "jsonl_path": exp_cfg.data.jsonl_path,
            },
        )

    result = {
        "experiment_name": exp_cfg.experiment_name,
        "optimizer_name": opt_cfg.name,
        "optimizer_kind": opt_cfg.kind if not opt_cfg.phases else "two_phase",
        "model_variant": exp_cfg.model.variant,
        "master_dtype": exp_cfg.model.master_dtype,
        "compute_dtype": exp_cfg.model.dtype,
        "num_params": bundle.num_params,
        "devices": [str(d) for d in jax.devices()],
        "init_train_loss": init_train_loss if init_train_loss is not None else 0.0,
        "init_val_loss": init_val,
        "best_val_loss": best_val_loss,
        "best_val_step": best_val_step,
        "best_val_evals": best_val_evals,
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

    del params
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

    base_train_loss_fn = make_accumulated_loss_fn(
        bundle.forward_fn, l2_reg=exp_cfg.training.l2_reg, mesh=bundle.mesh
    )
    eval_loss_fn = (
        base_train_loss_fn
        if exp_cfg.training.l2_reg == 0.0
        else make_accumulated_loss_fn(bundle.forward_fn, l2_reg=0.0, mesh=bundle.mesh)
    )
    eval_loss_jit = jax.jit(eval_loss_fn)
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
                base_train_loss_fn=base_train_loss_fn,
                eval_loss_jit=eval_loss_jit,
                sharded_val_batch=sharded_val_batch,
            )
            results.append(res)
        except Exception as exc:
            print(f"[{opt_cfg.name}] FAILED: {exc}", flush=True)
            traceback.print_exc()
            if exc.__traceback__ is not None:
                traceback.clear_frames(exc.__traceback__)
        finally:
            gc.collect()
            jax.clear_caches()
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run Gemma 4 + qqn-jax fine-tuning experiment"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML experiment config"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Override output directory"
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_experiment(cfg, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
