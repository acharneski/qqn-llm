"""Tests for OWL-like Half-Quantization QQN runner, PTQ baselines, and FSDP compatibility."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from qqn_llm.config import OptimizerConfig, load_config
from qqn_llm.optimizers.owl_quant import (
    OWLQuantRunner,
    PTQRunner,
    make_device_quant_telemetry_fn,
)


def _single_device_mesh_and_sharding():
    mesh = Mesh(np.array(jax.devices()[:1]), ("fsdp",))
    return mesh, NamedSharding(mesh, P())


def test_owl_quant_beats_rtz_and_midpoint_on_anisotropic_no_worse_problem():
    """Verify OWL-HalfQuant reduces L1, lands 100% on 16-bit boundaries, and obeys L_train <= L_0."""
    mesh, sharding = _single_device_mesh_and_sharding()
    rng = np.random.default_rng(0)

    # Anisotropic loss mimicking sensitive RMSNorm/attention weights vs flat MLP weights:
    # Truncating `sensitive` toward 0 sharply increases L_train, whereas `flat` weights
    # can shrink freely onto 16-bit boundaries!
    sens_target = jnp.asarray(
        1.0 + 0.02 * rng.standard_normal((32,)), dtype=jnp.float32
    )
    flat_init = jnp.asarray(
        0.25 + 0.1 * rng.standard_normal((256,)), dtype=jnp.float32
    )
    # Start slightly off-minimum on `sensitive` so shrinking `sensitive` toward 0 hurts loss!
    p0 = {
        "sensitive": sens_target - jnp.asarray(0.004123, dtype=jnp.float32),
        "flat": flat_init + jnp.asarray(0.001357, dtype=jnp.float32),
    }
    n_params = 32 + 256

    def train_loss_fn(p, _batch):
        return 500.0 * jnp.mean((p["sensitive"] - sens_target) ** 2) + 1e-3 * jnp.mean(
            (p["flat"] - 0.3) ** 2
        )

    telem_fn = make_device_quant_telemetry_fn(
        mesh, n_params, qtype="uniform", bits=8, lo=-1.5, hi=1.5
    )
    init_loss = float(train_loss_fn(p0, None))
    init_stats = telem_fn(p0)

    # 1. PTQ-RTZ (unconstrained L1 inside cells) VIOLATES the no-worse constraint!
    cfg_rtz = OptimizerConfig(
        name="rtz",
        kind="ptq",
        ptq_mode="rtz",
        quant_qtype="uniform",
        quant_bits=8,
        quant_lo=-1.5,
        quant_hi=1.5,
    )
    rtz_runner = PTQRunner(train_loss_fn, cfg_rtz, mesh=mesh, param_sharding=sharding)
    st_rtz = rtz_runner.init(p0, None)
    p_rtz, st_rtz, _ = rtz_runner.step(p0, st_rtz, None)
    assert st_rtz.value > init_loss * 1.15, (
        f"Expected RTZ to degrade train loss: {init_loss} -> {st_rtz.value}"
    )

    # 2. Midpoint-cell control lands on midpoints (high quant_l1_error, ~0% exact boundary)
    cfg_mid = OptimizerConfig(
        name="mid",
        kind="owl_quant",
        history_size=4,
        learning_rate=0.8,
        ls_max_iter=5,
        quant_qtype="uniform",
        quant_bits=8,
        quant_lo=-1.5,
        quant_hi=1.5,
        quant_region_mode="midpoint_cell",
        no_worse_tol=0.0,
        enforce_no_worse=True,
    )
    mid_runner = OWLQuantRunner(
        train_loss_fn, cfg_mid, mesh=mesh, param_sharding=sharding, num_params=n_params
    )
    st_mid = mid_runner.init(p0, None)
    p_mid = p0
    for i in range(10):
        p_mid, st_mid, _ = mid_runner.step(
            p_mid, st_mid, None, is_last_phase_step=(i == 9)
        )
    mid_stats = telem_fn(p_mid)

    # 3. Proposed OWL-HalfQuant (`sticky` + `final_snap`) achieves:
    #    - ZERO no-worse violation (L_train <= init_loss)
    #    - Strict L1 norm reduction below initial & RTN
    #    - 100.0% exact grid boundary alignment (0.0 quantization rounding error!)
    cfg_owl = OptimizerConfig(
        name="owl_hq",
        kind="owl_quant",
        history_size=4,
        learning_rate=0.8,
        ls_max_iter=5,
        quant_qtype="uniform",
        quant_bits=8,
        quant_lo=-1.5,
        quant_hi=1.5,
        quant_region_mode="half_quant",
        quant_boundary_mode="sticky",
        no_worse_tol=0.0,
        enforce_no_worse=True,
        final_snap=True,
    )
    owl_runner = OWLQuantRunner(
        train_loss_fn, cfg_owl, mesh=mesh, param_sharding=sharding, num_params=n_params
    )
    st_owl = owl_runner.init(p0, None)
    p_owl = p0
    for i in range(10):
        p_owl, st_owl, _ = owl_runner.step(
            p_owl, st_owl, None, is_last_phase_step=(i == 9)
        )
    owl_stats = telem_fn(p_owl)

    assert st_owl.value <= init_loss + 1e-6
    assert owl_stats["param_l1_norm"] < init_stats["param_l1_norm"]
    assert owl_stats["quant_boundary_frac"] == 1.0
    assert owl_stats["quant_l1_error"] == 0.0
    assert owl_stats["quant_boundary_frac"] > mid_stats["quant_boundary_frac"] + 0.5
    assert owl_stats["quant_l1_error"] < mid_stats["quant_l1_error"] * 1e-3


def test_owl_quant_bfloat16_fsdp_4device_cpu():
    """Verify 4-device CPU FSDP sharding with OWLQuantRunner (32-bit -> 16-bit bfloat16)."""
    script = """
import jax
import jax.numpy as jnp
from qqn_llm.config import ExperimentConfig, ModelConfig, DataConfig, TrainingConfig, OptimizerConfig
from qqn_llm.train import run_experiment

assert len(jax.devices()) == 4
cfg = ExperimentConfig(
    experiment_name="test_owl_fsdp",
    output_dir="runs/_test_owl_fsdp",
    seed=7,
    model=ModelConfig(
        variant="tiny_smoke",
        dtype="float32",
        master_dtype="float32",
        remat=False,
        tiny_num_layers=1,
        tiny_embed_dim=64,
        tiny_hidden_dim=128,
        tiny_num_heads=4,
        tiny_num_kv_heads=2,
        tiny_head_dim=16,
        tiny_vocab_size=256,
    ),
    data=DataConfig(
        source="synthetic",
        seq_len=32,
        train_examples=16,
        val_examples=8,
        micro_batch_size=8,
        batch_mode="fixed_subset",
    ),
    training=TrainingConfig(
        steps=4,
        eval_every=2,
        fsdp_axis_size=4,
        quant_eval_qtype="bfloat16",
    ),
    optimizers=[
        OptimizerConfig(
            name="owl_bf16_fsdp",
            kind="owl_quant",
            history_size=3,
            learning_rate=0.8,
            ls_max_iter=4,
            quant_qtype="bfloat16",
            quant_region_mode="half_quant",
            quant_boundary_mode="sticky",
            no_worse_tol=0.0,
            enforce_no_worse=True,
            final_snap=True,
        )
    ],
)
res = run_experiment(cfg, output_dir="runs/_test_owl_fsdp")[0]
assert res["final_no_worse_violation"] <= 1e-5
assert res["final_param_l1_delta"] < 0.0
assert res["final_quant_boundary_frac"] >= 0.9999
assert res["final_quant_l1_error"] <= 1e-6
print("OWL FSDP 4-device test PASSED:", res["final_param_l1_delta"], res["final_quant_boundary_frac"])
"""
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    env["PYTHONPATH"] = f"{root / 'src'}:{root / 'qqn-jax'}"
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"


def test_owl_quant_configs_parse_cleanly():
    for path in (
        "configs/owl_quant_local.yaml",
        "configs/owl_quant_stress_local.yaml",
        "configs/gemma4_e2b_aws_owl_quant.yaml",
    ):
        cfg = load_config(path)
        assert cfg.model.dtype == "float32"
        assert cfg.model.master_dtype == "float32"
        assert cfg.training.quant_eval_qtype in ("bfloat16", "float16", "uniform")
        assert len(cfg.optimizers) >= 5
