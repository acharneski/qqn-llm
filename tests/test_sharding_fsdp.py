"""Tier 2 smoke test: verify 4-way FSDP sharding with PyTree QQN on virtual CPU devices."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_fsdp_4device_cpu_smoke():
    """Run 4-virtual-device FSDP QQN in an isolated subprocess with XLA host devices=4."""
    script = """
import jax
import jax.numpy as jnp
from qqn_llm.config import load_config
from qqn_llm.train import run_experiment

assert len(jax.devices()) == 4, f"Expected 4 virtual CPU devices, got {jax.devices()}"
cfg = load_config("configs/smoke_fsdp_cpu.yaml")
results = run_experiment(cfg, output_dir="runs/_test_fsdp_cpu")
assert len(results) == 1
hist = results[0]["history"]
assert len(hist) == cfg.training.steps
assert hist[-1]["train_loss"] < hist[0]["train_loss"], (
    f"Expected loss decrease: {hist[0]['train_loss']} -> {hist[-1]['train_loss']}"
)
print("FSDP 4-device CPU smoke test PASSED:", hist[0]["train_loss"], "->", hist[-1]["train_loss"])
"""
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    env["PYTHONPATH"] = str(root / "src")
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
