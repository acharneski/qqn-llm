"""Tier 1 smoke test: end-to-end local GPU/CPU fine-tuning (QQN vs AdamW) + Orbax ckpt."""

from pathlib import Path

import jax
import jax.numpy as jnp

from qqn_llm.checkpoint import restore_checkpoint, save_checkpoint
from qqn_llm.config import load_config
from qqn_llm.model import build_model_and_params
from qqn_llm.train import run_experiment


def test_smoke_local_e2e(tmp_path: Path):
    cfg = load_config("configs/smoke_local.yaml")
    cfg.training.steps = 8
    cfg.training.eval_every = 4
    cfg.training.checkpoint_every = 8
    out_dir = tmp_path / "smoke_local"

    results = run_experiment(cfg, output_dir=out_dir)
    assert len(results) == len(cfg.optimizers)

    for res in results:
        hist = res["history"]
        assert len(hist) == 8
        assert all(jnp.isfinite(h["train_loss"]) for h in hist)
        assert hist[-1]["train_loss"] < hist[0]["train_loss"], (
            f"{res['optimizer_name']} failed to reduce loss: "
            f"{hist[0]['train_loss']} -> {hist[-1]['train_loss']}"
        )
        assert res["final_weight_l1_dist"] > 0.0 and jnp.isfinite(res["final_weight_l1_dist"])
        assert res["final_weight_l2_dist"] > 0.0 and jnp.isfinite(res["final_weight_l2_dist"])
        assert (out_dir / f"{res['optimizer_name']}.json").exists()

    # Verify Orbax checkpoint save & restore round-trip
    ckpt_path = out_dir / "checkpoints" / "qqn_lbfgs_m5_step_8"
    assert ckpt_path.exists()
    bundle = build_model_and_params(cfg.model, seed=cfg.seed, fsdp_axis_size=1)
    restored = restore_checkpoint(ckpt_path, bundle.params)
    assert jax.tree_util.tree_structure(restored) == jax.tree_util.tree_structure(bundle.params)
