"""Orbax checkpoint save and restore helpers with bfloat16 compaction & metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp


def save_checkpoint(
    params: Any,
    ckpt_dir: str | Path,
    *,
    dtype: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save a full parameter PyTree as an Orbax checkpoint (optionally cast to bfloat16)."""
    path = Path(ckpt_dir).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    to_save = params
    if dtype == "bfloat16":
        to_save = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), params)
    elif dtype == "float32":
        to_save = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), params)

    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(path, to_save, force=True)
    checkpointer.wait_until_finished()

    if metadata is not None:
        meta_file = path / "checkpoint_meta.json"
        meta_file.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    return path


def restore_checkpoint(ckpt_dir: str | Path, target_params: Any = None) -> Any:
    """Restore a parameter PyTree from an Orbax checkpoint."""
    path = Path(ckpt_dir).resolve()
    checkpointer = ocp.StandardCheckpointer()
    if target_params is not None:
        return checkpointer.restore(path, target_params)
    return checkpointer.restore(path)
