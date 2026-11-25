"""Orbax checkpoint save and restore helpers with bfloat16 compaction & metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


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


def restore_checkpoint(
    ckpt_dir: str | Path,
    target_params: Any = None,
    *,
    sharding: jax.sharding.Sharding | None = None,
) -> Any:
    """Restore a parameter PyTree from an Orbax checkpoint across any device topology.

    Defaults to placing restored arrays onto a single-device replicated ``NamedSharding``
    on ``jax.devices()[0]`` so an 8-GPU FSDP training checkpoint restores cleanly onto
    a 1-GPU inference server without a topology mismatch error.
    """
    path = Path(ckpt_dir).resolve()
    if sharding is None:
        dev0 = jax.devices()[0]
        sharding = NamedSharding(Mesh(np.asarray([dev0]), ("fsdp",)), P())

    target_with_sharding = None
    if target_params is not None:

        def _with_sharding(leaf):
            if isinstance(leaf, jax.ShapeDtypeStruct) and leaf.sharding is None:
                return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=sharding)
            if (
                hasattr(leaf, "shape")
                and hasattr(leaf, "dtype")
                and getattr(leaf, "sharding", None) is None
            ):
                return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=sharding)
            return leaf

        target_with_sharding = jax.tree_util.tree_map(_with_sharding, target_params)

    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    try:
        return checkpointer.restore(
            path,
            args=ocp.args.StandardRestore(
                item=target_with_sharding,
                fallback_sharding=sharding,
            ),
        )
    finally:
        checkpointer.close()
