"""Orbax checkpoint save and restore helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orbax.checkpoint as ocp


def save_checkpoint(params: Any, ckpt_dir: str | Path) -> Path:
    """Save a full parameter PyTree as an Orbax checkpoint."""
    path = Path(ckpt_dir).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(path, params, force=True)
    checkpointer.wait_until_finished()
    return path


def restore_checkpoint(ckpt_dir: str | Path, target_params: Any = None) -> Any:
    """Restore a parameter PyTree from an Orbax checkpoint."""
    path = Path(ckpt_dir).resolve()
    checkpointer = ocp.StandardCheckpointer()
    return checkpointer.restore(path, target_params)
