"""Typed configuration dataclasses and YAML loader for qqn-llm experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import yaml


@dataclass
class ModelConfig:
    variant: Literal["tiny_smoke", "gemma4_e2b", "gemma4_e4b"] = "tiny_smoke"
    checkpoint: Optional[str] = None
    dtype: Literal["bfloat16", "float32"] = "bfloat16"
    master_dtype: Literal["bfloat16", "float32"] = "bfloat16"
    remat: bool = True
    # Used only when variant == "tiny_smoke"
    tiny_num_layers: int = 2
    tiny_embed_dim: int = 256
    tiny_hidden_dim: int = 512
    tiny_num_heads: int = 4
    tiny_num_kv_heads: int = 2
    tiny_head_dim: int = 64
    tiny_vocab_size: int = 2048


@dataclass
class DataConfig:
    source: Literal["synthetic", "gsm8k", "jsonl"] = "synthetic"
    jsonl_path: Optional[str] = None
    seq_len: int = 128
    train_examples: int = 32
    val_examples: int = 16
    micro_batch_size: int = 8
    batch_mode: Literal["fixed_subset", "macro_batch"] = "fixed_subset"
    # Used only when batch_mode == "macro_batch" (Mode B)
    macro_batch_examples: int = 64
    inner_steps_per_batch: int = 3


@dataclass
class TrainingConfig:
    steps: int = 15
    eval_every: int = 5
    checkpoint_every: int = 0
    l2_reg: float = 0.0
    fsdp_axis_size: int = 1
    stop_on_train_loss: float = 0.0
    stop_on_consecutive_stalls: int = 0


@dataclass
class OptimizerConfig:
    name: str = "qqn_lbfgs_m5"
    kind: Literal["qqn", "adamw", "lbfgs_optax"] = "qqn"
    # QQN options
    oracle: str = "lbfgs"
    history_size: int = 5
    line_search: str = "backtracking"  # backtracking | strong_wolfe | null | fixed
    path_strategy: str = "quadratic"   # quadratic | linear | spline
    c1: float = 1e-6
    c2: float = 0.9
    ls_max_iter: int = 5
    max_t: float = 10.0
    remember_step_size: bool = True
    spline_refine_rounds: int = 2
    spline_max_control_points: int = 8
    # AdamW / AdamOracle options
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0


@dataclass
class ExperimentConfig:
    experiment_name: str = "smoke_local"
    output_dir: str = "runs/smoke_local"
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizers: list[OptimizerConfig] = field(default_factory=list)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an :class:`ExperimentConfig` from a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    model_cfg = ModelConfig(**raw.get("model", {}))
    data_cfg = DataConfig(**raw.get("data", {}))
    train_cfg = TrainingConfig(**raw.get("training", {}))
    opt_cfgs = [OptimizerConfig(**o) for o in raw.get("optimizers", [])]
    return ExperimentConfig(
        experiment_name=raw.get("experiment_name", Path(path).stem),
        output_dir=raw.get("output_dir", f"runs/{Path(path).stem}"),
        seed=int(raw.get("seed", 42)),
        model=model_cfg,
        data=data_cfg,
        training=train_cfg,
        optimizers=opt_cfgs,
    )
