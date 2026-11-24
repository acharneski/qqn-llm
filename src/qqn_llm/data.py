"""Dataset loaders, tokenization, prompt/response masking, and macro-batch packing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import jax.numpy as jnp
import numpy as np

from qqn_llm.config import DataConfig


def _pack_macro_batch(
    input_ids: np.ndarray,
    target_ids: np.ndarray,
    loss_mask: np.ndarray,
    micro_batch_size: int,
) -> dict[str, jnp.ndarray]:
    """Reshape ``(N, seq_len)`` arrays into ``(num_micro, micro_batch_size, seq_len)``."""
    n = input_ids.shape[0]
    eff_mb = min(micro_batch_size, n)
    num_micro = n // eff_mb
    usable = num_micro * eff_mb
    return {
        "input_ids": jnp.asarray(
            input_ids[:usable].reshape(num_micro, eff_mb, -1), dtype=jnp.int32
        ),
        "target_ids": jnp.asarray(
            target_ids[:usable].reshape(num_micro, eff_mb, -1), dtype=jnp.int32
        ),
        "loss_mask": jnp.asarray(
            loss_mask[:usable].reshape(num_micro, eff_mb, -1), dtype=jnp.float32
        ),
    }


def _make_synthetic_split(
    num_examples: int, seq_len: int, vocab_size: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a learnable patterned token dataset for deterministic smoke testing."""
    rng = np.random.default_rng(seed)
    # Build structured repeating + shifted patterns so any working optimizer drops loss quickly.
    pattern_len = 8
    inputs = np.zeros((num_examples, seq_len), dtype=np.int32)
    targets = np.zeros((num_examples, seq_len), dtype=np.int32)
    masks = np.zeros((num_examples, seq_len), dtype=np.float32)

    usable_vocab = max(16, min(vocab_size - 4, 256))
    for i in range(num_examples):
        motif = rng.integers(2, usable_vocab, size=(pattern_len,), dtype=np.int32)
        seq = np.tile(motif, (seq_len + 1) // pattern_len + 1)[: seq_len + 1]
        inputs[i] = seq[:-1]
        targets[i] = seq[1:]
        # Mask out first 25% as "prompt", compute loss on remaining 75% as "response"
        prompt_cut = seq_len // 4
        masks[i, prompt_cut:] = 1.0
    return inputs, targets, masks


def _get_gemma_tokenizer():
    from gemma import gm

    for cls_name in ("Gemma4Tokenizer", "Gemma3Tokenizer", "Gemma2Tokenizer"):
        cls = getattr(gm.text, cls_name, None)
        if cls is not None:
            return cls()
    raise RuntimeError("No Gemma tokenizer class found in gemma.gm.text")


def _format_chat_prompt(prompt: str, tokenizer) -> str:
    """Format a user prompt with Gemma 4's exact turn template used by Gemma4Sampler."""
    try:
        from gemma.gm.text import _sampler

        fmt = getattr(tokenizer, "FORMAT", None)
        if fmt is not None:
            normalized = _sampler._normalize_prompt(prompt, format=fmt)
            if isinstance(normalized, list) and normalized:
                return normalized[0]
            if isinstance(normalized, str):
                return normalized
    except Exception:
        pass
    return prompt


def _tokenize_pairs(
    pairs: list[tuple[str, str]],
    seq_len: int,
    tokenizer,
    *,
    chat_format: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(pairs)
    inputs = np.zeros((n, seq_len), dtype=np.int32)
    targets = np.zeros((n, seq_len), dtype=np.int32)
    masks = np.zeros((n, seq_len), dtype=np.float32)

    special = getattr(tokenizer, "special_tokens", None)
    eos_id = getattr(special, "EOS", getattr(tokenizer, "eos_id", 1))
    eot_id = getattr(special, "END_OF_TURN", None)

    for i, (prompt, response) in enumerate(pairs):
        formatted_prompt = _format_chat_prompt(prompt, tokenizer) if chat_format else prompt
        p_ids = list(tokenizer.encode(formatted_prompt, add_bos=True))
        r_ids = list(tokenizer.encode(response, add_bos=False))
        tail = [eot_id, eos_id] if (chat_format and eot_id is not None) else [eos_id]
        full = (p_ids + r_ids + tail)[: seq_len + 1]
        if len(full) < 2:
            continue
        seq_in = full[:-1]
        seq_tgt = full[1:]
        L = len(seq_in)
        inputs[i, :L] = seq_in
        targets[i, :L] = seq_tgt
        resp_start = max(0, min(len(p_ids) - 1, L - 1))
        masks[i, resp_start:L] = 1.0
    return inputs, targets, masks


def _load_text_pairs(
    cfg: DataConfig,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    if cfg.source == "gsm8k":
        from datasets import load_dataset

        train_ds = load_dataset("openai/gsm8k", "main", split="train")
        val_ds = load_dataset("openai/gsm8k", "main", split="test")
        train_pairs = [
            (f"Question: {row['question']}\nAnswer:", f" {row['answer']}")
            for row in train_ds.select(range(min(cfg.train_examples, len(train_ds))))
        ]
        val_pairs = [
            (f"Question: {row['question']}\nAnswer:", f" {row['answer']}")
            for row in val_ds.select(range(min(cfg.val_examples, len(val_ds))))
        ]
        return train_pairs, val_pairs

    if cfg.source == "jsonl":
        if not cfg.jsonl_path:
            raise ValueError("data.jsonl_path must be set when data.source == 'jsonl'")
        rows = [
            json.loads(line)
            for line in Path(cfg.jsonl_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        pairs = [(r["prompt"], r["response"]) for r in rows]
        n_train = min(cfg.train_examples, max(1, len(pairs) - cfg.val_examples))
        return pairs[:n_train], pairs[n_train : n_train + cfg.val_examples]

    raise ValueError(f"Unsupported data source: {cfg.source!r}")


class DatasetBundle:
    """Provides deterministic fixed-subset or consistent macro-batch iterators."""

    def __init__(
        self,
        train_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
        val_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
        cfg: DataConfig,
        seed: int,
    ):
        self.train_arrays = train_arrays
        self.val_arrays = val_arrays
        self.cfg = cfg
        self.seed = seed

        self.fixed_train_batch = _pack_macro_batch(
            *train_arrays, micro_batch_size=cfg.micro_batch_size
        )
        self.val_batch = _pack_macro_batch(
            *val_arrays, micro_batch_size=cfg.micro_batch_size
        )

    def macro_batch_iterator(
        self,
        num_outer_batches: int,
        macro_batch_examples: int | None = None,
        seed_offset: int = 0,
    ) -> Iterator[dict[str, jnp.ndarray]]:
        """Yield macro-batches for Mode B / multi-phase SFT."""
        inputs, targets, masks = self.train_arrays
        n = inputs.shape[0]
        bs = min(macro_batch_examples or self.cfg.macro_batch_examples, n)
        rng = np.random.default_rng(self.seed + seed_offset)
        for _ in range(num_outer_batches):
            idx = rng.choice(n, size=bs, replace=False)
            yield _pack_macro_batch(
                inputs[idx], targets[idx], masks[idx], self.cfg.micro_batch_size
            )


def load_dataset_bundle(
    cfg: DataConfig, vocab_size: int, seed: int
) -> DatasetBundle:
    if cfg.source == "synthetic":
        train_arr = _make_synthetic_split(
            cfg.train_examples, cfg.seq_len, vocab_size, seed
        )
        val_arr = _make_synthetic_split(
            cfg.val_examples, cfg.seq_len, vocab_size, seed + 1
        )
        return DatasetBundle(train_arr, val_arr, cfg, seed)

    print(f"[qqn-llm] Loading tokenizer and dataset ({cfg.source})...", flush=True)
    tokenizer = _get_gemma_tokenizer()
    train_pairs, val_pairs = _load_text_pairs(cfg)
    use_chat = cfg.source == "jsonl" and cfg.chat_format
    print(
        f"[qqn-llm] Tokenizing {len(train_pairs)} train / {len(val_pairs)} val examples "
        f"(seq_len={cfg.seq_len}, chat_format={use_chat})...",
        flush=True,
    )
    train_arr = _tokenize_pairs(train_pairs, cfg.seq_len, tokenizer, chat_format=use_chat)
    val_arr = _tokenize_pairs(val_pairs, cfg.seq_len, tokenizer, chat_format=use_chat)
    print("[qqn-llm] Dataset tokenization complete.", flush=True)
    return DatasetBundle(train_arr, val_arr, cfg, seed)
