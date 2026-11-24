"""Masked causal language-modeling loss with checkpointed lax.scan gradient accumulation."""

from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def sequence_cross_entropy(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    loss_mask: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute token-weighted masked cross-entropy in float32.

    Uses ``jax.nn.logsumexp`` + gather instead of materializing a full
    ``(batch, seq_len, vocab_size)`` ``log_softmax`` tensor, cutting peak
    logit memory by ~2x on 262k-vocab models like Gemma 4.
    """
    logits_f32 = logits.astype(jnp.float32)
    lse = jax.nn.logsumexp(logits_f32, axis=-1)
    target_logits = jnp.take_along_axis(logits_f32, targets[..., None], axis=-1)[..., 0]
    nll = lse - target_logits
    mask_f32 = loss_mask.astype(jnp.float32)
    loss_sum = jnp.sum(nll * mask_f32)
    token_count = jnp.sum(mask_f32)
    return loss_sum, token_count


def make_accumulated_loss_fn(
    forward_fn: Callable[[dict, jnp.ndarray], jnp.ndarray],
    l2_reg: float = 0.0,
    mesh: Optional[Mesh] = None,
    anchor_params: Optional[dict] = None,
    anchor_l2_reg: float = 0.0,
) -> Callable[[dict, dict[str, jnp.ndarray]], jnp.ndarray]:
    """Build a deterministic macro-batch loss using checkpointed ``jax.lax.scan``.

    Wrapping the micro-batch scan body with ``jax.checkpoint`` ensures JAX only
    retains 1 micro-batch of activations/logits during backward rather than
    ``num_micro_batches`` copies. When ``anchor_l2_reg > 0`` and ``anchor_params``
    is provided, adds an L2-SP prior ``0.5 * anchor_l2_reg * ‖θ - θ₀‖₂²`` that
    regularizes weights toward pretrained ``θ₀`` while keeping line searches and
    L-BFGS secant pairs strictly objective-consistent.
    """

    def loss_fn(params, macro_batch: dict[str, jnp.ndarray] | tuple) -> jnp.ndarray:
        if isinstance(macro_batch, tuple):
            scan_batch, runtime_anchor = macro_batch
        else:
            scan_batch, runtime_anchor = macro_batch, anchor_params

        use_batch_shard = (
            mesh is not None
            and mesh.shape["fsdp"] > 1
            and scan_batch["input_ids"].shape[1] % mesh.shape["fsdp"] == 0
        )

        @jax.checkpoint
        def body(carry, micro_batch):
            acc_loss, acc_tokens = carry
            inp = micro_batch["input_ids"]
            tgt = micro_batch["target_ids"]
            msk = micro_batch["loss_mask"]
            if use_batch_shard:
                s2 = NamedSharding(mesh, P("fsdp", None))
                s3 = NamedSharding(mesh, P("fsdp", None, None))
                inp = jax.lax.with_sharding_constraint(inp, s2)
                tgt = jax.lax.with_sharding_constraint(tgt, s2)
                msk = jax.lax.with_sharding_constraint(msk, s2)
                logits = forward_fn(params, inp)
                logits = jax.lax.with_sharding_constraint(logits, s3)
            else:
                logits = forward_fn(params, inp)
            mb_loss, mb_tokens = sequence_cross_entropy(logits, tgt, msk)
            return (acc_loss + mb_loss, acc_tokens + mb_tokens), None

        init_carry = (jnp.asarray(0.0, jnp.float32), jnp.asarray(0.0, jnp.float32))
        (total_loss, total_tokens), _ = jax.lax.scan(body, init_carry, scan_batch)
        mean_loss = total_loss / jnp.maximum(total_tokens, jnp.asarray(1.0, jnp.float32))

        if l2_reg > 0.0:
            leaves = jax.tree_util.tree_leaves(params)
            l2 = sum(jnp.sum(x.astype(jnp.float32) ** 2) for x in leaves)
            mean_loss = mean_loss + jnp.asarray(l2_reg, jnp.float32) * l2

        if anchor_l2_reg > 0.0 and runtime_anchor is not None:
            sq_diff = sum(
                jnp.sum((p.astype(jnp.float32) - p0.astype(jnp.float32)) ** 2)
                for p, p0 in zip(
                    jax.tree_util.tree_leaves(params),
                    jax.tree_util.tree_leaves(runtime_anchor),
                )
            )
            mean_loss = (
                mean_loss + jnp.asarray(0.5 * anchor_l2_reg, jnp.float32) * sq_diff
            )

        return mean_loss

    return loss_fn
