"""Gemma 4 model factory, FSDP sharding helpers, and tiny smoke-test architecture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from qqn_llm.config import ModelConfig


def resolve_dtype(name: str):
    if name == "bfloat16":
        return jnp.bfloat16
    if name == "float32":
        return jnp.float32
    raise ValueError(f"Unsupported dtype: {name!r}")


class _RMSNorm(nn.Module):
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", nn.initializers.ones, (x.shape[-1],), jnp.float32)
        rms = jnp.sqrt(
            jnp.mean(x.astype(jnp.float32) ** 2, axis=-1, keepdims=True) + self.eps
        )
        return (x.astype(jnp.float32) / rms * scale).astype(x.dtype)


class _TinyGemma4Block(nn.Module):
    """Compact Gemma-4-style transformer block (RMSNorm + GQA + GeGLU + PLE)."""

    embed_dim: int
    hidden_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    ple_dim: int
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x, ple_vec, causal_mask):
        b, t, _ = x.shape
        h = _RMSNorm()(x)

        q = nn.Dense(self.num_heads * self.head_dim, use_bias=False, dtype=self.dtype, param_dtype=self.dtype, name="q_proj")(h)
        k = nn.Dense(self.num_kv_heads * self.head_dim, use_bias=False, dtype=self.dtype, param_dtype=self.dtype, name="k_proj")(h)
        v = nn.Dense(self.num_kv_heads * self.head_dim, use_bias=False, dtype=self.dtype, param_dtype=self.dtype, name="v_proj")(h)

        q = q.reshape(b, t, self.num_heads, self.head_dim)
        k = k.reshape(b, t, self.num_kv_heads, self.head_dim)
        v = v.reshape(b, t, self.num_kv_heads, self.head_dim)

        repeats = self.num_heads // self.num_kv_heads
        if repeats > 1:
            k = jnp.repeat(k, repeats, axis=2)
            v = jnp.repeat(v, repeats, axis=2)

        scale = jnp.asarray(self.head_dim ** -0.5, dtype=jnp.float32)
        attn_logits = jnp.einsum("bthd,bshd->bhts", q.astype(jnp.float32), k.astype(jnp.float32)) * scale
        attn_logits = jnp.where(causal_mask, attn_logits, jnp.asarray(-1e9, jnp.float32))
        attn_weights = jax.nn.softmax(attn_logits, axis=-1).astype(self.dtype)
        attn_out = jnp.einsum("bhts,bshd->bthd", attn_weights, v).reshape(b, t, self.num_heads * self.head_dim)
        attn_out = nn.Dense(self.embed_dim, use_bias=False, dtype=self.dtype, param_dtype=self.dtype, name="o_proj")(attn_out)
        x = x + attn_out

        # GeGLU MLP
        h2 = _RMSNorm()(x)
        gate = nn.gelu(nn.Dense(self.hidden_dim, use_bias=False, dtype=self.dtype, param_dtype=self.dtype, name="gate_proj")(h2))
        up = nn.Dense(self.hidden_dim, use_bias=False, dtype=self.dtype, param_dtype=self.dtype, name="up_proj")(h2)
        down = nn.Dense(self.embed_dim, use_bias=False, dtype=self.dtype, param_dtype=self.dtype, name="down_proj")(gate * up)
        x = x + down

        # Per-Layer Embedding (PLE) residual gate (mirrors Gemma 4 E2B/E4B)
        ple_proj = nn.Dense(self.embed_dim, use_bias=False, dtype=self.dtype, param_dtype=self.dtype, name="ple_proj")(ple_vec)
        x = x + ple_proj
        return x


class TinyGemma4LM(nn.Module):
    """Miniaturized Gemma-4-structured LM for fast local RTX 2080 & CPU FSDP smoke tests."""

    num_layers: int = 2
    embed_dim: int = 256
    hidden_dim: int = 512
    num_heads: int = 4
    num_kv_heads: int = 2
    head_dim: int = 64
    vocab_size: int = 2048
    ple_dim: int = 64
    dtype: Any = jnp.bfloat16
    remat: bool = False

    @nn.compact
    def __call__(self, tokens: jnp.ndarray) -> jnp.ndarray:
        b, t = tokens.shape
        embed = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.embed_dim,
            dtype=self.dtype,
            param_dtype=self.dtype,
            name="token_embed",
        )
        x = embed(tokens)
        causal_mask = jnp.tril(jnp.ones((1, 1, t, t), dtype=bool))

        block_cls = nn.remat(_TinyGemma4Block) if self.remat else _TinyGemma4Block
        for i in range(self.num_layers):
            ple_table = nn.Embed(
                num_embeddings=self.vocab_size,
                features=self.ple_dim,
                dtype=self.dtype,
                param_dtype=self.dtype,
                name=f"ple_embed_{i}",
            )
            x = block_cls(
                embed_dim=self.embed_dim,
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                ple_dim=self.ple_dim,
                dtype=self.dtype,
                name=f"layer_{i}",
            )(x, ple_table(tokens), causal_mask)

        x = _RMSNorm(name="final_norm")(x)
        logits = embed.attend(x)
        return logits


@dataclass
class ModelBundle:
    forward_fn: Callable[[Any, jnp.ndarray], jnp.ndarray]
    params: Any
    vocab_size: int
    num_params: int
    mesh: Mesh
    param_sharding: Any


def build_mesh(fsdp_axis_size: int) -> Mesh:
    devices = jax.devices()
    n = len(devices)
    eff_fsdp = min(max(1, fsdp_axis_size), n)
    return Mesh(np.array(devices[:eff_fsdp]).reshape((eff_fsdp,)), ("fsdp",))


def make_fsdp_sharding(params, mesh: Mesh):
    """Build per-leaf NamedSharding that shards the largest divisible axis across ``fsdp``."""
    fsdp_size = mesh.shape["fsdp"]

    def _spec_for_leaf(leaf):
        if leaf is None or not hasattr(leaf, "ndim"):
            return NamedSharding(mesh, P())
        if fsdp_size <= 1 or leaf.ndim == 0:
            return NamedSharding(mesh, P())
        # Choose largest dimension divisible by fsdp_size
        best_axis = None
        best_dim = -1
        for axis, dim in enumerate(leaf.shape):
            if dim % fsdp_size == 0 and dim > best_dim:
                best_dim = dim
                best_axis = axis
        if best_axis is None:
            return NamedSharding(mesh, P())
        spec = [None] * leaf.ndim
        spec[best_axis] = "fsdp"
        return NamedSharding(mesh, P(*spec))

    return jax.tree_util.tree_map(_spec_for_leaf, params)


def shard_optimizer_state_sharding(state_shapes, mesh: Mesh):
    """Infer FSDP sharding for arbitrary optimizer state pytrees (including L-BFGS (m, *shape))."""
    return make_fsdp_sharding(state_shapes, mesh)


def count_params(params) -> int:
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))


def _resolve_gemma4_checkpoint(gm, ckpt_name: str | None, variant: str):
    cp = gm.ckpts.CheckpointPath
    if ckpt_name:
        key = ckpt_name.upper()
        if hasattr(cp, key):
            return getattr(cp, key)
        return ckpt_name
    default_key = "GEMMA4_E2B_IT" if variant == "gemma4_e2b" else "GEMMA4_E4B_IT"
    if hasattr(cp, default_key):
        return getattr(cp, default_key)
    fallback_key = "GEMMA4_E2B" if variant == "gemma4_e2b" else "GEMMA4_E4B"
    if hasattr(cp, fallback_key):
        return getattr(cp, fallback_key)
    raise ValueError(
        f"Could not resolve Gemma 4 checkpoint for variant={variant!r} in gm.ckpts.CheckpointPath. "
        f"Available attributes: {[a for a in dir(cp) if 'GEMMA' in a]}"
    )


def make_batch_sharding(macro_batch: dict[str, jnp.ndarray], mesh: Mesh):
    """Shard macro_batch along the micro-batch axis (dim 1) across ``fsdp`` when divisible."""
    fsdp_size = mesh.shape["fsdp"]
    mb_size = macro_batch["input_ids"].shape[1]
    if fsdp_size > 1 and mb_size % fsdp_size == 0:
        sharding = NamedSharding(mesh, P(None, "fsdp", None))
    else:
        sharding = NamedSharding(mesh, P())
    return jax.tree_util.tree_map(lambda _: sharding, macro_batch)


def _wrap_gemma4_with_remat(base_cls):
    from gemma.gm.nn.gemma4 import _modules

    remat_block = nn.remat(_modules.Block, prevent_cse=False)

    class _RematGemma4(base_cls):
        def setup(self):
            orig = _modules.Block
            _modules.Block = remat_block
            try:
                super().setup()
            finally:
                _modules.Block = orig

    _RematGemma4.__name__ = f"Remat{base_cls.__name__}"
    return _RematGemma4


def _filter_params_to_model(loaded_params: dict, model, seed: int) -> dict:
    """Keep only parameters actually used by ``model`` (e.g. drop dead vision/audio in text_only)."""
    abstract = jax.eval_shape(
        model.init, jax.random.PRNGKey(seed), jnp.zeros((1, 8), dtype=jnp.int32)
    )["params"]
    if isinstance(loaded_params, dict) and isinstance(abstract, dict):
        expected_keys = set(abstract.keys())
        extra = set(loaded_params.keys()) - expected_keys
        if extra:
            print(
                f"[qqn-llm] Dropping unused multimodal checkpoint keys for text_only model: {sorted(extra)}",
                flush=True,
            )
            return {k: v for k, v in loaded_params.items() if k in expected_keys}
    return loaded_params


def build_model_and_params(
    cfg: ModelConfig,
    seed: int,
    fsdp_axis_size: int = 1,
) -> ModelBundle:
    """Instantiate model, load/init full weight set, and shard across the FSDP mesh.

    When ``cfg.master_dtype == "float32"`` and ``cfg.dtype == "bfloat16"``,
    trainable weights are stored in ``float32`` (preventing bf16 update
    underflow and preserving exact L-BFGS curvature pairs ``s = x_new - x``)
    while casting to ``bfloat16`` inside ``forward_fn`` for fast compute.
    """
    compute_dtype = resolve_dtype(cfg.dtype)
    master_dtype = resolve_dtype(cfg.master_dtype)
    mesh = build_mesh(fsdp_axis_size)

    if cfg.variant == "tiny_smoke":
        model = TinyGemma4LM(
            num_layers=cfg.tiny_num_layers,
            embed_dim=cfg.tiny_embed_dim,
            hidden_dim=cfg.tiny_hidden_dim,
            num_heads=cfg.tiny_num_heads,
            num_kv_heads=cfg.tiny_num_kv_heads,
            head_dim=cfg.tiny_head_dim,
            vocab_size=cfg.tiny_vocab_size,
            dtype=compute_dtype,
            remat=cfg.remat,
        )
        key = jax.random.PRNGKey(seed)
        dummy = jnp.zeros((1, 16), dtype=jnp.int32)
        variables = model.init(key, dummy)
        params = jax.tree_util.tree_map(lambda x: x.astype(master_dtype), variables["params"])
        param_sharding = make_fsdp_sharding(params, mesh)
        params = jax.device_put(params, param_sharding)

        def forward_fn(p, tokens):
            if master_dtype != compute_dtype:
                p = jax.tree_util.tree_map(lambda x: x.astype(compute_dtype), p)
            return model.apply({"params": p}, tokens)

        return ModelBundle(
            forward_fn=forward_fn,
            params=params,
            vocab_size=cfg.tiny_vocab_size,
            num_params=count_params(params),
            mesh=mesh,
            param_sharding=param_sharding,
        )

    # Real Gemma 4 via google-deepmind/gemma
    from gemma import gm

    if cfg.variant == "gemma4_e2b":
        model_cls = getattr(gm.nn, "Gemma4_E2B", None)
    elif cfg.variant == "gemma4_e4b":
        model_cls = getattr(gm.nn, "Gemma4_E4B", None)
    else:
        raise ValueError(f"Unsupported Gemma 4 variant: {cfg.variant!r}")

    if model_cls is None:
        available = [a for a in dir(gm.nn) if "Gemma" in a]
        raise RuntimeError(
            f"Installed `gemma` package does not expose Gemma 4 model class for {cfg.variant!r}. "
            f"Available in gm.nn: {available}. Upgrade `gemma` (pip install -U gemma)."
        )

    if cfg.remat:
        model_cls = _wrap_gemma4_with_remat(model_cls)
        print(f"[qqn-llm] Enabled per-layer nn.remat gradient checkpointing on {model_cls.__name__}.", flush=True)

    model = model_cls()
    ckpt_path = _resolve_gemma4_checkpoint(gm, cfg.checkpoint, cfg.variant)
    print(f"[qqn-llm] Loading checkpoint {ckpt_path}...", flush=True)
    params = gm.ckpts.load_params(ckpt_path)
    params = _filter_params_to_model(params, model, seed)
    params = jax.tree_util.tree_map(lambda x: x.astype(master_dtype), params)
    param_sharding = make_fsdp_sharding(params, mesh)
    # Keep the master initial weights in CPU host RAM so each optimizer starts
    # with a completely clean GPU heap (no pinned copy sitting in GPU VRAM).
    cpu_dev = jax.devices("cpu")[0]
    params = jax.device_put(params, cpu_dev)
    print(
        f"[qqn-llm] Staged initial parameters on CPU RAM "
        f"(master={cfg.master_dtype}, compute={cfg.dtype}, target FSDP mesh={mesh}).",
        flush=True,
    )

    def forward_fn(p, tokens):
        if master_dtype != compute_dtype:
            p = jax.tree_util.tree_map(lambda x: x.astype(compute_dtype), p)
        out = model.apply({"params": p}, tokens=tokens)
        return out.logits if hasattr(out, "logits") else out

    vocab_size = int(getattr(model.config, "num_embed", 262144))
    return ModelBundle(
        forward_fn=forward_fn,
        params=params,
        vocab_size=vocab_size,
        num_params=count_params(params),
        mesh=mesh,
        param_sharding=param_sharding,
    )
