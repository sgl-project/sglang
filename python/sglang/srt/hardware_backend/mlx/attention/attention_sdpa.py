# Adapter from https://github.com/vllm-project/vllm-metal/blob/a06cd65a35b5c61c9a7f9d5f5ae00b30d9603379/vllm_metal/metal_kernel_backend/attention_sdpa.py
# SPDX-License-Identifier: Apache-2.0
"""Scaled dot-product attention (SDPA) on Metal.

Supports MHA, GQA, and MQA as variants of the same kernel — the head ratio
between ``n_heads`` (queries) and ``n_kv_heads`` (keys/values) is handled
transparently by the Metal paged attention kernel.

Handles models whose attention module exposes:
- ``q_proj``, ``k_proj``, ``v_proj``, ``o_proj`` linear projections
- ``rope`` for rotary position embeddings
- ``n_heads``, ``n_kv_heads`` head counts
- Optionally ``q_norm``, ``k_norm`` (Qwen3 per-head RMSNorm before RoPE)

Covers: Qwen3, Llama, Mistral, and other standard transformer architectures.

All operations use MLX arrays end-to-end — no PyTorch MPS bridge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

from sglang.srt.hardware_backend.mlx.attention.ops import get_ops
from sglang.srt.hardware_backend.mlx.attention.packed_rope import apply_packed_rope

if TYPE_CHECKING:
    from sglang.srt.hardware_backend.mlx.kv_cache.attention_wrapper import PagedAttentionContext


def is_sdpa(module: nn.Module) -> bool:
    """Return True if *module* is an SDPA attention layer (MHA, GQA, or MQA)."""
    return (
        hasattr(module, "q_proj")
        and hasattr(module, "k_proj")
        and hasattr(module, "v_proj")
        and hasattr(module, "o_proj")
    )


def sdpa_forward(
    inner: nn.Module,
    x: mx.array,
    ctx: PagedAttentionContext,
    layer_idx: int,
) -> mx.array:
    """Full SDPA forward pass: project → norm → RoPE → Metal kernel.

    Handles MHA, GQA, and MQA uniformly — the head ratio between
    ``inner.n_heads`` and ``inner.n_kv_heads`` is passed to the Metal
    kernel which handles the broadcast internally.
    """
    B, L, D = x.shape  # noqa: N806

    # --- Projections + reshape ---
    queries = inner.q_proj(x).reshape(B, L, inner.n_heads, -1)
    keys = inner.k_proj(x).reshape(B, L, inner.n_kv_heads, -1)
    values = inner.v_proj(x).reshape(B, L, inner.n_kv_heads, -1)

    # Qwen3 per-head RMSNorm before RoPE
    if hasattr(inner, "q_norm"):
        queries = inner.q_norm(queries)
    if hasattr(inner, "k_norm"):
        keys = inner.k_norm(keys)

    # transpose → (B, heads, L, head_dim)
    queries = queries.transpose(0, 2, 1, 3)
    keys = keys.transpose(0, 2, 1, 3)
    values = values.transpose(0, 2, 1, 3)

    # --- RoPE (per-request position reset) ---
    if not hasattr(inner, "rope"):
        raise NotImplementedError(
            f"Attention module {type(inner).__name__} does not have a 'rope' "
            "attribute. Only RoPE-based models are supported by paged attention."
        )

    queries, keys = apply_packed_rope(
        inner,
        queries,
        keys,
        ctx.cu_seqlens,
        offsets=ctx.offsets if ctx.offsets else None,
    )

    # --- Metal kernel dispatch ---
    n_heads = queries.shape[1]
    head_dim = queries.shape[3]

    # Reshape to 3D: (1, heads, L, hd) → (L, heads, hd)
    q_3d = mx.contiguous(queries[0].transpose(1, 0, 2).astype(ctx.kv_pool.dtype))
    k_3d = mx.contiguous(keys[0].transpose(1, 0, 2).astype(ctx.kv_pool.dtype))
    v_3d = mx.contiguous(values[0].transpose(1, 0, 2).astype(ctx.kv_pool.dtype))

    slot_mapping = mx.array(ctx.slot_mapping, dtype=mx.int64)

    # Build block_tables and seq_lens from context
    max_blocks_per_seq = max(len(bt) for bt in ctx.block_tables)
    block_tables_list = [
        bt + [0] * (max_blocks_per_seq - len(bt)) for bt in ctx.block_tables
    ]
    block_tables = mx.array(block_tables_list, dtype=mx.int32)
    seq_lens = mx.array(ctx.context_lens, dtype=mx.int32)
    cu_seqlens_q = mx.array(ctx.cu_seqlens, dtype=mx.int32)

    ops = get_ops()

    out = mx.array(0)

    # Cache write: MLX-native scatter (pure functional, graph-tracked)
    flat_k = ctx.kv_pool.k_buffer[layer_idx].reshape(-1, ctx.kv_pool.n_kv_heads, head_dim)
    flat_k[slot_mapping] = k_3d
    new_k_cache = flat_k.reshape(ctx.kv_pool.k_buffer[layer_idx].shape)

    flat_v = ctx.kv_pool.v_buffer[layer_idx].reshape(
        -1, ctx.kv_pool.n_kv_heads, head_dim
    )
    flat_v[slot_mapping] = v_3d
    new_v_cache = flat_v.reshape(ctx.kv_pool.v_buffer[layer_idx].shape)

    # Rebind so next layer / decode step uses the updated cache
    ctx.kv_pool.k_buffer[layer_idx] = new_k_cache
    ctx.kv_pool.v_buffer[layer_idx] = new_v_cache

    max_seq_len = max(ctx.context_lens) if ctx.context_lens else 0

    k_cache_view = new_k_cache[:, None, :, :]
    v_cache_view = new_v_cache[:, None, :, :]

    ops.paged_attention_primitive(
        q_3d,
        k_cache_view,
        v_cache_view,
        ctx.kv_pool.n_kv_heads,
        inner.scale,
        0.0,  # softcap (0 = disabled)
        block_tables,
        seq_lens,
        cu_seqlens_q,
        1,  # block_size = 1 for token-level radix cache
        max_seq_len,
        -1,  # sliding_window (-1 = disabled)
        out,
    )

    # output: (L, n_heads, head_dim) → (B, L, n_heads * head_dim)
    out = out.reshape(B, L, n_heads * head_dim)
    return inner.o_proj(out)
