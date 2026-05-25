"""Batched decode attention wrapper for MLX backend."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from sglang.srt.hardware_backend.mlx.kv_cache.contiguous_cache import ContiguousKVCache

_thread_local = threading.local()


# TODO: Move from threading to multiprocessing or asyncio
@dataclass
class BatchedDecodeContext:
    """Context set before batched decode, read by attention wrappers."""

    batch_size: int
    seq_lens: list[int]  # per-request token count before the new token
    # layer_caches[layer_idx][req_idx] = ContiguousKVCache
    layer_caches: list[list[ContiguousKVCache]]

    # Derived tensors/metadata, shared across all layers in one forward pass.
    offsets: mx.array = field(init=False)
    max_len: int = field(init=False)
    valid_lens: mx.array = field(init=False)
    needs_padding: bool = field(init=False)
    pad_sizes: list[int] = field(init=False)
    positions: Optional[mx.array] = field(init=False)

    def __post_init__(self) -> None:
        seq_lens = self.seq_lens
        max_seq_len = max(seq_lens)
        self.offsets = mx.array(seq_lens, dtype=mx.int32)
        self.max_len = max_seq_len + 1
        self.valid_lens = self.offsets + 1
        self.needs_padding = min(seq_lens) < max_seq_len
        self.pad_sizes = [max_seq_len - s for s in seq_lens]
        self.positions = mx.arange(self.max_len) if self.needs_padding else None


def set_context(ctx: Optional[BatchedDecodeContext]) -> None:
    _thread_local.batched_ctx = ctx


def get_context() -> Optional[BatchedDecodeContext]:
    return getattr(_thread_local, "batched_ctx", None)


def clear_context() -> None:
    _thread_local.batched_ctx = None


class MLXAttentionWrapper(nn.Module):
    """Wraps an mlx-lm Attention for batched decode (BS>1).

    When ``BatchedDecodeContext`` is set, performs per-request RoPE,
    cache writes, and batched SDPA.  Otherwise delegates to inner module.
    """

    def __init__(self, inner: nn.Module, layer_idx: int):
        super().__init__()
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_layer_idx", layer_idx)

    def __call__(self, x: mx.array, mask: Any = None, cache: Any = None) -> mx.array:
        ctx = get_context()
        if ctx is None:
            return self._inner(x, mask=mask, cache=cache)
        return self._batched_decode(x, ctx)

    def _batched_decode(self, x: mx.array, ctx: BatchedDecodeContext) -> mx.array:
        inner = self._inner
        layer_idx = self._layer_idx
        B = ctx.batch_size

        queries = inner.q_proj(x)
        keys = inner.k_proj(x)
        values = inner.v_proj(x)

        head_dim = queries.shape[-1] // inner.n_heads
        queries = queries.reshape(B, 1, inner.n_heads, head_dim)
        keys = keys.reshape(B, 1, inner.n_kv_heads, head_dim)
        values = values.reshape(B, 1, inner.n_kv_heads, head_dim)

        if hasattr(inner, "q_norm"):
            queries = inner.q_norm(queries)
        if hasattr(inner, "k_norm"):
            keys = inner.k_norm(keys)

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        # Vectorized RoPE with per-batch offsets
        offsets = ctx.offsets
        queries = inner.rope(queries, offset=offsets)
        keys = inner.rope(keys, offset=offsets)

        layer_caches = ctx.layer_caches[layer_idx]
        max_len = ctx.max_len
        pad_sizes = ctx.pad_sizes

        # TODO: replace per-request loop with native batched/ragged
        # attention once mx.fast.scaled_dot_product_attention supports
        # variable-length sequences.
        all_k = []
        all_v = []

        for i in range(B):
            layer_caches[i].write_token(keys[i : i + 1], values[i : i + 1])

            k_all, v_all = layer_caches[i].get_kv()

            pad = pad_sizes[i]
            if pad > 0:
                k_pad = mx.zeros(
                    (1, inner.n_kv_heads, pad, head_dim), dtype=k_all.dtype
                )
                v_pad = mx.zeros(
                    (1, inner.n_kv_heads, pad, head_dim), dtype=v_all.dtype
                )
                k_all = mx.concatenate([k_all, k_pad], axis=2)
                v_all = mx.concatenate([v_all, v_pad], axis=2)

            all_k.append(k_all)
            all_v.append(v_all)

        keys_b = mx.concatenate(all_k, axis=0)
        values_b = mx.concatenate(all_v, axis=0)

        attn_mask = None
        if ctx.needs_padding:
            mask_bool = ctx.positions[None, :] >= ctx.valid_lens[:, None]
            attn_mask = mx.where(
                mask_bool[:, None, None, :],
                mx.array(mx.finfo(queries.dtype).min, dtype=queries.dtype),
                mx.array(0.0, dtype=queries.dtype),
            )

        output = mx.fast.scaled_dot_product_attention(
            queries, keys_b, values_b, scale=inner.scale, mask=attn_mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, 1, -1)
        return inner.o_proj(output)
