"""Batched decode attention wrapper for MLX backend."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from sglang.srt.hardware_backend.mlx.aot import (
    MlxAOTKernelContext,
    MlxAOTKernelSet,
    MlxAOTRoPEContext,
)
from sglang.srt.hardware_backend.mlx.kv_cache.attention_contract import (
    get_attention_scale,
    get_head_dim,
    get_num_heads,
    get_num_kv_heads,
)
from sglang.srt.hardware_backend.mlx.kv_cache.attention_kv_cache import (
    ContiguousAttentionKVCache,
)

_thread_local = threading.local()


# TODO: Move from threading to multiprocessing or asyncio
@dataclass
class BatchedDecodeContext:
    """Context set before batched decode, read by attention wrappers."""

    batch_size: int
    seq_lens: list[int]  # per-request token count before the new token
    # attention_layer_caches[attention_pool_idx][req_idx] = ContiguousAttentionKVCache
    attention_layer_caches: list[list[ContiguousAttentionKVCache]]
    attention_pool_index_by_layer: dict[int, int] = field(default_factory=dict)

    # Optional AOT kernel state. Keep kernel-specific fields out of the regular
    # MLX decode path so future AOT kernels can be added without growing this
    # context one field at a time.
    aot: MlxAOTKernelContext = field(default_factory=MlxAOTKernelContext)

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
        if not self.attention_pool_index_by_layer:
            self.attention_pool_index_by_layer = {
                idx: idx for idx in range(len(self.attention_layer_caches))
            }

    @classmethod
    def from_decode(
        cls,
        *,
        caches: list[list[Any]],
        req_ids: list[str],
        aot_kernels: MlxAOTKernelSet,
        kv_pool: Any | None,
        req_pool_idx: dict[str, int],
        req_to_token_pool: Any | None,
        attention_layer_indices: list[int] | None = None,
        attention_pool_index_by_layer: dict[int, int] | None = None,
    ) -> BatchedDecodeContext:
        batch_size = len(req_ids)
        if attention_layer_indices is None:
            attention_layer_indices = list(range(len(caches[0])))
        seq_lens = [
            caches[i][attention_layer_indices[0]].offset for i in range(batch_size)
        ]
        attention_layer_caches = [
            [caches[i][layer_idx] for i in range(batch_size)]
            for layer_idx in attention_layer_indices
        ]
        return cls(
            batch_size=batch_size,
            seq_lens=seq_lens,
            attention_layer_caches=attention_layer_caches,
            attention_pool_index_by_layer=attention_pool_index_by_layer or {},
            aot=MlxAOTKernelContext.from_decode(
                aot_kernels=aot_kernels,
                kv_pool=kv_pool,
                req_ids=req_ids,
                req_pool_idx=req_pool_idx,
                req_to_token_pool=req_to_token_pool,
                layer_caches=attention_layer_caches,
            ),
        )


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

    ``window_size`` marks a sliding-window layer: the pool keeps the full
    KV history and the wrapper attends to the trailing window only, which
    is numerically identical to a rotating cache.
    """

    def __init__(
        self, inner: nn.Module, layer_idx: int, window_size: int | None = None
    ):
        super().__init__()
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_layer_idx", layer_idx)
        object.__setattr__(self, "_window_size", window_size)
        # Resolved once at patch time (weights are loaded before patching and
        # the inner module is never swapped afterwards), keeping the decode
        # hot path free of attribute scans and failing fast on a bad module.
        scale = get_attention_scale(inner)
        if scale is None:
            raise RuntimeError(
                f"Cannot determine attention scale for {type(inner).__name__}"
            )
        object.__setattr__(self, "_scale", scale)
        object.__setattr__(self, "_sinks", getattr(inner, "sinks", None))

    def __call__(self, x: mx.array, mask: Any = None, cache: Any = None) -> mx.array:
        ctx = get_context()
        if ctx is None:
            return self._inner(x, mask=mask, cache=cache)
        return self._batched_decode(x, ctx)

    def _batched_decode(self, x: mx.array, ctx: BatchedDecodeContext) -> mx.array:
        inner = self._inner
        layer_idx = self._layer_idx
        B = ctx.batch_size
        n_heads = get_num_heads(inner)
        n_kv_heads = get_num_kv_heads(inner)
        if n_heads is None or n_kv_heads is None:
            raise RuntimeError(
                f"Cannot determine attention head counts for {type(inner).__name__}"
            )

        q_proj_output = inner.q_proj(x)
        keys = inner.k_proj(x)
        values = inner.v_proj(x)

        head_dim = get_head_dim(inner)
        if head_dim is None:
            head_dim = keys.shape[-1] // n_kv_heads

        q_width = n_heads * head_dim
        gate = None
        if q_proj_output.shape[-1] == q_width:
            queries = q_proj_output.reshape(B, 1, n_heads, head_dim)
        elif q_proj_output.shape[-1] == 2 * q_width:
            queries, gate = mx.split(
                q_proj_output.reshape(B, 1, n_heads, 2 * head_dim), 2, axis=-1
            )
            gate = gate.reshape(B, 1, q_width)
        else:
            raise RuntimeError(
                f"Unexpected q_proj output shape {q_proj_output.shape} for "
                f"{type(inner).__name__}"
            )

        keys = keys.reshape(B, 1, n_kv_heads, head_dim)
        values = values.reshape(B, 1, n_kv_heads, head_dim)

        if hasattr(inner, "q_norm"):
            queries = inner.q_norm(queries)
        if hasattr(inner, "k_norm"):
            keys = inner.k_norm(keys)

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        # Vectorized RoPE with per-batch offsets (cached on the context).
        offsets = ctx.offsets
        attention_pool_idx = ctx.attention_pool_index_by_layer[layer_idx]

        if ctx.aot.rope is not None:
            # AOT path: real .metallib RoPE + fused KV pool scatter.
            queries, keys = self._rope_custom_aot(
                queries,
                keys,
                values,
                offsets,
                attention_pool_idx,
                ctx.aot.rope,
            )
        else:
            # Fallback: MLX's built-in mx.fast.rope (used when the AOT kernel
            # isn't built or the model uses an unsupported RoPE variant).
            queries = inner.rope(queries, offset=offsets)
            keys = inner.rope(keys, offset=offsets)

        layer_caches = ctx.attention_layer_caches[attention_pool_idx]
        window = self._window_size
        if window is None:
            pad_sizes = ctx.pad_sizes
        else:
            # Sliding-window layer: the cache keeps the full history but the
            # newest token only attends to the trailing ``window`` keys.  The
            # padding metadata shared on the context is full-length, so it is
            # rebuilt locally for the windowed lengths.
            eff_lens = [min(n + 1, window) for n in ctx.seq_lens]
            max_eff = max(eff_lens)
            pad_sizes = [max_eff - n for n in eff_lens]

        # TODO: replace per-request loop with native batched/ragged
        # attention once mx.fast.scaled_dot_product_attention supports
        # variable-length sequences.
        all_k = []
        all_v = []

        for i in range(B):
            layer_caches[i].write_token(keys[i : i + 1], values[i : i + 1])

            k_all, v_all = layer_caches[i].get_kv()
            if window is not None and k_all.shape[2] > window:
                k_all = k_all[:, :, -window:, :]
                v_all = v_all[:, :, -window:, :]

            pad = pad_sizes[i]
            if pad > 0:
                k_pad = mx.zeros((1, n_kv_heads, pad, head_dim), dtype=k_all.dtype)
                v_pad = mx.zeros((1, n_kv_heads, pad, head_dim), dtype=v_all.dtype)
                k_all = mx.concatenate([k_all, k_pad], axis=2)
                v_all = mx.concatenate([v_all, v_pad], axis=2)

            all_k.append(k_all)
            all_v.append(v_all)

        keys_b = mx.concatenate(all_k, axis=0)
        values_b = mx.concatenate(all_v, axis=0)

        pad_mask = None
        if window is None:
            if ctx.needs_padding:
                pad_mask = ctx.positions[None, :] >= ctx.valid_lens[:, None]
        elif max(pad_sizes) > 0:
            eff = mx.array(eff_lens, dtype=mx.int32)
            pad_mask = mx.arange(max_eff)[None, :] >= eff[:, None]

        attn_mask = None
        if pad_mask is not None:
            attn_mask = mx.where(
                pad_mask[:, None, None, :],
                mx.array(mx.finfo(queries.dtype).min, dtype=queries.dtype),
                mx.array(0.0, dtype=queries.dtype),
            )

        # Only pass sinks when the module has them: the kwarg requires a
        # recent mlx and must not constrain models without sinks.
        sink_kwargs = {}
        if self._sinks is not None:
            sink_kwargs["sinks"] = self._sinks
        output = mx.fast.scaled_dot_product_attention(
            queries,
            keys_b,
            values_b,
            scale=self._scale,
            mask=attn_mask,
            **sink_kwargs,
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, 1, -1)
        if gate is not None:
            output = output * mx.sigmoid(gate)
        return inner.o_proj(output)

    @staticmethod
    def _rope_custom_aot(
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        positions: mx.array,
        attention_pool_idx: int,
        rope_ctx: MlxAOTRoPEContext,
    ) -> tuple[mx.array, mx.array]:
        """AOT path: rotate Q/K and scatter K/V into the shared pool.

        The kernel call does RoPE on Q/K and scatters
        rotated K + (untouched) V into ``kv_pool`` at ``new_token_slots``
        for ``layer_idx``.

        If ``new_token_slots`` is None, slot=-1 sentinel is used (no pool
        write, RoPE-only mode). Returns rotated (queries, keys) in the
        original 4-D attention layout. ``values`` is unchanged by RoPE.
        """
        # (B, n_heads, 1, head_dim) -> (B, n_heads, head_dim) for kernel
        q_flat = queries[:, :, 0, :]
        k_flat = keys[:, :, 0, :]
        v_flat = values[:, :, 0, :]
        B = q_flat.shape[0]

        if rope_ctx.new_token_slots is None:
            slots = mx.full((B,), -1, dtype=mx.int32)
        else:
            slots = rope_ctx.new_token_slots.astype(mx.int32)

        k_pool = rope_ctx.kv_pool.k_buffer[attention_pool_idx]
        v_pool = rope_ctx.kv_pool.v_buffer[attention_pool_idx]

        q_rot, k_rot, k_pool_new, v_pool_new = rope_ctx.kernel.rope_pool_fused(
            q_flat,
            k_flat,
            v_flat,
            positions,
            slots,
            k_pool,
            v_pool,
            head_dim=rope_ctx.kernel.config["head_dim"],
            num_qo_heads=rope_ctx.kernel.config["num_qo_heads"],
            num_kv_heads=rope_ctx.kernel.config["num_kv_heads"],
            rope_base=rope_ctx.kernel.base,
        )
        # Rebind pool buffers (zero-copy donation result).
        rope_ctx.kv_pool.k_buffer[attention_pool_idx] = k_pool_new
        rope_ctx.kv_pool.v_buffer[attention_pool_idx] = v_pool_new

        # (B, n_heads, head_dim) -> (B, n_heads, 1, head_dim) for SDPA path
        return q_rot[:, :, None, :], k_rot[:, :, None, :]
