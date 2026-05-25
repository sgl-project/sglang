"""Batched decode attention wrapper for MLX backend."""

from __future__ import annotations

import logging
import sys
import threading
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from sglang.srt.hardware_backend.mlx.kv_cache.contiguous_cache import ContiguousKVCache

logger = logging.getLogger(__name__)
_thread_local = threading.local()


def _import_sgl_kernel_metal():
    try:
        metal = import_module("sgl_kernel.metal")
    except ModuleNotFoundError:
        for parent in Path(__file__).resolve().parents:
            candidate = parent / "sgl-kernel" / "python"
            if (candidate / "sgl_kernel" / "metal.py").is_file():
                sys.path.insert(0, str(candidate))
                metal = import_module("sgl_kernel.metal")
                break
        else:
            raise
    if getattr(metal, "_metal", None) is None:
        raise ImportError(
            "sgl_kernel._metal is not available. Build with "
            "`TOOLCHAINS=metal python sgl-kernel/setup_metal.py build_ext --inplace`."
        )
    return metal


@dataclass
class MlxAOTRoPEContext:
    config: dict
    base: float
    kv_pool: Any
    new_token_slots: Optional[mx.array] = None


@dataclass
class MlxAOTKernelContext:
    rope: Optional[MlxAOTRoPEContext] = None


@dataclass
class MlxAOTKernelBuildInfo:
    rope_config: dict = field(default_factory=dict)
    rope_base: float = 0.0
    kv_pool: Optional[Any] = None
    req_ids: Optional[list[str]] = None
    req_pool_idx: Optional[dict[str, int]] = None
    req_to_token_pool: Optional[Any] = None


# TODO: Move from threading to multiprocessing or asyncio
@dataclass
class BatchedDecodeContext:
    """Context set before batched decode, read by attention wrappers."""

    batch_size: int
    seq_lens: list[int]  # per-request token count before the new token
    # layer_caches[layer_idx][req_idx] = ContiguousKVCache
    layer_caches: list[list[ContiguousKVCache]]

    # Optional AOT kernel state. Keep kernel-specific fields out of the regular
    # MLX decode path so future AOT kernels can be added without growing this
    # context one field at a time.
    aot: MlxAOTKernelContext = field(default_factory=MlxAOTKernelContext)
    aot_build: Optional[MlxAOTKernelBuildInfo] = None

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
        if self.aot_build is not None and self.aot.rope is None:
            self.aot = self._build_aot_kernel_context()

    def _build_aot_kernel_context(self) -> MlxAOTKernelContext:
        info = self.aot_build
        if (
            info is None
            or not info.rope_config
            or info.rope_base <= 0.0
            or info.kv_pool is None
        ):
            return MlxAOTKernelContext()

        new_token_slots = None
        if (
            info.req_ids is not None
            and info.req_pool_idx is not None
            and info.req_to_token_pool is not None
        ):
            try:
                slot_ids = []
                for req_idx, req_id in enumerate(info.req_ids):
                    req_pool_idx = info.req_pool_idx.get(req_id)
                    if req_pool_idx is None:
                        raise KeyError(req_id)
                    slot = int(
                        info.req_to_token_pool.req_to_token[
                            req_pool_idx, self.layer_caches[0][req_idx].offset
                        ].item()
                    )
                    slot_ids.append(slot)
                new_token_slots = mx.array(slot_ids, dtype=mx.int32)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "AOT RoPE: failed to resolve new-token slots (%s); "
                    "falling back to RoPE-only for this decode step",
                    exc,
                )

        return MlxAOTKernelContext(
            rope=MlxAOTRoPEContext(
                config=info.rope_config,
                base=info.rope_base,
                kv_pool=info.kv_pool,
                new_token_slots=new_token_slots,
            )
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

        # Vectorized RoPE with per-batch offsets (cached on the context).
        offsets = ctx.offsets

        if ctx.aot.rope is not None:
            # AOT path: real .metallib RoPE + fused KV pool scatter.
            queries, keys = self._rope_custom_aot(
                queries,
                keys,
                values,
                offsets,
                layer_idx,
                ctx.aot.rope,
            )
        else:
            # Fallback: MLX's built-in mx.fast.rope (used when the AOT kernel
            # isn't built or the model uses an unsupported RoPE variant).
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

    @staticmethod
    def _rope_custom_aot(
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        positions: mx.array,
        layer_idx: int,
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
        rope_pool_fused = _import_sgl_kernel_metal().rope_pool_fused

        # (B, n_heads, 1, head_dim) -> (B, n_heads, head_dim) for kernel
        q_flat = queries[:, :, 0, :]
        k_flat = keys[:, :, 0, :]
        v_flat = values[:, :, 0, :]
        B = q_flat.shape[0]

        if rope_ctx.new_token_slots is None:
            slots = mx.full((B,), -1, dtype=mx.int32)
        else:
            slots = rope_ctx.new_token_slots.astype(mx.int32)

        k_pool = rope_ctx.kv_pool.k_buffer[layer_idx]
        v_pool = rope_ctx.kv_pool.v_buffer[layer_idx]

        q_rot, k_rot, k_pool_new, v_pool_new = rope_pool_fused(
            q_flat,
            k_flat,
            v_flat,
            positions,
            slots,
            k_pool,
            v_pool,
            head_dim=rope_ctx.config["head_dim"],
            num_qo_heads=rope_ctx.config["num_qo_heads"],
            num_kv_heads=rope_ctx.config["num_kv_heads"],
            rope_base=rope_ctx.base,
        )
        # Rebind pool buffers (zero-copy donation result).
        rope_ctx.kv_pool.k_buffer[layer_idx] = k_pool_new
        rope_ctx.kv_pool.v_buffer[layer_idx] = v_pool_new

        # (B, n_heads, head_dim) -> (B, n_heads, 1, head_dim) for SDPA path
        return q_rot[:, :, None, :], k_rot[:, :, None, :]
