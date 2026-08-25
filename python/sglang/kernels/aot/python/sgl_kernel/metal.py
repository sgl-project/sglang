"""Python entry points for the sgl_kernel Metal extension."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlx.core as mx

_METALLIB_NAME = "sgl_metal_kernels.metallib"

try:
    from . import _metal

    _metallib_path = Path(_metal.__file__).resolve().parent / _METALLIB_NAME
    if not _metallib_path.is_file():
        raise ImportError(
            f"{_METALLIB_NAME} not found next to the native Metal extension "
            f"at {_metallib_path}"
        )
    _metal.register_library(str(_metallib_path))
except ImportError as _exc:  # pragma: no cover - import guarded at call time
    _metal = None
    _IMPORT_ERROR: Exception | None = _exc
else:
    _IMPORT_ERROR = None

# Python wrappers for the compiled `_metal.*` entry points go below. Wrappers
# validate input shapes/dtypes and then invoke AOT C++ entry points. They do
# not force `mx.eval`, so MLX can keep these calls inside its lazy graph.


def rope_pool_fused(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    positions: mx.array,
    slots: mx.array,
    k_pool: mx.array,
    v_pool: mx.array,
    *,
    head_dim: int,
    num_qo_heads: int,
    num_kv_heads: int,
    rope_base: float,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Apply NeoX RoPE to Q/K and scatter K/V into the MLX KV pool.

    Args:
        q: Query tensor with shape `[num_tokens, num_qo_heads, head_dim]`.
        k: Key tensor with shape `[num_tokens, num_kv_heads, head_dim]`.
        v: Value tensor with shape `[num_tokens, num_kv_heads, head_dim]`.
        positions: int32 positions with shape `[num_tokens]`.
        slots: int32 KV-pool slots with shape `[num_tokens]`; values `< 0`
            skip the pool write for that token.
        k_pool: Existing K pool with shape `[pool_size, num_kv_heads, head_dim]`.
        v_pool: Existing V pool with shape `[pool_size, num_kv_heads, head_dim]`.

    Returns:
        `(q_rot, k_rot, k_pool_new, v_pool_new)`.
    """
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("rope_pool_fused expects q/k/v to be 3-D")
    if positions.ndim != 1 or slots.ndim != 1:
        raise ValueError("rope_pool_fused expects positions/slots to be 1-D")
    if k_pool.ndim != 3 or v_pool.ndim != 3:
        raise ValueError("rope_pool_fused expects pool tensors to be 3-D")
    q_shape = tuple(q.shape)
    k_shape = tuple(k.shape)
    v_shape = tuple(v.shape)
    positions_shape = tuple(positions.shape)
    slots_shape = tuple(slots.shape)
    k_pool_shape = tuple(k_pool.shape)
    v_pool_shape = tuple(v_pool.shape)

    if q_shape != (q_shape[0], num_qo_heads, head_dim):
        raise ValueError(
            "q shape must be [num_tokens, num_qo_heads, head_dim], " f"got {q.shape}"
        )
    if k_shape != (q_shape[0], num_kv_heads, head_dim):
        raise ValueError(
            "k shape must be [num_tokens, num_kv_heads, head_dim], " f"got {k.shape}"
        )
    if v_shape != k_shape:
        raise ValueError(f"v shape must match k shape, got {v.shape} vs {k.shape}")
    if positions_shape != (q_shape[0],) or slots_shape != (q_shape[0],):
        raise ValueError("positions/slots must have one entry per token")
    if k_pool_shape[1:] != (num_kv_heads, head_dim):
        raise ValueError(f"k_pool has incompatible shape {k_pool.shape}")
    if v_pool_shape != k_pool_shape:
        raise ValueError(
            f"v_pool shape must match k_pool shape, got {v_pool.shape} vs {k_pool.shape}"
        )
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q/k/v dtypes must match")
    if k_pool.dtype != q.dtype or v_pool.dtype != q.dtype:
        raise ValueError("pool dtypes must match q/k/v dtype")

    return _metal.rope_pool_fused(
        q,
        k,
        v,
        positions,
        slots,
        k_pool,
        v_pool,
        head_dim,
        num_qo_heads,
        num_kv_heads,
        float(rope_base),
    )


def block_paged_attention_decode(
    q: mx.array,
    k_blocks: mx.array,
    v_blocks: mx.array,
    block_tables: mx.array,
    seq_lens: mx.array,
    *,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    sm_scale: float,
) -> mx.array:
    """Decode attention over a block-table MLX KV layout.

    Args:
        q: Query tensor with shape `[batch, num_qo_heads, head_dim]`.
        k_blocks: Key blocks with shape
            `[num_blocks, block_size, num_kv_heads, head_dim]`.
        v_blocks: Value blocks with the same shape as `k_blocks`.
        block_tables: int32 physical block IDs with shape
            `[batch, max_num_blocks]`; `-1` entries are padding.
        seq_lens: int32 visible sequence lengths with shape `[batch]`.

    Returns:
        Attention output with shape `[batch, num_qo_heads, head_dim]`.
    """
    if q.ndim != 3:
        raise ValueError("block_paged_attention_decode expects q to be 3-D")
    if k_blocks.ndim != 4 or v_blocks.ndim != 4:
        raise ValueError("block_paged_attention_decode expects K/V blocks to be 4-D")
    if block_tables.ndim != 2 or seq_lens.ndim != 1:
        raise ValueError(
            "block_paged_attention_decode expects block_tables to be 2-D and seq_lens to be 1-D"
        )

    q_shape = tuple(q.shape)
    k_shape = tuple(k_blocks.shape)
    v_shape = tuple(v_blocks.shape)
    if q_shape[1:] != (num_qo_heads, head_dim):
        raise ValueError(
            "q shape must be [batch, num_qo_heads, head_dim], " f"got {q.shape}"
        )
    if k_shape[1:] != (block_size, num_kv_heads, head_dim):
        raise ValueError(
            "k_blocks must have shape [num_blocks, block_size, num_kv_heads, head_dim], "
            f"got {k_blocks.shape}"
        )
    if v_shape != k_shape:
        raise ValueError(
            f"v_blocks shape must match k_blocks shape, got {v_blocks.shape} vs {k_blocks.shape}"
        )
    if tuple(block_tables.shape)[0] != q_shape[0]:
        raise ValueError("block_tables must have one row per batch item")
    if tuple(seq_lens.shape) != (q_shape[0],):
        raise ValueError("seq_lens must have one entry per batch item")
    if num_qo_heads % num_kv_heads != 0:
        raise ValueError("num_qo_heads must be divisible by num_kv_heads")
    if head_dim > 256:
        raise ValueError(
            "block_paged_attention_decode currently supports head_dim <= 256"
        )
    if q.dtype != k_blocks.dtype or q.dtype != v_blocks.dtype:
        raise ValueError("q/k_blocks/v_blocks dtypes must match")

    return _metal.block_paged_attention_decode(
        q,
        k_blocks,
        v_blocks,
        block_tables,
        seq_lens,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        block_size,
        float(sm_scale),
    )
