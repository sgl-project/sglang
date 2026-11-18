"""MLU MLA cache gather with a bounded persistent grid."""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 1}, num_stages=3, num_warps=1),
        triton.Config({"BLOCK_M": 1}, num_stages=5, num_warps=1),
        triton.Config({"BLOCK_M": 16}, num_stages=3, num_warps=1),
        triton.Config({"BLOCK_M": 64}, num_stages=5, num_warps=1),
    ],
    key=[
        "n_loc_bucket",
        "nope_dim",
        "rope_dim",
        "buffer_stride",
        "nope_stride",
        "rope_stride",
    ],
)
@triton.heuristics(
    {
        "BLOCK_N": lambda a: triton.next_power_of_2(a["nope_dim"] + a["rope_dim"]),
        "SINGLE_TILE": lambda a: a["n_loc"] <= a["BLOCK_M"] * a["core_count"],
    }
)
@triton.jit(do_not_specialize=["n_loc", "n_loc_bucket"])
def _get_mla_kv_buffer_kernel(
    kv_buffer_ptr,
    cache_k_nope_ptr,
    cache_k_rope_ptr,
    loc_ptr,
    buffer_stride: tl.constexpr,
    nope_stride: tl.constexpr,
    rope_stride: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    n_loc,
    n_loc_bucket,
    core_count: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SINGLE_TILE: tl.constexpr,
):
    cols = tl.arange(0, BLOCK_N)
    nope_offsets = tl.arange(0, nope_dim)
    if SINGLE_TILE:
        iterations = 1
    else:
        iterations = tl.cdiv(tl.cdiv(n_loc, BLOCK_M), tl.num_programs(0))
    for i in range(iterations):
        block = tl.program_id(0) + i * tl.num_programs(0)
        block_offset = block.to(tl.int64) * BLOCK_M
        rows = block_offset + tl.arange(0, BLOCK_M)
        valid = rows < n_loc
        # Use int64 to avoid overflow when computing cache offsets.
        loc = tl.load(loc_ptr + rows, mask=valid, other=0).to(tl.int64)
        # Invalid rows read slot zero; only valid rows are written below.
        val = tl.load(
            kv_buffer_ptr + loc[:, None] * buffer_stride + cols[None, :],
            mask=(cols < nope_dim + rope_dim)[None, :],
        )
        tl.store(
            cache_k_nope_ptr + rows[:, None] * nope_stride + nope_offsets[None, :],
            val[:, 0:nope_dim],
            mask=valid[:, None],
        )
        if rope_dim > 0:
            rope_offsets = tl.arange(0, rope_dim)
            tl.store(
                cache_k_rope_ptr + rows[:, None] * rope_stride + rope_offsets[None, :],
                val[:, nope_dim : nope_dim + rope_dim],
                mask=valid[:, None],
            )


def get_mla_kv_buffer_mlu(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: Optional[torch.Tensor] = None,
    *,
    core_count: int,
) -> None:
    """Gather MLA cache rows with a bounded persistent grid on MLU."""
    n_loc = loc.numel()
    if n_loc == 0:
        return
    # Split each power-of-two interval in two to bound tuning without
    # sharing one configuration across a full 2x change in row count.
    bucket_size = max(1, triton.next_power_of_2(n_loc) // 4)
    n_loc_bucket = triton.cdiv(n_loc, bucket_size) * bucket_size
    # Each program loops over row tiles, avoiding a token-count-sized grid.
    grid = lambda meta: (min(triton.cdiv(n_loc, meta["BLOCK_M"]), core_count),)
    _get_mla_kv_buffer_kernel[grid](
        kv_buffer,
        cache_k_nope,
        cache_k_rope,
        loc,
        kv_buffer.stride(0),
        cache_k_nope.stride(0),
        cache_k_rope.stride(0) if cache_k_rope is not None else 0,
        cache_k_nope.shape[-1],
        cache_k_rope.shape[-1] if cache_k_rope is not None else 0,
        n_loc,
        n_loc_bucket,
        core_count,
    )
