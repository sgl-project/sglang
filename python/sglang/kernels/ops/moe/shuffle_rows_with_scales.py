"""Single-launch row gather for a quantized activation and its group scales.

The cutlass fp8 blockwise MoE quantizes its activation once and then replicates
rows per routed expert. That took two ``shuffle_rows`` launches walking the same
dst2src map: one for the fp8 values, one for the fp32 group scales. The scale
gather moves 1/32 of the bytes the value gather does (``k // 128`` fp32 against
``k`` fp8), so as its own launch it is almost pure latency -- which is exactly
the cost that matters at low concurrency, where the whole gather is a few tens
of KB. This kernel walks the map once and writes both.

The gather is a permutation of bytes -- rows are copied, never recomputed -- so
the result is bit-identical to the two calls it replaces.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl

# Bytes of the value row one program copies; the grid is
# (num_dst_rows, ceil(k / BLOCK_K)).
#
# This is a bytes-per-thread knob, not a parallelism knob, and that is what
# makes it load-bearing. At low concurrency the gather is a few tens of KB and
# every setting measures the same, because all that is being timed is the launch.
# At prefill sizes it decides everything: on B200 with k = 7168, rows = 8192,
# against the two shuffle_rows launches this replaces (33.9 us) --
#
#   BLOCK_K   512    1024   2048   4096   8192   16384
#   time     66.6   37.8   25.5   18.1   17.4   19.4   us   (num_warps=4)
#
# 512 is half the speed of the CUDA kernel it replaces: at num_warps=4 that is
# 4 bytes per thread, a quarter of the 128 bits per thread the CUDA kernel
# vectorizes to. 4096 puts 32 bytes in each thread and lands on the plateau.
#
# Columns past k are masked off, so a model narrower than BLOCK_K runs partly
# empty lanes: k = 2048 still measures 1.33x against the two launches, and
# nothing narrower has been measured. If a k of 1024 or less turns up on this
# path, re-run the sweep before assuming this setting still holds.
BLOCK_K = 4096
NUM_WARPS = 4


@triton.jit
def _shuffle_rows_with_scales_kernel(
    q_ptr,  # [num_src_rows, k] int8 view of the quantized values
    scale_ptr,  # [num_src_rows, num_groups] fp32 group scales
    q_out_ptr,  # [num_dst_rows, k] int8 view
    scale_out_ptr,  # [num_dst_rows, num_groups] fp32
    dst2src_ptr,  # [num_dst_rows] int32, out[i] = src[dst2src[i]]
    k,
    num_groups,
    BLOCK_K: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    dst_row = tl.program_id(0)
    tile = tl.program_id(1)
    # int64 row bases: rows * k overflows int32 well inside the shapes this path
    # serves (the CUDA shuffle_rows it replaces indexes in int64 for the same
    # reason).
    src_row = tl.load(dst2src_ptr + dst_row).to(tl.int64)
    dst_row64 = dst_row.to(tl.int64)

    offs_k = tile * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = offs_k < k
    vals = tl.load(q_ptr + src_row * k + offs_k, mask=mask_k)
    tl.store(q_out_ptr + dst_row64 * k + offs_k, vals, mask=mask_k)

    # The scale row is 1/32 of the value row, so one tile carries all of it
    # rather than the whole thing costing a second launch.
    if tile == 0:
        offs_g = tl.arange(0, BLOCK_G)
        mask_g = offs_g < num_groups
        scales = tl.load(scale_ptr + src_row * num_groups + offs_g, mask=mask_g)
        tl.store(scale_out_ptr + dst_row64 * num_groups + offs_g, scales, mask=mask_g)


def shuffle_rows_with_scales(
    q: torch.Tensor,
    scale: torch.Tensor,
    dst2src_map: torch.Tensor,
    num_dst_rows: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather ``num_dst_rows`` rows of ``q`` and ``scale`` through one map.

    Replaces a pair of ``shuffle_rows`` calls over the same ``dst2src_map``,
    with the same semantics for both tensors: ``out[i] = src[dst2src_map[i]]``.
    Returns the two gathered tensors, allocated here.

    ``q`` is any 1-byte dtype (it is moved as bytes, not interpreted) and
    ``scale`` is its row-major per-group scale tensor; both must be contiguous
    and share a row count.
    """
    assert q.dim() == 2 and scale.dim() == 2, "q and scale must be 2D"
    assert q.is_contiguous() and scale.is_contiguous(), "q and scale must be contiguous"
    assert q.element_size() == 1, f"q must be a 1-byte dtype, got {q.dtype}"
    assert q.shape[0] == scale.shape[0], (
        f"row count mismatch: q {q.shape[0]} vs scale {scale.shape[0]}"
    )
    assert dst2src_map.numel() >= num_dst_rows, (
        f"map holds {dst2src_map.numel()} rows, need {num_dst_rows}"
    )
    # The kernel reads the map as whatever dtype it carries and casts to int64,
    # so a float map would truncate into a plausible-looking row id instead of
    # failing.
    assert dst2src_map.dtype in (
        torch.int32,
        torch.int64,
    ), f"dst2src_map must hold integer row ids, got {dst2src_map.dtype}"
    assert q.device == scale.device == dst2src_map.device, (
        f"inputs must share a device: q {q.device}, scale {scale.device}, "
        f"map {dst2src_map.device}"
    )

    k = q.shape[1]
    num_groups = scale.shape[1]
    q_out = torch.empty((num_dst_rows, k), device=q.device, dtype=q.dtype)
    scale_out = torch.empty(
        (num_dst_rows, num_groups), device=scale.device, dtype=scale.dtype
    )
    if num_dst_rows == 0:
        return q_out, scale_out

    _shuffle_rows_with_scales_kernel[(num_dst_rows, triton.cdiv(k, BLOCK_K))](
        q.view(torch.int8),
        scale,
        q_out.view(torch.int8),
        scale_out,
        dst2src_map,
        k,
        num_groups,
        BLOCK_K=BLOCK_K,
        BLOCK_G=triton.next_power_of_2(max(num_groups, 1)),
        num_warps=NUM_WARPS,
    )
    return q_out, scale_out
