"""Split-KV combine for aiter's persistent MLA decode, over the real heads only.

Drop-in for aiter.mla_reduce_v1 on the head-padded verify path: same partial
buffers and reduce maps, same math (per-split normalized outputs, combined with
exp(lse_j - global_lse) weights), but one program per (row, head, latent slice)
instead of one per (row, head), and the padded heads are skipped.

Partial layout (aiter mla_decode_fwd, persistent mode):
    partial_output [P * max_q, 1, HP, DV] fp32,  partial_lse [P * max_q, 1, HP, 1] fp32
For reduce tile i the splits are reduce_partial_map[reduce_indptr[i]:reduce_indptr[i+1]];
row l of the tile (final row reduce_final_map[i][0] + l) reads partial row map + l.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _global_lse(max_lse, sum_lse):
    # aiter's guard: an empty or NaN sum gives +inf, which zeroes the row.
    return tl.where(
        (sum_lse == 0.0) | (sum_lse != sum_lse), float("inf"), tl.log(sum_lse) + max_lse
    )


@triton.jit
def _mla_split_reduce_kernel(
    partial_out_ptr,
    partial_lse_ptr,
    reduce_indptr_ptr,
    reduce_final_map_ptr,
    reduce_partial_map_ptr,
    out_ptr,
    stride_os,
    stride_oh,
    MAX_Q: tl.constexpr,
    HP: tl.constexpr,
    DV: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    head = tl.program_id(1)
    d0 = tl.program_id(2) * BLOCK_D
    tile = pid // MAX_Q
    l = pid % MAX_Q

    start = tl.load(reduce_indptr_ptr + tile)
    n = tl.load(reduce_indptr_ptr + tile + 1) - start
    q_start = tl.load(reduce_final_map_ptr + tile * 2)
    q_end = tl.load(reduce_final_map_ptr + tile * 2 + 1)
    if n <= 0 or q_start + l >= q_end:
        return

    offs_s = tl.arange(0, BLOCK_S)
    offs_d = d0 + tl.arange(0, BLOCK_D)
    if n <= BLOCK_S:
        # One chunk: every split's [BLOCK_D] slice in flight at once.
        mask_s = offs_s < n
        rows = (
            tl.load(reduce_partial_map_ptr + start + offs_s, mask=mask_s, other=0) + l
        )
        lse = tl.load(
            partial_lse_ptr + rows * HP + head, mask=mask_s, other=-float("inf")
        )
        max_lse = tl.max(lse, axis=0)
        sum_lse = tl.sum(tl.where(mask_s, tl.exp(lse - max_lse), 0.0), axis=0)
        global_lse = _global_lse(max_lse, sum_lse)
        scale = tl.where(mask_s, tl.exp(lse - global_lse), 0.0)
        part = tl.load(
            partial_out_ptr + (rows[:, None] * HP + head) * DV + offs_d[None, :],
            mask=mask_s[:, None],
            other=0.0,
        )
        acc = tl.sum(scale[:, None] * part, axis=0)
    else:
        # BLOCK_S chunks, two passes: the global LSE first, then the weighted sum.
        max_lse = -float("inf")
        for c0 in range(0, n, BLOCK_S):
            mask_s = c0 + offs_s < n
            rows = (
                tl.load(
                    reduce_partial_map_ptr + start + c0 + offs_s, mask=mask_s, other=0
                )
                + l
            )
            lse = tl.load(
                partial_lse_ptr + rows * HP + head, mask=mask_s, other=-float("inf")
            )
            max_lse = tl.maximum(max_lse, tl.max(lse, axis=0))
        sum_lse = 0.0
        for c0 in range(0, n, BLOCK_S):
            mask_s = c0 + offs_s < n
            rows = (
                tl.load(
                    reduce_partial_map_ptr + start + c0 + offs_s, mask=mask_s, other=0
                )
                + l
            )
            lse = tl.load(
                partial_lse_ptr + rows * HP + head, mask=mask_s, other=-float("inf")
            )
            sum_lse += tl.sum(tl.where(mask_s, tl.exp(lse - max_lse), 0.0), axis=0)
        global_lse = _global_lse(max_lse, sum_lse)
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        for c0 in range(0, n, BLOCK_S):
            mask_s = c0 + offs_s < n
            rows = (
                tl.load(
                    reduce_partial_map_ptr + start + c0 + offs_s, mask=mask_s, other=0
                )
                + l
            )
            lse = tl.load(
                partial_lse_ptr + rows * HP + head, mask=mask_s, other=-float("inf")
            )
            scale = tl.where(mask_s, tl.exp(lse - global_lse), 0.0)
            part = tl.load(
                partial_out_ptr + (rows[:, None] * HP + head) * DV + offs_d[None, :],
                mask=mask_s[:, None],
                other=0.0,
            )
            acc += tl.sum(scale[:, None] * part, axis=0)

    s = q_start + l
    tl.store(
        out_ptr + s * stride_os + head * stride_oh + offs_d,
        acc.to(out_ptr.dtype.element_ty),
    )


# One chunk covers SGLANG_AITER_MLA_MAX_SPLITS_PER_BATCH=128 splits; 64-wide
# latent slices keep that [128, 64] fp32 tile at 32 registers per lane.
_BLOCK_S = 128
_BLOCK_D = 64

# Above this many query rows aiter's reduce is faster (its one program per
# (row, head) already fills the GPU). Measured on MI350X at 16 / 64 / 128 splits.
MAX_ROWS: int = 16


def mla_split_reduce(
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    reduce_indptr: torch.Tensor,
    reduce_final_map: torch.Tensor,
    reduce_partial_map: torch.Tensor,
    max_seqlen_q: int,
    out: torch.Tensor,
    num_heads: int,
) -> None:
    """Combine the split partials of the first num_heads heads into out
    [S, HP, DV] (rows a reduce tile covers; other rows are left as stage 1
    wrote them)."""
    HP, DV = partial_output.shape[-2], partial_output.shape[-1]
    num_tiles = reduce_indptr.shape[0] - 1
    block_s, block_d = _BLOCK_S, _BLOCK_D
    assert DV % block_d == 0 and out.stride(-1) == 1
    assert reduce_final_map.is_contiguous() and reduce_final_map.shape[-1] == 2
    _mla_split_reduce_kernel[(num_tiles * max_seqlen_q, num_heads, DV // block_d)](
        partial_output,
        partial_lse,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        out,
        out.stride(0),
        out.stride(1),
        MAX_Q=max_seqlen_q,
        HP=HP,
        DV=DV,
        BLOCK_D=block_d,
        BLOCK_S=block_s,
        num_warps=4,
    )
