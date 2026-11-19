"""FP8 row and group quantization.

Whole-row scales use fast division on SM90. Nonzero scales are floored
at 2^-126 so their reciprocals stay finite. All-zero rows use zero codes.
Group amax is floored at 1e-10, and extreme amax values use finite codes.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from sglang.kernels.arch import current_arch

_WHOLE_ROW_FAST_DIV = tl.constexpr(current_arch().is_sm(9, 0))
# Smallest normal FP32: the floor of a nonzero whole-row scale.
_MIN_ROW_SCALE = tl.constexpr(2.0**-126)


@triton.jit
def quantize_fp8_groups(grouped, WHOLE_ROW: tl.constexpr):
    """Quantize [groups, group_size] and return the codes and scales."""
    amax = tl.max(tl.abs(grouped), axis=1)
    if WHOLE_ROW:
        if _WHOLE_ROW_FAST_DIV:
            scale = amax / 448.0
        else:
            scale = libdevice.div_rn(amax, 448.0)
        scale = tl.where(amax > 0.0, tl.maximum(scale, _MIN_ROW_SCALE), 0.0)
        if _WHOLE_ROW_FAST_DIV:
            inv = 1.0 / scale
        else:
            inv = libdevice.div_rn(1.0, scale)
        inv = tl.where(scale == 0.0, 0.0, inv)
        q = grouped * inv[:, None]
    else:
        amax = tl.maximum(amax, 1e-10)
        scale = amax * (1.0 / 448.0)
        q = grouped * (448.0 / amax)[:, None]
    return tl.clamp(q, -448.0, 448.0), scale


@triton.jit
def _quantize_rows_per_token_kernel(
    x_ptr,  # bf16 rows, viewed flat [rows, K]
    q_ptr,  # fp8e4m3 [rows, K]
    s_ptr,  # fp32 [rows]
    masked_m_ptr,  # [E] int32 live rows per slab (MASKED only)
    rows_per_slab,
    num_rows,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ROWS: tl.constexpr,
    MASKED: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    live = rows < num_rows
    if MASKED:
        slab_m = tl.load(masked_m_ptr + rows // rows_per_slab, mask=live, other=0)
        live = live & (rows % rows_per_slab < slab_m)
    if tl.sum(live.to(tl.int32), axis=0) == 0:
        return
    offs = tl.arange(0, BLOCK_K)
    mask = live[:, None] & (offs[None, :] < K)
    x = tl.load(x_ptr + rows[:, None] * K + offs[None, :], mask=mask, other=0).to(
        tl.float32
    )
    q, scale = quantize_fp8_groups(x, WHOLE_ROW=True)
    tl.store(
        q_ptr + rows[:, None] * K + offs[None, :],
        q.to(q_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(s_ptr + rows, scale, mask=live)


def quantize_rows_per_token_fp8(
    x: torch.Tensor,
    *,
    q_out: torch.Tensor,
    scale_out: torch.Tensor,
    masked_m: torch.Tensor | None = None,
) -> None:
    """Quantize each live row of [rows, K] or [E, m_max, K]."""
    k = x.shape[-1]
    rows = x.numel() // k
    if rows == 0:
        return
    block_k = triton.next_power_of_2(k)
    rows_per_program = max(1, min(8, 4096 // block_k))
    masked = masked_m is not None
    _quantize_rows_per_token_kernel[(triton.cdiv(rows, rows_per_program),)](
        x,
        q_out,
        scale_out,
        masked_m if masked else scale_out,
        x.shape[1] if masked else 1,
        rows,
        K=k,
        BLOCK_K=block_k,
        ROWS=rows_per_program,
        MASKED=masked,
        num_warps=4 if k <= 4096 else 8,
    )


@triton.jit
def _quantize_pair_rows_kernel(
    x_ptr,  # bf16 [rows, K]
    q_ptr,  # fp8e4m3 [rows, K]
    s_ptr,  # fp32 [rows, GROUPS]
    pair_to_row_ptr,  # [num_pairs] int32, -1 for a pair without a row
    num_pairs,
    K: tl.constexpr,
    GROUP: tl.constexpr,  # 128, or BLOCK_K for one scale per row
    GROUPS: tl.constexpr,  # K // GROUP, or 1 per row
    BLOCK_K: tl.constexpr,
    ROWS: tl.constexpr,
):
    pairs = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    rows = tl.load(pair_to_row_ptr + pairs, mask=pairs < num_pairs, other=-1)
    live = rows >= 0
    if tl.max(live.to(tl.int32), axis=0) == 0:
        return
    rows = tl.where(live, rows, 0).to(tl.int64)
    offs = tl.arange(0, BLOCK_K)
    mask = live[:, None] & (offs < K)[None, :]
    x = tl.load(x_ptr + rows[:, None] * K + offs[None, :], mask=mask, other=0)
    TILE_GROUPS: tl.constexpr = BLOCK_K // GROUP
    q, scale = quantize_fp8_groups(
        tl.reshape(x.to(tl.float32), (ROWS * TILE_GROUPS, GROUP)), GROUPS == 1
    )
    tl.store(
        q_ptr + rows[:, None] * K + offs[None, :],
        tl.reshape(q, (ROWS, BLOCK_K)).to(q_ptr.dtype.element_ty),
        mask=mask,
    )
    groups = tl.arange(0, TILE_GROUPS)
    tl.store(
        s_ptr + rows[:, None] * GROUPS + groups[None, :],
        tl.reshape(scale, (ROWS, TILE_GROUPS)),
        mask=live[:, None] & (groups < GROUPS)[None, :],
    )


def quantize_pair_rows_fp8(
    x: torch.Tensor,
    *,
    q_out: torch.Tensor,
    scale_out: torch.Tensor,
    pair_to_row: torch.Tensor,
    group: int,
) -> None:
    """Quantize only rows named by pair_to_row, leaving other bytes untouched."""
    num_pairs = pair_to_row.numel()
    if num_pairs == 0:
        return
    k = x.shape[-1]
    block_k = triton.next_power_of_2(k)
    rows_per_program = max(1, min(8, 4096 // block_k))
    _quantize_pair_rows_kernel[(triton.cdiv(num_pairs, rows_per_program),)](
        x,
        q_out,
        scale_out,
        pair_to_row,
        num_pairs,
        K=k,
        GROUP=block_k if group == k else group,
        GROUPS=k // group,
        BLOCK_K=block_k,
        ROWS=rows_per_program,
        num_warps=4 if k <= 4096 else 8,
    )
