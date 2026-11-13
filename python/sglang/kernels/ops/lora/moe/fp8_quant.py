"""Match upstream FP8 scale arithmetic, with zero codes for zero-scale rows.

Whole-row division uses fast math only on SM90; group quantization always does.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from sglang.kernels.arch import current_arch

_WHOLE_ROW_FAST_DIV = tl.constexpr(current_arch().is_sm(9, 0))


@triton.jit
def quantize_fp8_groups(grouped, WHOLE_ROW: tl.constexpr):
    """Return FP32 codes and per-group scales for [groups, group_size] input."""
    amax = tl.max(tl.abs(grouped), axis=1)
    if WHOLE_ROW:
        if _WHOLE_ROW_FAST_DIV:
            scale = amax / 448.0
            inv = tl.where(scale == 0.0, 0.0, 1.0 / scale)
        else:
            scale = libdevice.div_rn(amax, 448.0)
            inv = tl.where(scale == 0.0, 0.0, libdevice.div_rn(1.0, scale))
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
    live_ptr,  # [1] int32: leading rows to quantize (LIVE_BOUND only)
    masked_m_ptr,  # [E] int32 live rows per slab (MASKED only)
    rows_per_slab,
    num_rows,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ROWS: tl.constexpr,
    LIVE_BOUND: tl.constexpr,
    MASKED: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    live = rows < num_rows
    if LIVE_BOUND:
        live = live & (rows < tl.load(live_ptr))
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
    q, scale = quantize_fp8_groups(x, True)
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
    live_rows: torch.Tensor | None = None,
) -> None:
    """One FP8 scale per row of [rows, K] or [E, m_max, K]; masked slabs skip
    the rows past masked_m[e], and ``live_rows`` (a device int32 scalar) stops
    a flat layout after its leading rows. Programs take several rows each, so
    the padded tail of a layout costs nothing."""
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
        scale_out if live_rows is None else live_rows,
        masked_m if masked else scale_out,
        x.shape[1] if masked else 1,
        rows,
        K=k,
        BLOCK_K=block_k,
        ROWS=rows_per_program,
        LIVE_BOUND=live_rows is not None,
        MASKED=masked,
        num_warps=4 if k <= 4096 else 8,
    )
