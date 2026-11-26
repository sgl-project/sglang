"""Scatter DSV4 SWA K rows into the unified_kv pool (``unified_kv_triton``).

unified_kv[L] rows ``[0, swa_pages)`` are the SWA ring
(``state_slot * win + pos % win``).
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.layout import (
    check_two_pool_pair,
)


@triton.jit
def _swa_scatter_kernel(
    kv_ptr,  # [T, D] bf16
    state_slot_ptr,  # [T] int
    positions_ptr,  # [T] int
    final_pos_ptr,  # [T] int
    unified_ptr,  # [pages, D] bf16
    n_rows,
    ring_stride,  # SWA ring per-slot stride
    win: tl.constexpr,
    D: tl.constexpr,
    HAS_FINAL: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= n_rows:
        return
    pos = tl.load(positions_ptr + row)
    if HAS_FINAL:
        fp = tl.load(final_pos_ptr + row)
        if pos <= fp - win:
            return
    s = tl.load(state_slot_ptr + row)
    loc = s * ring_stride + (pos % ring_stride)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    vals = tl.load(kv_ptr + row * D + offs, mask=mask, other=0.0)
    tl.store(unified_ptr + loc * D + offs, vals, mask=mask)


def store_swa_into_unified(
    *,
    kv: torch.Tensor,  # [T, head_dim] bf16, or [T, nope_row_bytes] packed fp8
    state_slot: torch.Tensor,  # [T] int
    positions: torch.Tensor,  # [T] int
    unified_kv: torch.Tensor,  # [pages, ...] same dtype and row width as kv
    win: int,  # SWA attention window length
    ring_stride: int,  # SWA ring stride
    final_pos: Optional[torch.Tensor] = None,  # [T] req's last position
    kv_rope: Optional[torch.Tensor] = None,  # [T, rope_dim] bf16, fp8 layout only
    unified_kv_rope: Optional[torch.Tensor] = None,  # [pages, rope_dim] bf16
) -> None:
    """Scatter SWA K into ring row ``state_slot * ring_stride + pos % ring_stride``.

    Under the fp8 layout the latent is split over two pools, so ``kv`` carries the
    already-packed nope row (DSV4_FP8_NOPE_ROW_BYTES wide: values + E8M0 scales +
    pad; nothing is quantized here) and ``kv_rope`` the bf16 rope half. That width
    is a byte count that happens to equal the bf16 head_dim in elements -- the two
    are not the same thing. The scatter itself takes the row width off the tensor;
    only the pair check reads the constant.

    Both scatters recompute the row index from the same ``state_slot`` /
    ``positions``, so the two pools stay in lockstep with each other and with the
    bf16 layout. What a caller can still get wrong is passing two pools that aren't
    a pair, so the pair goes through ``check_two_pool_pair`` before the first launch
    -- shared with the fused store, which has the same coupling.
    """
    n_rows = kv.shape[0]
    if n_rows == 0:
        return

    has_final = final_pos is not None
    fp_arg = final_pos if has_final else positions
    assert kv.is_contiguous() and kv.dtype == unified_kv.dtype
    assert state_slot.is_contiguous() and positions.is_contiguous()
    assert fp_arg.is_contiguous()
    two_pool = kv_rope is not None
    assert two_pool == (unified_kv_rope is not None), (
        "kv_rope and unified_kv_rope come together"
    )
    if two_pool:
        assert kv_rope.is_contiguous(), (
            f"kv_rope must be contiguous, got strides {kv_rope.stride()}"
        )
        assert kv_rope.shape[0] == n_rows, (
            f"kv_rope holds {kv_rope.shape[0]} rows, kv holds {n_rows}"
        )
        check_two_pool_pair(
            unified_kv,
            unified_kv_rope,
            rope_width=kv_rope.shape[1],
            rope_dtype=kv_rope.dtype,
        )

    def _scatter(src: torch.Tensor, dst: torch.Tensor) -> None:
        D = src.shape[1]
        assert dst.shape[1] == D, f"row width {D} does not fit pool {dst.shape[1]}"
        _swa_scatter_kernel[(n_rows,)](
            src,
            state_slot,
            positions,
            fp_arg,
            dst,
            n_rows,
            ring_stride,
            win=win,
            D=D,
            HAS_FINAL=has_final,
            BLOCK_D=triton.next_power_of_2(D),
            num_warps=8,
        )

    if kv.element_size() == 1:
        # single-byte rows (any fp8 variant) are a pure byte move, and the E8M0
        # scale bytes aren't floats -- uint8 avoids a triton convert for `other=`
        assert unified_kv.is_contiguous()
        _scatter(kv.view(torch.uint8), unified_kv.view(torch.uint8))
    else:
        _scatter(kv, unified_kv)

    if two_pool:
        _scatter(kv_rope, unified_kv_rope)


@triton.jit
def _scatter_loc_kernel(
    kv_ptr,  # [T, D] bf16
    loc_ptr,  # [T] int (unified row index; <0 => skip)
    unified_ptr,  # [pages, D] bf16
    n_rows,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= n_rows:
        return
    loc = tl.load(loc_ptr + row).to(tl.int64)
    if loc < 0:
        return
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    vals = tl.load(kv_ptr + row * D + offs, mask=mask, other=0.0)
    tl.store(unified_ptr + loc * D + offs, vals, mask=mask)


def scatter_bf16_into_unified(
    *,
    kv: torch.Tensor,  # [T, head_dim] bf16 (already norm+rope'd)
    loc: torch.Tensor,  # [T] int32/int64 unified ring row; <0 => skip
    unified_kv: torch.Tensor,  # [pages, head_dim] bf16
) -> None:
    """Scatter already-norm+rope'd bf16 K into ``unified_kv[loc]`` (skip loc < 0).

    Companion to ``store_swa_into_unified`` for callers that already hold the
    precomputed ring row index (the DSpark draft: ``get_unified_swa_loc`` for the
    draft forward, or the commit-inject layout for target-hidden injection) and
    need per-row commit masking expressed as ``loc == -1``.
    """
    n_rows, D = kv.shape
    if n_rows == 0:
        return
    assert kv.is_contiguous() and kv.dtype == unified_kv.dtype
    assert loc.is_contiguous()
    assert unified_kv.is_contiguous()
    _scatter_loc_kernel[(n_rows,)](
        kv,
        loc,
        unified_kv,
        n_rows,
        D=D,
        BLOCK_D=triton.next_power_of_2(D),
        num_warps=8,
    )
