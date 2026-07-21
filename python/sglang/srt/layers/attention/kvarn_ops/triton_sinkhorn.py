# SPDX-License-Identifier: Apache-2.0
"""Triton fused log-domain iterative variance-normalization for KVarN.

One Triton program per ``[R, C]`` tile.

For ``R = C = 128`` the full tile is 64 KB fp32 — fits in a single Triton
block's register/SMEM budget.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.layers.quantization.kvarn.sinkhorn import (
    variance_normalize_batched,
)

_CLIP_STD_MIN = 1e-3
_CLIP_STD_MAX = 1e3
_LOG_S_MIN = -0.3
_LOG_S_MAX = 10.0


@triton.jit
def _sinkhorn_log_kernel(
    Tile_ptr,  # [N, R, C] fp32 input
    Balanced_ptr,  # [N, R, C] fp32 output
    SCol_ptr,  # [N, C] fp32 output
    SRow_ptr,  # [N, R] fp32 output
    stride_tn,
    stride_tr,
    stride_bn,
    stride_br,
    stride_sc_n,
    stride_sr_n,
    R: tl.constexpr,
    C: tl.constexpr,
    ITERATIONS: tl.constexpr,
    CLIP_STD_MIN: tl.constexpr,
    CLIP_STD_MAX: tl.constexpr,
    LOG_S_MIN: tl.constexpr,
    LOG_S_MAX: tl.constexpr,
):
    pid = tl.program_id(0)

    r_offs = tl.arange(0, R)
    c_offs = tl.arange(0, C)

    tile_base = pid * stride_tn
    tile_ptrs = Tile_ptr + tile_base + r_offs[:, None] * stride_tr + c_offs[None, :]
    tile = tl.load(tile_ptrs).to(tl.float32)

    log_s_col = tl.zeros([C], dtype=tl.float32)
    log_s_row = tl.zeros([R], dtype=tl.float32)

    cur = tile

    col_mean0 = tl.sum(cur, axis=0) / R
    col_var0 = tl.sum(cur * cur, axis=0) / R - col_mean0 * col_mean0
    col_std0 = tl.sqrt(tl.maximum(col_var0 * R / (R - 1), 0.0))
    row_mean0 = tl.sum(cur, axis=1) / C
    row_var0 = tl.sum(cur * cur, axis=1) / C - row_mean0 * row_mean0
    row_std0 = tl.sqrt(tl.maximum(row_var0 * C / (C - 1), 0.0))

    col_max0 = tl.max(col_std0)
    col_min0 = tl.maximum(tl.min(col_std0), 1e-8)
    row_max0 = tl.max(row_std0)
    row_min0 = tl.maximum(tl.min(row_std0), 1e-8)
    imb_best = col_max0 / col_min0 + row_max0 / row_min0

    sc_best = tl.exp(log_s_col)
    sr_best = tl.exp(log_s_row)

    for _ in tl.static_range(ITERATIONS):
        col_mean = tl.sum(cur, axis=0) / R
        col_var = tl.sum(cur * cur, axis=0) / R - col_mean * col_mean
        col_std = tl.sqrt(tl.maximum(col_var * R / (R - 1), 0.0))
        col_std_clipped = tl.maximum(tl.minimum(col_std, CLIP_STD_MAX), CLIP_STD_MIN)
        log_s_col = log_s_col + tl.log(col_std_clipped)
        log_s_col = tl.maximum(tl.minimum(log_s_col, LOG_S_MAX), LOG_S_MIN)
        s_col_lin = tl.exp(log_s_col)
        s_row_lin = tl.exp(log_s_row)
        cur = tile / s_col_lin[None, :] / s_row_lin[:, None]

        row_mean = tl.sum(cur, axis=1) / C
        row_var = tl.sum(cur * cur, axis=1) / C - row_mean * row_mean
        row_std = tl.sqrt(tl.maximum(row_var * C / (C - 1), 0.0))
        row_std_clipped = tl.maximum(tl.minimum(row_std, CLIP_STD_MAX), CLIP_STD_MIN)
        log_s_row = log_s_row + tl.log(row_std_clipped)
        log_s_row = tl.maximum(tl.minimum(log_s_row, LOG_S_MAX), LOG_S_MIN)
        s_col_lin = tl.exp(log_s_col)
        s_row_lin = tl.exp(log_s_row)
        cur = tile / s_col_lin[None, :] / s_row_lin[:, None]

        col_mean_n = tl.sum(cur, axis=0) / R
        col_var_n = tl.sum(cur * cur, axis=0) / R - col_mean_n * col_mean_n
        col_std_n = tl.sqrt(tl.maximum(col_var_n * R / (R - 1), 0.0))
        row_mean_n = tl.sum(cur, axis=1) / C
        row_var_n = tl.sum(cur * cur, axis=1) / C - row_mean_n * row_mean_n
        row_std_n = tl.sqrt(tl.maximum(row_var_n * C / (C - 1), 0.0))
        col_max_n = tl.max(col_std_n)
        col_min_n = tl.maximum(tl.min(col_std_n), 1e-8)
        row_max_n = tl.max(row_std_n)
        row_min_n = tl.maximum(tl.min(row_std_n), 1e-8)
        imb = col_max_n / col_min_n + row_max_n / row_min_n

        better = imb <= imb_best
        sc_best = tl.where(better, s_col_lin, sc_best)
        sr_best = tl.where(better, s_row_lin, sr_best)
        imb_best = tl.where(better, imb, imb_best)

    balanced = tile / sc_best[None, :] / sr_best[:, None]
    bal_ptrs = (
        Balanced_ptr + pid * stride_bn + r_offs[:, None] * stride_br + c_offs[None, :]
    )
    tl.store(bal_ptrs, balanced)
    tl.store(SCol_ptr + pid * stride_sc_n + c_offs, sc_best)
    tl.store(SRow_ptr + pid * stride_sr_n + r_offs, sr_best)


def kvarn_sinkhorn_triton(
    tiles: torch.Tensor,
    iterations: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triton driver for ``_sinkhorn_log_kernel``.

    Args:
        tiles: ``[N, R, C]`` fp32 (or any real dtype, cast inside). Both R
            and C must be compile-time-constant power-of-2 values.
        iterations: number of alternating col/row passes (default 16).

    Returns:
        balanced: ``[N, R, C]`` fp32.
        s_col:    ``[N, C]`` fp32.
        s_row:    ``[N, R]`` fp32.
    """
    assert tiles.ndim == 3
    N, R, C = tiles.shape
    tiles = tiles.contiguous().to(torch.float32)
    device = tiles.device

    if max(R, C) > 256:
        bal, s_col_b, s_row_b = variance_normalize_batched(tiles, iterations=iterations)
        return (
            bal.contiguous(),
            s_col_b.reshape(N, C).contiguous(),
            s_row_b.reshape(N, R).contiguous(),
        )

    balanced = torch.empty(N, R, C, dtype=torch.float32, device=device)
    s_col = torch.empty(N, C, dtype=torch.float32, device=device)
    s_row = torch.empty(N, R, dtype=torch.float32, device=device)

    _sinkhorn_log_kernel[(N,)](
        tiles,
        balanced,
        s_col,
        s_row,
        tiles.stride(0),
        tiles.stride(1),
        balanced.stride(0),
        balanced.stride(1),
        s_col.stride(0),
        s_row.stride(0),
        R=R,
        C=C,
        ITERATIONS=iterations,
        CLIP_STD_MIN=_CLIP_STD_MIN,
        CLIP_STD_MAX=_CLIP_STD_MAX,
        LOG_S_MIN=_LOG_S_MIN,
        LOG_S_MAX=_LOG_S_MAX,
        num_warps=8,
        num_stages=2,
    )
    return balanced, s_col, s_row
