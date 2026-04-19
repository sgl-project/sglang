# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Layout factory functions for Gluon extend-attention kernels.

Consolidates repeated layout definitions shared by the symmetric, DeepSeek,
and FP8 kernel files.  Each factory returns layout tuples suitable for
unpacking via indexed ``gl.constexpr`` assignments inside ``@gluon.jit``
kernels.

All factories must be decorated with ``@gluon.constexpr_function`` so the
Gluon JIT can evaluate them at compile time.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import AMDMFMALayout
from triton.experimental.gluon.language._layouts import (
    DistributedLinearLayout,
    DotOperandLayout,
    PaddedSharedLayout,
)

# ===-----------------------------------------------------------------------===#
# Phase 1 -- Header layouts (MFMA + dot operands + blocked + slices)
# ===-----------------------------------------------------------------------===#


@gluon.constexpr_function
def make_mfma_dot_layouts(num_warps, mma_m, mma_n, mma_k, qk_kw, pv_kw):
    """MFMA accumulator layout and QK / PV dot-operand layouts.

    Returns (mma_layout, q_dot, kt_dot, p_dot, v_dot).
    """
    mma = AMDMFMALayout(
        version=4,
        instr_shape=[mma_m, mma_n, mma_k],
        transposed=True,
        warps_per_cta=[num_warps, 1],
    )
    q_dot = DotOperandLayout(operand_index=0, parent=mma, k_width=qk_kw)
    kt_dot = DotOperandLayout(operand_index=1, parent=mma, k_width=qk_kw)
    p_dot = DotOperandLayout(operand_index=0, parent=mma, k_width=pv_kw)
    v_dot = DotOperandLayout(operand_index=1, parent=mma, k_width=pv_kw)
    return mma, q_dot, kt_dot, p_dot, v_dot


@gluon.constexpr_function
def make_fp8_dot_layouts(mma_layout, fp8_qk_kw, fp8_pv_kw):
    """Extra dot-operand layouts for the FP8 prefix MFMA path.

    Returns (fp8_q_dot, fp8_kt_dot, fp8_p_dot, fp8_v_dot).
    """
    fp8_q = DotOperandLayout(operand_index=0, parent=mma_layout, k_width=fp8_qk_kw)
    fp8_kt = DotOperandLayout(operand_index=1, parent=mma_layout, k_width=fp8_qk_kw)
    fp8_p = DotOperandLayout(operand_index=0, parent=mma_layout, k_width=fp8_pv_kw)
    fp8_v = DotOperandLayout(operand_index=1, parent=mma_layout, k_width=fp8_pv_kw)
    return fp8_q, fp8_kt, fp8_p, fp8_v


@gluon.constexpr_function
def make_blocked_and_slice_layouts(num_warps, mma_layout):
    """Output blocked layout and 1-D slice helpers.

    Returns (blocked, offs_m_layout, offs_d_layout,
             mma_offs_n_col, mma_offs_m_row, mma_m_layout).
    """
    blocked = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[16, 4],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )
    offs_m = gl.SliceLayout(dim=1, parent=blocked)
    offs_d = gl.SliceLayout(dim=0, parent=blocked)
    mma_n_col = gl.SliceLayout(dim=0, parent=mma_layout)
    mma_m_row = gl.SliceLayout(dim=1, parent=mma_layout)
    mma_m_ly = gl.SliceLayout(dim=1, parent=mma_layout)
    return blocked, offs_m, offs_d, mma_n_col, mma_m_row, mma_m_ly


# ===-----------------------------------------------------------------------===#
# Phase 2 -- Serial shared-memory layouts (SwizzledShared + serial BlockedLayout)
# ===-----------------------------------------------------------------------===#

SERIAL_KT_SMEM = gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=16, order=[0, 1])
SERIAL_V_SMEM = gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=16, order=[1, 0])
SERIAL_Q_SMEM = SERIAL_V_SMEM


# ===-----------------------------------------------------------------------===#
# Phase 3 -- PaddedSharedLayout factory
# ===-----------------------------------------------------------------------===#


@gluon.constexpr_function
def make_padded_smem(shape, offset_bases, padding_pairs):
    """Padded shared-memory layout for async DMA tiles.

    ``padding_pairs`` is e.g. ``[[512, pad]]`` (BF16) or
    ``[[1024, pad], [2048, 32]]`` (FP8).
    """
    return PaddedSharedLayout(
        interval_padding_pairs=padding_pairs,
        offset_bases=offset_bases,
        cga_layout=[],
        shape=shape,
    )


# ===-----------------------------------------------------------------------===#
# Phase 4 -- DistributedLinearLayout (DLL) wrapper
# ===-----------------------------------------------------------------------===#


@gluon.constexpr_function
def make_dll(shape, reg_bases, lane_bases, warp_bases):
    """Async DMA layout (DistributedLinearLayout) for K^T / V / KPE tiles."""
    return DistributedLinearLayout(
        reg_bases=reg_bases,
        lane_bases=lane_bases,
        warp_bases=warp_bases,
        block_bases=[],
        shape=shape,
    )


@gluon.constexpr_function
def make_serial_kt_blocked(num_warps):
    """Blocked layout for serial K^T tile loads (warps spread along N)."""
    return gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[16, 4],
        warps_per_cta=[1, num_warps],
        order=[0, 1],
    )


# ===-----------------------------------------------------------------------===#
# Phase 5 -- Offset-bases factory for PaddedSharedLayout
# ===-----------------------------------------------------------------------===#


@gluon.constexpr_function
def make_offset_bases(major_max, minor_coarse, minor_fine, major_dim):
    """Compute offset_bases for PaddedSharedLayout async DMA tiles.

    Generates a power-of-2 ladder ``[1 .. major_max]`` along ``major_dim``
    followed by two groups of minor-dimension basis vectors.

    Args:
        major_max: largest power-of-2 in the fast-varying dimension
                   (e.g. ``BLOCK_DMODEL // 2`` for K^T, ``BLOCK_DV // 2`` for V).
        minor_coarse: coarse basis values placed in the high address bits.
        minor_fine: fine basis values placed in the low address bits.
        major_dim: ``0`` for K^T / KPE (row-major), ``1`` for V (col-major).
    """
    bases = []
    d = 1
    while d <= major_max:
        bases.append([d, 0] if major_dim == 0 else [0, d])
        d *= 2
    for v in minor_coarse:
        bases.append([0, v] if major_dim == 0 else [v, 0])
    for v in minor_fine:
        bases.append([0, v] if major_dim == 0 else [v, 0])
    return bases


# ===-----------------------------------------------------------------------===#
# Phase 6 -- High-level async DMA layout factories
#
# K^T and V layouts are transposes of each other for symmetric attention:
# V = K^T with every basis vector [a, b] mirrored to [b, a] and the tile
# shape swapped from [D, N] to [N, D]. The private ``_kt_*_bases``
# helpers encode the single source of truth; public ``make_v_*`` just
# mirror them. The FP8 kernel needs a few non-standard V variants for
# shared-memory-pressure reasons -- those live below in Phase 7.
# ===-----------------------------------------------------------------------===#


@gluon.constexpr_function
def _minor_coarse_for_d(BLOCK_D, BLOCK_N):
    """Coarse swizzle bits for the fast-varying dim as a function of
    the slow-varying dim and block count."""
    if BLOCK_D >= 512:
        return [16, 32] if BLOCK_N >= 64 else [16]
    if BLOCK_D >= 256:
        return [16]
    return [16, 32, 64] if BLOCK_N >= 128 else [16, 32]


@gluon.constexpr_function
def _power_of_2_ladder(limit):
    """[1, 2, 4, ..., <=limit]."""
    out = []
    d = 1
    while d <= limit:
        out.append(d)
        d *= 2
    return out


@gluon.constexpr_function
def _transpose_bases(bases):
    """[a, b] -> [b, a] elementwise (K^T <-> V basis duality)."""
    return [[b, a] for (a, b) in bases]


@gluon.constexpr_function
def _kt_offset_bases(BLOCK_D, BLOCK_N):
    """PaddedSharedLayout offset bases for a [BLOCK_D, BLOCK_N] K^T tile.

    Layout: a power-of-2 ladder along the D axis, then coarse + fine
    basis vectors along N.
    """
    ob = [[d, 0] for d in _power_of_2_ladder(BLOCK_D // 2)]
    for v in _minor_coarse_for_d(BLOCK_D, BLOCK_N):
        ob.append([0, v])
    for v in [1, 2, 4, 8]:
        ob.append([0, v])
    return ob


@gluon.constexpr_function
def _kt_dll_bases_4w(BLOCK_D, BLOCK_N):
    """4-warp K^T (reg_bases, lane_bases, warp_bases)."""
    if BLOCK_D >= 512:
        reg = [[1,0],[2,0],[4,0],[0,4],[0,8],[0,16],[0,32]] if BLOCK_N >= 64 \
           else [[1,0],[2,0],[4,0],[0,4],[0,8],[0,16]]
        lane = [[8,0],[16,0],[32,0],[64,0],[128,0],[256,0]]
    elif BLOCK_D >= 256:
        reg  = [[1,0],[2,0],[4,0],[0,4],[0,8]]
        lane = [[8,0],[16,0],[32,0],[64,0],[128,0],[0,16]]
    else:
        reg = [[1,0],[2,0],[4,0],[0,4],[0,8],[0,64]] if BLOCK_N >= 128 \
           else [[1,0],[2,0],[4,0],[0,4],[0,8]]
        lane = [[8,0],[16,0],[32,0],[64,0],[0,16],[0,32]]
    return reg, lane, [[0,1],[0,2]]


@gluon.constexpr_function
def _kt_dll_bases_8w(BLOCK_D, BLOCK_N):
    """8-warp K^T (reg_bases, lane_bases, warp_bases)."""
    if BLOCK_D >= 512:
        lane = [[8,0],[16,0],[32,0],[64,0],[128,0],[256,0]]
        if BLOCK_N >= 64:
            return ([[1,0],[2,0],[4,0],[0,4],[0,8],[0,16]], lane,
                    [[0,1],[0,2],[0,32]])
        return ([[1,0],[2,0],[4,0],[0,8],[0,16]], lane,
                [[0,1],[0,2],[0,4]])
    if BLOCK_D >= 256:
        return ([[1,0],[2,0],[4,0],[0,8]],
                [[8,0],[16,0],[32,0],[64,0],[128,0],[0,16]],
                [[0,1],[0,2],[0,4]])
    if BLOCK_D >= 128:
        reg = [[1,0],[2,0],[4,0],[0,8],[0,64]] if BLOCK_N >= 128 \
           else [[1,0],[2,0],[4,0],[0,8]]
        return (reg,
                [[8,0],[16,0],[32,0],[64,0],[0,16],[0,32]],
                [[0,1],[0,2],[0,4]])
    reg = [[1,0],[2,0],[4,0],[0,64]] if BLOCK_N >= 128 \
       else [[1,0],[2,0],[4,0]]
    return (reg,
            [[8,0],[16,0],[32,0],[0,16],[0,32],[0,1]],
            [[0,2],[0,4],[0,8]])


@gluon.constexpr_function
def _kt_dll_bases(num_warps, BLOCK_D, BLOCK_N):
    """(reg, lane, warp) for an async K^T tile, dispatching on warp count."""
    if num_warps < 8:
        return _kt_dll_bases_4w(BLOCK_D, BLOCK_N)
    return _kt_dll_bases_8w(BLOCK_D, BLOCK_N)


@gluon.constexpr_function
def make_kt_offset_bases(BLOCK_DMODEL, BLOCK_N):
    """PaddedSharedLayout offset bases for a [BLOCK_DMODEL, BLOCK_N] K^T tile."""
    return _kt_offset_bases(BLOCK_DMODEL, BLOCK_N)


@gluon.constexpr_function
def make_v_offset_bases(BLOCK_DV, BLOCK_N):
    """PaddedSharedLayout offset bases for a [BLOCK_N, BLOCK_DV] V tile
    (transpose of the K^T layout)."""
    return _transpose_bases(_kt_offset_bases(BLOCK_DV, BLOCK_N))


@gluon.constexpr_function
def make_kt_dll(num_warps, BLOCK_DMODEL, BLOCK_N):
    """Async DMA DistributedLinearLayout for a [BLOCK_DMODEL, BLOCK_N] K^T tile."""
    reg, lane, warp = _kt_dll_bases(num_warps, BLOCK_DMODEL, BLOCK_N)
    return DistributedLinearLayout(
        reg_bases=reg, lane_bases=lane, warp_bases=warp,
        block_bases=[], shape=[BLOCK_DMODEL, BLOCK_N],
    )


@gluon.constexpr_function
def make_v_dll(num_warps, BLOCK_DV, BLOCK_N):
    """Async DMA DistributedLinearLayout for a [BLOCK_N, BLOCK_DV] V tile.

    Pure transpose of the K^T layout: every basis [a, b] is mirrored to
    [b, a] and the tile shape is [N, Dv] instead of [Dv, N].
    """
    reg, lane, warp = _kt_dll_bases(num_warps, BLOCK_DV, BLOCK_N)
    return DistributedLinearLayout(
        reg_bases=_transpose_bases(reg),
        lane_bases=_transpose_bases(lane),
        warp_bases=_transpose_bases(warp),
        block_bases=[], shape=[BLOCK_N, BLOCK_DV],
    )


# ===-----------------------------------------------------------------------===#
# Phase 7 -- FP8 kernel BF16-extend V layout factories
#
# The FP8 symmetric kernel's BF16 extend phase needs a non-standard V
# layout in three (num_warps, BLOCK_DV, BLOCK_N) corners for shared-
# memory-pressure reasons. Standard V layout is used everywhere else.
# BF16 K^T in the FP8 kernel uses the standard ``make_kt_dll``.
# ===-----------------------------------------------------------------------===#


@gluon.constexpr_function
def _fp8_bf16_v_dll_bases_4w(BLOCK_DV, BLOCK_N):
    """4-warp FP8-kernel BF16-V (reg, lane, warp), with special-case overrides."""
    if BLOCK_DV >= 256:
        reg = ([[0,1],[0,2],[0,4],[4,0],[8,0],[32,0],[64,0]] if BLOCK_N >= 128
               else [[0,1],[0,2],[0,4],[4,0],[8,0]])
        return (reg,
                [[0,8],[0,16],[0,32],[0,64],[0,128],[16,0]],
                [[1,0],[2,0]])
    if BLOCK_N >= 128:
        return ([[0,1],[0,2],[0,4],[4,0],[8,0],[64,0]],
                [[0,8],[0,16],[0,32],[0,64],[16,0],[32,0]],
                [[1,0],[2,0]])
    return ([[0,1],[0,2],[0,4],[0,8],[8,0]],
            [[0,16],[0,32],[0,64],[1,0],[2,0],[4,0]],
            [[16,0],[32,0]])


@gluon.constexpr_function
def _fp8_bf16_v_dll_bases_8w(BLOCK_DV, BLOCK_N):
    """8-warp FP8-kernel BF16-V (reg, lane, warp), with special-case overrides."""
    if BLOCK_DV >= 256:
        return ([[0,1],[0,2],[0,4],[8,0]],
                [[0,8],[0,16],[0,32],[0,64],[0,128],[16,0]],
                [[1,0],[2,0],[4,0]])
    if BLOCK_DV >= 128:
        if BLOCK_N >= 128:
            return ([[0,1],[0,2],[0,4],[8,0],[64,0]],
                    [[0,8],[0,16],[0,32],[0,64],[16,0],[32,0]],
                    [[1,0],[2,0],[4,0]])
        return ([[0,1],[0,2],[0,4],[0,8]],
                [[0,16],[0,32],[0,64],[1,0],[2,0],[4,0]],
                [[8,0],[16,0],[32,0]])
    reg = [[0,1],[0,2],[0,4],[64,0]] if BLOCK_N >= 128 \
       else [[0,1],[0,2],[0,4]]
    return (reg,
            [[0,8],[0,16],[0,32],[16,0],[32,0],[1,0]],
            [[2,0],[4,0],[8,0]])


@gluon.constexpr_function
def make_fp8_bf16_v_offset_bases(num_warps, BLOCK_DV, BLOCK_N):
    """PaddedSharedLayout offset bases for the FP8 kernel's BF16 V tile.

    Coarse swizzle ladder differs from the standard V layout:
    - 4w Dv>=256 N>=128: [16, 32, 64]   (extra swizzle for shared-mem pressure)
    - any   Dv>=256    : [16]           (no D>=512 widening like the standard V)
    - Dv<256           : [16, 32, 64] if N>=128 else [16, 32]
    """
    is_4w = num_warps < 8
    if is_4w and BLOCK_DV >= 256 and BLOCK_N >= 128:
        mc = [16, 32, 64]
    elif BLOCK_DV >= 256:
        mc = [16]
    else:
        mc = [16, 32, 64] if BLOCK_N >= 128 else [16, 32]
    ob = [[0, d] for d in _power_of_2_ladder(BLOCK_DV // 2)]
    for v in mc:
        ob.append([v, 0])
    for v in [1, 2, 4, 8]:
        ob.append([v, 0])
    return ob


@gluon.constexpr_function
def make_fp8_bf16_v_dll(num_warps, BLOCK_DV, BLOCK_N):
    """Async DMA DLL for the FP8 kernel's BF16-extend V tile [BLOCK_N, BLOCK_DV].

    Non-standard corners vs ``make_v_dll``:
    - 4w Dv>=256 N>=128: extra ``[32,0], [64,0]`` in reg
    - 4w Dv<256  N<128:  shuffled lane/warp for FP8 shared-memory pressure
    - 8w Dv<256  N<128:  same shuffled pattern, 3-warp stride
    """
    if num_warps < 8:
        reg, lane, warp = _fp8_bf16_v_dll_bases_4w(BLOCK_DV, BLOCK_N)
    else:
        reg, lane, warp = _fp8_bf16_v_dll_bases_8w(BLOCK_DV, BLOCK_N)
    return DistributedLinearLayout(
        reg_bases=reg, lane_bases=lane, warp_bases=warp,
        block_bases=[], shape=[BLOCK_N, BLOCK_DV],
    )
