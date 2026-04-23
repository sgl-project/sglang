# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon extend-attention kernel for gfx950 (MI350X / CDNA4).

Three dispatch paths -- basic request-centric, mask-split, and persistent-CTA
(work-centric scheduling) -- are selected at launch time based on workload shape.
Supports D=64/128/256, causal/non-causal, logit cap, sliding window, custom mask,
FP8 KV dequant (k_scale/v_scale), and XAI temperature.
"""

import torch
import triton
import triton.language as tl

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import AMDMFMALayout, warp_pipeline_stage
from triton.experimental.gluon.language.amd.cdna4 import async_copy as cdna4_async
from triton.experimental.gluon.language.amd.cdna4 import (
    buffer_load as cdna4_buffer_load,
    buffer_store as cdna4_buffer_store,
)
from triton.experimental.gluon.language.amd.cdna4 import mfma as mfma_cdna4
from triton.experimental.gluon.language._layouts import (
    DistributedLinearLayout,
    DotOperandLayout,
    PaddedSharedLayout,
)

LOG2E = tl.constexpr(1.4426950408889634)


# ===-----------------------------------------------------------------------===#
# Layouts
# ===-----------------------------------------------------------------------===#
#
# Every layout the kernels consume is built here as a @gluon.constexpr_function
# (evaluated at JIT compile time, no runtime cost). Kernels call the individual
# factories directly.
#
# For symmetric attention, V is the transpose of K^T: every basis ``[a, b]``
# is mirrored to ``[b, a]`` and the tile shape swaps ``[D, N]`` <-> ``[N, D]``.
# The `_kt_*_bases` functions are the single source of truth; public
# `make_v_*` helpers transpose them. The FP8 kernel's BF16-extend V layout
# diverges in three corners (num_warps, BLOCK_DV, BLOCK_N) to relieve LDS
# pressure — `make_fp8_bf16_v_*` (and its aliases `make_fp8_v_*` /
# `make_fp8_extend_v_*`) handle that.


@gluon.constexpr_function
def make_mfma_dot_layouts(num_warps, mma_m, mma_n, mma_k, qk_kw, pv_kw):
    """MFMA accumulator + QK / PV dot-operand layouts.

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
    """Output blocked layout + 1-D slice helpers.

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


# Shared-memory layouts for the serial (non-DMA) KT / V / Q path.
SERIAL_KT_SMEM = gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=16, order=[0, 1])
SERIAL_V_SMEM = gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=16, order=[1, 0])
SERIAL_Q_SMEM = SERIAL_V_SMEM


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


@gluon.constexpr_function
def make_serial_kt_blocked(num_warps):
    """Blocked layout for serial K^T tile loads (warps spread along N)."""
    return gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[16, 4],
        warps_per_cta=[1, num_warps],
        order=[0, 1],
    )


@gluon.constexpr_function
def make_offset_bases(major_max, minor_coarse, minor_fine, major_dim):
    """Compute offset_bases for a PaddedSharedLayout async DMA tile.

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


@gluon.constexpr_function
def _minor_coarse_for_d(BLOCK_D, BLOCK_N):
    """Coarse swizzle bits along the fast-varying dim as a function of
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
    """PaddedSharedLayout offset bases for a [BLOCK_D, BLOCK_N] K^T tile."""
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


# FP8 kernel's BF16-extend V layout factories.
#
# The FP8 kernel's BF16 extend phase uses a non-standard V layout in
# three (num_warps, BLOCK_DV, BLOCK_N) corners for shared-memory-pressure
# reasons; elsewhere it uses the standard layout. BF16 K^T in the FP8
# kernel uses the standard `make_kt_dll`.


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
    - 4w Dv>=256 N>=128: [16, 32, 64]   (extra swizzle for LDS pressure)
    - any   Dv>=256    : [16]           (no D>=512 widening like standard V)
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

    Non-standard corners vs `make_v_dll`:
    - 4w Dv>=256 N>=128: extra [32,0], [64,0] in reg
    - 4w Dv<256  N<128:  shuffled lane/warp for FP8 LDS pressure
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


# ===-----------------------------------------------------------------------===#
# FP8 prefix async-DMA layout factories
#
# The FP8 kernel runs two async-DMA streams: prefix (FP8 K/V loads, 1 byte)
# and extend (BF16 K/V loads, 2 bytes). The prefix stream packs more elements
# per vector register along the major axis, so its DistributedLinearLayouts
# differ from the BF16 factories above. Offset bases coincide with BF16 for
# K^T; V and KPE use their own ladders.
#
#   Factory                              | Used for
#   -------------------------------------|-------------------------------------
#   make_fp8_kt_dll                      | FP8 prefix K^T DLL   (D>=128; D<128 falls back to BF16)
#   make_fp8_v_dll                       | FP8 prefix V   DLL
#   make_fp8_v_offset_bases              | FP8 prefix V   offset bases (alias)
#   make_fp8_extend_v_dll                | FP8 kernel BF16 extend V DLL (alias for make_fp8_bf16_v_dll)
#   make_ext_kt_offset_bases / _dll      | BF16 extend K^T when EXT_BLOCK_N != BLOCK_N (rare)
#   make_ext_v_offset_bases  / _dll      | BF16 extend V   when EXT_BLOCK_N != BLOCK_N (rare)
# ===-----------------------------------------------------------------------===#


@gluon.constexpr_function
def make_fp8_kt_dll(num_warps, BLOCK_DMODEL, BLOCK_N):
    """Async DMA DLL for FP8 prefix K^T tile [BLOCK_DMODEL, BLOCK_N].

    FP8 packs 8 elements per vector register along D (1 byte each = 8 bytes,
    matches a dword load). For D<128 the partition matches BF16 KT, so we
    delegate to ``make_kt_dll`` in that case.
    """
    is_4w = num_warps < 8
    if BLOCK_DMODEL < 128:
        return make_kt_dll(num_warps, BLOCK_DMODEL, BLOCK_N)
    if is_4w:
        if BLOCK_DMODEL >= 256:
            reg  = [[1,0],[2,0],[4,0],[0,4],[0,8]]
            lane = [[8,0],[16,0],[32,0],[64,0],[128,0],[0,16]]
            warp = [[0,1],[0,2]]
        else:  # D=128
            if BLOCK_N >= 128:
                reg  = [[1,0],[2,0],[4,0],[8,0],[0,4],[0,8]]
                lane = [[16,0],[32,0],[64,0],[0,16],[0,32],[0,64]]
                warp = [[0,1],[0,2]]
            else:
                reg  = [[1,0],[2,0],[4,0],[8,0],[0,8]]
                lane = [[16,0],[32,0],[64,0],[0,1],[0,2],[0,4]]
                warp = [[0,16],[0,32]]
    else:  # 8-warp
        if BLOCK_DMODEL >= 256:
            reg  = [[1,0],[2,0],[4,0],[0,8]]
            lane = [[8,0],[16,0],[32,0],[64,0],[128,0],[0,16]]
            warp = [[0,1],[0,2],[0,4]]
        else:  # D=128
            if BLOCK_N >= 128:
                reg  = [[1,0],[2,0],[4,0],[8,0],[0,8]]
                lane = [[16,0],[32,0],[64,0],[0,16],[0,32],[0,64]]
                warp = [[0,1],[0,2],[0,4]]
            else:
                reg  = [[1,0],[2,0],[4,0],[8,0]]
                lane = [[16,0],[32,0],[64,0],[0,1],[0,2],[0,4]]
                warp = [[0,8],[0,16],[0,32]]
    return DistributedLinearLayout(
        reg_bases=reg, lane_bases=lane, warp_bases=warp,
        block_bases=[], shape=[BLOCK_DMODEL, BLOCK_N],
    )


@gluon.constexpr_function
def make_fp8_v_dll(num_warps, BLOCK_DMODEL, BLOCK_DV, BLOCK_N):
    """Async DMA DLL for FP8 prefix V tile [BLOCK_N, BLOCK_DV].

    The 8-warp layout depends on (D, Dv, N) jointly because K^T shared-memory
    pressure (driven by D) constrains the V warp partition. For D<128 this
    matches BF16 V, so we delegate to ``make_v_dll``. The 4-warp path depends
    on (Dv, N) only.
    """
    if BLOCK_DMODEL < 128:
        return make_v_dll(num_warps, BLOCK_DV, BLOCK_N)
    is_4w = num_warps < 8
    if is_4w:
        if BLOCK_DV >= 256:
            if BLOCK_N >= 128:
                reg  = [[0,1],[0,2],[0,4],[0,8],[4,0],[8,0],[64,0]]
                lane = [[0,16],[0,32],[0,64],[0,128],[16,0],[32,0]]
            else:
                reg  = [[0,1],[0,2],[0,4],[4,0],[8,0]]
                lane = [[0,8],[0,16],[0,32],[0,64],[0,128],[16,0]]
            warp = [[1,0],[2,0]]
        else:  # Dv<256
            if BLOCK_N >= 128:
                reg  = [[0,1],[0,2],[0,4],[0,8],[4,0],[8,0]]
                lane = [[0,16],[0,32],[0,64],[16,0],[32,0],[64,0]]
                warp = [[1,0],[2,0]]
            else:
                reg  = [[0,1],[0,2],[0,4],[0,8],[8,0]]
                lane = [[0,16],[0,32],[0,64],[1,0],[2,0],[4,0]]
                warp = [[16,0],[32,0]]
    else:  # 8-warp
        if BLOCK_DMODEL >= 256:
            if BLOCK_DV >= 256:
                reg  = [[0,1],[0,2],[0,4],[8,0]]
                lane = [[0,8],[0,16],[0,32],[0,64],[0,128],[16,0]]
            else:
                reg  = [[0,1],[0,2],[0,4],[8,0]]
                lane = [[0,8],[0,16],[0,32],[0,64],[16,0],[32,0]]
            warp = [[1,0],[2,0],[4,0]]
        else:  # D=128  (Dv=128 implied at every callsite)
            if BLOCK_N >= 128:
                reg  = [[0,1],[0,2],[0,4],[0,8],[8,0]]
                lane = [[0,16],[0,32],[0,64],[16,0],[32,0],[64,0]]
                warp = [[1,0],[2,0],[4,0]]
            else:
                reg  = [[0,1],[0,2],[0,4],[0,8]]
                lane = [[0,16],[0,32],[0,64],[1,0],[2,0],[4,0]]
                warp = [[8,0],[16,0],[32,0]]
    return DistributedLinearLayout(
        reg_bases=reg, lane_bases=lane, warp_bases=warp,
        block_bases=[], shape=[BLOCK_N, BLOCK_DV],
    )


# FP8 prefix V uses the same offset bases as the FP8 kernel's BF16 extend V
# tile -- the FP8 vs BF16 distinction shows up only in the DLL partition.
make_fp8_v_offset_bases = make_fp8_bf16_v_offset_bases

# Alias: the FP8 kernel's BF16-extend V tile. Semantically the FP8 kernel's
# "extend" counterpart to ``make_fp8_v_dll``.
make_fp8_extend_v_dll = make_fp8_bf16_v_dll


@gluon.constexpr_function
def make_ext_kt_offset_bases(num_warps, BLOCK_DMODEL, EXT_BLOCK_N):
    """BF16 extend K^T offset bases for ``EXT_BLOCK_N != BLOCK_N`` (rare).

    Used only inside the FP8 kernel's 4-warp DMA path when the extend tile
    size differs from the prefix tile size. All other extend paths reuse
    ``make_kt_offset_bases``.
    """
    if BLOCK_DMODEL >= 256:
        return make_offset_bases(128, [16,32], [1,2,4,8], 0)
    return make_offset_bases(64, [16,32], [1,2,4,8], 0)


@gluon.constexpr_function
def make_ext_kt_dll(num_warps, BLOCK_DMODEL, EXT_BLOCK_N):
    """BF16 extend K^T DLL for ``EXT_BLOCK_N != BLOCK_N`` (rare)."""
    if BLOCK_DMODEL >= 256:
        reg = [[1,0],[2,0],[4,0],[128,0],[0,4],[0,8]]
    else:
        reg = [[1,0],[2,0],[4,0],[0,4],[0,8]]
    lane = [[8,0],[16,0],[32,0],[64,0],[0,16],[0,32]]
    warp = [[0,1],[0,2]]
    return DistributedLinearLayout(
        reg_bases=reg, lane_bases=lane, warp_bases=warp,
        block_bases=[], shape=[BLOCK_DMODEL, EXT_BLOCK_N],
    )


@gluon.constexpr_function
def make_ext_v_offset_bases(num_warps, BLOCK_DV, EXT_BLOCK_N):
    """BF16 extend V offset bases for ``EXT_BLOCK_N != BLOCK_N`` (rare)."""
    if BLOCK_DV >= 256:
        return make_offset_bases(128, [16,32], [1,2,4,8], 1)
    return make_offset_bases(64, [16,32], [1,2,4,8], 1)


@gluon.constexpr_function
def make_ext_v_dll(num_warps, BLOCK_DV, EXT_BLOCK_N):
    """BF16 extend V DLL for ``EXT_BLOCK_N != BLOCK_N`` (rare)."""
    if BLOCK_DV >= 256:
        reg = [[0,1],[0,2],[0,4],[0,128],[4,0],[8,0]]
    else:
        reg = [[0,1],[0,2],[0,4],[4,0],[8,0]]
    lane = [[0,8],[0,16],[0,32],[0,64],[16,0],[32,0]]
    warp = [[1,0],[2,0]]
    return DistributedLinearLayout(
        reg_bases=reg, lane_bases=lane, warp_bases=warp,
        block_bases=[], shape=[EXT_BLOCK_N, BLOCK_DV],
    )


# ===-----------------------------------------------------------------------===#
# Primitives (inlined from f16_extend_attention_gfx950)
# ===-----------------------------------------------------------------------===#


@gluon.jit
def _nan_propagating_maximum(a, b):
    # Elementwise binary-op used as the combinator for `nan_propagating_max`
    # below. Named `maximum` (not `max`) to match the triton/numpy convention:
    # `maximum` is the elementwise binary-op, `max` is the reduction.
    return gl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@gluon.jit
def nan_propagating_max(x, axis):
    # Reduction along `axis` using NaN-propagating maximum as the combinator.
    return gl.reduce(x, axis, _nan_propagating_maximum)


@gluon.jit
def do_mma(a, b, c):
    if b.dtype == tl.float8e4b8 or b.dtype == tl.float8e4nv:
        a_fp8 = tl.cast(a, tl.float8e4nv, bitcast=(a.dtype != tl.bfloat16 and a.dtype != tl.float16))
        b_fp8 = tl.cast(b, tl.float8e4nv, bitcast=True)
        return mfma_cdna4(a_fp8, b_fp8, c)
    return mfma_cdna4(a.to(tl.bfloat16), b.to(tl.bfloat16), c)


@gluon.jit
def _buffer_load_to_shared_cast_safe(dst_smem, src_base, offsets, mask, other):
    """Type-safe global->shared load for mixed fp8/bf16 buffers."""
    if dst_smem.dtype == src_base.dtype.element_ty:
        if mask is not None:
            cdna4_async.buffer_load_to_shared(
                dst_smem, src_base, offsets, mask=mask, other=other
            )
        else:
            cdna4_async.buffer_load_to_shared(
                dst_smem, src_base, offsets
            )
    else:
        if mask is not None:
            reg = cdna4_buffer_load(src_base, offsets, mask=mask, other=other)
        else:
            reg = cdna4_buffer_load(src_base, offsets)
        dst_smem.store(reg.to(dst_smem.dtype))
    cdna4_async.commit_group()


@gluon.jit
def issue_async_load_k_prefix(
    kt_smem,
    k_prefix_base,  #
    kv_indices,
    kv_start,
    start_n,
    seq_len_prefix,  #
    stride_buf_kbs,  #
    BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
    kt_async_layout: gl.constexpr,  #
):
    kt_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=kt_async_layout)
    kt_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=kt_async_layout)
    kt_offs_d = gl.arange(0, BLOCK_DMODEL, layout=kt_offs_d_layout)
    kt_offs_n = gl.arange(0, BLOCK_N, layout=kt_offs_n_layout)

    n_idx = start_n + kt_offs_n
    mask_n = n_idx < seq_len_prefix
    kv_locs = gl.load(kv_indices + kv_start + n_idx, mask=mask_n, other=0).to(tl.int32)

    kt_offsets = (kt_offs_d[:, None] + kv_locs[None, :] * stride_buf_kbs).to(tl.int32)

    kt_mask = mask_n[None, :]
    _buffer_load_to_shared_cast_safe(
        kt_smem, k_prefix_base, kt_offsets, mask=kt_mask, other=0.0
    )


@gluon.jit
def issue_async_load_v_prefix(
    v_smem,
    v_prefix_base,  #
    kv_indices,
    kv_start,
    start_n,
    seq_len_prefix,  #
    stride_buf_vbs,  #
    BLOCK_N: gl.constexpr,
    BLOCK_DV: gl.constexpr,
    v_async_layout: gl.constexpr,  #
):
    v_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=v_async_layout)
    v_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=v_async_layout)
    v_offs_n = gl.arange(0, BLOCK_N, layout=v_offs_n_layout)
    v_offs_d = gl.arange(0, BLOCK_DV, layout=v_offs_d_layout)

    n_idx = start_n + v_offs_n
    mask_n = n_idx < seq_len_prefix
    kv_locs = gl.load(kv_indices + kv_start + n_idx, mask=mask_n, other=0).to(tl.int32)

    v_offsets = (kv_locs[:, None] * stride_buf_vbs + v_offs_d[None, :]).to(tl.int32)

    v_mask = mask_n[:, None]
    _buffer_load_to_shared_cast_safe(
        v_smem, v_prefix_base, v_offsets, mask=v_mask, other=0.0
    )


@gluon.jit
def issue_dma_k_prefix_from_locs(
    kt_smem,
    k_prefix_base,  #
    kv_locs,
    mask_n,
    stride_buf_kbs,  #
    BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
    kt_async_layout: gl.constexpr,  #
):
    kt_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=kt_async_layout)
    kt_offs_d = gl.arange(0, BLOCK_DMODEL, layout=kt_offs_d_layout)
    kt_offsets = (kt_offs_d[:, None] + kv_locs[None, :] * stride_buf_kbs).to(tl.int32)
    kt_mask = mask_n[None, :]
    _buffer_load_to_shared_cast_safe(
        kt_smem, k_prefix_base, kt_offsets, mask=kt_mask, other=0.0
    )


@gluon.jit
def issue_dma_v_prefix_from_locs(
    v_smem,
    v_prefix_base,  #
    kv_locs,
    mask_n,
    stride_buf_vbs,  #
    BLOCK_N: gl.constexpr,
    BLOCK_DV: gl.constexpr,
    v_async_layout: gl.constexpr,  #
):
    v_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=v_async_layout)
    v_offs_d = gl.arange(0, BLOCK_DV, layout=v_offs_d_layout)
    v_offsets = (kv_locs[:, None] * stride_buf_vbs + v_offs_d[None, :]).to(tl.int32)
    v_mask = mask_n[:, None]
    _buffer_load_to_shared_cast_safe(
        v_smem, v_prefix_base, v_offsets, mask=v_mask, other=0.0
    )


@gluon.jit
def issue_dma_k_prefix_from_locs_hot(
    kt_smem,
    k_prefix_base,  #
    kv_locs,
    scalar_mask,
    stride_buf_kbs,  #
    BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
    kt_async_layout: gl.constexpr,  #
):
    kt_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=kt_async_layout)
    kt_offs_d = gl.arange(0, BLOCK_DMODEL, layout=kt_offs_d_layout)
    kt_offsets = (kt_offs_d[:, None] + kv_locs[None, :] * stride_buf_kbs).to(tl.int32)
    _buffer_load_to_shared_cast_safe(
        kt_smem, k_prefix_base, kt_offsets, mask=scalar_mask, other=0.0
    )


@gluon.jit
def issue_dma_v_prefix_from_locs_hot(
    v_smem,
    v_prefix_base,  #
    kv_locs,
    scalar_mask,
    stride_buf_vbs,  #
    BLOCK_N: gl.constexpr,
    BLOCK_DV: gl.constexpr,
    v_async_layout: gl.constexpr,  #
):
    v_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=v_async_layout)
    v_offs_d = gl.arange(0, BLOCK_DV, layout=v_offs_d_layout)
    v_offsets = (kv_locs[:, None] * stride_buf_vbs + v_offs_d[None, :]).to(tl.int32)
    _buffer_load_to_shared_cast_safe(
        v_smem, v_prefix_base, v_offsets, mask=scalar_mask, other=0.0
    )


@gluon.jit
def issue_async_load_k_extend(
    kt_smem,
    k_base,
    start_n,
    seq_len_extend,  #
    stride_kbs,  #
    BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
    kt_async_layout: gl.constexpr,  #
    SKIP_BOUNDS_CHECK: gl.constexpr = False,
):
    kt_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=kt_async_layout)
    kt_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=kt_async_layout)
    kt_offs_d = gl.arange(0, BLOCK_DMODEL, layout=kt_offs_d_layout)
    kt_offs_n = gl.arange(0, BLOCK_N, layout=kt_offs_n_layout)
    kt_offsets = (kt_offs_d[:, None] + (start_n + kt_offs_n[None, :]) * stride_kbs).to(
        tl.int32
    )

    if SKIP_BOUNDS_CHECK:
        kt_mask = None
    else:
        kt_mask = (start_n + kt_offs_n[None, :]) < seq_len_extend
    _buffer_load_to_shared_cast_safe(
        kt_smem, k_base, kt_offsets, mask=kt_mask, other=0.0
    )


@gluon.jit
def issue_async_load_v_extend(
    v_smem,
    v_base,
    start_n,
    seq_len_extend,  #
    stride_vbs,  #
    BLOCK_N: gl.constexpr,
    BLOCK_DV: gl.constexpr,
    v_async_layout: gl.constexpr,  #
    SKIP_BOUNDS_CHECK: gl.constexpr = False,
):
    v_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=v_async_layout)
    v_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=v_async_layout)
    v_offs_n = gl.arange(0, BLOCK_N, layout=v_offs_n_layout)
    v_offs_d = gl.arange(0, BLOCK_DV, layout=v_offs_d_layout)
    v_offsets = ((start_n + v_offs_n[:, None]) * stride_vbs + v_offs_d[None, :]).to(
        tl.int32
    )

    if SKIP_BOUNDS_CHECK:
        v_mask = None
    else:
        v_mask = (start_n + v_offs_n[:, None]) < seq_len_extend
    _buffer_load_to_shared_cast_safe(
        v_smem, v_base, v_offsets, mask=v_mask, other=0.0
    )


@gluon.jit
def compute_softmax_prefix(
    acc,
    l_i,
    m_i,
    qk,
    start_n,
    seq_len_prefix,  #
    qk_scale,
    LOGIT_CAP: gl.constexpr,  #
    xai_temperature_reg,
    XAI_TEMPERATURE_LEN: gl.constexpr,  #
    q_abs_pos,
    SLIDING_WINDOW_SIZE: gl.constexpr,  #
    Mask,
    mask_base_idx,
    mask_row_stride,
    q_extend_offs,  #
    USE_CUSTOM_MASK: gl.constexpr,
    SKIP_PREFIX_CUSTOM_MASK: gl.constexpr,  #
    ENABLE_PREFIX_UNMASKED: gl.constexpr,  #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,  #
    mma_layout: gl.constexpr,
    mma_offs_n_col: gl.constexpr,  #
):
    qk_scaled = qk * qk_scale
    if LOGIT_CAP > 0:
        log2_cap: gl.constexpr = LOGIT_CAP * LOG2E
        inv_cap: gl.constexpr = 2.0 / LOGIT_CAP
        e_neg = tl.math.exp2(-qk_scaled * inv_cap)
        sig = 1.0 / (1.0 + e_neg)
        qk_scaled = log2_cap * (2.0 * sig - 1.0)
    if XAI_TEMPERATURE_LEN > 0:
        qk_scaled = qk_scaled * xai_temperature_reg[:, None]
    bound_offs = start_n + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
    use_custom_prefix_mask = USE_CUSTOM_MASK and (not SKIP_PREFIX_CUSTOM_MASK)
    is_partial_tail = (start_n + BLOCK_N) > seq_len_prefix
    if SLIDING_WINDOW_SIZE > 0:
        swa_safe = (tl.max(q_abs_pos) <= start_n + SLIDING_WINDOW_SIZE)
    else:
        swa_safe = True
    use_unmasked_path = (
        ENABLE_PREFIX_UNMASKED
        and swa_safe
        and (not use_custom_prefix_mask)
        and (not is_partial_tail)
    )

    if use_unmasked_path:
        m_ij = nan_propagating_max(qk_scaled, axis=1)
        m_new = gl.maximum(m_i, m_ij, propagate_nan=tl.PropagateNan.ALL)
        p = gl.exp2(qk_scaled - m_new[:, None])
    else:
        bound_mask = (q_abs_pos[:, None] >= 0) & (bound_offs[None, :] < seq_len_prefix)
        if SLIDING_WINDOW_SIZE > 0:
            bound_mask = bound_mask & (
                q_abs_pos[:, None] <= bound_offs[None, :] + SLIDING_WINDOW_SIZE
            )
        if use_custom_prefix_mask:
            mask_ptrs = (
                Mask
                + mask_base_idx
                + q_extend_offs[:, None] * mask_row_stride
                + start_n
                + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)[None, :]
            )
            mask_vals = gl.load(mask_ptrs, mask=bound_mask, other=0)
            bound_mask = bound_mask & (mask_vals != 0)
        qk_scaled = gl.where(
            bound_mask,
            qk_scaled,
            gl.full(
                [BLOCK_M, BLOCK_N], float("-inf"), dtype=gl.float32, layout=mma_layout
            ),
        )

        m_ij = nan_propagating_max(qk_scaled, axis=1)
        m_new = gl.maximum(m_i, m_ij, propagate_nan=tl.PropagateNan.ALL)
        if SLIDING_WINDOW_SIZE > 0 or use_custom_prefix_mask:
            m_new = gl.maximum(
                m_new,
                gl.full(
                    [BLOCK_M],
                    -1e20,
                    dtype=gl.float32,
                    layout=gl.SliceLayout(dim=1, parent=mma_layout),
                ),
            )
        p = gl.exp2(qk_scaled - m_new[:, None])
    l_ij = gl.sum(p, axis=1)
    alpha = gl.exp2(m_i - m_new)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    m_i = m_new
    return acc, l_i, m_i, p


@gluon.jit
def compute_softmax_extend(
    acc,
    l_i,
    m_i,
    qk,
    start_n,
    cur_block_m,
    seq_len_extend,  #
    qk_scale,
    LOGIT_CAP: gl.constexpr,  #
    xai_temperature_reg,
    XAI_TEMPERATURE_LEN: gl.constexpr,  #
    SLIDING_WINDOW_SIZE: gl.constexpr,  #
    IS_CAUSAL: gl.constexpr,  #
    Mask,
    mask_base_idx,
    mask_row_stride,
    mask_kv_col_offset,  #
    USE_CUSTOM_MASK: gl.constexpr,  #
    MASK_STEPS: gl.constexpr,  #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,  #
    mma_layout: gl.constexpr,
    mma_offs_n_col: gl.constexpr,
    mma_offs_m_row: gl.constexpr,  #
):
    """Fused softmax step: ``compute_softmax_extend_part0`` composed with
    ``compute_softmax_extend_part1``. Returns ``(acc, l_i, m_i, p)``.

    Used by callers that don't need the ``_part0``/``_part1`` split for
    warp-pipelined overlap between QK MMA and V DMA.
    """
    p, alpha, m_new = compute_softmax_extend_part0(
        m_i,
        qk,
        start_n,
        cur_block_m,
        seq_len_extend,
        qk_scale,
        LOGIT_CAP,
        xai_temperature_reg,
        XAI_TEMPERATURE_LEN,
        SLIDING_WINDOW_SIZE,
        IS_CAUSAL,
        Mask,
        mask_base_idx,
        mask_row_stride,
        mask_kv_col_offset,
        USE_CUSTOM_MASK,
        MASK_STEPS,
        BLOCK_M,
        BLOCK_N,
        mma_layout,
        mma_offs_n_col,
        mma_offs_m_row,
    )
    acc, l_i, m_i = compute_softmax_extend_part1(acc, l_i, p, alpha, m_new)
    return acc, l_i, m_i, p


@gluon.jit
def compute_softmax_extend_part0(
    m_i,
    qk,
    start_n,
    cur_block_m,
    seq_len_extend,  #
    qk_scale,
    LOGIT_CAP: gl.constexpr,  #
    xai_temperature_reg,
    XAI_TEMPERATURE_LEN: gl.constexpr,  #
    SLIDING_WINDOW_SIZE: gl.constexpr,  #
    IS_CAUSAL: gl.constexpr,  #
    Mask,
    mask_base_idx,
    mask_row_stride,
    mask_kv_col_offset,  #
    USE_CUSTOM_MASK: gl.constexpr,  #
    MASK_STEPS: gl.constexpr,  #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,  #
    mma_layout: gl.constexpr,
    mma_offs_n_col: gl.constexpr,
    mma_offs_m_row: gl.constexpr,  #
):
    """Scale, optional mask, max reduction, exp2, alpha.

    Produces (p, alpha, m_new) without touching acc or l_i.
    Intended to run between QK MMA and V wait so the VALU work
    overlaps with the V global→LDS DMA.
    """
    qk_scaled = qk * qk_scale
    if LOGIT_CAP > 0:
        log2_cap: gl.constexpr = LOGIT_CAP * LOG2E
        inv_cap: gl.constexpr = 2.0 / LOGIT_CAP
        e_neg = tl.math.exp2(-qk_scaled * inv_cap)
        sig = 1.0 / (1.0 + e_neg)
        qk_scaled = log2_cap * (2.0 * sig - 1.0)
    if XAI_TEMPERATURE_LEN > 0:
        qk_scaled = qk_scaled * xai_temperature_reg[:, None]
    if MASK_STEPS:
        bound_offs = start_n + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
        q_offs = cur_block_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=mma_offs_m_row)
        valid_mask = q_offs[:, None] < seq_len_extend
        valid_mask = valid_mask & (bound_offs[None, :] < seq_len_extend)
        if USE_CUSTOM_MASK:
            mask_ptrs = (
                Mask
                + mask_base_idx
                + q_offs[:, None] * mask_row_stride
                + mask_kv_col_offset
                + start_n
                + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)[None, :]
            )
            mask_vals = gl.load(mask_ptrs, mask=valid_mask, other=0)
            valid_mask = valid_mask & (mask_vals != 0)
        elif IS_CAUSAL:
            valid_mask = valid_mask & (q_offs[:, None] >= bound_offs[None, :])
        if SLIDING_WINDOW_SIZE > 0:
            valid_mask = valid_mask & (
                q_offs[:, None] <= bound_offs[None, :] + SLIDING_WINDOW_SIZE
            )
        qk_scaled = gl.where(
            valid_mask,
            qk_scaled,
            gl.full(
                [BLOCK_M, BLOCK_N], float("-inf"), dtype=gl.float32, layout=mma_layout
            ),
        )

        m_ij = nan_propagating_max(qk_scaled, axis=1)
        m_new = gl.maximum(m_i, m_ij, propagate_nan=tl.PropagateNan.ALL)
        if SLIDING_WINDOW_SIZE > 0 or USE_CUSTOM_MASK:
            m_new = gl.maximum(
                m_new,
                gl.full(
                    [BLOCK_M],
                    -1e20,
                    dtype=gl.float32,
                    layout=gl.SliceLayout(dim=1, parent=mma_layout),
                ),
            )
        p = gl.exp2(qk_scaled - m_new[:, None])
    else:
        m_ij = nan_propagating_max(qk_scaled, axis=1)
        m_new = gl.maximum(m_i, m_ij, propagate_nan=tl.PropagateNan.ALL)
        p = gl.exp2(qk_scaled - m_new[:, None])
    alpha = gl.exp2(m_i - m_new)
    return p, alpha, m_new


@gluon.jit
def compute_softmax_extend_part1(acc, l_i, p, alpha, m_new):
    """Row-sum, rescale acc and l_i, update m_i.

    Runs after V is available so the acc rescale feeds directly
    into the P*V MMA that follows.
    """
    l_ij = gl.sum(p, axis=1)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    m_i = m_new
    return acc, l_i, m_i


# ===-----------------------------------------------------------------------===#
# Prefix inner loops
#
# Four helpers, selected by (warp count, NUM_STAGES, BLOCK_DMODEL):
#
#   attn_fwd_inner_prefix_pingpong_8w     -- 8-warp pingpong (warp_pipeline_stage), NS>=2, any D
#   attn_fwd_inner_prefix_sw_pipeline_4w  -- 4-warp sw-pipeline (manual async DMA), NS>=2, D>=128
#   attn_fwd_inner_prefix_serial_4w       -- 4-warp synchronous (no async), NS=1 or D<128
#   attn_fwd_inner_prefix_unpipelined     -- 1-stage fallback; also the
#                                            short-prefix path (n_full_prefix < NUM_STAGES)
#
# Callers always pass the full prefix length as ``seq_len_prefix``. The
# pingpong and sw-pipeline helpers floor-align to ``BLOCK_N`` internally
# and run a single masked-tail block (via the unpipelined helper) when
# the full length isn't a multiple of ``BLOCK_N``. On the persistent
# path ``seq_len_prefix`` is already aligned (split-K partitions on
# ``BLOCK_N`` boundaries), so the tail is a runtime no-op.
# ===-----------------------------------------------------------------------===#


@gluon.jit
def attn_fwd_inner_prefix_pingpong_8w(
    acc,
    l_i,
    m_i,
    q_dot,  #
    K_Buffer,
    V_Buffer,
    kv_indices,  #
    kv_start,
    cur_kv_head,
    seq_len_prefix,  # full prefix length; helper floor-aligns to BLOCK_N for the pipelined bulk
    stride_buf_kbs,
    stride_buf_kh,  #
    stride_buf_vbs,
    stride_buf_vh,  #
    kt_smem,
    v_smem,  #
    qk_scale,
    LOGIT_CAP: gl.constexpr,  #
    xai_temperature_reg,
    XAI_TEMPERATURE_LEN: gl.constexpr,  #
    q_abs_pos,
    SLIDING_WINDOW_SIZE: gl.constexpr,  #
    Mask,
    mask_base_idx,
    mask_row_stride,
    q_extend_offs,  #
    USE_CUSTOM_MASK: gl.constexpr,
    SKIP_PREFIX_CUSTOM_MASK: gl.constexpr,  #
    ENABLE_PREFIX_UNMASKED: gl.constexpr,  #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,  #
    BLOCK_DMODEL: gl.constexpr,
    BLOCK_DV: gl.constexpr,
    NUM_STAGES: gl.constexpr,  #
    kt_async_layout: gl.constexpr,
    v_async_layout: gl.constexpr,  #
    kt_dot_layout: gl.constexpr,
    p_dot_layout: gl.constexpr,
    v_dot_layout: gl.constexpr,  #
    mma_layout: gl.constexpr,
    mma_offs_n_col: gl.constexpr,  #
    SCALAR_MASK: gl.constexpr = False,  # True: _hot DMA with scalar seq>0 mask (fewer insts, +22% geomean on persistent)
):
    """8-warp pingpong prefix: warp_pipeline_stage-scheduled async DMA loop.

    Runs compute ("compute*") and memory ("memory*") on alternating warp
    groups; requires NUM_STAGES>=2 physical LDS buffers for deterministic
    pipeline scheduling. Callers pass the full prefix length -- the helper
    processes the BLOCK_N-aligned bulk via the pipelined loop and then
    runs a single masked block covering any partial tail. On the
    persistent path ``seq_len_prefix`` is already BLOCK_N-aligned, so
    that tail branch is a runtime no-op.
    """
    STREAMS: gl.constexpr = 2
    # warp_pipeline_stage requires >=2 physical LDS buffers. With NS=1 the
    # stage_idx collapses to 0, and iter N's DMA write to smem[0] races
    # iter N+1's relaxed read from smem[0] because membarFilter skips
    # barriers between BufferLoadToLocalOp and syncedViaAsyncWait loads.
    # Callers wanting NS=1 must route to attn_fwd_inner_prefix_unpipelined
    # (or the 4-warp sw_pipeline / serial variants).
    tl.static_assert(
        NUM_STAGES >= 2,
        "attn_fwd_inner_prefix_pingpong_8w requires NUM_STAGES>=2 for "
        "determinism (warp_pipeline_stage needs multiple LDS buffers).",
    )

    # BLOCK_N-aligned bulk: the pingpong loop only touches whole BLOCK_N
    # slices. Any partial tail is handled after the pipeline drains.
    aligned_prefix_len = (seq_len_prefix // BLOCK_N) * BLOCK_N
    n_prefix_blocks = aligned_prefix_len // BLOCK_N
    k_prefix_base = K_Buffer + cur_kv_head * stride_buf_kh
    v_prefix_base = V_Buffer + cur_kv_head * stride_buf_vh

    kt_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=kt_async_layout)
    v_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=v_async_layout)
    kt_offs_n_pf = gl.arange(0, BLOCK_N, layout=kt_offs_n_layout)
    v_offs_n_pf = gl.arange(0, BLOCK_N, layout=v_offs_n_layout)

    for stage in gl.static_range(NUM_STAGES):
        pf_init_n = stage * BLOCK_N
        issue_async_load_k_prefix(
            kt_smem.index(stage),
            k_prefix_base,
            kv_indices,
            kv_start,
            pf_init_n,
            seq_len_prefix,
            stride_buf_kbs,
            BLOCK_N,
            BLOCK_DMODEL,
            kt_async_layout,
        )
        issue_async_load_v_prefix(
            v_smem.index(stage),
            v_prefix_base,
            kv_indices,
            kv_start,
            pf_init_n,
            seq_len_prefix,
            stride_buf_vbs,
            BLOCK_N,
            BLOCK_DV,
            v_async_layout,
        )

    pf_start_n = NUM_STAGES * BLOCK_N
    n_idx_kt_pf = pf_start_n + kt_offs_n_pf
    mask_n_kt_pf = n_idx_kt_pf < seq_len_prefix
    kv_locs_kt_pf = gl.load(
        kv_indices + kv_start + n_idx_kt_pf, mask=mask_n_kt_pf, other=0
    ).to(tl.int32)
    n_idx_v_pf = pf_start_n + v_offs_n_pf
    mask_n_v_pf = n_idx_v_pf < seq_len_prefix
    kv_locs_v_pf = gl.load(
        kv_indices + kv_start + n_idx_v_pf, mask=mask_n_v_pf, other=0
    ).to(tl.int32)

    WAIT_K: gl.constexpr = STREAMS * NUM_STAGES - (STREAMS - 1)
    WAIT_V: gl.constexpr = STREAMS * NUM_STAGES - STREAMS

    # Scalar-mask fast path: replaces the per-element `n < seq_len_prefix`
    # vector mask on K/V DMAs with a single scalar `seq_len_prefix > 0`
    # check. Safe whenever `n_prefix_blocks * BLOCK_N` bounds were already
    # established at prologue time (which is the case here).
    dma_mask = seq_len_prefix > 0

    cdna4_async.wait_group(WAIT_K)
    kt_dot = cdna4_async.load_shared_relaxed(kt_smem.index(0), kt_dot_layout)

    main_loop_end = n_prefix_blocks - NUM_STAGES
    for block_n in tl.range(0, main_loop_end, loop_unroll_factor=2):

        with warp_pipeline_stage("compute0", priority=0):
            stage_idx = (block_n % NUM_STAGES).to(tl.int32)
            start_n = (block_n * BLOCK_N).to(tl.int32)
            qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
            qk = do_mma(q_dot, kt_dot, qk)

        cdna4_async.wait_group(WAIT_V)

        with warp_pipeline_stage("memory0", priority=1):
            v_dot = cdna4_async.load_shared_relaxed(
                v_smem.index(stage_idx), v_dot_layout
            )
            if SCALAR_MASK:
                issue_dma_k_prefix_from_locs_hot(
                    kt_smem.index(stage_idx),
                    k_prefix_base,
                    kv_locs_kt_pf,
                    dma_mask,
                    stride_buf_kbs,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    kt_async_layout,
                )
            else:
                issue_dma_k_prefix_from_locs(
                    kt_smem.index(stage_idx),
                    k_prefix_base,
                    kv_locs_kt_pf,
                    mask_n_kt_pf,
                    stride_buf_kbs,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    kt_async_layout,
                )

        with warp_pipeline_stage("compute1", priority=0):
            acc, l_i, m_i, p = compute_softmax_prefix(
                acc,
                l_i,
                m_i,
                qk,
                start_n,
                seq_len_prefix,
                qk_scale,
                LOGIT_CAP,
                xai_temperature_reg,
                XAI_TEMPERATURE_LEN,
                q_abs_pos,
                SLIDING_WINDOW_SIZE,
                Mask,
                mask_base_idx,
                mask_row_stride,
                q_extend_offs,
                USE_CUSTOM_MASK,
                SKIP_PREFIX_CUSTOM_MASK,
                ENABLE_PREFIX_UNMASKED,
                BLOCK_M,
                BLOCK_N,
                mma_layout,
                mma_offs_n_col,
            )

        with warp_pipeline_stage("memory1", priority=1):
            if SCALAR_MASK:
                issue_dma_v_prefix_from_locs_hot(
                    v_smem.index(stage_idx),
                    v_prefix_base,
                    kv_locs_v_pf,
                    dma_mask,
                    stride_buf_vbs,
                    BLOCK_N,
                    BLOCK_DV,
                    v_async_layout,
                )
            else:
                issue_dma_v_prefix_from_locs(
                    v_smem.index(stage_idx),
                    v_prefix_base,
                    kv_locs_v_pf,
                    mask_n_v_pf,
                    stride_buf_vbs,
                    BLOCK_N,
                    BLOCK_DV,
                    v_async_layout,
                )
            nf_start_n = ((block_n + NUM_STAGES + 1) * BLOCK_N).to(tl.int32)
            n_idx_kt_nf = nf_start_n + kt_offs_n_pf
            mask_n_kt_pf = n_idx_kt_nf < seq_len_prefix
            kv_locs_kt_pf = gl.load(
                kv_indices + kv_start + n_idx_kt_nf, mask=mask_n_kt_pf, other=0
            ).to(tl.int32)

        with warp_pipeline_stage("compute2", priority=0):
            p_cast = p.to(v_dot.dtype)
            p_dot_reg = gl.convert_layout(p_cast, p_dot_layout)
            acc = do_mma(p_dot_reg, v_dot, acc)

        cdna4_async.wait_group(WAIT_K)

        with warp_pipeline_stage("memory2", priority=1):
            next_stage_idx = ((block_n + 1) % NUM_STAGES).to(tl.int32)
            kt_dot = cdna4_async.load_shared_relaxed(
                kt_smem.index(next_stage_idx), kt_dot_layout
            )
            n_idx_v_nf = nf_start_n + v_offs_n_pf
            mask_n_v_pf = n_idx_v_nf < seq_len_prefix
            kv_locs_v_pf = gl.load(
                kv_indices + kv_start + n_idx_v_nf, mask=mask_n_v_pf, other=0
            ).to(tl.int32)

    for tail_i in gl.static_range(NUM_STAGES):
        cdna4_async.wait_group(STREAMS * (NUM_STAGES - tail_i) - (STREAMS - 1))
        stage_idx = ((main_loop_end + tail_i) % NUM_STAGES).to(tl.int32)
        start_n = (main_loop_end + tail_i) * BLOCK_N

        kt_dot_tail = cdna4_async.load_shared_relaxed(
            kt_smem.index(stage_idx), kt_dot_layout
        )
        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        qk = do_mma(q_dot, kt_dot_tail, qk)

        acc, l_i, m_i, p = compute_softmax_prefix(
            acc,
            l_i,
            m_i,
            qk,
            start_n,
            seq_len_prefix,
            qk_scale,
            LOGIT_CAP,
            xai_temperature_reg,
            XAI_TEMPERATURE_LEN,
            q_abs_pos,
            SLIDING_WINDOW_SIZE,
            Mask,
            mask_base_idx,
            mask_row_stride,
            q_extend_offs,
            USE_CUSTOM_MASK,
            SKIP_PREFIX_CUSTOM_MASK,
            ENABLE_PREFIX_UNMASKED,
            BLOCK_M,
            BLOCK_N,
            mma_layout,
            mma_offs_n_col,
        )

        cdna4_async.wait_group(STREAMS * (NUM_STAGES - tail_i) - STREAMS)
        v_dot_tail = cdna4_async.load_shared_relaxed(
            v_smem.index(stage_idx), v_dot_layout
        )
        p_cast = p.to(v_dot_tail.dtype)
        p_dot_tail = gl.convert_layout(p_cast, p_dot_layout)
        acc = do_mma(p_dot_tail, v_dot_tail, acc)

    # Masked tail: one partial block covering [aligned_prefix_len, seq_len_prefix).
    # Runs *after* the pingpong flush drained every async op, so smem slot 0
    # is free to reuse via the unpipelined helper. Branch is False on the
    # persistent path (seq_len_prefix is BLOCK_N-aligned there).
    if seq_len_prefix > aligned_prefix_len:
        acc, l_i, m_i = attn_fwd_inner_prefix_unpipelined(
            acc,
            l_i,
            m_i,
            q_dot,
            K_Buffer,
            V_Buffer,
            kv_indices,
            kv_start,
            cur_kv_head,
            seq_len_prefix,
            stride_buf_kbs,
            stride_buf_kh,
            stride_buf_vbs,
            stride_buf_vh,
            kt_smem,
            v_smem,
            qk_scale,
            LOGIT_CAP,
            xai_temperature_reg,
            XAI_TEMPERATURE_LEN,
            q_abs_pos,
            SLIDING_WINDOW_SIZE,
            Mask,
            mask_base_idx,
            mask_row_stride,
            q_extend_offs,
            USE_CUSTOM_MASK,
            SKIP_PREFIX_CUSTOM_MASK,
            ENABLE_PREFIX_UNMASKED,
            BLOCK_M,
            BLOCK_N,
            BLOCK_DMODEL,
            BLOCK_DV,
            kt_async_layout,
            v_async_layout,
            kt_dot_layout,
            p_dot_layout,
            v_dot_layout,
            mma_layout,
            mma_offs_n_col,
            block_start=n_prefix_blocks,
        )

    return acc, l_i, m_i


@gluon.jit
def attn_fwd_inner_prefix_unpipelined(
    acc,
    l_i,
    m_i,
    q_dot,  #
    K_Buffer,
    V_Buffer,
    kv_indices,  #
    kv_start,
    cur_kv_head,
    seq_len_prefix,  #
    stride_buf_kbs,
    stride_buf_kh,  #
    stride_buf_vbs,
    stride_buf_vh,  #
    kt_smem,
    v_smem,  #
    qk_scale,
    LOGIT_CAP: gl.constexpr,  #
    xai_temperature_reg,
    XAI_TEMPERATURE_LEN: gl.constexpr,  #
    q_abs_pos,
    SLIDING_WINDOW_SIZE: gl.constexpr,  #
    Mask,
    mask_base_idx,
    mask_row_stride,
    q_extend_offs,  #
    USE_CUSTOM_MASK: gl.constexpr,
    SKIP_PREFIX_CUSTOM_MASK: gl.constexpr,  #
    ENABLE_PREFIX_UNMASKED: gl.constexpr,  #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,  #
    BLOCK_DMODEL: gl.constexpr,
    BLOCK_DV: gl.constexpr,
    kt_async_layout: gl.constexpr,
    v_async_layout: gl.constexpr,  #
    kt_dot_layout: gl.constexpr,
    p_dot_layout: gl.constexpr,
    v_dot_layout: gl.constexpr,  #
    mma_layout: gl.constexpr,
    mma_offs_n_col: gl.constexpr,  #
    block_start=0,
):
    n_prefix_blocks = (seq_len_prefix + BLOCK_N - 1) // BLOCK_N
    k_prefix_base = K_Buffer + cur_kv_head * stride_buf_kh
    v_prefix_base = V_Buffer + cur_kv_head * stride_buf_vh

    for block_n in tl.range(block_start, n_prefix_blocks):
        start_n = block_n * BLOCK_N

        issue_async_load_k_prefix(
            kt_smem.index(0),
            k_prefix_base,
            kv_indices,
            kv_start,
            start_n,
            seq_len_prefix,
            stride_buf_kbs,
            BLOCK_N,
            BLOCK_DMODEL,
            kt_async_layout,
        )
        issue_async_load_v_prefix(
            v_smem.index(0),
            v_prefix_base,
            kv_indices,
            kv_start,
            start_n,
            seq_len_prefix,
            stride_buf_vbs,
            BLOCK_N,
            BLOCK_DV,
            v_async_layout,
        )

        cdna4_async.wait_group(1)
        kt_dot = cdna4_async.load_shared_relaxed(kt_smem.index(0), kt_dot_layout)

        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        qk = do_mma(q_dot, kt_dot, qk)

        acc, l_i, m_i, p = compute_softmax_prefix(
            acc,
            l_i,
            m_i,
            qk,
            start_n,
            seq_len_prefix,
            qk_scale,
            LOGIT_CAP,
            xai_temperature_reg,
            XAI_TEMPERATURE_LEN,
            q_abs_pos,
            SLIDING_WINDOW_SIZE,
            Mask,
            mask_base_idx,
            mask_row_stride,
            q_extend_offs,
            USE_CUSTOM_MASK,
            SKIP_PREFIX_CUSTOM_MASK,
            ENABLE_PREFIX_UNMASKED,
            BLOCK_M,
            BLOCK_N,
            mma_layout,
            mma_offs_n_col,
        )

        cdna4_async.wait_group(0)
        v_dot = cdna4_async.load_shared_relaxed(v_smem.index(0), v_dot_layout)
        p_cast = p.to(v_dot.dtype)
        p_dot = gl.convert_layout(p_cast, p_dot_layout)
        acc = do_mma(p_dot, v_dot, acc)

    return acc, l_i, m_i


@gluon.jit
def attn_fwd_inner_prefix_sw_pipeline_4w(
    acc,
    l_i,
    m_i,
    q_dot,  #
    K_Buffer,
    V_Buffer,
    kv_indices,  #
    kv_start,
    cur_kv_head,
    seq_len_prefix,  # full prefix length; helper floor-aligns to BLOCK_N for the pipelined bulk
    stride_buf_kbs,
    stride_buf_kh,  #
    stride_buf_vbs,
    stride_buf_vh,  #
    kt_smem,
    v_smem,  #
    qk_scale,
    LOGIT_CAP: gl.constexpr,  #
    xai_temperature_reg,
    XAI_TEMPERATURE_LEN: gl.constexpr,  #
    q_abs_pos,
    SLIDING_WINDOW_SIZE: gl.constexpr,  #
    Mask,
    mask_base_idx,
    mask_row_stride,
    q_extend_offs,  #
    USE_CUSTOM_MASK: gl.constexpr,
    SKIP_PREFIX_CUSTOM_MASK: gl.constexpr,  #
    ENABLE_PREFIX_UNMASKED: gl.constexpr,  #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,  #
    BLOCK_DMODEL: gl.constexpr,
    BLOCK_DV: gl.constexpr,
    NUM_STAGES: gl.constexpr,  #
    kt_async_layout: gl.constexpr,
    v_async_layout: gl.constexpr,  #
    kt_dot_layout: gl.constexpr,
    p_dot_layout: gl.constexpr,
    v_dot_layout: gl.constexpr,  #
    mma_layout: gl.constexpr,
    mma_offs_n_col: gl.constexpr,  #
):
    """4-warp sw-pipeline prefix: manually-scheduled async DMA loop (NS>=2, D>=128).

    Explicitly issues K_future / V_future loads and consumes them with
    ``cdna4_async.wait_group`` counters (no warp_pipeline_stage). Callers
    pass the full prefix length; the helper floor-aligns to ``BLOCK_N``
    for the pipelined bulk and runs one masked tail block if the full
    length isn't a multiple of ``BLOCK_N``.
    """
    STREAMS: gl.constexpr = 2

    # Pipelined bulk runs over whole BLOCK_N slices only; any unaligned
    # tail is handled after the sw-pipeline drains (runtime no-op on the
    # persistent path, since seq_len_prefix is BLOCK_N-aligned there).
    aligned_prefix_len = (seq_len_prefix // BLOCK_N) * BLOCK_N
    n_prefix_blocks = aligned_prefix_len // BLOCK_N
    k_prefix_base = K_Buffer + cur_kv_head * stride_buf_kh
    v_prefix_base = V_Buffer + cur_kv_head * stride_buf_vh
    kv_indices_base = kv_indices + kv_start

    kt_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=kt_async_layout)
    v_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=v_async_layout)
    kt_offs_n = gl.arange(0, BLOCK_N, layout=kt_offs_n_layout)
    v_offs_n = gl.arange(0, BLOCK_N, layout=v_offs_n_layout)

    WAIT_K: gl.constexpr = STREAMS * NUM_STAGES - (STREAMS - 1)
    WAIT_V: gl.constexpr = STREAMS * NUM_STAGES - STREAMS

    for stage in gl.static_range(NUM_STAGES):
        cdna4_async.wait_group(0)
        pf_init_n = stage * BLOCK_N
        n_idx_k = pf_init_n + kt_offs_n
        mask_n_k = n_idx_k < seq_len_prefix
        kv_locs_k = cdna4_buffer_load(
            kv_indices_base, n_idx_k.to(tl.int32), mask=mask_n_k, other=0
        ).to(tl.int32)
        n_idx_v = pf_init_n + v_offs_n
        mask_n_v = n_idx_v < seq_len_prefix
        kv_locs_v = cdna4_buffer_load(
            kv_indices_base, n_idx_v.to(tl.int32), mask=mask_n_v, other=0
        ).to(tl.int32)
        issue_dma_k_prefix_from_locs(
            kt_smem.index(stage),
            k_prefix_base,
            kv_locs_k,
            mask_n_k,
            stride_buf_kbs,
            BLOCK_N,
            BLOCK_DMODEL,
            kt_async_layout,
        )
        issue_dma_v_prefix_from_locs(
            v_smem.index(stage),
            v_prefix_base,
            kv_locs_v,
            mask_n_v,
            stride_buf_vbs,
            BLOCK_N,
            BLOCK_DV,
            v_async_layout,
        )

    cdna4_async.wait_group(0)
    pf_start_n = NUM_STAGES * BLOCK_N
    n_idx_k_pf = pf_start_n + kt_offs_n
    mask_k_pf = n_idx_k_pf < seq_len_prefix
    kv_locs_k_pf = cdna4_buffer_load(
        kv_indices_base, n_idx_k_pf.to(tl.int32), mask=mask_k_pf, other=0
    ).to(tl.int32)
    n_idx_v_pf = pf_start_n + v_offs_n
    mask_v_pf = n_idx_v_pf < seq_len_prefix
    kv_locs_v_pf = cdna4_buffer_load(
        kv_indices_base, n_idx_v_pf.to(tl.int32), mask=mask_v_pf, other=0
    ).to(tl.int32)

    cdna4_async.wait_group(WAIT_K)
    kt_dot = cdna4_async.load_shared_relaxed(kt_smem.index(0), kt_dot_layout)

    main_loop_end = n_prefix_blocks - NUM_STAGES
    for block_n in tl.range(0, main_loop_end):
        stage_idx = (block_n % NUM_STAGES).to(tl.int32)
        start_n = (block_n * BLOCK_N).to(tl.int32)

        nf_start_n = ((block_n + NUM_STAGES + 1) * BLOCK_N).to(tl.int32)
        n_idx_k_nf = nf_start_n + kt_offs_n
        mask_k_nf = n_idx_k_nf < seq_len_prefix
        kv_locs_k_nf = cdna4_buffer_load(
            kv_indices_base, n_idx_k_nf.to(tl.int32), mask=mask_k_nf, other=0
        ).to(tl.int32)
        n_idx_v_nf = nf_start_n + v_offs_n
        mask_v_nf = n_idx_v_nf < seq_len_prefix
        kv_locs_v_nf = cdna4_buffer_load(
            kv_indices_base, n_idx_v_nf.to(tl.int32), mask=mask_v_nf, other=0
        ).to(tl.int32)

        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        qk = do_mma(q_dot, kt_dot, qk)

        cdna4_async.wait_group(WAIT_V)
        v_dot = cdna4_async.load_shared_relaxed(v_smem.index(stage_idx), v_dot_layout)

        # NOTE: issue K_future AFTER v_dot read; writing back into kt_smem slot
        # after the load avoids the DMA race that was observable once the
        # ``_buffer_load_to_shared_cast_safe`` dtype check was fixed (d1a61f1).
        issue_dma_k_prefix_from_locs(
            kt_smem.index(stage_idx),
            k_prefix_base,
            kv_locs_k_pf,
            mask_k_pf,
            stride_buf_kbs,
            BLOCK_N,
            BLOCK_DMODEL,
            kt_async_layout,
        )

        acc, l_i, m_i, p = compute_softmax_prefix(
            acc,
            l_i,
            m_i,
            qk,
            start_n,
            seq_len_prefix,
            qk_scale,
            LOGIT_CAP,
            xai_temperature_reg,
            XAI_TEMPERATURE_LEN,
            q_abs_pos,
            SLIDING_WINDOW_SIZE,
            Mask,
            mask_base_idx,
            mask_row_stride,
            q_extend_offs,
            USE_CUSTOM_MASK,
            SKIP_PREFIX_CUSTOM_MASK,
            ENABLE_PREFIX_UNMASKED,
            BLOCK_M,
            BLOCK_N,
            mma_layout,
            mma_offs_n_col,
        )

        p_cast = p.to(v_dot.dtype)
        p_dot_reg = gl.convert_layout(p_cast, p_dot_layout)
        acc = do_mma(p_dot_reg, v_dot, acc)

        next_stage_idx = ((block_n + 1) % NUM_STAGES).to(tl.int32)
        cdna4_async.wait_group(WAIT_K)
        kt_dot = cdna4_async.load_shared_relaxed(
            kt_smem.index(next_stage_idx), kt_dot_layout
        )

        # V_future must be issued after the p*V MMA has consumed v_dot and
        # after the next kt_dot has been read.
        issue_dma_v_prefix_from_locs(
            v_smem.index(stage_idx),
            v_prefix_base,
            kv_locs_v_pf,
            mask_v_pf,
            stride_buf_vbs,
            BLOCK_N,
            BLOCK_DV,
            v_async_layout,
        )

        kv_locs_k_pf = kv_locs_k_nf
        kv_locs_v_pf = kv_locs_v_nf
        mask_k_pf = mask_k_nf
        mask_v_pf = mask_v_nf

    for tail_i in gl.static_range(NUM_STAGES):
        cdna4_async.wait_group(STREAMS * (NUM_STAGES - tail_i) - (STREAMS - 1))
        stage_idx = ((main_loop_end + tail_i) % NUM_STAGES).to(tl.int32)
        start_n = (main_loop_end + tail_i) * BLOCK_N

        kt_dot_tail = cdna4_async.load_shared_relaxed(
            kt_smem.index(stage_idx), kt_dot_layout
        )
        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        qk = do_mma(q_dot, kt_dot_tail, qk)

        acc, l_i, m_i, p = compute_softmax_prefix(
            acc,
            l_i,
            m_i,
            qk,
            start_n,
            seq_len_prefix,
            qk_scale,
            LOGIT_CAP,
            xai_temperature_reg,
            XAI_TEMPERATURE_LEN,
            q_abs_pos,
            SLIDING_WINDOW_SIZE,
            Mask,
            mask_base_idx,
            mask_row_stride,
            q_extend_offs,
            USE_CUSTOM_MASK,
            SKIP_PREFIX_CUSTOM_MASK,
            ENABLE_PREFIX_UNMASKED,
            BLOCK_M,
            BLOCK_N,
            mma_layout,
            mma_offs_n_col,
        )

        cdna4_async.wait_group(STREAMS * (NUM_STAGES - tail_i) - STREAMS)
        v_dot_tail = cdna4_async.load_shared_relaxed(
            v_smem.index(stage_idx), v_dot_layout
        )
        p_cast = p.to(v_dot_tail.dtype)
        p_dot_tail = gl.convert_layout(p_cast, p_dot_layout)
        acc = do_mma(p_dot_tail, v_dot_tail, acc)

    # Masked tail: one partial block covering [aligned_prefix_len, seq_len_prefix).
    # Runs after the sw-pipeline drain so smem slot 0 is free to reuse.
    if seq_len_prefix > aligned_prefix_len:
        acc, l_i, m_i = attn_fwd_inner_prefix_unpipelined(
            acc,
            l_i,
            m_i,
            q_dot,
            K_Buffer,
            V_Buffer,
            kv_indices,
            kv_start,
            cur_kv_head,
            seq_len_prefix,
            stride_buf_kbs,
            stride_buf_kh,
            stride_buf_vbs,
            stride_buf_vh,
            kt_smem,
            v_smem,
            qk_scale,
            LOGIT_CAP,
            xai_temperature_reg,
            XAI_TEMPERATURE_LEN,
            q_abs_pos,
            SLIDING_WINDOW_SIZE,
            Mask,
            mask_base_idx,
            mask_row_stride,
            q_extend_offs,
            USE_CUSTOM_MASK,
            SKIP_PREFIX_CUSTOM_MASK,
            ENABLE_PREFIX_UNMASKED,
            BLOCK_M,
            BLOCK_N,
            BLOCK_DMODEL,
            BLOCK_DV,
            kt_async_layout,
            v_async_layout,
            kt_dot_layout,
            p_dot_layout,
            v_dot_layout,
            mma_layout,
            mma_offs_n_col,
            block_start=n_prefix_blocks,
        )

    return acc, l_i, m_i


# ===-----------------------------------------------------------------------===#
# Extend inner loops
# ===-----------------------------------------------------------------------===#


@gluon.jit
def attn_fwd_inner_extend_sw_pipeline_4w(
    acc,
    l_i,
    m_i,
    q_dot,  #
    k_base,
    v_base,  #
    cur_block_m,
    seq_len_extend,  #
    stride_kbs,
    stride_vbs,  #
    block_start,
    block_end,  #
    kt_smem,
    v_smem,  #
    qk_scale,
    LOGIT_CAP: gl.constexpr,  #
    xai_temperature_reg,
    XAI_TEMPERATURE_LEN: gl.constexpr,  #
    SLIDING_WINDOW_SIZE: gl.constexpr,  #
    IS_CAUSAL: gl.constexpr,  #
    Mask,
    mask_base_idx,
    mask_row_stride,
    mask_kv_col_offset,  #
    USE_CUSTOM_MASK: gl.constexpr,  #
    MASK_STEPS: gl.constexpr,  #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,  #
    BLOCK_DMODEL: gl.constexpr,
    BLOCK_DV: gl.constexpr,
    NUM_STAGES: gl.constexpr,  #
    kt_async_layout: gl.constexpr,
    v_async_layout: gl.constexpr,  #
    kt_dot_layout: gl.constexpr,
    p_dot_layout: gl.constexpr,
    v_dot_layout: gl.constexpr,  #
    mma_layout: gl.constexpr,
    mma_offs_n_col: gl.constexpr,
    mma_offs_m_row: gl.constexpr,  #
    SKIP_BOUNDS_CHECK: gl.constexpr = False,
):
    STREAMS: gl.constexpr = 2
    cdna4_async.wait_group(0)

    for stage in gl.static_range(NUM_STAGES):
        pf_start_n = (block_start + stage) * BLOCK_N
        issue_async_load_k_extend(
            kt_smem.index(stage),
            k_base,
            pf_start_n,
            seq_len_extend,
            stride_kbs,
            BLOCK_N,
            BLOCK_DMODEL,
            kt_async_layout,
            SKIP_BOUNDS_CHECK,
        )
        issue_async_load_v_extend(
            v_smem.index(stage),
            v_base,
            pf_start_n,
            seq_len_extend,
            stride_vbs,
            BLOCK_N,
            BLOCK_DV,
            v_async_layout,
            SKIP_BOUNDS_CHECK,
        )

    WAIT_K: gl.constexpr = STREAMS * NUM_STAGES - 1
    WAIT_V: gl.constexpr = STREAMS * NUM_STAGES - 2
    cdna4_async.wait_group(WAIT_K)
    kt_dot = cdna4_async.load_shared_relaxed(kt_smem.index(0), kt_dot_layout)

    main_loop_end = block_end - NUM_STAGES
    for block_n in tl.range(block_start, main_loop_end, loop_unroll_factor=2):
        stage_idx = ((block_n - block_start) % NUM_STAGES).to(tl.int32)
        start_n = (block_n * BLOCK_N).to(tl.int32)
        future_start_n = ((block_n + NUM_STAGES) * BLOCK_N).to(tl.int32)

        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        qk = do_mma(q_dot, kt_dot, qk)

        cdna4_async.wait_group(WAIT_V)
        v_dot = cdna4_async.load_shared_relaxed(v_smem.index(stage_idx), v_dot_layout)
        issue_async_load_k_extend(
            kt_smem.index(stage_idx),
            k_base,
            future_start_n,
            seq_len_extend,
            stride_kbs,
            BLOCK_N,
            BLOCK_DMODEL,
            kt_async_layout,
            SKIP_BOUNDS_CHECK,
        )

        acc, l_i, m_i, p = compute_softmax_extend(
            acc,
            l_i,
            m_i,
            qk,
            start_n,
            cur_block_m,
            seq_len_extend,
            qk_scale,
            LOGIT_CAP,
            xai_temperature_reg,
            XAI_TEMPERATURE_LEN,
            SLIDING_WINDOW_SIZE,
            IS_CAUSAL,
            Mask,
            mask_base_idx,
            mask_row_stride,
            mask_kv_col_offset,
            USE_CUSTOM_MASK,
            MASK_STEPS,
            BLOCK_M,
            BLOCK_N,
            mma_layout,
            mma_offs_n_col,
            mma_offs_m_row,
        )
        p_cast = p.to(v_dot.dtype)
        p_dot_reg = gl.convert_layout(p_cast, p_dot_layout)
        acc = do_mma(p_dot_reg, v_dot, acc)

        cdna4_async.wait_group(WAIT_V)
        next_stage_idx = ((block_n + 1 - block_start) % NUM_STAGES).to(tl.int32)
        kt_dot = cdna4_async.load_shared_relaxed(
            kt_smem.index(next_stage_idx), kt_dot_layout
        )
        issue_async_load_v_extend(
            v_smem.index(stage_idx),
            v_base,
            future_start_n,
            seq_len_extend,
            stride_vbs,
            BLOCK_N,
            BLOCK_DV,
            v_async_layout,
            SKIP_BOUNDS_CHECK,
        )

    for tail_i in gl.static_range(NUM_STAGES):
        cdna4_async.wait_group(STREAMS * (NUM_STAGES - tail_i) - 1)
        stage_idx = ((main_loop_end + tail_i - block_start) % NUM_STAGES).to(tl.int32)
        start_n = (main_loop_end + tail_i) * BLOCK_N

        kt_dot_tail = cdna4_async.load_shared_relaxed(
            kt_smem.index(stage_idx), kt_dot_layout
        )
        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        qk = do_mma(q_dot, kt_dot_tail, qk)

        p, alpha, m_new = compute_softmax_extend_part0(
            m_i,
            qk,
            start_n,
            cur_block_m,
            seq_len_extend,
            qk_scale,
            LOGIT_CAP,
            xai_temperature_reg,
            XAI_TEMPERATURE_LEN,
            SLIDING_WINDOW_SIZE,
            IS_CAUSAL,
            Mask,
            mask_base_idx,
            mask_row_stride,
            mask_kv_col_offset,
            USE_CUSTOM_MASK,
            MASK_STEPS,
            BLOCK_M,
            BLOCK_N,
            mma_layout,
            mma_offs_n_col,
            mma_offs_m_row,
        )

        cdna4_async.wait_group(STREAMS * (NUM_STAGES - tail_i) - 2)
        v_dot_tail = cdna4_async.load_shared_relaxed(
            v_smem.index(stage_idx), v_dot_layout
        )
        acc, l_i, m_i = compute_softmax_extend_part1(acc, l_i, p, alpha, m_new)
        p_cast = p.to(v_dot_tail.dtype)
        p_dot_tail = gl.convert_layout(p_cast, p_dot_layout)
        acc = do_mma(p_dot_tail, v_dot_tail, acc)

    return acc, l_i, m_i


@gluon.jit
def attn_fwd_inner_extend_pingpong_8w(
    acc,
    l_i,
    m_i,
    q_dot,  #
    k_base,
    v_base,  #
    cur_block_m,
    seq_len_extend,  #
    stride_kbs,
    stride_vbs,  #
    block_start,
    block_end,  #
    kt_smem,
    v_smem,  #
    qk_scale,
    LOGIT_CAP: gl.constexpr,  #
    xai_temperature_reg,
    XAI_TEMPERATURE_LEN: gl.constexpr,  #
    SLIDING_WINDOW_SIZE: gl.constexpr,  #
    IS_CAUSAL: gl.constexpr,  #
    Mask,
    mask_base_idx,
    mask_row_stride,
    mask_kv_col_offset,  #
    USE_CUSTOM_MASK: gl.constexpr,  #
    MASK_STEPS: gl.constexpr,  #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,  #
    BLOCK_DMODEL: gl.constexpr,
    BLOCK_DV: gl.constexpr,
    NUM_STAGES: gl.constexpr,  #
    kt_async_layout: gl.constexpr,
    v_async_layout: gl.constexpr,  #
    kt_dot_layout: gl.constexpr,
    p_dot_layout: gl.constexpr,
    v_dot_layout: gl.constexpr,  #
    mma_layout: gl.constexpr,
    mma_offs_n_col: gl.constexpr,
    mma_offs_m_row: gl.constexpr,  #
    SKIP_BOUNDS_CHECK: gl.constexpr = False,
):
    STREAMS: gl.constexpr = 2
    # warp_pipeline_stage requires >=2 physical LDS buffers. With NS=1 the
    # stage_idx collapses to 0, and iter N's DMA write to smem[0] races
    # iter N+1's relaxed read from smem[0] because membarFilter skips
    # barriers between BufferLoadToLocalOp and syncedViaAsyncWait loads.
    # Callers wanting NS=1 must route to attn_fwd_inner_extend_unpipelined
    # or the 4-warp sw_pipeline / serial variants.
    tl.static_assert(
        NUM_STAGES >= 2,
        "attn_fwd_inner_extend_pingpong_8w requires NUM_STAGES>=2 for "
        "determinism (warp_pipeline_stage needs multiple LDS buffers).",
    )
    cdna4_async.wait_group(0)

    for stage in gl.static_range(NUM_STAGES):
        pf_start_n = (block_start + stage) * BLOCK_N
        issue_async_load_k_extend(
            kt_smem.index(stage),
            k_base,
            pf_start_n,
            seq_len_extend,
            stride_kbs,
            BLOCK_N,
            BLOCK_DMODEL,
            kt_async_layout,
            SKIP_BOUNDS_CHECK,
        )
        issue_async_load_v_extend(
            v_smem.index(stage),
            v_base,
            pf_start_n,
            seq_len_extend,
            stride_vbs,
            BLOCK_N,
            BLOCK_DV,
            v_async_layout,
            SKIP_BOUNDS_CHECK,
        )

    WAIT_INIT: gl.constexpr = STREAMS * NUM_STAGES - (STREAMS - 1)
    WAIT_LOOP: gl.constexpr = STREAMS * NUM_STAGES - STREAMS
    cdna4_async.wait_group(WAIT_INIT)
    kt_dot = cdna4_async.load_shared_relaxed(kt_smem.index(0), kt_dot_layout)

    main_loop_end = block_end - NUM_STAGES
    for block_n in tl.range(block_start, main_loop_end, loop_unroll_factor=2):

        with warp_pipeline_stage("dot1", priority=0):
            stage_idx = ((block_n - block_start) % NUM_STAGES).to(tl.int32)
            start_n = (block_n * BLOCK_N).to(tl.int32)
            future_start_n = ((block_n + NUM_STAGES) * BLOCK_N).to(tl.int32)
            qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
            qk = do_mma(q_dot, kt_dot, qk)
            p, alpha, m_new = compute_softmax_extend_part0(
                m_i,
                qk,
                start_n,
                cur_block_m,
                seq_len_extend,
                qk_scale,
                LOGIT_CAP,
                xai_temperature_reg,
                XAI_TEMPERATURE_LEN,
                SLIDING_WINDOW_SIZE,
                IS_CAUSAL,
                Mask,
                mask_base_idx,
                mask_row_stride,
                mask_kv_col_offset,
                USE_CUSTOM_MASK,
                MASK_STEPS,
                BLOCK_M,
                BLOCK_N,
                mma_layout,
                mma_offs_n_col,
                mma_offs_m_row,
            )

        cdna4_async.wait_group(WAIT_LOOP)

        with warp_pipeline_stage("mem1", priority=1):
            v_dot = cdna4_async.load_shared_relaxed(
                v_smem.index(stage_idx), v_dot_layout
            )
            issue_async_load_k_extend(
                kt_smem.index(stage_idx),
                k_base,
                future_start_n,
                seq_len_extend,
                stride_kbs,
                BLOCK_N,
                BLOCK_DMODEL,
                kt_async_layout,
                SKIP_BOUNDS_CHECK,
            )

        with warp_pipeline_stage("dot2a", priority=0):
            acc, l_i, m_i = compute_softmax_extend_part1(
                acc, l_i, p, alpha, m_new
            )

        with warp_pipeline_stage("dot2b", priority=0):
            p_cast = p.to(v_dot.dtype)
            p_dot_reg = gl.convert_layout(p_cast, p_dot_layout)
            acc = do_mma(p_dot_reg, v_dot, acc)

        cdna4_async.wait_group(WAIT_LOOP)

        with warp_pipeline_stage("mem2", priority=1):
            next_stage_idx = ((block_n + 1 - block_start) % NUM_STAGES).to(tl.int32)
            kt_dot = cdna4_async.load_shared_relaxed(
                kt_smem.index(next_stage_idx), kt_dot_layout
            )
            issue_async_load_v_extend(
                v_smem.index(stage_idx),
                v_base,
                future_start_n,
                seq_len_extend,
                stride_vbs,
                BLOCK_N,
                BLOCK_DV,
                v_async_layout,
                SKIP_BOUNDS_CHECK,
            )

    for tail_i in gl.static_range(NUM_STAGES):
        cdna4_async.wait_group(STREAMS * (NUM_STAGES - tail_i) - (STREAMS - 1))
        stage_idx = ((main_loop_end + tail_i - block_start) % NUM_STAGES).to(tl.int32)
        start_n = (main_loop_end + tail_i) * BLOCK_N

        kt_dot_tail = cdna4_async.load_shared_relaxed(
            kt_smem.index(stage_idx), kt_dot_layout
        )
        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        qk = do_mma(q_dot, kt_dot_tail, qk)

        p, alpha, m_new = compute_softmax_extend_part0(
            m_i,
            qk,
            start_n,
            cur_block_m,
            seq_len_extend,
            qk_scale,
            LOGIT_CAP,
            xai_temperature_reg,
            XAI_TEMPERATURE_LEN,
            SLIDING_WINDOW_SIZE,
            IS_CAUSAL,
            Mask,
            mask_base_idx,
            mask_row_stride,
            mask_kv_col_offset,
            USE_CUSTOM_MASK,
            MASK_STEPS,
            BLOCK_M,
            BLOCK_N,
            mma_layout,
            mma_offs_n_col,
            mma_offs_m_row,
        )

        cdna4_async.wait_group(STREAMS * (NUM_STAGES - tail_i) - STREAMS)
        v_dot_tail = cdna4_async.load_shared_relaxed(
            v_smem.index(stage_idx), v_dot_layout
        )
        acc, l_i, m_i = compute_softmax_extend_part1(acc, l_i, p, alpha, m_new)
        p_cast = p.to(v_dot_tail.dtype)
        p_dot_tail = gl.convert_layout(p_cast, p_dot_layout)
        acc = do_mma(p_dot_tail, v_dot_tail, acc)

    return acc, l_i, m_i


# ===-----------------------------------------------------------------------===#
# Serial inner loops (4-warp path)
# ===-----------------------------------------------------------------------===#


@gluon.jit
def attn_fwd_inner_prefix_serial_4w(
    acc,
    l_i,
    m_i,
    q_dot,  #
    K_Buffer,
    V_Buffer,
    kv_indices,  #
    kv_start,
    kv_head,
    seq_len_prefix,  #
    stride_buf_kbs,
    stride_buf_kh,  #
    stride_buf_vbs,
    stride_buf_vh,  #
    kt_serial_smem,
    v_serial_smem,  #
    qk_scale,
    LOGIT_CAP: gl.constexpr,  #
    xai_temperature_reg,
    XAI_TEMPERATURE_LEN: gl.constexpr,  #
    q_abs_pos,
    SLIDING_WINDOW_SIZE: gl.constexpr,  #
    Mask,
    mask_base_idx,
    mask_row_stride,
    q_extend_offs,  #
    USE_CUSTOM_MASK: gl.constexpr,
    SKIP_PREFIX_CUSTOM_MASK: gl.constexpr,  #
    ENABLE_PREFIX_UNMASKED: gl.constexpr,  #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,  #
    BLOCK_DMODEL: gl.constexpr,
    BLOCK_DV: gl.constexpr,
    kt_blocked_layout: gl.constexpr,
    blocked_layout: gl.constexpr,  #
    kt_dot_layout: gl.constexpr,
    p_dot_layout: gl.constexpr,
    v_dot_layout: gl.constexpr,  #
    mma_layout: gl.constexpr,
    mma_offs_n_col: gl.constexpr,  #
):
    kt_offs_d = gl.arange(
        0, BLOCK_DMODEL, layout=gl.SliceLayout(dim=1, parent=kt_blocked_layout)
    )
    kt_offs_n = gl.arange(
        0, BLOCK_N, layout=gl.SliceLayout(dim=0, parent=kt_blocked_layout)
    )
    v_offs_n = gl.arange(
        0, BLOCK_N, layout=gl.SliceLayout(dim=1, parent=blocked_layout)
    )
    v_offs_d = gl.arange(0, BLOCK_DV, layout=gl.SliceLayout(dim=0, parent=blocked_layout))

    k_prefix_base = K_Buffer + kv_head * stride_buf_kh
    v_prefix_base = V_Buffer + kv_head * stride_buf_vh
    n_prefix_blocks = (seq_len_prefix + BLOCK_N - 1) // BLOCK_N

    for block_n in tl.range(0, n_prefix_blocks):
        start_n = block_n * BLOCK_N

        n_idx_k = start_n + kt_offs_n
        mask_n_k = n_idx_k < seq_len_prefix
        kv_locs_k = gl.load(kv_indices + kv_start + n_idx_k, mask=mask_n_k, other=0).to(
            tl.int32
        )

        kt_ptrs = (
            k_prefix_base + kt_offs_d[:, None] + kv_locs_k[None, :] * stride_buf_kbs
        )
        kt_mask = mask_n_k[None, :]
        kt_global = gl.load(kt_ptrs, mask=kt_mask, other=0.0)
        kt_serial_smem.store(kt_global)
        kt_dot = kt_serial_smem.load(kt_dot_layout)

        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        qk = do_mma(q_dot, kt_dot, qk)

        acc, l_i, m_i, p = compute_softmax_prefix(
            acc,
            l_i,
            m_i,
            qk,
            start_n,
            seq_len_prefix,
            qk_scale,
            LOGIT_CAP,
            xai_temperature_reg,
            XAI_TEMPERATURE_LEN,
            q_abs_pos,
            SLIDING_WINDOW_SIZE,
            Mask,
            mask_base_idx,
            mask_row_stride,
            q_extend_offs,
            USE_CUSTOM_MASK,
            SKIP_PREFIX_CUSTOM_MASK,
            ENABLE_PREFIX_UNMASKED,
            BLOCK_M,
            BLOCK_N,
            mma_layout,
            mma_offs_n_col,
        )

        n_idx_v = start_n + v_offs_n
        mask_n_v = n_idx_v < seq_len_prefix
        kv_locs_v = gl.load(
            kv_indices + kv_start + n_idx_v, mask=mask_n_v, other=0
        ).to(tl.int32)
        v_ptrs = (
            v_prefix_base + kv_locs_v[:, None] * stride_buf_vbs + v_offs_d[None, :]
        )
        v_mask = mask_n_v[:, None]
        v_global = gl.load(v_ptrs, mask=v_mask, other=0.0)
        v_serial_smem.store(v_global)
        v_dot = v_serial_smem.load(v_dot_layout)

        p_cast = p.to(v_dot.dtype)
        p_dot = gl.convert_layout(p_cast, p_dot_layout)
        acc = do_mma(p_dot, v_dot, acc)

    return acc, l_i, m_i


@gluon.jit
def attn_fwd_inner_extend_serial_4w(
    acc,
    l_i,
    m_i,
    q_dot,  #
    k_extend_base,
    v_extend_base,  #
    cur_block_m,
    seq_len_extend,  #
    stride_kbs,
    stride_vbs,  #
    block_start,
    block_end,  #
    kt_serial_smem,
    v_serial_smem,  #
    qk_scale,
    LOGIT_CAP: gl.constexpr,  #
    xai_temperature_reg,
    XAI_TEMPERATURE_LEN: gl.constexpr,  #
    SLIDING_WINDOW_SIZE: gl.constexpr,  #
    IS_CAUSAL: gl.constexpr,  #
    Mask,
    mask_base_idx,
    mask_row_stride,
    mask_kv_col_offset,  #
    USE_CUSTOM_MASK: gl.constexpr,  #
    MASK_STEPS: gl.constexpr,  #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,  #
    BLOCK_DMODEL: gl.constexpr,
    BLOCK_DV: gl.constexpr,
    kt_blocked_layout: gl.constexpr,
    blocked_layout: gl.constexpr,  #
    kt_dot_layout: gl.constexpr,
    p_dot_layout: gl.constexpr,
    v_dot_layout: gl.constexpr,  #
    mma_layout: gl.constexpr,
    mma_offs_n_col: gl.constexpr,
    mma_offs_m_row: gl.constexpr,  #
):
    kt_offs_d = gl.arange(
        0, BLOCK_DMODEL, layout=gl.SliceLayout(dim=1, parent=kt_blocked_layout)
    )
    kt_offs_n = gl.arange(
        0, BLOCK_N, layout=gl.SliceLayout(dim=0, parent=kt_blocked_layout)
    )
    v_offs_n = gl.arange(
        0, BLOCK_N, layout=gl.SliceLayout(dim=1, parent=blocked_layout)
    )
    v_offs_d = gl.arange(0, BLOCK_DV, layout=gl.SliceLayout(dim=0, parent=blocked_layout))

    for block_n in tl.range(block_start, block_end):
        start_n = block_n * BLOCK_N

        kt_ptrs = (
            k_extend_base
            + kt_offs_d[:, None]
            + (start_n + kt_offs_n[None, :]) * stride_kbs
        )
        kt_mask = (start_n + kt_offs_n[None, :]) < seq_len_extend
        kt_global = gl.load(kt_ptrs, mask=kt_mask, other=0.0)
        kt_serial_smem.store(kt_global)
        kt_dot = kt_serial_smem.load(kt_dot_layout)

        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        qk = do_mma(q_dot, kt_dot, qk)

        acc, l_i, m_i, p = compute_softmax_extend(
            acc,
            l_i,
            m_i,
            qk,
            start_n,
            cur_block_m,
            seq_len_extend,
            qk_scale,
            LOGIT_CAP,
            xai_temperature_reg,
            XAI_TEMPERATURE_LEN,
            SLIDING_WINDOW_SIZE,
            IS_CAUSAL,
            Mask,
            mask_base_idx,
            mask_row_stride,
            mask_kv_col_offset,
            USE_CUSTOM_MASK,
            MASK_STEPS,
            BLOCK_M,
            BLOCK_N,
            mma_layout,
            mma_offs_n_col,
            mma_offs_m_row,
        )

        v_ptrs = (
            v_extend_base
            + (start_n + v_offs_n[:, None]) * stride_vbs
            + v_offs_d[None, :]
        )
        v_mask = (start_n + v_offs_n[:, None]) < seq_len_extend
        v_global = gl.load(v_ptrs, mask=v_mask, other=0.0)
        v_serial_smem.store(v_global)
        v_dot = v_serial_smem.load(v_dot_layout)

        p_cast = p.to(v_dot.dtype)
        p_dot = gl.convert_layout(p_cast, p_dot_layout)
        acc = do_mma(p_dot, v_dot, acc)

    return acc, l_i, m_i


# ===-----------------------------------------------------------------------===#
# Extend short path (preload-all, defined in this file)
# ===-----------------------------------------------------------------------===#


@gluon.jit
def attn_fwd_inner_extend_unpipelined(
    acc,
    l_i,
    m_i,
    q_dot,  #
    k_base,
    v_base,  #
    cur_block_m,
    seq_len_extend,  #
    stride_kbs,
    stride_vbs,  #
    block_start,
    block_end,  #
    kt_smem,
    v_smem,  #
    qk_scale,
    LOGIT_CAP: gl.constexpr,  #
    xai_temperature_reg,
    XAI_TEMPERATURE_LEN: gl.constexpr,  #
    SLIDING_WINDOW_SIZE: gl.constexpr,  #
    IS_CAUSAL: gl.constexpr,  #
    Mask,
    mask_base_idx,
    mask_row_stride,
    mask_kv_col_offset,  #
    USE_CUSTOM_MASK: gl.constexpr,  #
    MASK_STEPS: gl.constexpr,  #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,  #
    BLOCK_DMODEL: gl.constexpr,
    BLOCK_DV: gl.constexpr,
    kt_async_layout: gl.constexpr,
    v_async_layout: gl.constexpr,  #
    kt_dot_layout: gl.constexpr,
    p_dot_layout: gl.constexpr,
    v_dot_layout: gl.constexpr,  #
    mma_layout: gl.constexpr,
    mma_offs_n_col: gl.constexpr,
    mma_offs_m_row: gl.constexpr,  #
    SKIP_BOUNDS_CHECK: gl.constexpr = False,  #
):
    """Preload all K/V blocks into distinct SMEM slots, then compute.

    Requires (block_end - block_start) <= NUM_STAGES so each block gets its own buffer.
    Phase 1: issue K[i]->smem[i] and V[i]->smem[i] for all i.
    Phase 2: wait_group(0) -- everything fully drained.
    Phase 3: compute QK.softmax.PV block-by-block from resident SMEM.
    """
    cdna4_async.wait_group(0)

    # Phase 1: bulk-issue all DMAs
    n_local_blocks = block_end - block_start
    for local_idx in tl.range(0, n_local_blocks):
        buf_idx = local_idx.to(gl.int32)
        start_n = (block_start + local_idx) * BLOCK_N
        issue_async_load_k_extend(
            kt_smem.index(buf_idx),
            k_base,
            start_n,
            seq_len_extend,
            stride_kbs,
            BLOCK_N,
            BLOCK_DMODEL,
            kt_async_layout,
            SKIP_BOUNDS_CHECK,
        )
        issue_async_load_v_extend(
            v_smem.index(buf_idx),
            v_base,
            start_n,
            seq_len_extend,
            stride_vbs,
            BLOCK_N,
            BLOCK_DV,
            v_async_layout,
            SKIP_BOUNDS_CHECK,
        )

    # Phase 2: drain every outstanding DMA
    cdna4_async.wait_group(0)

    # Phase 3: compute from fully-resident SMEM (no DMA during this loop)
    for local_idx in tl.range(0, n_local_blocks):
        buf_idx = local_idx.to(gl.int32)
        start_n = (block_start + local_idx) * BLOCK_N

        kt_dot = cdna4_async.load_shared_relaxed(
            kt_smem.index(buf_idx), kt_dot_layout
        )

        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        qk = do_mma(q_dot, kt_dot, qk)

        acc, l_i, m_i, p = compute_softmax_extend(
            acc,
            l_i,
            m_i,
            qk,
            start_n,
            cur_block_m,
            seq_len_extend,
            qk_scale,
            LOGIT_CAP,
            xai_temperature_reg,
            XAI_TEMPERATURE_LEN,
            SLIDING_WINDOW_SIZE,
            IS_CAUSAL,
            Mask,
            mask_base_idx,
            mask_row_stride,
            mask_kv_col_offset,
            USE_CUSTOM_MASK,
            MASK_STEPS,
            BLOCK_M,
            BLOCK_N,
            mma_layout,
            mma_offs_n_col,
            mma_offs_m_row,
        )

        v_dot = cdna4_async.load_shared_relaxed(v_smem.index(buf_idx), v_dot_layout)
        p_cast = p.to(v_dot.dtype)
        p_dot = gl.convert_layout(p_cast, p_dot_layout)
        acc = do_mma(p_dot, v_dot, acc)

    return acc, l_i, m_i
