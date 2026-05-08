# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon extend-attention aggregates for gfx950 (MI350X / CDNA4).

The runtime and layout state for the three extend-attention schedules
(serial / 4w software pipeline / 8w pingpong) lives in the aggregates
below, following the upstream mxfp gemm/fa "aggregate-as-program" style:

    * mxfp_gemm_gfx1250.MXFPGEMMPipelinedProgram.pipeline(K)
    * mxfp_fa_gfx1250.GlobalScaledAttentionProgram.compute(...)

The three per-kernel ``@gluon.jit`` functions in
``extend_attention_gfx950.py`` are thin wrappers that build
``ExtendAttnProgram`` and pass it to one of:

    ExtendAttnSerialProgram.run()
    ExtendAttnSwPipeline4WProgram.run()
    ExtendAttnPingpong8WProgram.run()

Aggregates defined here:

    ExtendAttentionLayouts -- dtype-aware high-level layout facade.
    ExtendAttnConfig       -- bundles layouts + every compile-time shape
                              / mode constexpr so state and schedule
                              programs share one config object
                              (analogous to ``MXFPGEMMConfig``).
    KVSmemBank             -- (kt_smem, v_smem) plus async layouts for
                              prefix and extend K/V staging.
    AsyncKVLoader          -- one K or V async-DMA tap bundling the
                              multi-slot smem with its global ptr /
                              stride / layout. Covers prefix gather and
                              row-major extend loads.
    ExtendAttnProgram        -- shared runtime state. ``@composition`` over
                              ``ExtendAttnConfig``; owns tensors, masks,
                              loaders, softmax helpers, prefix/extend
                              loops, and epilogue helpers.
    ExtendAttnSerialProgram,
    ExtendAttnSwPipeline4WProgram,
    ExtendAttnPingpong8WProgram
                            -- schedule-specific entry bodies over the
                              shared state.

Gluon conventions used here:

    * Aggregate fields that aren't always present (Mask, MaskIndptr,
      WindowKvOffsets, Sinks, split-K workspace) are stored as
      ``gl.tensor | gl.constexpr`` and the outer kernel passes
      ``gl.constexpr(0)`` placeholders when the feature is off.
    * ``@composition`` forwards attribute lookups to aggregate-typed
      fields so ``self.BLOCK_M`` resolves via ``self.cfg.BLOCK_M``.
    * ``@gluon.constexpr_function __init__`` is the canonical aggregate
      constructor. ``@gluon.jit initialize(...)`` is the runtime
      factory that does program_id math / strides / returns the
      aggregate.
    * Layout factory methods intentionally stay on ``ExtendAttentionLayouts``.
      The class is long, but keeping the helpers in one namespace makes the
      gfx950-specific K^T/V symmetry and FP8 exceptions easier to audit than a
      set of forwarding-only policy aggregates.
"""

# ruff: noqa

import inspect

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language._layouts import (
    DistributedLinearLayout,
    DotOperandLayout,
    PaddedSharedLayout,
)
from triton.experimental.gluon.language.amd import AMDMFMALayout, warp_pipeline_stage
from triton.experimental.gluon.language.amd.cdna4 import async_copy as cdna4_async
from triton.experimental.gluon.language.amd.cdna4 import (
    buffer_load as cdna4_buffer_load,
)
from triton.experimental.gluon.language.amd.cdna4 import (
    buffer_store as cdna4_buffer_store,
)
from triton.experimental.gluon.language.amd.cdna4 import mfma as mfma_cdna4
from triton.language.core import _aggregate as aggregate

# ===---------------------------------------------------------------------===#
# Module-level constants
# ===---------------------------------------------------------------------===#

LOG2E = tl.constexpr(1.4426950408889634)

# ``vec`` must match the dot-operand ``kWidth`` of the consumer (the
# ``ttg.local_load`` that feeds the MFMA). A mismatched vec lowers
# ds_reads at the smaller width (extra instructions) or triggers LDS
# bank conflicts. The MFMA operand widths are kWidth=8 for K^T and
# kWidth=4 for V on PV. Defined at module scope because
# ``@aggregate`` strips non-annotated class bindings.
SERIAL_KT_SMEM = gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=16, order=[0, 1])
SERIAL_V_SMEM = gl.SwizzledSharedLayout(vec=4, per_phase=1, max_phase=16, order=[1, 0])


@gluon.constexpr_function
def _wrap_rt_scalar(x):
    """Re-wrap Python scalars that Triton specialized before aggregate init."""
    if isinstance(x, (int, float)):
        return gl.constexpr(x)
    return x


# The PaddedSharedLayout constructor spells the ``block_bases`` kwarg as
# ``cga_layout`` on older Triton builds. Detect once at import time.
try:
    _PADDED_SHARED_LAYOUT_BLOCK_ARG = (
        "block_bases"
        if "block_bases" in inspect.signature(PaddedSharedLayout).parameters
        else "cga_layout"
    )
except (TypeError, ValueError):
    _PADDED_SHARED_LAYOUT_BLOCK_ARG = "block_bases"


# ===---------------------------------------------------------------------===#
# Leaf primitives (stateless @gluon.jit helpers)
# ===---------------------------------------------------------------------===#
#
# These four primitives are pure tensor ops with no state to bundle and no
# layouts to bind. ``nan_propagating_max`` passes ``_nan_propagating_maximum``
# to ``gl.reduce`` as the combinator, which requires a top-level
# ``@gluon.jit`` function reference -- so both must live at module scope.


@gluon.jit
def _nan_propagating_maximum(a, b):
    # Elementwise binary-op used as the combinator for ``nan_propagating_max``
    # below. Named ``maximum`` (not ``max``) to match the triton/numpy
    # convention: ``maximum`` is the elementwise binary-op, ``max`` is the
    # reduction.
    return gl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@gluon.jit
def nan_propagating_max(x, axis):
    # Reduction along ``axis`` using NaN-propagating maximum as the combinator.
    return gl.reduce(x, axis, _nan_propagating_maximum)


@gluon.jit
def do_mma(a, b, c):
    if b.dtype == tl.float8e4b8 or b.dtype == tl.float8e4nv:
        a_fp8 = tl.cast(
            a, tl.float8e4nv, bitcast=(a.dtype != tl.bfloat16 and a.dtype != tl.float16)
        )
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
            cdna4_async.buffer_load_to_shared(dst_smem, src_base, offsets)
    else:
        if mask is not None:
            reg = cdna4_buffer_load(src_base, offsets, mask=mask, other=other)
        else:
            reg = cdna4_buffer_load(src_base, offsets)
        dst_smem.store(reg.to(dst_smem.dtype))
    cdna4_async.commit_group()


# ===---------------------------------------------------------------------===#
# XCD-aware program_id remapping (MI350X: 8 XCDs x 32 CUs x 4 MB L2)
# ===---------------------------------------------------------------------===#
#
# Port of the gemm/f16_gemm_common_gfx950.py::get_pid_m_n swizzle, lifted to
# operate on a 1-D tile_idx (attention has no M/N notion; tiles are already
# linear). Upstream analogue: triton-upstream-main/third_party/amd/python/
# examples/gluon/f16_gemm_common_gfx1250.py::chiplet_transform_chunked.
#
# Contract:
#   * For pid < full = (domain_size // (NUM_XCDS * XCD_CHUNK)) * (NUM_XCDS * XCD_CHUNK)
#     the remapped id guarantees that pids sharing the same (pid % NUM_XCDS)
#     (i.e. the hardware XCD they'll land on under HIP's wg_id%NUM_XCDS
#     scheduler default) receive a *contiguous* range of XCD_CHUNK tile ids.
#   * For pid >= full (the ragged tail) we pass through unchanged; the
#     number of tail tiles is < NUM_XCDS * XCD_CHUNK so locality loss is
#     bounded and the formula stays branch-free.
#   * XCD_CHUNK == 1 is the identity (no-op) -- serves as the A/B "off" toggle.
#
# Caller passes the rectangular launch domain size that covers every valid pid
# (for WCA: total_valid_tiles; for data-centric: B * H * max_NM).


@gluon.jit
def remap_xcd_chunked(
    pid,
    domain_size,
    NUM_XCDS: gl.constexpr,
    XCD_CHUNK: gl.constexpr,
):
    """XCD-aware chunked swizzle for a 1-D program_id.

    See module comment above. When ``XCD_CHUNK == 1`` this is the identity
    mapping (guaranteed to compile away via the constexpr branch).
    """
    if XCD_CHUNK == 1:
        return pid

    full = (domain_size // (NUM_XCDS * XCD_CHUNK)) * (NUM_XCDS * XCD_CHUNK)

    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    chunk_id = local_pid // XCD_CHUNK
    pos = local_pid - chunk_id * XCD_CHUNK
    new_pid = chunk_id * (NUM_XCDS * XCD_CHUNK) + xcd * XCD_CHUNK + pos

    return gl.where(pid < full, new_pid, pid)


@gluon.jit
def remap_xcd_attention(
    pid,
    domain_size,
    NUM_XCDS: gl.constexpr,
):
    """SGLang decode-attention style contiguous-per-XCD swizzle."""
    pids_per_xcd = (domain_size + NUM_XCDS - 1) // NUM_XCDS
    tall_xcds = domain_size % NUM_XCDS
    tall_xcds = tl.where(tall_xcds == 0, NUM_XCDS, tall_xcds)
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    tall_pid = xcd * pids_per_xcd + local_pid
    short_pid = (
        tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid
    )
    return tl.where(xcd < tall_xcds, tall_pid, short_pid)


# ===---------------------------------------------------------------------===#
# @composition decorator
# ===---------------------------------------------------------------------===#
#
# Lets an aggregate forward attribute lookups to its aggregate members so
# ``self.BLOCK_M`` resolves through ``self.cfg.BLOCK_M``. Matches the
# upstream ``gfx1250_utils.composition`` helper inlined here so this file
# has no internal mxfp dependency.


def composition(cls):
    """Let an aggregate forward attribute reads to its aggregate members."""

    def __getattr__(self, name):
        if name in self.__dict__:
            return object.__getattribute__(self, name)
        for member in self.__dict__.values():
            if getattr(member, "__triton_aggregate__", False) and hasattr(member, name):
                return getattr(member, name)
        raise AttributeError(f"{type(self).__name__} object has no attribute '{name}'")

    cls.__getattr__ = __getattr__
    return cls


# ===---------------------------------------------------------------------===#
# ExtendAttentionLayouts -- dtype-aware layout bundle (BF16 / FP8 KV)
# ===---------------------------------------------------------------------===#
#
# Bundles every layout the unified extend kernel consumes behind one
# IS_FP8 gate, so the kernel body calls ``layouts.pfx_kt_dot_layout``
# (FP8 or BF16) instead of picking between two local constexprs. Every
# layout factory lives here too as a static ``@gluon.constexpr_function``
# method (upstream ``KVMemory.preshuffle`` pattern). Inner calls resolve
# via ``ExtendAttentionLayouts.<name>(...)`` -- the class name is a
# module global by the time any method body runs.
#
# For symmetric attention, V is the transpose of K^T: every basis
# ``[a, b]`` is mirrored to ``[b, a]`` and the tile shape swaps
# ``[D, N]`` <-> ``[N, D]``. The ``_kt_*_bases`` helpers are the single
# source of truth; ``make_v_*`` helpers transpose them. The FP8 kernel's
# BF16-extend V layout diverges in three ``(num_warps, BLOCK_DV,
# BLOCK_N)`` corners to relieve LDS pressure -- ``make_fp8_bf16_v_*``
# (and its aliases ``make_fp8_v_*`` / ``make_fp8_extend_v_*``) handle
# that.
#


@aggregate
class ExtendAttentionLayouts:
    """Dot-operand + MMA + async-DMA layouts for BF16 / FP8 extend attention.

    Extend phase is always BF16 (the FP8 kernel runs BF16 MFMA on the
    live extend tensors), so ``kt_dot_layout`` / ``p_dot_layout`` /
    ``v_dot_layout`` always hold the BF16-k_width flavor. The ``pfx_*``
    counterparts flip to FP8-k_width when ``IS_FP8`` is True.

    The static ``make_*`` / ``prefix_*`` factories are grouped by section:
    dot layouts, padded smem, BF16 async tiles, FP8 prefix tiles, rare FP8
    extend overrides, and IS_FP8-aware prefix policy. Call sites use this
    facade directly, with local comments for the policy boundary being built.
    """

    # ==== Shared MFMA accumulator ====
    mma_layout: gl.constexpr

    # ==== Extend-phase dot operands (always BF16) ====
    q_dot_layout: gl.constexpr
    kt_dot_layout: gl.constexpr
    p_dot_layout: gl.constexpr
    v_dot_layout: gl.constexpr

    # ==== Prefix-phase dot operands (FP8 when IS_FP8, else mirrors extend) ====
    pfx_q_dot_layout: gl.constexpr
    pfx_kt_dot_layout: gl.constexpr
    pfx_p_dot_layout: gl.constexpr
    pfx_v_dot_layout: gl.constexpr

    # ==== Shared blocked / slice layouts for output + row/col helpers ====
    blocked_layout: gl.constexpr
    offs_m_layout: gl.constexpr
    offs_d_layout: gl.constexpr
    mma_offs_n_col: gl.constexpr
    mma_offs_m_row: gl.constexpr
    mma_m_layout: gl.constexpr

    # ==== Policy ====
    IS_FP8: gl.constexpr
    PFX_SMEM_TY: gl.constexpr  # element type for prefix smem allocation
    ASYNC_PAD_K: gl.constexpr
    ASYNC_PAD_V: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, IS_FP8, num_warps, q_dtype, k_buffer_dtype):
        mma, q_dot, kt_dot, p_dot, v_dot = ExtendAttentionLayouts.make_mfma_dot_layouts(
            num_warps, 16, 16, 32, 8, 4
        )
        blocked, offs_m, offs_d, mma_n_col, mma_m_row, mma_m_ly = (
            ExtendAttentionLayouts.make_blocked_and_slice_layouts(num_warps, mma)
        )

        self.mma_layout = gl.constexpr(mma)
        self.q_dot_layout = gl.constexpr(q_dot)
        self.kt_dot_layout = gl.constexpr(kt_dot)
        self.p_dot_layout = gl.constexpr(p_dot)
        self.v_dot_layout = gl.constexpr(v_dot)
        self.blocked_layout = gl.constexpr(blocked)
        self.offs_m_layout = gl.constexpr(offs_m)
        self.offs_d_layout = gl.constexpr(offs_d)
        self.mma_offs_n_col = gl.constexpr(mma_n_col)
        self.mma_offs_m_row = gl.constexpr(mma_m_row)
        self.mma_m_layout = gl.constexpr(mma_m_ly)

        if IS_FP8:
            fp8_q, fp8_kt, fp8_p, fp8_v = ExtendAttentionLayouts.make_fp8_dot_layouts(
                mma, 16, 8
            )
            self.pfx_q_dot_layout = gl.constexpr(fp8_q)
            self.pfx_kt_dot_layout = gl.constexpr(fp8_kt)
            self.pfx_p_dot_layout = gl.constexpr(fp8_p)
            self.pfx_v_dot_layout = gl.constexpr(fp8_v)
            self.PFX_SMEM_TY = gl.constexpr(k_buffer_dtype)
        else:
            self.pfx_q_dot_layout = gl.constexpr(q_dot)
            self.pfx_kt_dot_layout = gl.constexpr(kt_dot)
            self.pfx_p_dot_layout = gl.constexpr(p_dot)
            self.pfx_v_dot_layout = gl.constexpr(v_dot)
            self.PFX_SMEM_TY = gl.constexpr(q_dtype)

        self.IS_FP8 = gl.constexpr(IS_FP8)
        self.ASYNC_PAD_K = gl.constexpr(16)
        self.ASYNC_PAD_V = gl.constexpr(16)

    # ==== MMA + dot-operand factories ========================================

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

    # ==== Async tile padded-SMEM factory ======================================

    @gluon.constexpr_function
    def make_padded_smem(shape, offset_bases, padding_pairs):
        """Padded shared-memory layout for async DMA tiles.

        ``padding_pairs`` is e.g. ``[[512, pad]]`` (BF16) or
        ``[[1024, pad], [2048, 32]]`` (FP8).
        """
        if _PADDED_SHARED_LAYOUT_BLOCK_ARG == "block_bases":
            return PaddedSharedLayout(
                interval_padding_pairs=padding_pairs,
                offset_bases=offset_bases,
                block_bases=[],
                shape=shape,
            )
        return PaddedSharedLayout(
            interval_padding_pairs=padding_pairs,
            offset_bases=offset_bases,
            cga_layout=[],
            shape=shape,
        )

    @gluon.constexpr_function
    def make_serial_kt_blocked(num_warps):
        """Blocked layout for serial K^T tile loads.

        ``K^T`` is stored as [BLOCK_DMODEL, BLOCK_N] with D contiguous.
        Threads therefore hold consecutive D elements, which lowers to wide
        buffer loads and wide LDS stores instead of scalar bf16 traffic along
        the slower N dimension.
        """
        return gl.BlockedLayout(
            size_per_thread=[8, 1],
            threads_per_warp=[8, 8],
            warps_per_cta=[1, num_warps],
            order=[0, 1],
        )

    # ==== Offset-bases primitives =============================================

    @gluon.constexpr_function
    def make_offset_bases(major_max, minor_coarse, minor_fine, major_dim):
        """Compute offset_bases for a PaddedSharedLayout async DMA tile.

        Generates a power-of-2 ladder ``[1 .. major_max]`` along
        ``major_dim`` followed by two groups of minor-dimension basis
        vectors.

        Args:
            major_max: largest power-of-2 in the fast-varying dimension
                       (e.g. ``BLOCK_DMODEL // 2`` for K^T, ``BLOCK_DV // 2``
                       for V).
            minor_coarse: coarse basis values placed in the high address
                          bits.
            minor_fine: fine basis values placed in the low address bits.
            major_dim: ``0`` for K^T / KPE (row-major), ``1`` for V
                       (col-major).
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
        ob = [[d, 0] for d in ExtendAttentionLayouts._power_of_2_ladder(BLOCK_D // 2)]
        for v in ExtendAttentionLayouts._minor_coarse_for_d(BLOCK_D, BLOCK_N):
            ob.append([0, v])
        for v in [1, 2, 4, 8]:
            ob.append([0, v])
        return ob

    # ==== BF16 K^T async-DMA DLL bases ========================================

    @gluon.constexpr_function
    def _kt_dll_bases_4w(BLOCK_D, BLOCK_N):
        """4-warp K^T (reg_bases, lane_bases, warp_bases)."""
        if BLOCK_D >= 512:
            reg = (
                [[1, 0], [2, 0], [4, 0], [0, 4], [0, 8], [0, 16], [0, 32]]
                if BLOCK_N >= 64
                else [[1, 0], [2, 0], [4, 0], [0, 4], [0, 8], [0, 16]]
            )
            lane = [[8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [256, 0]]
        elif BLOCK_D >= 256:
            reg = [[1, 0], [2, 0], [4, 0], [0, 4], [0, 8]]
            lane = [[8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [0, 16]]
        else:
            reg = (
                [[1, 0], [2, 0], [4, 0], [0, 4], [0, 8], [0, 64]]
                if BLOCK_N >= 128
                else [[1, 0], [2, 0], [4, 0], [0, 4], [0, 8]]
            )
            lane = [[8, 0], [16, 0], [32, 0], [64, 0], [0, 16], [0, 32]]
        return reg, lane, [[0, 1], [0, 2]]

    @gluon.constexpr_function
    def _kt_dll_bases_8w(BLOCK_D, BLOCK_N):
        """8-warp K^T (reg_bases, lane_bases, warp_bases)."""
        if BLOCK_D >= 512:
            lane = [[8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [256, 0]]
            if BLOCK_N >= 64:
                return (
                    [[1, 0], [2, 0], [4, 0], [0, 4], [0, 8], [0, 16]],
                    lane,
                    [[0, 1], [0, 2], [0, 32]],
                )
            return (
                [[1, 0], [2, 0], [4, 0], [0, 8], [0, 16]],
                lane,
                [[0, 1], [0, 2], [0, 4]],
            )
        if BLOCK_D >= 256:
            return (
                [[1, 0], [2, 0], [4, 0], [0, 8]],
                [[8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [0, 16]],
                [[0, 1], [0, 2], [0, 4]],
            )
        if BLOCK_D >= 128:
            reg = (
                [[1, 0], [2, 0], [4, 0], [0, 8], [0, 64]]
                if BLOCK_N >= 128
                else [[1, 0], [2, 0], [4, 0], [0, 8]]
            )
            return (
                reg,
                [[8, 0], [16, 0], [32, 0], [64, 0], [0, 16], [0, 32]],
                [[0, 1], [0, 2], [0, 4]],
            )
        reg = (
            [[1, 0], [2, 0], [4, 0], [0, 64]]
            if BLOCK_N >= 128
            else [[1, 0], [2, 0], [4, 0]]
        )
        return (
            reg,
            [[8, 0], [16, 0], [32, 0], [0, 16], [0, 32], [0, 1]],
            [[0, 2], [0, 4], [0, 8]],
        )

    @gluon.constexpr_function
    def _kt_dll_bases(num_warps, BLOCK_D, BLOCK_N):
        """(reg, lane, warp) for an async K^T tile, dispatching on warp count."""
        if num_warps < 8:
            return ExtendAttentionLayouts._kt_dll_bases_4w(BLOCK_D, BLOCK_N)
        return ExtendAttentionLayouts._kt_dll_bases_8w(BLOCK_D, BLOCK_N)

    # ==== BF16 async K^T / V tile factories ===================================

    @gluon.constexpr_function
    def make_kt_offset_bases(BLOCK_DMODEL, BLOCK_N):
        """PaddedSharedLayout offset bases for a [BLOCK_DMODEL, BLOCK_N] K^T tile."""
        return ExtendAttentionLayouts._kt_offset_bases(BLOCK_DMODEL, BLOCK_N)

    @gluon.constexpr_function
    def make_v_offset_bases(BLOCK_DV, BLOCK_N):
        """PaddedSharedLayout offset bases for a [BLOCK_N, BLOCK_DV] V tile
        (transpose of the K^T layout)."""
        return ExtendAttentionLayouts._transpose_bases(
            ExtendAttentionLayouts._kt_offset_bases(BLOCK_DV, BLOCK_N)
        )

    @gluon.constexpr_function
    def make_kt_dll(num_warps, BLOCK_DMODEL, BLOCK_N):
        """Async DMA DistributedLinearLayout for a [BLOCK_DMODEL, BLOCK_N] K^T tile."""
        reg, lane, warp = ExtendAttentionLayouts._kt_dll_bases(
            num_warps, BLOCK_DMODEL, BLOCK_N
        )
        return DistributedLinearLayout(
            reg_bases=reg,
            lane_bases=lane,
            warp_bases=warp,
            block_bases=[],
            shape=[BLOCK_DMODEL, BLOCK_N],
        )

    @gluon.constexpr_function
    def make_v_dll(num_warps, BLOCK_DV, BLOCK_N):
        """Async DMA DistributedLinearLayout for a [BLOCK_N, BLOCK_DV] V tile.

        Pure transpose of the K^T layout: every basis [a, b] is mirrored to
        [b, a] and the tile shape is [N, Dv] instead of [Dv, N].
        """
        reg, lane, warp = ExtendAttentionLayouts._kt_dll_bases(
            num_warps, BLOCK_DV, BLOCK_N
        )
        return DistributedLinearLayout(
            reg_bases=ExtendAttentionLayouts._transpose_bases(reg),
            lane_bases=ExtendAttentionLayouts._transpose_bases(lane),
            warp_bases=ExtendAttentionLayouts._transpose_bases(warp),
            block_bases=[],
            shape=[BLOCK_N, BLOCK_DV],
        )

    # ==== FP8 extend exceptions: BF16 V layout inside the FP8 kernel ==========
    #
    # The FP8 kernel's BF16 extend phase uses a non-standard V layout in
    # three (num_warps, BLOCK_DV, BLOCK_N) corners for shared-memory-pressure
    # reasons; elsewhere it uses the standard layout. BF16 K^T in the FP8
    # kernel uses the standard ``make_kt_dll``.

    @gluon.constexpr_function
    def _fp8_bf16_v_dll_bases_4w(BLOCK_DV, BLOCK_N):
        """4-warp FP8-kernel BF16-V (reg, lane, warp), with special-case overrides."""
        if BLOCK_DV >= 256:
            reg = (
                [[0, 1], [0, 2], [0, 4], [4, 0], [8, 0], [32, 0], [64, 0]]
                if BLOCK_N >= 128
                else [[0, 1], [0, 2], [0, 4], [4, 0], [8, 0]]
            )
            return (
                reg,
                [[0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [16, 0]],
                [[1, 0], [2, 0]],
            )
        if BLOCK_N >= 128:
            return (
                [[0, 1], [0, 2], [0, 4], [4, 0], [8, 0], [64, 0]],
                [[0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0]],
                [[1, 0], [2, 0]],
            )
        return (
            [[0, 1], [0, 2], [0, 4], [0, 8], [8, 0]],
            [[0, 16], [0, 32], [0, 64], [1, 0], [2, 0], [4, 0]],
            [[16, 0], [32, 0]],
        )

    @gluon.constexpr_function
    def _fp8_bf16_v_dll_bases_8w(BLOCK_DV, BLOCK_N):
        """8-warp FP8-kernel BF16-V (reg, lane, warp), with special-case overrides."""
        if BLOCK_DV >= 256:
            return (
                [[0, 1], [0, 2], [0, 4], [8, 0]],
                [[0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [16, 0]],
                [[1, 0], [2, 0], [4, 0]],
            )
        if BLOCK_DV >= 128:
            if BLOCK_N >= 128:
                return (
                    [[0, 1], [0, 2], [0, 4], [8, 0], [64, 0]],
                    [[0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0]],
                    [[1, 0], [2, 0], [4, 0]],
                )
            return (
                [[0, 1], [0, 2], [0, 4], [0, 8]],
                [[0, 16], [0, 32], [0, 64], [1, 0], [2, 0], [4, 0]],
                [[8, 0], [16, 0], [32, 0]],
            )
        reg = (
            [[0, 1], [0, 2], [0, 4], [64, 0]]
            if BLOCK_N >= 128
            else [[0, 1], [0, 2], [0, 4]]
        )
        return (
            reg,
            [[0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [1, 0]],
            [[2, 0], [4, 0], [8, 0]],
        )

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
        ob = [[0, d] for d in ExtendAttentionLayouts._power_of_2_ladder(BLOCK_DV // 2)]
        for v in mc:
            ob.append([v, 0])
        for v in [1, 2, 4, 8]:
            ob.append([v, 0])
        return ob

    @gluon.constexpr_function
    def make_fp8_bf16_v_dll(num_warps, BLOCK_DV, BLOCK_N):
        """Async DMA DLL for the FP8 kernel's BF16-extend V tile [BLOCK_N, BLOCK_DV].

        Non-standard corners vs ``make_v_dll``:
        - 4w Dv>=256 N>=128: extra [32,0], [64,0] in reg
        - 4w Dv<256  N<128:  shuffled lane/warp for FP8 LDS pressure
        - 8w Dv<256  N<128:  same shuffled pattern, 3-warp stride
        """
        if num_warps < 8:
            reg, lane, warp = ExtendAttentionLayouts._fp8_bf16_v_dll_bases_4w(
                BLOCK_DV, BLOCK_N
            )
        else:
            reg, lane, warp = ExtendAttentionLayouts._fp8_bf16_v_dll_bases_8w(
                BLOCK_DV, BLOCK_N
            )
        return DistributedLinearLayout(
            reg_bases=reg,
            lane_bases=lane,
            warp_bases=warp,
            block_bases=[],
            shape=[BLOCK_N, BLOCK_DV],
        )

    # ==== FP8 prefix async-DMA tile factories =================================
    #
    # The FP8 kernel runs two async-DMA streams: prefix (FP8 K/V loads, 1 byte)
    # and extend (BF16 K/V loads, 2 bytes). The prefix stream packs more
    # elements per vector register along the major axis, so its
    # DistributedLinearLayouts differ from the BF16 factories above. Offset
    # bases coincide with BF16 for K^T; V and KPE use their own ladders.
    #
    #   Factory                              | Used for
    #   -------------------------------------|-------------------------------------
    #   make_fp8_kt_dll                      | FP8 prefix K^T DLL   (D>=128; D<128 falls back to BF16)
    #   make_fp8_v_dll                       | FP8 prefix V   DLL
    #   make_fp8_v_offset_bases              | FP8 prefix V   offset bases (alias)
    #   make_fp8_extend_v_dll                | FP8 kernel BF16 extend V DLL (alias for make_fp8_bf16_v_dll)
    #   make_ext_kt_offset_bases / _dll      | BF16 extend K^T when EXT_BLOCK_N != BLOCK_N (rare)
    #   make_ext_v_offset_bases  / _dll      | BF16 extend V   when EXT_BLOCK_N != BLOCK_N (rare)

    @gluon.constexpr_function
    def make_fp8_kt_dll(num_warps, BLOCK_DMODEL, BLOCK_N):
        """Async DMA DLL for FP8 prefix K^T tile [BLOCK_DMODEL, BLOCK_N].

        FP8 packs 8 elements per vector register along D (1 byte each = 8 bytes,
        matches a dword load). For D<128 the partition matches BF16 KT, so we
        delegate to ``make_kt_dll`` in that case.
        """
        is_4w = num_warps < 8
        if BLOCK_DMODEL < 128:
            return ExtendAttentionLayouts.make_kt_dll(num_warps, BLOCK_DMODEL, BLOCK_N)
        if is_4w:
            if BLOCK_DMODEL >= 256:
                reg = [[1, 0], [2, 0], [4, 0], [0, 4], [0, 8]]
                lane = [[8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [0, 16]]
                warp = [[0, 1], [0, 2]]
            else:  # D=128
                if BLOCK_N >= 128:
                    reg = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]]
                    lane = [[16, 0], [32, 0], [64, 0], [0, 16], [0, 32], [0, 64]]
                    warp = [[0, 1], [0, 2]]
                else:
                    reg = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 8]]
                    lane = [[16, 0], [32, 0], [64, 0], [0, 1], [0, 2], [0, 4]]
                    warp = [[0, 16], [0, 32]]
        else:  # 8-warp
            if BLOCK_DMODEL >= 256:
                reg = [[1, 0], [2, 0], [4, 0], [0, 8]]
                lane = [[8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [0, 16]]
                warp = [[0, 1], [0, 2], [0, 4]]
            else:  # D=128
                if BLOCK_N >= 128:
                    reg = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 8]]
                    lane = [[16, 0], [32, 0], [64, 0], [0, 16], [0, 32], [0, 64]]
                    warp = [[0, 1], [0, 2], [0, 4]]
                else:
                    reg = [[1, 0], [2, 0], [4, 0], [8, 0]]
                    lane = [[16, 0], [32, 0], [64, 0], [0, 1], [0, 2], [0, 4]]
                    warp = [[0, 8], [0, 16], [0, 32]]
        return DistributedLinearLayout(
            reg_bases=reg,
            lane_bases=lane,
            warp_bases=warp,
            block_bases=[],
            shape=[BLOCK_DMODEL, BLOCK_N],
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
            return ExtendAttentionLayouts.make_v_dll(num_warps, BLOCK_DV, BLOCK_N)
        is_4w = num_warps < 8
        if is_4w:
            if BLOCK_DV >= 256:
                if BLOCK_N >= 128:
                    reg = [[0, 1], [0, 2], [0, 4], [0, 8], [4, 0], [8, 0], [64, 0]]
                    lane = [[0, 16], [0, 32], [0, 64], [0, 128], [16, 0], [32, 0]]
                else:
                    reg = [[0, 1], [0, 2], [0, 4], [4, 0], [8, 0]]
                    lane = [[0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [16, 0]]
                warp = [[1, 0], [2, 0]]
            else:  # Dv<256
                if BLOCK_N >= 128:
                    reg = [[0, 1], [0, 2], [0, 4], [0, 8], [4, 0], [8, 0]]
                    lane = [[0, 16], [0, 32], [0, 64], [16, 0], [32, 0], [64, 0]]
                    warp = [[1, 0], [2, 0]]
                else:
                    reg = [[0, 1], [0, 2], [0, 4], [0, 8], [8, 0]]
                    lane = [[0, 16], [0, 32], [0, 64], [1, 0], [2, 0], [4, 0]]
                    warp = [[16, 0], [32, 0]]
        else:  # 8-warp
            if BLOCK_DMODEL >= 256:
                if BLOCK_DV >= 256:
                    reg = [[0, 1], [0, 2], [0, 4], [8, 0]]
                    lane = [[0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [16, 0]]
                else:
                    reg = [[0, 1], [0, 2], [0, 4], [8, 0]]
                    lane = [[0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0]]
                warp = [[1, 0], [2, 0], [4, 0]]
            else:  # D=128  (Dv=128 implied at every callsite)
                if BLOCK_N >= 128:
                    reg = [[0, 1], [0, 2], [0, 4], [0, 8], [8, 0]]
                    lane = [[0, 16], [0, 32], [0, 64], [16, 0], [32, 0], [64, 0]]
                    warp = [[1, 0], [2, 0], [4, 0]]
                else:
                    reg = [[0, 1], [0, 2], [0, 4], [0, 8]]
                    lane = [[0, 16], [0, 32], [0, 64], [1, 0], [2, 0], [4, 0]]
                    warp = [[8, 0], [16, 0], [32, 0]]
        return DistributedLinearLayout(
            reg_bases=reg,
            lane_bases=lane,
            warp_bases=warp,
            block_bases=[],
            shape=[BLOCK_N, BLOCK_DV],
        )

    @gluon.constexpr_function
    def make_fp8_v_offset_bases(num_warps, BLOCK_DV, BLOCK_N):
        """FP8 prefix V offset bases -- same ladder as BF16 extend V (alias)."""
        return ExtendAttentionLayouts.make_fp8_bf16_v_offset_bases(
            num_warps, BLOCK_DV, BLOCK_N
        )

    @gluon.constexpr_function
    def make_fp8_extend_v_dll(num_warps, BLOCK_DV, BLOCK_N):
        """FP8 kernel's BF16-extend V tile DLL -- alias for make_fp8_bf16_v_dll."""
        return ExtendAttentionLayouts.make_fp8_bf16_v_dll(num_warps, BLOCK_DV, BLOCK_N)

    # ==== FP8 extend exceptions: rare BF16 EXT_BLOCK_N tile factories =========

    @gluon.constexpr_function
    def make_ext_kt_offset_bases(num_warps, BLOCK_DMODEL, EXT_BLOCK_N):
        """BF16 extend K^T offset bases for ``EXT_BLOCK_N != BLOCK_N`` (rare).

        Used only inside the FP8 kernel's 4-warp DMA path when the extend tile
        size differs from the prefix tile size. All other extend paths reuse
        ``make_kt_offset_bases``.
        """
        if BLOCK_DMODEL >= 256:
            return ExtendAttentionLayouts.make_offset_bases(
                128, [16, 32], [1, 2, 4, 8], 0
            )
        return ExtendAttentionLayouts.make_offset_bases(64, [16, 32], [1, 2, 4, 8], 0)

    @gluon.constexpr_function
    def make_ext_kt_dll(num_warps, BLOCK_DMODEL, EXT_BLOCK_N):
        """BF16 extend K^T DLL for ``EXT_BLOCK_N != BLOCK_N`` (rare)."""
        if BLOCK_DMODEL >= 256:
            reg = [[1, 0], [2, 0], [4, 0], [128, 0], [0, 4], [0, 8]]
        else:
            reg = [[1, 0], [2, 0], [4, 0], [0, 4], [0, 8]]
        lane = [[8, 0], [16, 0], [32, 0], [64, 0], [0, 16], [0, 32]]
        warp = [[0, 1], [0, 2]]
        return DistributedLinearLayout(
            reg_bases=reg,
            lane_bases=lane,
            warp_bases=warp,
            block_bases=[],
            shape=[BLOCK_DMODEL, EXT_BLOCK_N],
        )

    @gluon.constexpr_function
    def make_ext_v_offset_bases(num_warps, BLOCK_DV, EXT_BLOCK_N):
        """BF16 extend V offset bases for ``EXT_BLOCK_N != BLOCK_N`` (rare)."""
        if BLOCK_DV >= 256:
            return ExtendAttentionLayouts.make_offset_bases(
                128, [16, 32], [1, 2, 4, 8], 1
            )
        return ExtendAttentionLayouts.make_offset_bases(64, [16, 32], [1, 2, 4, 8], 1)

    @gluon.constexpr_function
    def make_ext_v_dll(num_warps, BLOCK_DV, EXT_BLOCK_N):
        """BF16 extend V DLL for ``EXT_BLOCK_N != BLOCK_N`` (rare)."""
        if BLOCK_DV >= 256:
            reg = [[0, 1], [0, 2], [0, 4], [0, 128], [4, 0], [8, 0]]
        else:
            reg = [[0, 1], [0, 2], [0, 4], [4, 0], [8, 0]]
        lane = [[0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0]]
        warp = [[1, 0], [2, 0]]
        return DistributedLinearLayout(
            reg_bases=reg,
            lane_bases=lane,
            warp_bases=warp,
            block_bases=[],
            shape=[EXT_BLOCK_N, BLOCK_DV],
        )

    # ==== Prefix smem policy and IS_FP8-aware prefix async factories ==========

    @gluon.constexpr_function
    def prefix_v_offset_bases(layouts, num_warps, BLOCK_DV, BLOCK_N):
        """V offset bases for an async prefix tile (FP8-packed layout when IS_FP8)."""
        if layouts.IS_FP8:
            return ExtendAttentionLayouts.make_fp8_v_offset_bases(
                num_warps, BLOCK_DV, BLOCK_N
            )
        return ExtendAttentionLayouts.make_v_offset_bases(BLOCK_DV, BLOCK_N)

    @gluon.constexpr_function
    def prefix_kt_dll(layouts, num_warps, BLOCK_DMODEL, BLOCK_N):
        """Prefix K^T async DMA DLL (FP8-native partition when IS_FP8)."""
        if layouts.IS_FP8:
            return ExtendAttentionLayouts.make_fp8_kt_dll(
                num_warps, BLOCK_DMODEL, BLOCK_N
            )
        return ExtendAttentionLayouts.make_kt_dll(num_warps, BLOCK_DMODEL, BLOCK_N)

    @gluon.constexpr_function
    def prefix_v_dll(layouts, num_warps, BLOCK_DMODEL, BLOCK_DV, BLOCK_N):
        """Prefix V async DMA DLL (FP8-native partition when IS_FP8)."""
        if layouts.IS_FP8:
            return ExtendAttentionLayouts.make_fp8_v_dll(
                num_warps, BLOCK_DMODEL, BLOCK_DV, BLOCK_N
            )
        return ExtendAttentionLayouts.make_v_dll(num_warps, BLOCK_DV, BLOCK_N)

    @gluon.constexpr_function
    def prefix_kt_smem_layout(layouts, BLOCK_DMODEL, BLOCK_N, offset_bases):
        """Prefix K^T padded smem (1024-byte interval on FP8, 512 on BF16)."""
        if layouts.IS_FP8:
            pad_pairs = [[1024, layouts.ASYNC_PAD_K.value], [2048, 32]]
        else:
            pad_pairs = [[512, layouts.ASYNC_PAD_K.value]]
        return ExtendAttentionLayouts.make_padded_smem(
            [BLOCK_DMODEL, BLOCK_N], offset_bases, pad_pairs
        )

    @gluon.constexpr_function
    def prefix_v_smem_layout(layouts, BLOCK_N, BLOCK_DV, offset_bases):
        """Prefix V padded smem (1024-byte interval on FP8, 512 on BF16)."""
        if layouts.IS_FP8:
            pad_pairs = [[1024, layouts.ASYNC_PAD_V.value], [2048, 32]]
        else:
            pad_pairs = [[512, layouts.ASYNC_PAD_V.value]]
        return ExtendAttentionLayouts.make_padded_smem(
            [BLOCK_N, BLOCK_DV], offset_bases, pad_pairs
        )


# ===---------------------------------------------------------------------===#
# ExtendAttnConfig -- bundles layouts + all compile-time shapes/modes
# ===---------------------------------------------------------------------===#
#
# Analogous to upstream ``MXFPGEMMConfig`` / ``AttentionConfigBase``. Groups
# everything constexpr so the state and schedule aggregates can thread one
# constexpr field through their methods. All three kernel variants share the
# same config; the outer wrapper selects the schedule aggregate.


@aggregate
class ExtendAttnConfig:
    # Layouts aggregate (already @aggregate in `_common.py`). Annotated
    # with the aggregate class directly so the @aggregate type check
    # recognizes it as a nested aggregate (not a plain constexpr). This
    # mirrors ``MXFPGEMMPipelinedProgram.cfg: MXFPGEMMConfig`` in upstream.
    layouts: ExtendAttentionLayouts

    # Shape constexprs.
    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    BLOCK_DMODEL: gl.constexpr
    BLOCK_DV: gl.constexpr
    NUM_STAGES: gl.constexpr
    EXT_BLOCK_N: gl.constexpr
    EXT_NUM_STAGES: gl.constexpr
    num_warps: gl.constexpr

    # Mode constexprs.
    IS_CAUSAL: gl.constexpr
    USE_CUSTOM_MASK: gl.constexpr
    ENABLE_PREFIX_UNMASKED: gl.constexpr
    LOGIT_CAP: gl.constexpr
    XAI_TEMPERATURE_LEN: gl.constexpr
    SLIDING_WINDOW_SIZE: gl.constexpr
    HAS_WINDOW_OFFSETS: gl.constexpr
    IS_FP8: gl.constexpr
    HAS_SINK: gl.constexpr
    IS_WCA: gl.constexpr
    SPLIT_K: gl.constexpr

    # XCD-aware PID remap (MI350X has 8 XCDs x 32 CUs). ``XCD_REMAP`` is
    # the A/B gate, ``NUM_XCDS`` is the hardware constant (8 on gfx950),
    # and ``XCD_CHUNK`` controls how many consecutive tile_idx values
    # land on one XCD before the next chunk rotates to the next XCD.
    # ``XCD_CHUNK=1`` is the identity mapping (no-op). The dispatcher
    # picks these from the host-side WCA launch policy.
    XCD_REMAP: gl.constexpr
    NUM_XCDS: gl.constexpr
    XCD_CHUNK: gl.constexpr
    XCD_MODE: gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
        layouts,
        BLOCK_M,
        BLOCK_N,
        BLOCK_DMODEL,
        BLOCK_DV,
        NUM_STAGES,
        EXT_BLOCK_N,
        EXT_NUM_STAGES,
        num_warps,
        IS_CAUSAL,
        USE_CUSTOM_MASK,
        ENABLE_PREFIX_UNMASKED,
        LOGIT_CAP,
        XAI_TEMPERATURE_LEN,
        SLIDING_WINDOW_SIZE,
        HAS_WINDOW_OFFSETS,
        IS_FP8,
        HAS_SINK,
        IS_WCA,
        SPLIT_K,
        XCD_REMAP=False,
        NUM_XCDS=8,
        XCD_CHUNK=1,
        XCD_MODE=1,
    ):
        # NOTE: @gluon.constexpr_function unwraps gl.constexpr args to raw
        # Python values before calling this body, so every constexpr field
        # must be re-wrapped on assignment (the aggregate __setattr__ type
        # checker rejects raw ints / bools otherwise). This matches the
        # upstream MXFPGEMMConfig.__init__ pattern. ``layouts`` is a nested
        # aggregate (ExtendAttentionLayouts) and is stored as-is.
        self.layouts = layouts
        self.BLOCK_M = gl.constexpr(BLOCK_M)
        self.BLOCK_N = gl.constexpr(BLOCK_N)
        self.BLOCK_DMODEL = gl.constexpr(BLOCK_DMODEL)
        self.BLOCK_DV = gl.constexpr(BLOCK_DV)
        self.NUM_STAGES = gl.constexpr(NUM_STAGES)
        self.EXT_BLOCK_N = gl.constexpr(EXT_BLOCK_N)
        self.EXT_NUM_STAGES = gl.constexpr(EXT_NUM_STAGES)
        self.num_warps = gl.constexpr(num_warps)
        self.IS_CAUSAL = gl.constexpr(IS_CAUSAL)
        self.USE_CUSTOM_MASK = gl.constexpr(USE_CUSTOM_MASK)
        self.ENABLE_PREFIX_UNMASKED = gl.constexpr(ENABLE_PREFIX_UNMASKED)
        self.LOGIT_CAP = gl.constexpr(LOGIT_CAP)
        self.XAI_TEMPERATURE_LEN = gl.constexpr(XAI_TEMPERATURE_LEN)
        self.SLIDING_WINDOW_SIZE = gl.constexpr(SLIDING_WINDOW_SIZE)
        self.HAS_WINDOW_OFFSETS = gl.constexpr(HAS_WINDOW_OFFSETS)
        self.IS_FP8 = gl.constexpr(IS_FP8)
        self.HAS_SINK = gl.constexpr(HAS_SINK)
        self.IS_WCA = gl.constexpr(IS_WCA)
        self.SPLIT_K = gl.constexpr(SPLIT_K)
        self.XCD_REMAP = gl.constexpr(XCD_REMAP)
        self.NUM_XCDS = gl.constexpr(NUM_XCDS)
        self.XCD_CHUNK = gl.constexpr(XCD_CHUNK)
        self.XCD_MODE = gl.constexpr(XCD_MODE)


# ===---------------------------------------------------------------------===#
# KVSmemBank -- K/V smem staging across prefix and extend phases
# ===---------------------------------------------------------------------===#
#
# Wraps the (kt_smem, v_smem) pair and the async layouts needed by
# prefix or extend K/V staging. FP8 prefix kernels call
# ``transition_to_extend(...)`` to release the FP8 bank and return a
# fresh BF16 bank sized for the extend phase. BF16 callers simply reuse
# the prefix bank in-place.
#
# This encapsulates what the pre-port kernel spelled out inline as:
#
#     if IS_FP8:
#         kt_smem._keep_alive(); v_smem._keep_alive()
#         kt_smem = gl.allocate_shared_memory(...)
#         v_smem  = gl.allocate_shared_memory(...)
#         for _s in gl.static_range(_EXT_NS):
#             v_smem.index(_s).store(zeros(...))
#         gl.barrier()


@aggregate
class KVSmemBank:
    # (kt_smem, v_smem) + the two async-copy layouts that the prefix
    # and extend helpers consume. Storing the async layouts here (rather than
    # re-passing them through every call site) is the mxfp-fa
    # ``KVMemory``/descriptor-bundle pattern: per-phase constexpr
    # metadata rides on the same aggregate as the backing smem, so
    # helper signatures only take the bank.
    kt_smem: gl.shared_memory_descriptor
    v_smem: gl.shared_memory_descriptor
    kt_async_layout: gl.constexpr
    v_async_layout: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, kt_smem, v_smem, kt_async_layout, v_async_layout):
        self.kt_smem = kt_smem
        self.v_smem = v_smem
        self.kt_async_layout = gl.constexpr(kt_async_layout)
        self.v_async_layout = gl.constexpr(v_async_layout)

    @gluon.jit
    def initialize(
        cfg,
        kt_smem_layout: gl.constexpr,
        v_smem_layout: gl.constexpr,
        kt_async_layout: gl.constexpr,
        v_async_layout: gl.constexpr,
        zero_fill: gl.constexpr,
    ):
        """Allocate the prefix-phase smem + (optionally) zero-fill V.

        Zero-fill is required when:
          * IS_FP8 (the prefix -> BF16 transition reads all NUM_STAGES
            slots even on small prefixes), or
          * is_valid_tile for the non-FP8 case (skipped tiles bail
            before consuming the smem, so no zero-fill needed).

        Callers pass the OR of those at the call site as a constexpr
        (``zero_fill=True`` if IS_FP8 else is_valid_tile).
        """
        kt_smem = gl.allocate_shared_memory(
            cfg.layouts.PFX_SMEM_TY,
            [cfg.NUM_STAGES, cfg.BLOCK_DMODEL, cfg.BLOCK_N],
            layout=kt_smem_layout,
        )
        v_smem = gl.allocate_shared_memory(
            cfg.layouts.PFX_SMEM_TY,
            [cfg.NUM_STAGES, cfg.BLOCK_N, cfg.BLOCK_DV],
            layout=v_smem_layout,
        )
        if zero_fill:
            for _s in gl.static_range(cfg.NUM_STAGES):
                v_zero = gl.zeros(
                    [cfg.BLOCK_N, cfg.BLOCK_DV],
                    dtype=cfg.layouts.PFX_SMEM_TY,
                    layout=v_async_layout,
                )
                v_smem.index(_s).store(v_zero)
            gl.barrier()
        return KVSmemBank(kt_smem, v_smem, kt_async_layout, v_async_layout)

    @gluon.jit
    def transition_to_extend(
        self,
        cfg,
        extend_dtype,
        ext_kt_smem_layout: gl.constexpr,
        ext_v_smem_layout: gl.constexpr,
        ext_kt_async_layout: gl.constexpr,
        ext_v_async_layout: gl.constexpr,
        EXT_N: gl.constexpr,
        EXT_NS: gl.constexpr,
    ):
        """FP8 prefix smem -> BF16 extend smem handoff.

        Releases the FP8 smem via ``_keep_alive()`` (kernel-local
        dealloc), re-allocates the bank in the extend dtype, and
        zero-fills V. Returns a fresh ``KVSmemBank`` that the
        extend hot-loop consumes with the extend-phase async layouts.

        BF16 callers never invoke this; the prefix smem is reused
        in-place for the extend phase.
        """
        self.kt_smem._keep_alive()
        self.v_smem._keep_alive()
        kt_smem = gl.allocate_shared_memory(
            extend_dtype,
            [EXT_NS, cfg.BLOCK_DMODEL, EXT_N],
            layout=ext_kt_smem_layout,
        )
        v_smem = gl.allocate_shared_memory(
            extend_dtype,
            [EXT_NS, EXT_N, cfg.BLOCK_DV],
            layout=ext_v_smem_layout,
        )
        for _s in gl.static_range(EXT_NS):
            v_zero = gl.zeros(
                [EXT_N, cfg.BLOCK_DV],
                dtype=extend_dtype,
                layout=ext_v_async_layout,
            )
            v_smem.index(_s).store(v_zero)
        gl.barrier()
        return KVSmemBank(kt_smem, v_smem, ext_kt_async_layout, ext_v_async_layout)


# ===---------------------------------------------------------------------===#
# AsyncKVLoader -- single K-or-V async-DMA tap with prefix/extend variants
# ===---------------------------------------------------------------------===#
#
# Analog of upstream ``ScaleAsyncCopyDescriptor``. One instance wraps a
# single tap (either K or V, either prefix-gathered or extend-row-major).
# The ``IS_PREFIX`` constexpr selects the DMA helper used by ``.issue()``:
#   IS_PREFIX=True  -> ``issue_async_load_{k,v}_prefix`` (gather via kv_indices)
#   IS_PREFIX=False -> ``issue_async_load_{k,v}_extend`` (row-major stride walk)
#
# For the hot-path WCA scalar-mask variant that currently has its
# own helper pair (``*_from_locs_hot``), callers use
# ``.issue_from_locs(...)`` and pass ``IS_SCALAR_MASK=True``.
#
# Prefix pipelines choose the normal or scalar-mask gather path at the
# call site, where the loop already knows whether the BLOCK_N tile is
# fully in range.


@aggregate
class AsyncKVLoader:
    # Multi-slot smem bank: ``.index(slot)`` picks the destination buffer
    # at ``issue()`` time, matching the mxfp-fa ``KVMemory`` pattern
    # where ``self.k_mem.smem.index(buf)`` threads the pipeline buffer
    # index through a runtime arg. Callers construct one loader per
    # (K|V, prefix|extend) tap and reuse it across the pipelined bulk,
    # rotating ``slot`` each iteration.
    smem: gl.shared_memory_descriptor
    gbl_base: gl.tensor
    kv_indices: gl.tensor | gl.constexpr  # gl.constexpr(0) when IS_PREFIX=False
    kv_start: gl.tensor | gl.constexpr  # gl.constexpr(0) when IS_PREFIX=False
    stride_n: gl.tensor | gl.constexpr
    IS_PREFIX: gl.constexpr
    IS_K: gl.constexpr
    BLOCK_N: gl.constexpr
    BLOCK_D: gl.constexpr
    async_layout: gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
        smem,
        gbl_base,
        kv_indices,
        kv_start,
        stride_n,
        IS_PREFIX,
        IS_K,
        BLOCK_N,
        BLOCK_D,
        async_layout,
    ):
        # Runtime scalars (kv_indices, kv_start, stride_n) may arrive as
        # Python ints when Triton specializes them on the kernel
        # boundary; re-wrap for the union-typed aggregate field.
        self.smem = smem
        self.gbl_base = gbl_base
        self.kv_indices = _wrap_rt_scalar(kv_indices)
        self.kv_start = _wrap_rt_scalar(kv_start)
        self.stride_n = _wrap_rt_scalar(stride_n)
        self.IS_PREFIX = gl.constexpr(IS_PREFIX)
        self.IS_K = gl.constexpr(IS_K)
        self.BLOCK_N = gl.constexpr(BLOCK_N)
        self.BLOCK_D = gl.constexpr(BLOCK_D)
        self.async_layout = gl.constexpr(async_layout)

    @gluon.jit
    def for_prefix_k(
        kt_smem,
        K_Buffer,
        cur_kv_head,
        kv_indices,
        kv_start,
        stride_buf_kbs,
        stride_buf_kh,
        BLOCK_N: gl.constexpr,
        BLOCK_DMODEL: gl.constexpr,
        kt_async_layout: gl.constexpr,
    ):
        """Build a loader for the prefix-K gather tap (multi-slot)."""
        k_prefix_base = K_Buffer + cur_kv_head * stride_buf_kh
        return AsyncKVLoader(
            kt_smem,
            k_prefix_base,
            kv_indices,
            kv_start,
            stride_buf_kbs,
            IS_PREFIX=True,
            IS_K=True,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_DMODEL,
            async_layout=kt_async_layout,
        )

    @gluon.jit
    def for_prefix_v(
        v_smem,
        V_Buffer,
        cur_kv_head,
        kv_indices,
        kv_start,
        stride_buf_vbs,
        stride_buf_vh,
        BLOCK_N: gl.constexpr,
        BLOCK_DV: gl.constexpr,
        v_async_layout: gl.constexpr,
    ):
        """Build a loader for the prefix-V gather tap (multi-slot)."""
        v_prefix_base = V_Buffer + cur_kv_head * stride_buf_vh
        return AsyncKVLoader(
            v_smem,
            v_prefix_base,
            kv_indices,
            kv_start,
            stride_buf_vbs,
            IS_PREFIX=True,
            IS_K=False,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_DV,
            async_layout=v_async_layout,
        )

    @gluon.jit
    def for_extend_k(
        kt_smem,
        K_Extend,
        cur_seq_q_start_idx,
        cur_kv_head,
        stride_kbs,
        stride_kh,
        BLOCK_N: gl.constexpr,
        BLOCK_DMODEL: gl.constexpr,
        kt_async_layout: gl.constexpr,
    ):
        """Build a loader for the extend-K row-major tap (multi-slot)."""
        k_extend_base = (
            K_Extend + cur_seq_q_start_idx * stride_kbs + cur_kv_head * stride_kh
        )
        return AsyncKVLoader(
            kt_smem,
            k_extend_base,
            0,
            0,
            stride_kbs,
            IS_PREFIX=False,
            IS_K=True,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_DMODEL,
            async_layout=kt_async_layout,
        )

    @gluon.jit
    def for_extend_v(
        v_smem,
        V_Extend,
        cur_seq_q_start_idx,
        cur_kv_head,
        stride_vbs,
        stride_vh,
        BLOCK_N: gl.constexpr,
        BLOCK_DV: gl.constexpr,
        v_async_layout: gl.constexpr,
    ):
        """Build a loader for the extend-V row-major tap (multi-slot)."""
        v_extend_base = (
            V_Extend + cur_seq_q_start_idx * stride_vbs + cur_kv_head * stride_vh
        )
        return AsyncKVLoader(
            v_smem,
            v_extend_base,
            0,
            0,
            stride_vbs,
            IS_PREFIX=False,
            IS_K=False,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_DV,
            async_layout=v_async_layout,
        )

    @gluon.jit
    def issue(self, start_n, seq_len, slot=0):
        """Kick off one async-DMA load into ``self.smem.index(slot)``.

        Selects between the prefix (gathered via ``self.kv_indices``)
        and extend (row-major via ``start_n + offs_n``) taps based on
        the ``IS_PREFIX`` constexpr. Callers that need
        ``SKIP_BOUNDS_CHECK`` on the extend side use ``issue_nomask``.
        The ``slot`` kwarg selects which pipeline buffer the DMA writes
        to; defaults to 0 for single-slot (unpipelined) callers.
        """
        slot_smem = self.smem.index(slot)
        if self.IS_K:
            offs_d = gl.arange(
                0,
                self.BLOCK_D,
                layout=gl.SliceLayout(dim=1, parent=self.async_layout),
            )
            offs_n = gl.arange(
                0,
                self.BLOCK_N,
                layout=gl.SliceLayout(dim=0, parent=self.async_layout),
            )
            if self.IS_PREFIX:
                n_idx = start_n + offs_n
                mask_n = n_idx < seq_len
                kv_locs = gl.load(
                    self.kv_indices + self.kv_start + n_idx,
                    mask=mask_n,
                    other=0,
                ).to(tl.int32)
                kt_offsets = (offs_d[:, None] + kv_locs[None, :] * self.stride_n).to(
                    tl.int32
                )
                _buffer_load_to_shared_cast_safe(
                    slot_smem,
                    self.gbl_base,
                    kt_offsets,
                    mask=mask_n[None, :],
                    other=0.0,
                )
            else:
                kt_offsets = (
                    offs_d[:, None] + (start_n + offs_n[None, :]) * self.stride_n
                ).to(tl.int32)
                kt_mask = (start_n + offs_n[None, :]) < seq_len
                _buffer_load_to_shared_cast_safe(
                    slot_smem,
                    self.gbl_base,
                    kt_offsets,
                    mask=kt_mask,
                    other=0.0,
                )
        else:
            offs_n = gl.arange(
                0,
                self.BLOCK_N,
                layout=gl.SliceLayout(dim=1, parent=self.async_layout),
            )
            offs_d = gl.arange(
                0,
                self.BLOCK_D,
                layout=gl.SliceLayout(dim=0, parent=self.async_layout),
            )
            if self.IS_PREFIX:
                n_idx = start_n + offs_n
                mask_n = n_idx < seq_len
                kv_locs = gl.load(
                    self.kv_indices + self.kv_start + n_idx,
                    mask=mask_n,
                    other=0,
                ).to(tl.int32)
                v_offsets = (kv_locs[:, None] * self.stride_n + offs_d[None, :]).to(
                    tl.int32
                )
                _buffer_load_to_shared_cast_safe(
                    slot_smem,
                    self.gbl_base,
                    v_offsets,
                    mask=mask_n[:, None],
                    other=0.0,
                )
            else:
                v_offsets = (
                    (start_n + offs_n[:, None]) * self.stride_n + offs_d[None, :]
                ).to(tl.int32)
                v_mask = (start_n + offs_n[:, None]) < seq_len
                _buffer_load_to_shared_cast_safe(
                    slot_smem,
                    self.gbl_base,
                    v_offsets,
                    mask=v_mask,
                    other=0.0,
                )

    @gluon.jit
    def issue_nomask(self, start_n, seq_len, slot=0):
        """Extend-only issue that skips the bounds check (UNMASKED bulk).

        Bulk blocks are statically known to be fully in-range
        (``bulk_end = min(n_extend_blocks - masked_tail, ...)`` excludes
        the partial last block), so the per-element mask collapses to
        a compile-time ``None`` and the DMA can skip the predicate
        lanes entirely.
        """
        tl.static_assert(not self.IS_PREFIX, "issue_nomask is extend-only")
        slot_smem = self.smem.index(slot)
        if self.IS_K:
            offs_d_layout: gl.constexpr = gl.SliceLayout(
                dim=1, parent=self.async_layout
            )
            offs_n_layout: gl.constexpr = gl.SliceLayout(
                dim=0, parent=self.async_layout
            )
            offs_d = gl.arange(0, self.BLOCK_D, layout=offs_d_layout)
            offs_n = gl.arange(0, self.BLOCK_N, layout=offs_n_layout)
            kt_offsets = (
                offs_d[:, None] + (start_n + offs_n[None, :]) * self.stride_n
            ).to(tl.int32)
            _buffer_load_to_shared_cast_safe(
                slot_smem,
                self.gbl_base,
                kt_offsets,
                mask=None,
                other=0.0,
            )
        else:
            offs_n_layout: gl.constexpr = gl.SliceLayout(
                dim=1, parent=self.async_layout
            )
            offs_d_layout: gl.constexpr = gl.SliceLayout(
                dim=0, parent=self.async_layout
            )
            offs_n = gl.arange(0, self.BLOCK_N, layout=offs_n_layout)
            offs_d = gl.arange(0, self.BLOCK_D, layout=offs_d_layout)
            v_offsets = (
                (start_n + offs_n[:, None]) * self.stride_n + offs_d[None, :]
            ).to(tl.int32)
            _buffer_load_to_shared_cast_safe(
                slot_smem,
                self.gbl_base,
                v_offsets,
                mask=None,
                other=0.0,
            )

    @gluon.jit
    def issue_from_locs(
        self,
        kv_locs,
        mask_n_or_scalar,
        slot=0,
        IS_SCALAR_MASK: gl.constexpr = False,
    ):
        """Prefix-only: issue DMA using a pre-loaded ``kv_locs`` vector.

        Skips the ``gl.load(kv_indices + ...)`` step; the caller has
        already fetched the gather locations (e.g. the pingpong
        pipeline preloads ``kv_locs`` one iteration ahead). When
        ``IS_SCALAR_MASK`` is true, ``mask_n_or_scalar`` is a single
        scalar predicate instead of a per-element vector -- the
        WCA pingpong path can use this because
        ``pfx_seq_len`` is BLOCK_N-aligned, so per-element masking
        collapses to a scalar.
        """
        tl.static_assert(self.IS_PREFIX, "issue_from_locs is prefix-only")
        slot_smem = self.smem.index(slot)
        if IS_SCALAR_MASK:
            mask = mask_n_or_scalar
        else:
            mask = mask_n_or_scalar[None, :] if self.IS_K else mask_n_or_scalar[:, None]
        if self.IS_K:
            offs_d_layout: gl.constexpr = gl.SliceLayout(
                dim=1, parent=self.async_layout
            )
            offs_d = gl.arange(0, self.BLOCK_D, layout=offs_d_layout)
            kt_offsets = (offs_d[:, None] + kv_locs[None, :] * self.stride_n).to(
                tl.int32
            )
            _buffer_load_to_shared_cast_safe(
                slot_smem,
                self.gbl_base,
                kt_offsets,
                mask=mask,
                other=0.0,
            )
        else:
            offs_d_layout: gl.constexpr = gl.SliceLayout(
                dim=0, parent=self.async_layout
            )
            offs_d = gl.arange(0, self.BLOCK_D, layout=offs_d_layout)
            v_offsets = (kv_locs[:, None] * self.stride_n + offs_d[None, :]).to(
                tl.int32
            )
            _buffer_load_to_shared_cast_safe(
                slot_smem,
                self.gbl_base,
                v_offsets,
                mask=mask,
                other=0.0,
            )


# ===---------------------------------------------------------------------===#
# ExtendAttnProgram -- the orchestrator
# ===---------------------------------------------------------------------===#
#
# Bundles every tensor/stride/scalar/constexpr a single extend-attention
# tile needs. Schedule-specific program aggregates consume the same state
# through @composition, so shared helpers stay here while control flow lives
# with the schedule object.
#
#     state = ExtendAttnProgram.initialize(...)
#     pgm = ExtendAttnSerialProgram(state)
#     pgm.run()
#
# Private helpers (``_schedule_*``, ``_load_q``, ``_compute_*``,
# ``_apply_sinks``, ``_normalize_and_store``, ``_splitk_*``) factor the
# shared preamble/epilogue.
#
# Field layout mirrors the upstream ``GlobalScaledAttentionProgram`` +
# ``BlockScaledAttentionProgram`` pattern: a single config sub-aggregate
# holds every constexpr; ``@composition`` lets ``self.BLOCK_M`` etc
# resolve via ``cfg``.


@composition
@aggregate
class ExtendAttnProgram:
    cfg: ExtendAttnConfig

    # Input/output tensors.
    Q_Extend: gl.tensor
    K_Extend: gl.tensor
    V_Extend: gl.tensor
    O_Extend: gl.tensor
    K_Buffer: gl.tensor
    V_Buffer: gl.tensor

    # Ragged indexing tensors.
    qo_indptr: gl.tensor
    kv_indptr: gl.tensor
    kv_indices: gl.tensor

    # Mask / window tensors: real tensors only when the corresponding
    # constexpr gates say they can be read. Otherwise the dispatcher passes
    # ``0`` and the aggregate stores a ``gl.constexpr(0)`` placeholder.
    Mask: gl.tensor | gl.constexpr
    MaskIndptr: gl.tensor | gl.constexpr
    WindowKvOffsets: gl.tensor | gl.constexpr

    # Sinks may arrive as ``None`` from the dispatcher when HAS_SINK=False.
    # Union-typed so the wrapper can substitute ``gl.constexpr(0)`` and the
    # aggregate still accepts the assignment. Mirrors upstream
    # ``MXFPGEMMPipelinedProgram.a_scale_buffer: ... | gl.constexpr``.
    Sinks: gl.tensor | gl.constexpr

    # Strides (int32 registers). Triton specializes strides divisible by
    # 16 or equal to 1 to ``constexpr`` on the kernel boundary, so after
    # ``constexpr_function``'s unwrap these can arrive as raw Python ints
    # instead of tensors. Union-typed so the ``__init__`` can re-wrap
    # with ``gl.constexpr`` without failing the attribute check.
    stride_qbs: gl.tensor | gl.constexpr
    stride_qh: gl.tensor | gl.constexpr
    stride_kbs: gl.tensor | gl.constexpr
    stride_kh: gl.tensor | gl.constexpr
    stride_vbs: gl.tensor | gl.constexpr
    stride_vh: gl.tensor | gl.constexpr
    stride_obs: gl.tensor | gl.constexpr
    stride_oh: gl.tensor | gl.constexpr
    stride_buf_kbs: gl.tensor | gl.constexpr
    stride_buf_kh: gl.tensor | gl.constexpr
    stride_buf_vbs: gl.tensor | gl.constexpr
    stride_buf_vh: gl.tensor | gl.constexpr

    # Runtime scalars. Union-typed because the dispatcher passes Python
    # ``int`` / ``float`` values (e.g. ``kv_group_num = head_num //
    # H_KV``) that arrive at ``constexpr_function __init__`` unwrapped.
    # Arithmetic contexts (``cur_head // self.kv_group_num`` etc) still
    # auto-promote to tensors. Upstream mxfp-fa tags the same slot
    # ``ttgl.tensor`` but relies on the caller always passing tensors;
    # the current llvm limitation documented there also applies here.
    sm_scale: gl.tensor | gl.constexpr
    kv_group_num: gl.tensor | gl.constexpr
    v_scale: gl.tensor | gl.constexpr

    # WCA-loop / split-K workspace. When the data-centric non-WCA
    # path is taken (IS_WCA=False and SPLIT_K=1) the dispatcher
    # does not pass these kwargs; the wrapper substitutes
    # ``gl.constexpr(0)`` placeholders. Union-typed accordingly.
    num_heads: gl.tensor | gl.constexpr
    total_valid_tiles: gl.tensor | gl.constexpr
    total_programs: gl.tensor | gl.constexpr
    partial_out: gl.tensor | gl.constexpr
    partial_lse: gl.tensor | gl.constexpr
    tile_done: gl.tensor | gl.constexpr
    actual_batch_size: gl.tensor | gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
        cfg,
        Q_Extend,
        K_Extend,
        V_Extend,
        O_Extend,
        K_Buffer,
        V_Buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        Mask,
        MaskIndptr,
        WindowKvOffsets,
        Sinks,
        stride_qbs,
        stride_qh,
        stride_kbs,
        stride_kh,
        stride_vbs,
        stride_vh,
        stride_obs,
        stride_oh,
        stride_buf_kbs,
        stride_buf_kh,
        stride_buf_vbs,
        stride_buf_vh,
        sm_scale,
        kv_group_num,
        v_scale,
        num_heads,
        total_valid_tiles,
        total_programs,
        partial_out,
        partial_lse,
        tile_done,
        actual_batch_size,
    ):
        self.cfg = cfg
        self.Q_Extend = Q_Extend
        self.K_Extend = K_Extend
        self.V_Extend = V_Extend
        self.O_Extend = O_Extend
        self.K_Buffer = K_Buffer
        self.V_Buffer = V_Buffer
        self.qo_indptr = qo_indptr
        self.kv_indptr = kv_indptr
        self.kv_indices = kv_indices
        if cfg.USE_CUSTOM_MASK:
            self.Mask = Mask
            self.MaskIndptr = MaskIndptr
        else:
            self.Mask = gl.constexpr(Mask)
            self.MaskIndptr = gl.constexpr(MaskIndptr)
        if cfg.HAS_WINDOW_OFFSETS:
            self.WindowKvOffsets = WindowKvOffsets
        else:
            self.WindowKvOffsets = gl.constexpr(WindowKvOffsets)

        # Sinks: real bf16 tensor when HAS_SINK; ``gl.constexpr(0)``
        # placeholder otherwise. Guarding the wrap with ``cfg.HAS_SINK``
        # (rather than unconditionally wrapping) keeps the tensor path
        # free of a spurious ``constexpr(<tensor>)`` call, matching the
        # upstream pattern.
        if cfg.HAS_SINK:
            self.Sinks = Sinks
        else:
            self.Sinks = gl.constexpr(Sinks)

        # Strides. Triton sometimes specializes integer scalars divisible
        # by 16 as constexpr, which arrives here as a raw Python int after
        # the constexpr_function unwrap. ``_wrap_rt_scalar`` re-wraps
        # those back to ``gl.constexpr`` so the union-typed
        # aggregate field accepts the assignment; real tensors pass
        # through unchanged.
        self.stride_qbs = _wrap_rt_scalar(stride_qbs)
        self.stride_qh = _wrap_rt_scalar(stride_qh)
        self.stride_kbs = _wrap_rt_scalar(stride_kbs)
        self.stride_kh = _wrap_rt_scalar(stride_kh)
        self.stride_vbs = _wrap_rt_scalar(stride_vbs)
        self.stride_vh = _wrap_rt_scalar(stride_vh)
        self.stride_obs = _wrap_rt_scalar(stride_obs)
        self.stride_oh = _wrap_rt_scalar(stride_oh)
        self.stride_buf_kbs = _wrap_rt_scalar(stride_buf_kbs)
        self.stride_buf_kh = _wrap_rt_scalar(stride_buf_kh)
        self.stride_buf_vbs = _wrap_rt_scalar(stride_buf_vbs)
        self.stride_buf_vh = _wrap_rt_scalar(stride_buf_vh)
        self.sm_scale = _wrap_rt_scalar(sm_scale)
        self.kv_group_num = _wrap_rt_scalar(kv_group_num)
        self.v_scale = _wrap_rt_scalar(v_scale)

        # WCA metadata is needed for every WCA launch. Split-K workspace
        # tensors are real only when ``SPLIT_K > 1``; otherwise they stay
        # ``gl.constexpr(0)`` placeholders even on WCA.
        #
        # ``num_heads`` / ``total_valid_tiles`` / ``total_programs`` /
        # ``actual_batch_size`` are runtime Python ints from the
        # dispatcher, but Triton's argument specializer may promote
        # divisible-by-16 values to constexpr before dispatch; after the
        # constexpr_function unwrap we see them as plain ``int`` which
        # fails the union-typed field assignment. Re-wrap with
        # ``_wrap_rt_scalar`` (same trick used on strides above) so both
        # int and tensor forms flow through.
        if cfg.IS_WCA or cfg.XCD_REMAP:
            self.num_heads = _wrap_rt_scalar(num_heads)
            self.total_valid_tiles = _wrap_rt_scalar(total_valid_tiles)
            self.total_programs = _wrap_rt_scalar(total_programs)
            self.actual_batch_size = _wrap_rt_scalar(actual_batch_size)
        else:
            self.num_heads = gl.constexpr(num_heads)
            self.total_valid_tiles = gl.constexpr(total_valid_tiles)
            self.total_programs = gl.constexpr(total_programs)
            self.actual_batch_size = gl.constexpr(actual_batch_size)
        if cfg.SPLIT_K > 1:
            self.partial_out = partial_out
            self.partial_lse = partial_lse
            self.tile_done = tile_done
        else:
            self.partial_out = gl.constexpr(partial_out)
            self.partial_lse = gl.constexpr(partial_lse)
            self.tile_done = gl.constexpr(tile_done)

    # --------------------------------------------------------------------- #
    # Factory: one call site builds the whole aggregate.
    # --------------------------------------------------------------------- #

    @gluon.jit
    def initialize(
        cfg,
        Q_Extend,
        K_Extend,
        V_Extend,
        O_Extend,
        K_Buffer,
        V_Buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        Mask,
        MaskIndptr,
        WindowKvOffsets,
        Sinks,
        stride_qbs,
        stride_qh,
        stride_kbs,
        stride_kh,
        stride_vbs,
        stride_vh,
        stride_obs,
        stride_oh,
        stride_buf_kbs,
        stride_buf_kh,
        stride_buf_vbs,
        stride_buf_vh,
        sm_scale,
        kv_group_num,
        v_scale,
        num_heads,
        total_valid_tiles,
        total_programs,
        partial_out,
        partial_lse,
        tile_done,
        actual_batch_size,
    ):
        return ExtendAttnProgram(
            cfg,
            Q_Extend,
            K_Extend,
            V_Extend,
            O_Extend,
            K_Buffer,
            V_Buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            Mask,
            MaskIndptr,
            WindowKvOffsets,
            Sinks,
            stride_qbs,
            stride_qh,
            stride_kbs,
            stride_kh,
            stride_vbs,
            stride_vh,
            stride_obs,
            stride_oh,
            stride_buf_kbs,
            stride_buf_kh,
            stride_buf_vbs,
            stride_buf_vh,
            sm_scale,
            kv_group_num,
            v_scale,
            num_heads,
            total_valid_tiles,
            total_programs,
            partial_out,
            partial_lse,
            tile_done,
            actual_batch_size,
        )

    # --------------------------------------------------------------------- #
    # Tile scheduling (data-centric 3D grid vs WCA inline scan)
    # --------------------------------------------------------------------- #

    @gluon.jit
    def _schedule_data_centric(self):
        """Serial NS=1 + data-centric pipelined launches: program_id math only.

        The launch grid is rectangular ``(batch, heads, max_m_tiles)``.
        Every CTA owns exactly the tile named by its three program ids;
        invalid padding tiles are turned into zero-length work below.
        """
        cfg: gl.constexpr = self.cfg
        cur_seq = gl.program_id(0)
        cur_head = gl.program_id(1)
        cur_block_m = gl.program_id(2)
        if cfg.XCD_REMAP:
            seq_head_domain = (
                cur_seq - cur_seq + self.actual_batch_size
            ) * self.num_heads
            tile_idx = cur_seq + self.actual_batch_size * (
                cur_head + self.num_heads * cur_block_m
            )
            if cfg.XCD_MODE == 2:
                tile_idx = remap_xcd_attention(
                    tile_idx,
                    self.total_valid_tiles,
                    cfg.NUM_XCDS,
                )
            else:
                tile_idx = remap_xcd_chunked(
                    tile_idx,
                    self.total_valid_tiles,
                    cfg.NUM_XCDS,
                    cfg.XCD_CHUNK,
                )
            cur_block_m = tile_idx // seq_head_domain
            seq_head_idx = tile_idx - cur_block_m * seq_head_domain
            cur_head = seq_head_idx // self.actual_batch_size
            cur_seq = seq_head_idx - cur_head * self.actual_batch_size
        cur_kv_head = cur_head // self.kv_group_num

        cur_seq_q_start_idx = gl.load(self.qo_indptr + cur_seq)
        seq_len_extend = (
            gl.load(self.qo_indptr + cur_seq + 1) - cur_seq_q_start_idx
        ).to(tl.int32)
        is_valid_tile = cur_block_m * cfg.BLOCK_M < seq_len_extend
        seq_len_extend = tl.where(is_valid_tile, seq_len_extend, 0)
        cur_seq_kv_start_idx = gl.load(self.kv_indptr + cur_seq)
        seq_len_prefix_raw = (
            gl.load(self.kv_indptr + cur_seq + 1) - cur_seq_kv_start_idx
        ).to(tl.int32)
        seq_len_prefix = tl.where(is_valid_tile, seq_len_prefix_raw, 0)

        return (
            cur_seq,
            cur_head,
            cur_block_m,
            cur_kv_head,
            cur_seq_q_start_idx,
            seq_len_extend,
            cur_seq_kv_start_idx,
            seq_len_prefix,
            is_valid_tile,
            0,  # output_tile (unused for data-centric)
            0,  # k_split_id (unused for data-centric)
        )

    @gluon.jit
    def _schedule_wca(self, tile_idx):
        """WCA CTA: inline linear scan over qo_indptr resolves tile_idx.

        WCA is work-centric, not data-centric with a different grid shape:
        ``tile_idx`` names a slot in the compact WCA logical range. With
        ``SPLIT_K == 1`` a valid slot maps directly to ``output_tile``;
        with ``SPLIT_K > 1`` adjacent slots are K-splits of the same
        output tile. The caller's outer loop may pass several tile_idx
        values through one CTA by striding with ``total_programs``.

        With XCD_REMAP on, we first swizzle the incoming 1-D tile_idx so
        that consecutive tile_ids land on the same hardware XCD (MI350X
        has 8 XCDs x 32 CUs, each with 4 MB L2). The remap is a pure
        scheduling permutation -- math, accumulation order, and split-K
        partitioning are all unchanged. The remap happens *before* the
        SPLIT_K decomposition so that the SPLIT_K natural adjacency
        (splits of one output_tile are adjacent in tile_idx) survives
        the swizzle as one XCD-local chunk.
        """
        cfg: gl.constexpr = self.cfg

        if cfg.XCD_REMAP:
            if cfg.XCD_MODE == 2:
                tile_idx = remap_xcd_attention(
                    tile_idx,
                    self.total_valid_tiles,
                    cfg.NUM_XCDS,
                )
            else:
                tile_idx = remap_xcd_chunked(
                    tile_idx,
                    self.total_valid_tiles,
                    cfg.NUM_XCDS,
                    cfg.XCD_CHUNK,
                )

        if cfg.SPLIT_K > 1:
            output_tile = tile_idx // cfg.SPLIT_K
            k_split_id = tile_idx % cfg.SPLIT_K
        else:
            output_tile = tile_idx
            k_split_id = 0

        # WCA tile map: inline linear scan over qo_indptr to resolve
        # output_tile -> (seq, head, block_m). The grid is sized with a
        # tight upper bound on sum(ceil(ext_i/BM)), so some CTAs may claim
        # an over-provisioned slot for which no seq has enough tiles; the
        # scan walks off the end with found=0 and we mark the tile invalid.
        cur_seq = 0
        cum_tiles = 0
        found = 0
        while (cur_seq < self.actual_batch_size) & (found == 0):
            _s_start = gl.load(self.qo_indptr + cur_seq)
            _s_end = gl.load(self.qo_indptr + cur_seq + 1)
            s_ext = (_s_end - _s_start).to(tl.int32)
            s_tiles = (
                tl.maximum((s_ext + cfg.BLOCK_M - 1) // cfg.BLOCK_M, 0) * self.num_heads
            )
            next_cum = cum_tiles + s_tiles
            if next_cum > output_tile:
                found = 1
            else:
                cum_tiles = next_cum
                cur_seq = cur_seq + 1
        is_valid_tile = found == 1
        cur_seq = tl.minimum(cur_seq, self.actual_batch_size - 1)

        local_tile = output_tile - cum_tiles
        seq_ext_len = (
            gl.load(self.qo_indptr + cur_seq + 1) - gl.load(self.qo_indptr + cur_seq)
        ).to(tl.int32)
        tiles_per_head = tl.maximum((seq_ext_len + cfg.BLOCK_M - 1) // cfg.BLOCK_M, 1)
        cur_head = local_tile // tiles_per_head
        cur_block_m = local_tile % tiles_per_head
        cur_kv_head = cur_head // self.kv_group_num

        cur_seq_q_start_idx = gl.load(self.qo_indptr + cur_seq)
        seq_len_extend = tl.where(is_valid_tile, seq_ext_len, 0)
        cur_seq_kv_start_idx = gl.load(self.kv_indptr + cur_seq)
        seq_len_prefix_raw = (
            gl.load(self.kv_indptr + cur_seq + 1) - cur_seq_kv_start_idx
        ).to(tl.int32)
        seq_len_prefix = tl.where(is_valid_tile, seq_len_prefix_raw, 0)

        return (
            cur_seq,
            cur_head,
            cur_block_m,
            cur_kv_head,
            cur_seq_q_start_idx,
            seq_len_extend,
            cur_seq_kv_start_idx,
            seq_len_prefix,
            is_valid_tile,
            output_tile,
            k_split_id,
        )

    # --------------------------------------------------------------------- #
    # Per-tile preamble: mask state, Q load, softmax state, XAI, SWA, extend bounds
    # --------------------------------------------------------------------- #

    @gluon.jit
    def _compute_mask_state(self, cur_seq, seq_len_extend, seq_len_prefix):
        """Custom-mask base index / stride / kv-col offset derivation.

        Zero when USE_CUSTOM_MASK is off; the constexpr gate leaves the
        non-taken branch fully DCE'd.
        """
        cfg: gl.constexpr = self.cfg
        if cfg.USE_CUSTOM_MASK:
            mask_base_idx = gl.load(self.MaskIndptr + cur_seq).to(tl.int64)
            window_kv_offset = 0
            if cfg.SLIDING_WINDOW_SIZE > 0 and cfg.HAS_WINDOW_OFFSETS:
                window_kv_offset = gl.load(self.WindowKvOffsets + cur_seq)
            cur_seq_len = seq_len_prefix + seq_len_extend
            mask_row_stride = (cur_seq_len + window_kv_offset).to(tl.int64)
            mask_base_idx = mask_base_idx + window_kv_offset.to(tl.int64)
            mask_kv_col_offset = seq_len_prefix.to(tl.int64)
        else:
            mask_base_idx = tl.cast(0, tl.int64)
            mask_row_stride = tl.cast(0, tl.int64)
            mask_kv_col_offset = tl.cast(0, tl.int64)
        return mask_base_idx, mask_row_stride, mask_kv_col_offset

    @gluon.jit
    def _load_q(
        self, cur_seq_q_start_idx, cur_head, cur_block_m, seq_len_extend, offs_m, offs_d
    ):
        """Q load shared by serial and pipelined kernels.

        ``q_dot`` and the optional FP8 ``pfx_q`` are layout-converted here
        so callers don't repeat the convert_layout boilerplate.
        """
        cfg: gl.constexpr = self.cfg
        q_ptrs = (
            self.Q_Extend
            + (cur_seq_q_start_idx + cur_block_m * cfg.BLOCK_M + offs_m[:, None])
            * self.stride_qbs
            + cur_head * self.stride_qh
            + offs_d[None, :]
        )
        q_mask = (cur_block_m * cfg.BLOCK_M + offs_m[:, None]) < seq_len_extend
        q = gl.load(q_ptrs, mask=q_mask, other=0.0)
        q_dot = gl.convert_layout(q, cfg.layouts.q_dot_layout)
        if cfg.IS_FP8:
            pfx_q = gl.convert_layout(q.to(tl.float8e4nv), cfg.layouts.pfx_q_dot_layout)
        else:
            pfx_q = q_dot
        return q_dot, pfx_q

    @gluon.jit
    def _init_softmax(self):
        """m_i (-inf), l_i (1.0), acc (zeros) in the MMA layouts."""
        cfg: gl.constexpr = self.cfg
        m_i = gl.full(
            [cfg.BLOCK_M],
            float("-inf"),
            dtype=gl.float32,
            layout=cfg.layouts.mma_m_layout,
        )
        l_i = gl.full(
            [cfg.BLOCK_M], 1.0, dtype=gl.float32, layout=cfg.layouts.mma_m_layout
        )
        acc = gl.zeros(
            [cfg.BLOCK_M, cfg.BLOCK_DV], dtype=gl.float32, layout=cfg.layouts.mma_layout
        )
        return m_i, l_i, acc

    @gluon.jit
    def _compute_q_abs_pos_and_xai(self, seq_len_prefix, cur_block_m):
        """q_abs_pos + optional XAI temperature register vector."""
        cfg: gl.constexpr = self.cfg
        q_abs_pos = (
            seq_len_prefix
            + cur_block_m * cfg.BLOCK_M
            + gl.arange(0, cfg.BLOCK_M, layout=cfg.layouts.mma_offs_m_row)
        )
        if cfg.XAI_TEMPERATURE_LEN > 0:
            inv_log2_len = 1.0 / tl.log2(float(cfg.XAI_TEMPERATURE_LEN))
            xai_temperature_reg = gl.where(
                q_abs_pos > cfg.XAI_TEMPERATURE_LEN,
                tl.log2(q_abs_pos.to(gl.float32)) * inv_log2_len,
                gl.full(
                    [cfg.BLOCK_M],
                    1.0,
                    dtype=gl.float32,
                    layout=cfg.layouts.mma_offs_m_row,
                ),
            )
        else:
            xai_temperature_reg = gl.full(
                [cfg.BLOCK_M],
                1.0,
                dtype=gl.float32,
                layout=cfg.layouts.mma_offs_m_row,
            )
        return q_abs_pos, xai_temperature_reg

    @gluon.jit
    def _compute_swa_skip(
        self, cur_seq_kv_start_idx, seq_len_prefix, cur_block_m, q_abs_pos
    ):
        """SWA prefix skip: jump past prefix tiles entirely outside the window.

        For the M-tile, min q_abs_pos = seq_len_prefix + cur_block_m * BLOCK_M.
        Any prefix block whose last key position < (min_q - SWS) is fully masked.
        """
        cfg: gl.constexpr = self.cfg
        pfx_kv_start = cur_seq_kv_start_idx
        pfx_seq_len = seq_len_prefix
        pfx_q_abs_pos = q_abs_pos
        if cfg.SLIDING_WINDOW_SIZE > 0:
            q_min_abs = seq_len_prefix + cur_block_m * cfg.BLOCK_M
            first_useful_pos = tl.maximum(q_min_abs - cfg.SLIDING_WINDOW_SIZE, 0)
            prefix_skip_n = (first_useful_pos // cfg.BLOCK_N) * cfg.BLOCK_N
            pfx_kv_start = cur_seq_kv_start_idx + prefix_skip_n
            pfx_seq_len = seq_len_prefix - prefix_skip_n
            pfx_q_abs_pos = q_abs_pos - prefix_skip_n
        return pfx_kv_start, pfx_seq_len, pfx_q_abs_pos

    @gluon.jit
    def _apply_splitk_prefix_partition(
        self, pfx_kv_start, pfx_seq_len, pfx_q_abs_pos, k_split_id, seq_len_extend
    ):
        """Select this CTA's prefix slice for WCA split-K.

        ``SPLIT_K > 1`` partitions prefix KV blocks across WCA CTAs. All
        CTAs but the last one return ``seq_len_extend=0`` so only the
        terminal CTA executes the extend phase and split-K reduction.
        """
        cfg: gl.constexpr = self.cfg
        if cfg.IS_WCA and cfg.SPLIT_K > 1:
            n_pfx_blocks = (pfx_seq_len + cfg.BLOCK_N - 1) // cfg.BLOCK_N
            blocks_per_split = (n_pfx_blocks + cfg.SPLIT_K - 1) // cfg.SPLIT_K
            my_block_start = k_split_id * blocks_per_split
            my_block_end = tl.minimum((k_split_id + 1) * blocks_per_split, n_pfx_blocks)
            split_start_offset = my_block_start * cfg.BLOCK_N
            pfx_kv_start = pfx_kv_start + split_start_offset
            pfx_seq_len = (
                tl.minimum(my_block_end * cfg.BLOCK_N, pfx_seq_len) - split_start_offset
            )
            pfx_seq_len = tl.maximum(pfx_seq_len, 0)
            pfx_q_abs_pos = pfx_q_abs_pos - split_start_offset
            if k_split_id < cfg.SPLIT_K - 1:
                seq_len_extend = 0
        return pfx_kv_start, pfx_seq_len, pfx_q_abs_pos, seq_len_extend

    @gluon.jit
    def _compute_extend_bounds(self, seq_len_extend, cur_block_m, EXT_N: gl.constexpr):
        """Causal/SWA clamp + unmasked-bulk / masked-tail split.

        Returns (effective_end, n_extend_blocks, n_full_blocks,
        swa_skip_n_blocks). The pre-port kernel inlined this in each
        pipelined kernel with the same shape; only the _EXT_N
        constexpr (4w vs 8w) differed.
        """
        cfg: gl.constexpr = self.cfg
        if cfg.IS_CAUSAL:
            causal_kv_end = (cur_block_m + 1) * cfg.BLOCK_M
            effective_end = tl.minimum(seq_len_extend, causal_kv_end)
        else:
            effective_end = seq_len_extend
        n_extend_blocks = (effective_end + EXT_N - 1) // EXT_N

        if cfg.BLOCK_DMODEL >= 256 or cfg.USE_CUSTOM_MASK:
            # One combined dispatch covers the whole range with per-step masking.
            n_full_blocks = 0
        elif cfg.SLIDING_WINDOW_SIZE > 0 and (
            cfg.IS_FP8 or effective_end > cfg.SLIDING_WINDOW_SIZE
        ):
            # Mixed SWA coverage: fall back to masked dispatch for all blocks.
            n_full_blocks = 0
        else:
            partial_block = ((effective_end % EXT_N) != 0).to(tl.int32)
            if cfg.IS_CAUSAL:
                masked_blocks = ((cfg.BLOCK_M + EXT_N - 1) // EXT_N) + partial_block
            else:
                masked_blocks = partial_block
            masked_blocks = tl.minimum(masked_blocks, n_extend_blocks)
            n_full_blocks = n_extend_blocks - masked_blocks

        if cfg.SLIDING_WINDOW_SIZE > 0 and cfg.IS_CAUSAL:
            swa_first_useful = tl.maximum(
                cur_block_m * cfg.BLOCK_M - cfg.SLIDING_WINDOW_SIZE, 0
            )
            swa_skip_n_blocks = swa_first_useful // EXT_N
        else:
            swa_skip_n_blocks = 0

        return effective_end, n_extend_blocks, n_full_blocks, swa_skip_n_blocks

    # --------------------------------------------------------------------- #
    # Online softmax: scale + optional mask + exp2 + stream-update of
    # (acc, l_i, m_i). The prefix variant and the fused extend variant
    # both take ``qk`` as the input tensor; the extend split variants
    # let the caller overlap vector ALU work with V DMA.
    # --------------------------------------------------------------------- #

    @gluon.jit
    def compute_softmax_prefix(
        self,
        acc,
        l_i,
        m_i,
        qk,
        start_n,
        seq_len_prefix,
        qk_scale,
        xai_temperature_reg,
        q_abs_pos,
        BLOCK_N: gl.constexpr,
        ENABLE_PREFIX_UNMASKED: gl.constexpr,
    ):
        """Prefix-phase online softmax.

        Prefix tokens are only gated by the sequence-length bound and,
        when ``cfg.SLIDING_WINDOW_SIZE > 0``, the sliding window.
        Custom-mask application lives in the extend phase.
        ``LOGIT_CAP > 0`` enables Gemma-style logit capping via the
        log2-space sigmoid identity (``tanh(x) = 2*sig(2x) - 1``); the
        branch folds away at compile time when capping is off.
        """
        cfg: gl.constexpr = self.cfg
        qk_scaled = qk * qk_scale
        if cfg.LOGIT_CAP > 0:
            log2_cap: gl.constexpr = cfg.LOGIT_CAP * LOG2E
            inv_cap: gl.constexpr = 2.0 / cfg.LOGIT_CAP
            e_neg = tl.math.exp2(-qk_scaled * inv_cap)
            sig = 1.0 / (1.0 + e_neg)
            qk_scaled = log2_cap * (2.0 * sig - 1.0)
        if cfg.XAI_TEMPERATURE_LEN > 0:
            qk_scaled = qk_scaled * xai_temperature_reg[:, None]
        bound_offs = start_n + gl.arange(0, BLOCK_N, layout=cfg.layouts.mma_offs_n_col)
        is_partial_tail = (start_n + BLOCK_N) > seq_len_prefix
        if cfg.SLIDING_WINDOW_SIZE > 0:
            swa_safe = tl.max(q_abs_pos) <= start_n + cfg.SLIDING_WINDOW_SIZE
        else:
            swa_safe = True
        use_unmasked_path = (
            ENABLE_PREFIX_UNMASKED and swa_safe and (not is_partial_tail)
        )

        if use_unmasked_path:
            m_ij = nan_propagating_max(qk_scaled, axis=1)
            m_new = gl.maximum(m_i, m_ij, propagate_nan=tl.PropagateNan.ALL)
            p = gl.exp2(qk_scaled - m_new[:, None])
        else:
            bound_mask = (q_abs_pos[:, None] >= 0) & (
                bound_offs[None, :] < seq_len_prefix
            )
            if cfg.SLIDING_WINDOW_SIZE > 0:
                bound_mask = bound_mask & (
                    q_abs_pos[:, None] <= bound_offs[None, :] + cfg.SLIDING_WINDOW_SIZE
                )
            qk_scaled = gl.where(
                bound_mask,
                qk_scaled,
                gl.full(
                    [cfg.BLOCK_M, BLOCK_N],
                    float("-inf"),
                    dtype=gl.float32,
                    layout=cfg.layouts.mma_layout,
                ),
            )

            m_ij = nan_propagating_max(qk_scaled, axis=1)
            m_new = gl.maximum(m_i, m_ij, propagate_nan=tl.PropagateNan.ALL)
            if cfg.SLIDING_WINDOW_SIZE > 0:
                m_new = gl.maximum(
                    m_new,
                    gl.full(
                        [cfg.BLOCK_M],
                        -1e20,
                        dtype=gl.float32,
                        layout=gl.SliceLayout(dim=1, parent=cfg.layouts.mma_layout),
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
    def softmax_part0(
        self,
        m_i,
        qk,
        start_n,
        cur_block_m,
        seq_len_extend,
        qk_scale,
        xai_temperature_reg,
        mask_base_idx,
        mask_row_stride,
        mask_kv_col_offset,
        BLOCK_N: gl.constexpr,
        MASK_STEPS: gl.constexpr,
    ):
        """Prepare extend probabilities before the V wait.

        Produces ``(p, alpha, m_new)`` without touching ``acc`` or ``l_i``.
        This is the QK-side half of online softmax: scale, apply optional
        masks/logit transforms, reduce max, and compute exp2/alpha while
        V DMA is still in flight.
        """
        cfg: gl.constexpr = self.cfg
        qk_scaled = qk * qk_scale
        if cfg.LOGIT_CAP > 0:
            log2_cap: gl.constexpr = cfg.LOGIT_CAP * LOG2E
            inv_cap: gl.constexpr = 2.0 / cfg.LOGIT_CAP
            e_neg = tl.math.exp2(-qk_scaled * inv_cap)
            sig = 1.0 / (1.0 + e_neg)
            qk_scaled = log2_cap * (2.0 * sig - 1.0)
        if cfg.XAI_TEMPERATURE_LEN > 0:
            qk_scaled = qk_scaled * xai_temperature_reg[:, None]
        if MASK_STEPS:
            bound_offs = start_n + gl.arange(
                0, BLOCK_N, layout=cfg.layouts.mma_offs_n_col
            )
            q_offs = cur_block_m * cfg.BLOCK_M + gl.arange(
                0, cfg.BLOCK_M, layout=cfg.layouts.mma_offs_m_row
            )
            valid_mask = q_offs[:, None] < seq_len_extend
            valid_mask = valid_mask & (bound_offs[None, :] < seq_len_extend)
            if cfg.USE_CUSTOM_MASK:
                mask_ptrs = (
                    self.Mask
                    + mask_base_idx
                    + q_offs[:, None] * mask_row_stride
                    + mask_kv_col_offset
                    + start_n
                    + gl.arange(0, BLOCK_N, layout=cfg.layouts.mma_offs_n_col)[None, :]
                )
                mask_vals = gl.load(mask_ptrs, mask=valid_mask, other=0)
                valid_mask = valid_mask & (mask_vals != 0)
            elif cfg.IS_CAUSAL:
                valid_mask = valid_mask & (q_offs[:, None] >= bound_offs[None, :])
            if cfg.SLIDING_WINDOW_SIZE > 0:
                valid_mask = valid_mask & (
                    q_offs[:, None] <= bound_offs[None, :] + cfg.SLIDING_WINDOW_SIZE
                )
            qk_scaled = gl.where(
                valid_mask,
                qk_scaled,
                gl.full(
                    [cfg.BLOCK_M, BLOCK_N],
                    float("-inf"),
                    dtype=gl.float32,
                    layout=cfg.layouts.mma_layout,
                ),
            )

            m_ij = nan_propagating_max(qk_scaled, axis=1)
            m_new = gl.maximum(m_i, m_ij, propagate_nan=tl.PropagateNan.ALL)
            if cfg.SLIDING_WINDOW_SIZE > 0 or cfg.USE_CUSTOM_MASK:
                m_new = gl.maximum(
                    m_new,
                    gl.full(
                        [cfg.BLOCK_M],
                        -1e20,
                        dtype=gl.float32,
                        layout=gl.SliceLayout(dim=1, parent=cfg.layouts.mma_layout),
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
    def softmax_part1(self, acc, l_i, p, alpha, m_new):
        """Finish the extend online-softmax update after the V wait.

        Row-sums ``p``, rescales ``acc`` / ``l_i``, and returns the new
        ``m_i`` so the caller can immediately feed ``p`` into the P*V MMA.
        """
        l_ij = gl.sum(p, axis=1)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        m_i = m_new
        return acc, l_i, m_i

    @gluon.jit
    def compute_softmax_extend(
        self,
        acc,
        l_i,
        m_i,
        qk,
        start_n,
        cur_block_m,
        seq_len_extend,
        qk_scale,
        xai_temperature_reg,
        mask_base_idx,
        mask_row_stride,
        mask_kv_col_offset,
        BLOCK_N: gl.constexpr,
        MASK_STEPS: gl.constexpr,
    ):
        """Fused extend online-softmax step: prepare then finish.

        Returns ``(acc, l_i, m_i, p)``. Used by callers that do not need
        the split form for warp-pipelined overlap between QK MMA and V DMA.
        """
        p, alpha, m_new = self.softmax_part0(
            m_i,
            qk,
            start_n,
            cur_block_m,
            seq_len_extend,
            qk_scale,
            xai_temperature_reg,
            mask_base_idx,
            mask_row_stride,
            mask_kv_col_offset,
            BLOCK_N,
            MASK_STEPS,
        )
        acc, l_i, m_i = self.softmax_part1(acc, l_i, p, alpha, m_new)
        return acc, l_i, m_i, p

    # --------------------------------------------------------------------- #
    # Prefix-phase inner loops (one block of KV at a time, gathered via
    # kv_indices). Each method reads the base pointers, strides, and
    # constexpr shape/feature gates off ``self`` / ``self.cfg``, and
    # takes the (kt_smem, v_smem, kt_async_layout, v_async_layout)
    # bundle via the ``KVSmemBank`` ``bank`` arg. DMAs go through
    # ``AsyncKVLoader`` so the hot-loop call is a 2-3 arg issue.
    # --------------------------------------------------------------------- #

    @gluon.jit
    def attn_fwd_inner_prefix_unpipelined(
        self,
        acc,
        l_i,
        m_i,
        q_dot,
        kv_start,
        cur_kv_head,
        seq_len_prefix,
        bank,
        qk_scale,
        xai_temperature_reg,
        q_abs_pos,
        block_start=0,
    ):
        """Serial prefix inner: one-slot synchronous-ish DMA loop.

        Used on its own by the 4w program's ``use_pipe_prefix=False`` branch
        and the 8w program's serial-prefix branch, and as the masked-tail
        helper for both pipelined variants (slot 0 is free to reuse
        after the pipeline drains).
        """
        cfg: gl.constexpr = self.cfg
        n_prefix_blocks = (seq_len_prefix + cfg.BLOCK_N - 1) // cfg.BLOCK_N

        kt_loader = AsyncKVLoader.for_prefix_k(
            bank.kt_smem,
            self.K_Buffer,
            cur_kv_head,
            self.kv_indices,
            kv_start,
            self.stride_buf_kbs,
            self.stride_buf_kh,
            cfg.BLOCK_N,
            cfg.BLOCK_DMODEL,
            bank.kt_async_layout,
        )
        v_loader = AsyncKVLoader.for_prefix_v(
            bank.v_smem,
            self.V_Buffer,
            cur_kv_head,
            self.kv_indices,
            kv_start,
            self.stride_buf_vbs,
            self.stride_buf_vh,
            cfg.BLOCK_N,
            cfg.BLOCK_DV,
            bank.v_async_layout,
        )

        for block_n in tl.range(block_start, n_prefix_blocks):
            start_n = block_n * cfg.BLOCK_N

            kt_loader.issue(start_n, seq_len_prefix)
            v_loader.issue(start_n, seq_len_prefix)

            # K and V are issued back-to-back. Keep V in flight while the
            # QK MMA runs, then drain it before the P*V MMA below.
            cdna4_async.wait_group(1)
            kt_dot = cdna4_async.load_shared_relaxed(
                bank.kt_smem.index(0),
                cfg.layouts.pfx_kt_dot_layout,
            )

            qk = gl.zeros(
                [cfg.BLOCK_M, cfg.BLOCK_N],
                dtype=gl.float32,
                layout=cfg.layouts.mma_layout,
            )
            qk = do_mma(q_dot, kt_dot, qk)

            acc, l_i, m_i, p = self.compute_softmax_prefix(
                acc,
                l_i,
                m_i,
                qk,
                start_n,
                seq_len_prefix,
                qk_scale,
                xai_temperature_reg,
                q_abs_pos,
                cfg.BLOCK_N,
                cfg.ENABLE_PREFIX_UNMASKED,
            )

            cdna4_async.wait_group(0)
            v_dot = cdna4_async.load_shared_relaxed(
                bank.v_smem.index(0),
                cfg.layouts.pfx_v_dot_layout,
            )
            p_cast = p.to(v_dot.dtype)
            p_dot = gl.convert_layout(p_cast, cfg.layouts.pfx_p_dot_layout)
            acc = do_mma(p_dot, v_dot, acc)

        return acc, l_i, m_i

    # --------------------------------------------------------------------- #
    # Extend-phase inner loops. Same pattern as prefix but row-major
    # (no kv_indices gather). Each fused outer wrapper picks between
    # the pipelined block-span helper and the unpipelined preload-all
    # fallback depending on block count vs NUM_STAGES. Callers pass
    # the post-transition ``bank`` (which now carries the extend-phase
    # async layouts) plus the ragged cursor for base-pointer math.
    # --------------------------------------------------------------------- #

    @gluon.jit
    def _attn_fwd_inner_extend_unpipelined_block_span(
        self,
        acc,
        l_i,
        m_i,
        q_dot,
        cur_seq_q_start_idx,
        cur_kv_head,
        cur_block_m,
        seq_len_extend,
        block_start,
        block_end,
        bank,
        qk_scale,
        xai_temperature_reg,
        mask_base_idx,
        mask_row_stride,
        mask_kv_col_offset,
        EXT_N: gl.constexpr,
        MASK_STEPS: gl.constexpr,
        SKIP_BOUNDS_CHECK: gl.constexpr = False,
    ):
        """Preload one half-open block span into SMEM, then compute.

        Processes ``[block_start, block_end)``. Requires
        ``block_end - block_start <= NUM_STAGES`` so each block
        gets its own buffer. ``EXT_N`` may differ from ``cfg.BLOCK_N``
        (FP8 uses a shrunken block during the extend phase).
          * Phase 1: issue K[i]->smem[i] and V[i]->smem[i] for all i.
          * Phase 2: wait_group(0) -- everything fully drained.
          * Phase 3: compute QK.softmax.PV block-by-block from resident SMEM.
        """
        cfg: gl.constexpr = self.cfg

        kt_loader = AsyncKVLoader.for_extend_k(
            bank.kt_smem,
            self.K_Extend,
            cur_seq_q_start_idx,
            cur_kv_head,
            self.stride_kbs,
            self.stride_kh,
            EXT_N,
            cfg.BLOCK_DMODEL,
            bank.kt_async_layout,
        )
        v_loader = AsyncKVLoader.for_extend_v(
            bank.v_smem,
            self.V_Extend,
            cur_seq_q_start_idx,
            cur_kv_head,
            self.stride_vbs,
            self.stride_vh,
            EXT_N,
            cfg.BLOCK_DV,
            bank.v_async_layout,
        )

        cdna4_async.wait_group(0)

        # Phase 1: bulk-issue all DMAs.
        n_local_blocks = block_end - block_start
        for local_idx in tl.range(0, n_local_blocks):
            buf_idx = local_idx.to(gl.int32)
            start_n = (block_start + local_idx) * EXT_N
            if SKIP_BOUNDS_CHECK:
                kt_loader.issue_nomask(start_n, seq_len_extend, slot=buf_idx)
                v_loader.issue_nomask(start_n, seq_len_extend, slot=buf_idx)
            else:
                kt_loader.issue(start_n, seq_len_extend, slot=buf_idx)
                v_loader.issue(start_n, seq_len_extend, slot=buf_idx)

        # Phase 2: drain every outstanding DMA.
        cdna4_async.wait_group(0)

        # Phase 3: compute from fully-resident SMEM (no DMA during this loop).
        for local_idx in tl.range(0, n_local_blocks):
            buf_idx = local_idx.to(gl.int32)
            start_n = (block_start + local_idx) * EXT_N

            kt_dot = cdna4_async.load_shared_relaxed(
                bank.kt_smem.index(buf_idx),
                cfg.layouts.kt_dot_layout,
            )

            qk = gl.zeros(
                [cfg.BLOCK_M, EXT_N], dtype=gl.float32, layout=cfg.layouts.mma_layout
            )
            qk = do_mma(q_dot, kt_dot, qk)

            acc, l_i, m_i, p = self.compute_softmax_extend(
                acc,
                l_i,
                m_i,
                qk,
                start_n,
                cur_block_m,
                seq_len_extend,
                qk_scale,
                xai_temperature_reg,
                mask_base_idx,
                mask_row_stride,
                mask_kv_col_offset,
                EXT_N,
                MASK_STEPS,
            )

            v_dot = cdna4_async.load_shared_relaxed(
                bank.v_smem.index(buf_idx),
                cfg.layouts.v_dot_layout,
            )
            p_cast = p.to(v_dot.dtype)
            p_dot = gl.convert_layout(p_cast, cfg.layouts.p_dot_layout)
            acc = do_mma(p_dot, v_dot, acc)

        return acc, l_i, m_i

    # --------------------------------------------------------------------- #
    # Epilogue: sinks + normalize/store + split-K reduce
    # --------------------------------------------------------------------- #

    @gluon.jit
    def _apply_sinks(self, cur_head, l_i, m_i):
        """Augment l_i with the sink token: l_i += 2^(sink * log2(e) - m_i)."""
        cfg: gl.constexpr = self.cfg
        if cfg.HAS_SINK:
            cur_sink = gl.load(self.Sinks + cur_head)
            l_i = l_i + gl.exp2(cur_sink * LOG2E - m_i)
        return l_i

    @gluon.jit
    def _normalize_and_store(
        self,
        cur_seq_q_start_idx,
        cur_head,
        cur_block_m,
        seq_len_extend,
        acc,
        l_i,
        offs_m,
        offs_dv,
    ):
        """Data-centric path (no split-K): normalize + v_scale + buffer_store output."""
        cfg: gl.constexpr = self.cfg
        l_recip = 1.0 / l_i
        acc = acc * l_recip[:, None]
        acc = acc * self.v_scale

        o_base = (
            self.O_Extend
            + cur_seq_q_start_idx * self.stride_obs
            + cur_head * self.stride_oh
        )
        o_offsets = (
            (cur_block_m * cfg.BLOCK_M + offs_m[:, None]) * self.stride_obs
            + offs_dv[None, :]
        ).to(tl.int32)
        o_mask = (cur_block_m * cfg.BLOCK_M + offs_m[:, None]) < seq_len_extend
        out = gl.convert_layout(acc, cfg.layouts.blocked_layout).to(
            self.O_Extend.dtype.element_ty
        )
        cdna4_buffer_store(out, o_base, o_offsets, mask=o_mask)

    @gluon.jit
    def _splitk_partial_store_and_reduce(
        self,
        output_tile,
        k_split_id,
        cur_seq_q_start_idx,
        cur_head,
        cur_block_m,
        orig_seq_len_extend,
        acc,
        l_i,
        m_i,
        offs_m,
        offs_dv,
    ):
        """Split-K partial store + atomic-counter-gated final reduce.

        The partial store always runs. The final reduction runs only on
        the CTA whose ``atomic_add`` returns ``SPLIT_K - 1`` (i.e. the
        last split to finish); that CTA owns the final output write.
        """
        cfg: gl.constexpr = self.cfg

        l_recip_sk = 1.0 / l_i
        acc_normed = acc * l_recip_sk[:, None]
        lse = m_i + tl.log2(l_i)
        split_idx = output_tile * cfg.SPLIT_K + k_split_id

        po_base = self.partial_out + split_idx * cfg.BLOCK_M * cfg.BLOCK_DV
        po_ptrs = po_base + offs_m[:, None] * cfg.BLOCK_DV + offs_dv[None, :]
        po_mask = (cur_block_m * cfg.BLOCK_M + offs_m[:, None]) < orig_seq_len_extend
        po_val = gl.convert_layout(acc_normed, cfg.layouts.blocked_layout)
        gl.store(po_ptrs, po_val, mask=po_mask)

        pl_base = self.partial_lse + split_idx * cfg.BLOCK_M
        pl_ptrs = pl_base + offs_m
        pl_mask = (cur_block_m * cfg.BLOCK_M + offs_m) < orig_seq_len_extend
        lse_val = gl.convert_layout(lse, cfg.layouts.offs_m_layout)
        gl.store(pl_ptrs, lse_val, mask=pl_mask)

        done = tl.atomic_add(self.tile_done + output_tile, 1)
        if done == cfg.SPLIT_K - 1:
            r_m_mask = (cur_block_m * cfg.BLOCK_M + offs_m) < orig_seq_len_extend

            r_base_0 = output_tile * cfg.SPLIT_K
            r_lse = gl.load(
                self.partial_lse + r_base_0 * cfg.BLOCK_M + offs_m,
                mask=r_m_mask,
                other=float("-inf"),
            )
            r_acc = gl.load(
                self.partial_out
                + r_base_0 * cfg.BLOCK_M * cfg.BLOCK_DV
                + offs_m[:, None] * cfg.BLOCK_DV
                + offs_dv[None, :],
                mask=r_m_mask[:, None],
                other=0.0,
            )
            for _sk in tl.static_range(1, cfg.SPLIT_K):
                r_base_k = r_base_0 + _sk
                lse_k = gl.load(
                    self.partial_lse + r_base_k * cfg.BLOCK_M + offs_m,
                    mask=r_m_mask,
                    other=float("-inf"),
                )
                acc_k = gl.load(
                    self.partial_out
                    + r_base_k * cfg.BLOCK_M * cfg.BLOCK_DV
                    + offs_m[:, None] * cfg.BLOCK_DV
                    + offs_dv[None, :],
                    mask=r_m_mask[:, None],
                    other=0.0,
                )
                max_lse = gl.maximum(r_lse, lse_k)
                w_old = gl.exp2(r_lse - max_lse)
                w_new = gl.exp2(lse_k - max_lse)
                denom = w_old + w_new
                r_acc = (r_acc * w_old[:, None] + acc_k * w_new[:, None]) / denom[
                    :, None
                ]
                r_lse = max_lse + tl.log2(denom)

            r_acc = r_acc * self.v_scale
            r_o_base = (
                self.O_Extend
                + cur_seq_q_start_idx * self.stride_obs
                + cur_head * self.stride_oh
            )
            r_o_offsets = (
                (cur_block_m * cfg.BLOCK_M + offs_m[:, None]) * self.stride_obs
                + offs_dv[None, :]
            ).to(tl.int32)
            r_o_mask = r_m_mask[:, None]
            out_r = gl.convert_layout(r_acc, cfg.layouts.blocked_layout).to(
                self.O_Extend.dtype.element_ty
            )
            cdna4_buffer_store(out_r, r_o_base, r_o_offsets, mask=r_o_mask)


# ===---------------------------------------------------------------------===#
# Schedule-specific program wrappers
# ===---------------------------------------------------------------------===#


@composition
@aggregate
class ExtendAttnSerialProgram:
    state: ExtendAttnProgram

    @gluon.constexpr_function
    def __init__(self, state):
        self.state = state

    @gluon.jit
    def run(self):
        """Body of ``gluon_extend_attn_serial_fwd`` (NS=1, NW in {2, 4}).

        Serial kernel: single CTA per (seq, head, m_tile),
        synchronous smem, no async DMA, no masked-tail split. Always
        launched as a 3D data-centric grid; there is no WCA variant.
        """
        cfg: gl.constexpr = self.cfg
        tl.static_assert(cfg.NUM_STAGES == 1, "serial program requires NUM_STAGES=1")
        tl.static_assert(
            cfg.num_warps == 2 or cfg.num_warps == 4,
            "serial program serves NW in {2, 4}; NW>=8 routes to 8w pingpong",
        )
        tl.static_assert(
            cfg.BLOCK_DMODEL == 64
            or cfg.BLOCK_DMODEL == 128
            or cfg.BLOCK_DMODEL == 256,
            "BLOCK_DMODEL must be in {64, 128, 256}",
        )
        tl.static_assert(
            not cfg.IS_FP8 or cfg.BLOCK_DMODEL != 256,
            "FP8 D=256 unsupported (MFMA_F8 gfx950 codegen failure)",
        )

        qk_scale = self.sm_scale * LOG2E

        # Tile coords (data-centric only for serial).
        (
            cur_seq,
            cur_head,
            cur_block_m,
            cur_kv_head,
            cur_seq_q_start_idx,
            seq_len_extend,
            cur_seq_kv_start_idx,
            seq_len_prefix,
            is_valid_tile,
            _,
            _,
        ) = self._schedule_data_centric()

        offs_m = gl.arange(0, cfg.BLOCK_M, layout=cfg.layouts.offs_m_layout)
        offs_d = gl.arange(0, cfg.BLOCK_DMODEL, layout=cfg.layouts.offs_d_layout)
        offs_dv = gl.arange(0, cfg.BLOCK_DV, layout=cfg.layouts.offs_d_layout)

        kt_blocked_layout: gl.constexpr = ExtendAttentionLayouts.make_serial_kt_blocked(
            cfg.num_warps
        )
        kt_prefix_smem = gl.allocate_shared_memory(
            cfg.layouts.PFX_SMEM_TY,
            [cfg.BLOCK_DMODEL, cfg.BLOCK_N],
            layout=SERIAL_KT_SMEM,
        )
        v_prefix_smem = gl.allocate_shared_memory(
            cfg.layouts.PFX_SMEM_TY,
            [cfg.BLOCK_N, cfg.BLOCK_DV],
            layout=SERIAL_V_SMEM,
        )
        if cfg.IS_FP8:
            kt_extend_smem = gl.allocate_shared_memory(
                self.Q_Extend.dtype.element_ty,
                [cfg.BLOCK_DMODEL, cfg.BLOCK_N],
                layout=SERIAL_KT_SMEM,
            )
            v_extend_smem = gl.allocate_shared_memory(
                self.Q_Extend.dtype.element_ty,
                [cfg.BLOCK_N, cfg.BLOCK_DV],
                layout=SERIAL_V_SMEM,
            )
        else:
            kt_extend_smem = kt_prefix_smem
            v_extend_smem = v_prefix_smem

        kt_offs_d = gl.arange(
            0, cfg.BLOCK_DMODEL, layout=gl.SliceLayout(dim=1, parent=kt_blocked_layout)
        )
        kt_offs_n = gl.arange(
            0, cfg.BLOCK_N, layout=gl.SliceLayout(dim=0, parent=kt_blocked_layout)
        )
        v_offs_n = gl.arange(
            0,
            cfg.BLOCK_N,
            layout=gl.SliceLayout(dim=1, parent=cfg.layouts.blocked_layout),
        )
        v_offs_d = gl.arange(
            0,
            cfg.BLOCK_DV,
            layout=gl.SliceLayout(dim=0, parent=cfg.layouts.blocked_layout),
        )

        mask_base_idx, mask_row_stride, mask_kv_col_offset = self._compute_mask_state(
            cur_seq, seq_len_extend, seq_len_prefix
        )

        q_dot, pfx_q = self._load_q(
            cur_seq_q_start_idx,
            cur_head,
            cur_block_m,
            seq_len_extend,
            offs_m,
            offs_d,
        )

        m_i, l_i, acc = self._init_softmax()
        q_abs_pos, xai_temperature_reg = self._compute_q_abs_pos_and_xai(
            seq_len_prefix, cur_block_m
        )

        # Serial kernel uses the same SWA prefix-skip as pipelined kernels.
        pfx_kv_start, pfx_seq_len, pfx_q_abs_pos = self._compute_swa_skip(
            cur_seq_kv_start_idx,
            seq_len_prefix,
            cur_block_m,
            q_abs_pos,
        )

        # Prefix loop: serial, synchronous gather.
        kv_indices_base = self.kv_indices + pfx_kv_start
        k_prefix_base = self.K_Buffer + cur_kv_head * self.stride_buf_kh
        v_prefix_base = self.V_Buffer + cur_kv_head * self.stride_buf_vh
        n_prefix_blocks = (pfx_seq_len + cfg.BLOCK_N - 1) // cfg.BLOCK_N
        for block_n in tl.range(0, n_prefix_blocks):
            start_n = block_n * cfg.BLOCK_N

            n_idx_k = start_n + kt_offs_n
            mask_n_k = n_idx_k < pfx_seq_len
            kv_locs_k = cdna4_buffer_load(
                kv_indices_base,
                n_idx_k.to(tl.int32),
                mask=mask_n_k,
                other=0,
            ).to(tl.int32)
            kt_offsets = (
                kt_offs_d[:, None] + kv_locs_k[None, :] * self.stride_buf_kbs
            ).to(tl.int32)
            kt_global = cdna4_buffer_load(
                k_prefix_base,
                kt_offsets,
                mask=mask_n_k[None, :],
                other=0.0,
            )
            kt_prefix_smem.store(kt_global)
            kt_dot = kt_prefix_smem.load(cfg.layouts.pfx_kt_dot_layout)

            qk = gl.zeros(
                [cfg.BLOCK_M, cfg.BLOCK_N],
                dtype=gl.float32,
                layout=cfg.layouts.mma_layout,
            )
            qk = do_mma(pfx_q, kt_dot, qk)
            acc, l_i, m_i, p = self.compute_softmax_prefix(
                acc,
                l_i,
                m_i,
                qk,
                start_n,
                pfx_seq_len,
                qk_scale,
                xai_temperature_reg,
                pfx_q_abs_pos,
                cfg.BLOCK_N,
                ENABLE_PREFIX_UNMASKED=cfg.IS_CAUSAL,
            )

            n_idx_v = start_n + v_offs_n
            mask_n_v = n_idx_v < pfx_seq_len
            kv_locs_v = cdna4_buffer_load(
                kv_indices_base,
                n_idx_v.to(tl.int32),
                mask=mask_n_v,
                other=0,
            ).to(tl.int32)
            v_offsets = (
                kv_locs_v[:, None] * self.stride_buf_vbs + v_offs_d[None, :]
            ).to(tl.int32)
            v_global = cdna4_buffer_load(
                v_prefix_base,
                v_offsets,
                mask=mask_n_v[:, None],
                other=0.0,
            )
            v_prefix_smem.store(v_global)
            v_dot = v_prefix_smem.load(cfg.layouts.pfx_v_dot_layout)
            p_cast = p.to(v_dot.dtype)
            p_dot = gl.convert_layout(p_cast, cfg.layouts.pfx_p_dot_layout)
            acc = do_mma(p_dot, v_dot, acc)

        # Extend loop: serial, synchronous row-major.
        cur_block_m_end = (
            seq_len_extend
            if not cfg.IS_CAUSAL
            else tl.minimum(seq_len_extend, (cur_block_m + 1) * cfg.BLOCK_M)
        )
        k_extend_base = (
            self.K_Extend
            + cur_seq_q_start_idx * self.stride_kbs
            + cur_kv_head * self.stride_kh
        )
        v_extend_base = (
            self.V_Extend
            + cur_seq_q_start_idx * self.stride_vbs
            + cur_kv_head * self.stride_vh
        )
        n_extend_blocks = (cur_block_m_end + cfg.BLOCK_N - 1) // cfg.BLOCK_N
        for block_n in tl.range(0, n_extend_blocks):
            start_n = block_n * cfg.BLOCK_N

            kt_idx = start_n + kt_offs_n[None, :]
            kt_offsets = (kt_offs_d[:, None] + kt_idx * self.stride_kbs).to(tl.int32)
            kt_global = cdna4_buffer_load(
                k_extend_base,
                kt_offsets,
                mask=kt_idx < cur_block_m_end,
                other=0.0,
            )
            kt_extend_smem.store(kt_global)
            kt_dot = kt_extend_smem.load(cfg.layouts.kt_dot_layout)

            qk = gl.zeros(
                [cfg.BLOCK_M, cfg.BLOCK_N],
                dtype=gl.float32,
                layout=cfg.layouts.mma_layout,
            )
            qk = do_mma(q_dot, kt_dot, qk)
            acc, l_i, m_i, p = self.compute_softmax_extend(
                acc,
                l_i,
                m_i,
                qk,
                start_n,
                cur_block_m,
                seq_len_extend,
                qk_scale,
                xai_temperature_reg,
                mask_base_idx,
                mask_row_stride,
                mask_kv_col_offset,
                cfg.BLOCK_N,
                MASK_STEPS=True,
            )

            v_idx = start_n + v_offs_n[:, None]
            v_offsets = (v_idx * self.stride_vbs + v_offs_d[None, :]).to(tl.int32)
            v_global = cdna4_buffer_load(
                v_extend_base,
                v_offsets,
                mask=v_idx < cur_block_m_end,
                other=0.0,
            )
            v_extend_smem.store(v_global)
            v_dot = v_extend_smem.load(cfg.layouts.v_dot_layout)
            p_cast = p.to(v_dot.dtype)
            p_dot = gl.convert_layout(p_cast, cfg.layouts.p_dot_layout)
            acc = do_mma(p_dot, v_dot, acc)

        l_i = self._apply_sinks(cur_head, l_i, m_i)
        self._normalize_and_store(
            cur_seq_q_start_idx,
            cur_head,
            cur_block_m,
            seq_len_extend,
            acc,
            l_i,
            offs_m,
            offs_dv,
        )


@composition
@aggregate
class ExtendAttnSwPipeline4WProgram:
    state: ExtendAttnProgram

    @gluon.constexpr_function
    def __init__(self, state):
        self.state = state

    # --------------------------------------------------------------------- #
    # 4w schedule-private prefix/extend loops
    # --------------------------------------------------------------------- #

    @gluon.jit
    def attn_fwd_inner_prefix_sw_pipeline_4w(
        self,
        acc,
        l_i,
        m_i,
        q_dot,
        kv_start,
        cur_kv_head,
        seq_len_prefix,
        bank,
        qk_scale,
        xai_temperature_reg,
        q_abs_pos,
    ):
        """4-warp sw-pipeline prefix: manually-scheduled async DMA loop.

        NS>=2, D>=128. Explicitly issues K_future / V_future loads and
        consumes them with ``cdna4_async.wait_group`` counters
        (no warp_pipeline_stage). Callers pass the full prefix length;
        the helper floor-aligns to ``BLOCK_N`` for the pipelined bulk
        and runs one masked tail block if the full length isn't a
        multiple of ``BLOCK_N``.
        """
        cfg: gl.constexpr = self.cfg
        STREAMS: gl.constexpr = 2

        aligned_prefix_len = (seq_len_prefix // cfg.BLOCK_N) * cfg.BLOCK_N
        n_prefix_blocks = aligned_prefix_len // cfg.BLOCK_N
        kv_indices_base = self.kv_indices + kv_start

        kt_async_layout: gl.constexpr = bank.kt_async_layout
        v_async_layout: gl.constexpr = bank.v_async_layout
        kt_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=kt_async_layout)
        v_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=v_async_layout)
        kt_offs_n = gl.arange(0, cfg.BLOCK_N, layout=kt_offs_n_layout)
        v_offs_n = gl.arange(0, cfg.BLOCK_N, layout=v_offs_n_layout)

        # Each logical tile issues K then V. The wait counts leave just
        # enough groups in flight to consume K first and V after QK softmax.
        WAIT_K: gl.constexpr = STREAMS * cfg.NUM_STAGES - (STREAMS - 1)
        WAIT_V: gl.constexpr = STREAMS * cfg.NUM_STAGES - STREAMS

        kt_loader = AsyncKVLoader.for_prefix_k(
            bank.kt_smem,
            self.K_Buffer,
            cur_kv_head,
            self.kv_indices,
            kv_start,
            self.stride_buf_kbs,
            self.stride_buf_kh,
            cfg.BLOCK_N,
            cfg.BLOCK_DMODEL,
            kt_async_layout,
        )
        v_loader = AsyncKVLoader.for_prefix_v(
            bank.v_smem,
            self.V_Buffer,
            cur_kv_head,
            self.kv_indices,
            kv_start,
            self.stride_buf_vbs,
            self.stride_buf_vh,
            cfg.BLOCK_N,
            cfg.BLOCK_DV,
            v_async_layout,
        )

        for stage in gl.static_range(cfg.NUM_STAGES):
            cdna4_async.wait_group(0)
            pf_init_n = stage * cfg.BLOCK_N
            n_idx_k = pf_init_n + kt_offs_n
            mask_n_k = n_idx_k < seq_len_prefix
            kv_locs_k = cdna4_buffer_load(
                kv_indices_base,
                n_idx_k.to(tl.int32),
                mask=mask_n_k,
                other=0,
            ).to(tl.int32)
            n_idx_v = pf_init_n + v_offs_n
            mask_n_v = n_idx_v < seq_len_prefix
            kv_locs_v = cdna4_buffer_load(
                kv_indices_base,
                n_idx_v.to(tl.int32),
                mask=mask_n_v,
                other=0,
            ).to(tl.int32)
            kt_loader.issue_from_locs(kv_locs_k, mask_n_k, slot=stage)
            v_loader.issue_from_locs(kv_locs_v, mask_n_v, slot=stage)

        cdna4_async.wait_group(0)
        pf_start_n = cfg.NUM_STAGES * cfg.BLOCK_N
        n_idx_k_pf = pf_start_n + kt_offs_n
        mask_k_pf = n_idx_k_pf < seq_len_prefix
        kv_locs_k_pf = cdna4_buffer_load(
            kv_indices_base,
            n_idx_k_pf.to(tl.int32),
            mask=mask_k_pf,
            other=0,
        ).to(tl.int32)
        n_idx_v_pf = pf_start_n + v_offs_n
        mask_v_pf = n_idx_v_pf < seq_len_prefix
        kv_locs_v_pf = cdna4_buffer_load(
            kv_indices_base,
            n_idx_v_pf.to(tl.int32),
            mask=mask_v_pf,
            other=0,
        ).to(tl.int32)

        cdna4_async.wait_group(WAIT_K)
        kt_dot = cdna4_async.load_shared_relaxed(
            bank.kt_smem.index(0),
            cfg.layouts.pfx_kt_dot_layout,
        )

        main_loop_end = n_prefix_blocks - cfg.NUM_STAGES
        for block_n in tl.range(0, main_loop_end):
            stage_idx = (block_n % cfg.NUM_STAGES).to(tl.int32)
            start_n = (block_n * cfg.BLOCK_N).to(tl.int32)

            nf_start_n = ((block_n + cfg.NUM_STAGES + 1) * cfg.BLOCK_N).to(tl.int32)
            n_idx_k_nf = nf_start_n + kt_offs_n
            mask_k_nf = n_idx_k_nf < seq_len_prefix
            kv_locs_k_nf = cdna4_buffer_load(
                kv_indices_base,
                n_idx_k_nf.to(tl.int32),
                mask=mask_k_nf,
                other=0,
            ).to(tl.int32)
            n_idx_v_nf = nf_start_n + v_offs_n
            mask_v_nf = n_idx_v_nf < seq_len_prefix
            kv_locs_v_nf = cdna4_buffer_load(
                kv_indices_base,
                n_idx_v_nf.to(tl.int32),
                mask=mask_v_nf,
                other=0,
            ).to(tl.int32)

            qk = gl.zeros(
                [cfg.BLOCK_M, cfg.BLOCK_N],
                dtype=gl.float32,
                layout=cfg.layouts.mma_layout,
            )
            qk = do_mma(q_dot, kt_dot, qk)

            cdna4_async.wait_group(WAIT_V)
            v_dot = cdna4_async.load_shared_relaxed(
                bank.v_smem.index(stage_idx),
                cfg.layouts.pfx_v_dot_layout,
            )

            # NOTE: issue K_future AFTER v_dot read; writing back into
            # kt_smem slot after the load avoids the DMA race that
            # was observable once the dtype check was fixed (d1a61f1).
            kt_loader.issue_from_locs(kv_locs_k_pf, mask_k_pf, slot=stage_idx)

            acc, l_i, m_i, p = self.compute_softmax_prefix(
                acc,
                l_i,
                m_i,
                qk,
                start_n,
                seq_len_prefix,
                qk_scale,
                xai_temperature_reg,
                q_abs_pos,
                cfg.BLOCK_N,
                cfg.ENABLE_PREFIX_UNMASKED,
            )

            p_cast = p.to(v_dot.dtype)
            p_dot_reg = gl.convert_layout(p_cast, cfg.layouts.pfx_p_dot_layout)
            acc = do_mma(p_dot_reg, v_dot, acc)

            next_stage_idx = ((block_n + 1) % cfg.NUM_STAGES).to(tl.int32)
            cdna4_async.wait_group(WAIT_K)
            kt_dot = cdna4_async.load_shared_relaxed(
                bank.kt_smem.index(next_stage_idx),
                cfg.layouts.pfx_kt_dot_layout,
            )

            # V_future must be issued after the p*V MMA has consumed
            # v_dot and after the next kt_dot has been read.
            v_loader.issue_from_locs(kv_locs_v_pf, mask_v_pf, slot=stage_idx)

            kv_locs_k_pf = kv_locs_k_nf
            kv_locs_v_pf = kv_locs_v_nf
            mask_k_pf = mask_k_nf
            mask_v_pf = mask_v_nf

        for tail_i in gl.static_range(cfg.NUM_STAGES):
            cdna4_async.wait_group(STREAMS * (cfg.NUM_STAGES - tail_i) - (STREAMS - 1))
            stage_idx = ((main_loop_end + tail_i) % cfg.NUM_STAGES).to(tl.int32)
            start_n = (main_loop_end + tail_i) * cfg.BLOCK_N

            kt_dot_tail = cdna4_async.load_shared_relaxed(
                bank.kt_smem.index(stage_idx),
                cfg.layouts.pfx_kt_dot_layout,
            )
            qk = gl.zeros(
                [cfg.BLOCK_M, cfg.BLOCK_N],
                dtype=gl.float32,
                layout=cfg.layouts.mma_layout,
            )
            qk = do_mma(q_dot, kt_dot_tail, qk)

            acc, l_i, m_i, p = self.compute_softmax_prefix(
                acc,
                l_i,
                m_i,
                qk,
                start_n,
                seq_len_prefix,
                qk_scale,
                xai_temperature_reg,
                q_abs_pos,
                cfg.BLOCK_N,
                cfg.ENABLE_PREFIX_UNMASKED,
            )

            cdna4_async.wait_group(STREAMS * (cfg.NUM_STAGES - tail_i) - STREAMS)
            v_dot_tail = cdna4_async.load_shared_relaxed(
                bank.v_smem.index(stage_idx),
                cfg.layouts.pfx_v_dot_layout,
            )
            p_cast = p.to(v_dot_tail.dtype)
            p_dot_tail = gl.convert_layout(p_cast, cfg.layouts.pfx_p_dot_layout)
            acc = do_mma(p_dot_tail, v_dot_tail, acc)

        # Masked tail: one partial block covering
        # [aligned_prefix_len, seq_len_prefix). Runs after the
        # sw-pipeline drain so smem slot 0 is free to reuse.
        if seq_len_prefix > aligned_prefix_len:
            acc, l_i, m_i = self.attn_fwd_inner_prefix_unpipelined(
                acc,
                l_i,
                m_i,
                q_dot,
                kv_start,
                cur_kv_head,
                seq_len_prefix,
                bank,
                qk_scale,
                xai_temperature_reg,
                q_abs_pos,
                block_start=n_prefix_blocks,
            )

        return acc, l_i, m_i

    @gluon.jit
    def _attn_fwd_inner_extend_sw_pipeline_4w_block_span(
        self,
        acc,
        l_i,
        m_i,
        q_dot,
        cur_seq_q_start_idx,
        cur_kv_head,
        cur_block_m,
        seq_len_extend,
        block_start,
        block_end,
        bank,
        qk_scale,
        xai_temperature_reg,
        mask_base_idx,
        mask_row_stride,
        mask_kv_col_offset,
        EXT_N: gl.constexpr,
        EXT_NS: gl.constexpr,
        MASK_STEPS: gl.constexpr,
        SKIP_BOUNDS_CHECK: gl.constexpr = False,
    ):
        """Run one half-open extend block span with the 4w pipeline.

        Processes ``[block_start, block_end)``. Private; used by
        ``attn_fwd_inner_extend_sw_pipeline_4w``. ``EXT_N`` / ``EXT_NS``
        may differ from ``cfg.BLOCK_N`` / ``cfg.NUM_STAGES`` (FP8 path).
        """
        cfg: gl.constexpr = self.cfg
        STREAMS: gl.constexpr = 2

        kt_loader = AsyncKVLoader.for_extend_k(
            bank.kt_smem,
            self.K_Extend,
            cur_seq_q_start_idx,
            cur_kv_head,
            self.stride_kbs,
            self.stride_kh,
            EXT_N,
            cfg.BLOCK_DMODEL,
            bank.kt_async_layout,
        )
        v_loader = AsyncKVLoader.for_extend_v(
            bank.v_smem,
            self.V_Extend,
            cur_seq_q_start_idx,
            cur_kv_head,
            self.stride_vbs,
            self.stride_vh,
            EXT_N,
            cfg.BLOCK_DV,
            bank.v_async_layout,
        )

        cdna4_async.wait_group(0)

        for stage in gl.static_range(EXT_NS):
            pf_start_n = (block_start + stage) * EXT_N
            if SKIP_BOUNDS_CHECK:
                kt_loader.issue_nomask(pf_start_n, seq_len_extend, slot=stage)
                v_loader.issue_nomask(pf_start_n, seq_len_extend, slot=stage)
            else:
                kt_loader.issue(pf_start_n, seq_len_extend, slot=stage)
                v_loader.issue(pf_start_n, seq_len_extend, slot=stage)

        # K is consumed before the softmax work; V stays in flight until
        # the probabilities are ready for the P*V MMA.
        WAIT_K: gl.constexpr = STREAMS * EXT_NS - 1
        WAIT_V: gl.constexpr = STREAMS * EXT_NS - 2
        cdna4_async.wait_group(WAIT_K)
        kt_dot = cdna4_async.load_shared_relaxed(
            bank.kt_smem.index(0),
            cfg.layouts.kt_dot_layout,
        )

        main_loop_end = block_end - EXT_NS
        for block_n in tl.range(block_start, main_loop_end, loop_unroll_factor=2):
            stage_idx = ((block_n - block_start) % EXT_NS).to(tl.int32)
            start_n = (block_n * EXT_N).to(tl.int32)
            future_start_n = ((block_n + EXT_NS) * EXT_N).to(tl.int32)

            qk = gl.zeros(
                [cfg.BLOCK_M, EXT_N], dtype=gl.float32, layout=cfg.layouts.mma_layout
            )
            qk = do_mma(q_dot, kt_dot, qk)

            cdna4_async.wait_group(WAIT_V)
            v_dot = cdna4_async.load_shared_relaxed(
                bank.v_smem.index(stage_idx),
                cfg.layouts.v_dot_layout,
            )
            if SKIP_BOUNDS_CHECK:
                kt_loader.issue_nomask(future_start_n, seq_len_extend, slot=stage_idx)
            else:
                kt_loader.issue(future_start_n, seq_len_extend, slot=stage_idx)

            acc, l_i, m_i, p = self.compute_softmax_extend(
                acc,
                l_i,
                m_i,
                qk,
                start_n,
                cur_block_m,
                seq_len_extend,
                qk_scale,
                xai_temperature_reg,
                mask_base_idx,
                mask_row_stride,
                mask_kv_col_offset,
                EXT_N,
                MASK_STEPS,
            )
            p_cast = p.to(v_dot.dtype)
            p_dot_reg = gl.convert_layout(p_cast, cfg.layouts.p_dot_layout)
            acc = do_mma(p_dot_reg, v_dot, acc)

            cdna4_async.wait_group(WAIT_V)
            next_stage_idx = ((block_n + 1 - block_start) % EXT_NS).to(tl.int32)
            kt_dot = cdna4_async.load_shared_relaxed(
                bank.kt_smem.index(next_stage_idx),
                cfg.layouts.kt_dot_layout,
            )
            if SKIP_BOUNDS_CHECK:
                v_loader.issue_nomask(future_start_n, seq_len_extend, slot=stage_idx)
            else:
                v_loader.issue(future_start_n, seq_len_extend, slot=stage_idx)

        for tail_i in gl.static_range(EXT_NS):
            cdna4_async.wait_group(STREAMS * (EXT_NS - tail_i) - 1)
            stage_idx = ((main_loop_end + tail_i - block_start) % EXT_NS).to(tl.int32)
            start_n = (main_loop_end + tail_i) * EXT_N

            kt_dot_tail = cdna4_async.load_shared_relaxed(
                bank.kt_smem.index(stage_idx),
                cfg.layouts.kt_dot_layout,
            )
            qk = gl.zeros(
                [cfg.BLOCK_M, EXT_N], dtype=gl.float32, layout=cfg.layouts.mma_layout
            )
            qk = do_mma(q_dot, kt_dot_tail, qk)

            p, alpha, m_new = self.softmax_part0(
                m_i,
                qk,
                start_n,
                cur_block_m,
                seq_len_extend,
                qk_scale,
                xai_temperature_reg,
                mask_base_idx,
                mask_row_stride,
                mask_kv_col_offset,
                EXT_N,
                MASK_STEPS,
            )

            cdna4_async.wait_group(STREAMS * (EXT_NS - tail_i) - 2)
            v_dot_tail = cdna4_async.load_shared_relaxed(
                bank.v_smem.index(stage_idx),
                cfg.layouts.v_dot_layout,
            )
            acc, l_i, m_i = self.softmax_part1(acc, l_i, p, alpha, m_new)
            p_cast = p.to(v_dot_tail.dtype)
            p_dot_tail = gl.convert_layout(p_cast, cfg.layouts.p_dot_layout)
            acc = do_mma(p_dot_tail, v_dot_tail, acc)

        return acc, l_i, m_i

    @gluon.jit
    def attn_fwd_inner_extend_sw_pipeline_4w(
        self,
        acc,
        l_i,
        m_i,
        q_dot,
        cur_seq_q_start_idx,
        cur_kv_head,
        cur_block_m,
        seq_len_extend,
        bulk_end,
        tail_start,
        tail_end,
        bank,
        qk_scale,
        xai_temperature_reg,
        mask_base_idx,
        mask_row_stride,
        mask_kv_col_offset,
        EXT_N: gl.constexpr,
        EXT_NS: gl.constexpr,
    ):
        """Fused unmasked-bulk + masked-tail extend loop (4w sw-pipeline).

        Unmasked bulk is ``[0, bulk_end)``; masked tail is
        ``[tail_start, tail_end)``. ``tail_start`` is normally
        ``bulk_end`` but can be larger when the caller has
        fast-forwarded past SWA windows entirely outside the active
        region. Callers that want every block masked (D>=256,
        USE_CUSTOM_MASK, mixed-SWA) pass ``bulk_end=0`` and
        ``tail_start=0``.

        Each phase picks between the async-DMA pipelined helper (when
        the block count >= EXT_NS -- enough to cover the DMA latency)
        and the synchronous preload-all unpipelined helper (otherwise
        -- serial runs skip the DMA setup cost). The underlying block-span
        helpers stay compile-time specialised on MASK_STEPS=False/True
        so the unmasked bulk keeps its cheaper softmax inner-body.
        Bulk phase hardcodes ``SKIP_BOUNDS_CHECK=True``; tail hardcodes
        ``SKIP_BOUNDS_CHECK=False``.

        ``EXT_N`` / ``EXT_NS`` let the FP8 path swap in a shrunken
        extend block (``EXT_BLOCK_N`` / ``EXT_NUM_STAGES``); BF16
        callers pass ``cfg.BLOCK_N`` / ``cfg.NUM_STAGES``.
        """
        if bulk_end >= EXT_NS:
            acc, l_i, m_i = self._attn_fwd_inner_extend_sw_pipeline_4w_block_span(
                acc,
                l_i,
                m_i,
                q_dot,
                cur_seq_q_start_idx,
                cur_kv_head,
                cur_block_m,
                seq_len_extend,
                0,
                bulk_end,
                bank,
                qk_scale,
                xai_temperature_reg,
                mask_base_idx,
                mask_row_stride,
                mask_kv_col_offset,
                EXT_N,
                EXT_NS,
                MASK_STEPS=False,
                SKIP_BOUNDS_CHECK=True,
            )
        elif bulk_end > 0:
            acc, l_i, m_i = self._attn_fwd_inner_extend_unpipelined_block_span(
                acc,
                l_i,
                m_i,
                q_dot,
                cur_seq_q_start_idx,
                cur_kv_head,
                cur_block_m,
                seq_len_extend,
                0,
                bulk_end,
                bank,
                qk_scale,
                xai_temperature_reg,
                mask_base_idx,
                mask_row_stride,
                mask_kv_col_offset,
                EXT_N,
                MASK_STEPS=False,
                SKIP_BOUNDS_CHECK=True,
            )
        remaining_blocks = tail_end - tail_start
        if remaining_blocks >= EXT_NS:
            acc, l_i, m_i = self._attn_fwd_inner_extend_sw_pipeline_4w_block_span(
                acc,
                l_i,
                m_i,
                q_dot,
                cur_seq_q_start_idx,
                cur_kv_head,
                cur_block_m,
                seq_len_extend,
                tail_start,
                tail_end,
                bank,
                qk_scale,
                xai_temperature_reg,
                mask_base_idx,
                mask_row_stride,
                mask_kv_col_offset,
                EXT_N,
                EXT_NS,
                MASK_STEPS=True,
                SKIP_BOUNDS_CHECK=False,
            )
        elif remaining_blocks > 0:
            acc, l_i, m_i = self._attn_fwd_inner_extend_unpipelined_block_span(
                acc,
                l_i,
                m_i,
                q_dot,
                cur_seq_q_start_idx,
                cur_kv_head,
                cur_block_m,
                seq_len_extend,
                tail_start,
                tail_end,
                bank,
                qk_scale,
                xai_temperature_reg,
                mask_base_idx,
                mask_row_stride,
                mask_kv_col_offset,
                EXT_N,
                MASK_STEPS=True,
                SKIP_BOUNDS_CHECK=False,
            )
        return acc, l_i, m_i

    @gluon.jit
    def run(self):
        """Body of ``gluon_extend_attn_fwd_4w`` (NS>=2, D>=128, NW<8)."""
        cfg: gl.constexpr = self.cfg
        tl.static_assert(
            cfg.NUM_STAGES >= 2, "4w sw-pipeline program requires NUM_STAGES>=2"
        )
        tl.static_assert(
            cfg.BLOCK_DMODEL >= 128, "4w sw-pipeline program requires BLOCK_DMODEL>=128"
        )
        tl.static_assert(
            cfg.num_warps < 8,
            "4w sw-pipeline program requires num_warps<8; NW>=8 routes to 8w pingpong",
        )
        if cfg.IS_FP8:
            tl.static_assert(
                cfg.BLOCK_DMODEL != 256,
                "FP8 D=256 unsupported on gfx950 (MFMA_F8 codegen failure)",
            )

        offs_m = gl.arange(0, cfg.BLOCK_M, layout=cfg.layouts.offs_m_layout)
        offs_d = gl.arange(0, cfg.BLOCK_DMODEL, layout=cfg.layouts.offs_d_layout)
        offs_dv = gl.arange(0, cfg.BLOCK_DV, layout=cfg.layouts.offs_d_layout)
        qk_scale = self.sm_scale * LOG2E

        if cfg.IS_WCA:
            tile_idx = gl.program_id(0)
        else:
            tile_idx = 0

        # Data-centric launches execute this loop once: each CTA owns the
        # rectangular-grid tile encoded by program_id(0/1/2). WCA launches
        # a compact 1D grid and reuses CTAs persistently; pid p handles
        # p, p + total_programs, p + 2 * total_programs, ...
        # SPLIT_K == 1 only removes the split-K reduction path; it does
        # not turn WCA into the data-centric scheduler.
        while tile_idx < (self.total_valid_tiles if cfg.IS_WCA else 1):
            # Tile coords.
            if cfg.IS_WCA:
                (
                    cur_seq,
                    cur_head,
                    cur_block_m,
                    cur_kv_head,
                    cur_seq_q_start_idx,
                    seq_len_extend,
                    cur_seq_kv_start_idx,
                    seq_len_prefix,
                    is_valid_tile,
                    output_tile,
                    k_split_id,
                ) = self._schedule_wca(tile_idx)
            else:
                (
                    cur_seq,
                    cur_head,
                    cur_block_m,
                    cur_kv_head,
                    cur_seq_q_start_idx,
                    seq_len_extend,
                    cur_seq_kv_start_idx,
                    seq_len_prefix,
                    is_valid_tile,
                    output_tile,
                    k_split_id,
                ) = self._schedule_data_centric()

            mask_base_idx, mask_row_stride, mask_kv_col_offset = (
                self._compute_mask_state(cur_seq, seq_len_extend, seq_len_prefix)
            )

            q_dot, pfx_q = self._load_q(
                cur_seq_q_start_idx,
                cur_head,
                cur_block_m,
                seq_len_extend,
                offs_m,
                offs_d,
            )
            m_i, l_i, acc = self._init_softmax()
            q_abs_pos, xai_temperature_reg = self._compute_q_abs_pos_and_xai(
                seq_len_prefix, cur_block_m
            )

            pfx_kv_start, pfx_seq_len, pfx_q_abs_pos = self._compute_swa_skip(
                cur_seq_kv_start_idx,
                seq_len_prefix,
                cur_block_m,
                q_abs_pos,
            )
            orig_seq_len_extend = seq_len_extend
            pfx_kv_start, pfx_seq_len, pfx_q_abs_pos, seq_len_extend = (
                self._apply_splitk_prefix_partition(
                    pfx_kv_start,
                    pfx_seq_len,
                    pfx_q_abs_pos,
                    k_split_id,
                    seq_len_extend,
                )
            )

            # FP8 EXT_N / EXT_NS override (4w only; BF16 reuses BLOCK_N / NUM_STAGES).
            _EXT_N: gl.constexpr = (
                cfg.EXT_BLOCK_N if (cfg.IS_FP8 and cfg.EXT_BLOCK_N > 0) else cfg.BLOCK_N
            )
            _EXT_NS: gl.constexpr = (
                cfg.EXT_NUM_STAGES
                if (cfg.IS_FP8 and cfg.EXT_NUM_STAGES > 0)
                else cfg.NUM_STAGES
            )

            effective_end, n_extend_blocks, n_full_blocks, swa_skip_n_blocks = (
                self._compute_extend_bounds(seq_len_extend, cur_block_m, _EXT_N)
            )

            # Dot layouts live on cfg.layouts; build the async tile layouts and
            # prefix smem policy directly from the same facade.
            kt_offset_bases: gl.constexpr = ExtendAttentionLayouts.make_kt_offset_bases(
                cfg.BLOCK_DMODEL, cfg.BLOCK_N
            )
            v_offset_bases: gl.constexpr = ExtendAttentionLayouts.prefix_v_offset_bases(
                cfg.layouts, cfg.num_warps, cfg.BLOCK_DV, cfg.BLOCK_N
            )
            kt_async_layout: gl.constexpr = ExtendAttentionLayouts.prefix_kt_dll(
                cfg.layouts, cfg.num_warps, cfg.BLOCK_DMODEL, cfg.BLOCK_N
            )
            v_async_layout: gl.constexpr = ExtendAttentionLayouts.prefix_v_dll(
                cfg.layouts, cfg.num_warps, cfg.BLOCK_DMODEL, cfg.BLOCK_DV, cfg.BLOCK_N
            )
            kt_smem_layout: gl.constexpr = ExtendAttentionLayouts.prefix_kt_smem_layout(
                cfg.layouts, cfg.BLOCK_DMODEL, cfg.BLOCK_N, kt_offset_bases
            )
            v_smem_layout: gl.constexpr = ExtendAttentionLayouts.prefix_v_smem_layout(
                cfg.layouts, cfg.BLOCK_N, cfg.BLOCK_DV, v_offset_bases
            )

            if cfg.IS_FP8:
                # FP8 prefix is native FP8, but extend K/V are BF16; these
                # branches select the BF16-extend exceptions and rare EXT_N
                # variants without adding another forwarding policy class.
                if _EXT_N == cfg.BLOCK_N:
                    ext_kt_offset_bases: gl.constexpr = kt_offset_bases
                    ext_v_offset_bases: gl.constexpr = v_offset_bases
                    ext_kt_async_layout: gl.constexpr = (
                        ExtendAttentionLayouts.make_kt_dll(
                            cfg.num_warps, cfg.BLOCK_DMODEL, cfg.BLOCK_N
                        )
                    )
                    ext_v_async_layout: gl.constexpr = (
                        ExtendAttentionLayouts.make_fp8_extend_v_dll(
                            cfg.num_warps, cfg.BLOCK_DV, cfg.BLOCK_N
                        )
                    )
                else:
                    ext_kt_offset_bases: gl.constexpr = (
                        ExtendAttentionLayouts.make_ext_kt_offset_bases(
                            cfg.num_warps, cfg.BLOCK_DMODEL, _EXT_N
                        )
                    )
                    ext_kt_async_layout: gl.constexpr = (
                        ExtendAttentionLayouts.make_ext_kt_dll(
                            cfg.num_warps, cfg.BLOCK_DMODEL, _EXT_N
                        )
                    )
                    ext_v_offset_bases: gl.constexpr = (
                        ExtendAttentionLayouts.make_ext_v_offset_bases(
                            cfg.num_warps, cfg.BLOCK_DV, _EXT_N
                        )
                    )
                    ext_v_async_layout: gl.constexpr = (
                        ExtendAttentionLayouts.make_ext_v_dll(
                            cfg.num_warps, cfg.BLOCK_DV, _EXT_N
                        )
                    )
                ext_kt_smem_layout: gl.constexpr = (
                    ExtendAttentionLayouts.make_padded_smem(
                        [cfg.BLOCK_DMODEL, _EXT_N],
                        ext_kt_offset_bases,
                        [[512, 16]],
                    )
                )
                ext_v_smem_layout: gl.constexpr = (
                    ExtendAttentionLayouts.make_padded_smem(
                        [_EXT_N, cfg.BLOCK_DV],
                        ext_v_offset_bases,
                        [[512, 16]],
                    )
                )
            else:
                ext_kt_async_layout: gl.constexpr = kt_async_layout
                ext_v_async_layout: gl.constexpr = v_async_layout
                ext_kt_smem_layout: gl.constexpr = kt_smem_layout
                ext_v_smem_layout: gl.constexpr = v_smem_layout

            # Allocate the K/V staging bank and carry the async layouts with
            # it so prefix and extend helpers share the same descriptor bundle.
            bank = KVSmemBank.initialize(
                cfg,
                kt_smem_layout,
                v_smem_layout,
                kt_async_layout,
                v_async_layout,
                zero_fill=(cfg.IS_FP8 or is_valid_tile),
            )

            # Prefix loop: use the pipeline only when there are enough full
            # blocks to fill it. Logit-capped shapes with long extend tails stay
            # on the serial prefix path to keep the overlap schedule simple.
            if pfx_seq_len > 0:
                n_full_prefix = pfx_seq_len // cfg.BLOCK_N
                n_extend_est = (seq_len_extend + cfg.BLOCK_N - 1) // cfg.BLOCK_N
                use_pipe_prefix = n_full_prefix >= cfg.NUM_STAGES
                if cfg.LOGIT_CAP > 0:
                    use_pipe_prefix = use_pipe_prefix and (
                        n_extend_est < cfg.NUM_STAGES
                    )
                if use_pipe_prefix:
                    acc, l_i, m_i = self.attn_fwd_inner_prefix_sw_pipeline_4w(
                        acc,
                        l_i,
                        m_i,
                        pfx_q,
                        pfx_kv_start,
                        cur_kv_head,
                        pfx_seq_len,
                        bank,
                        qk_scale,
                        xai_temperature_reg,
                        pfx_q_abs_pos,
                    )
                else:
                    acc, l_i, m_i = self.attn_fwd_inner_prefix_unpipelined(
                        acc,
                        l_i,
                        m_i,
                        pfx_q,
                        pfx_kv_start,
                        cur_kv_head,
                        pfx_seq_len,
                        bank,
                        qk_scale,
                        xai_temperature_reg,
                        pfx_q_abs_pos,
                    )

            cdna4_async.wait_group(0)

            # FP8 smem transition (BF16 skips).
            if cfg.IS_FP8:
                bank = bank.transition_to_extend(
                    cfg,
                    self.Q_Extend.dtype.element_ty,
                    ext_kt_smem_layout,
                    ext_v_smem_layout,
                    ext_kt_async_layout,
                    ext_v_async_layout,
                    _EXT_N,
                    _EXT_NS,
                )

            # Extend hot-loop (fused unmasked-bulk + masked-tail 4w sw-pipeline).
            masked_start = tl.maximum(n_full_blocks, swa_skip_n_blocks)
            acc, l_i, m_i = self.attn_fwd_inner_extend_sw_pipeline_4w(
                acc,
                l_i,
                m_i,
                q_dot,
                cur_seq_q_start_idx,
                cur_kv_head,
                cur_block_m,
                seq_len_extend,
                n_full_blocks,
                masked_start,
                n_extend_blocks,
                bank,
                qk_scale,
                xai_temperature_reg,
                mask_base_idx,
                mask_row_stride,
                mask_kv_col_offset,
                _EXT_N,
                _EXT_NS,
            )

            l_i = self._apply_sinks(cur_head, l_i, m_i)

            if cfg.IS_WCA and cfg.SPLIT_K > 1:
                self._splitk_partial_store_and_reduce(
                    output_tile,
                    k_split_id,
                    cur_seq_q_start_idx,
                    cur_head,
                    cur_block_m,
                    orig_seq_len_extend,
                    acc,
                    l_i,
                    m_i,
                    offs_m,
                    offs_dv,
                )
            else:
                self._normalize_and_store(
                    cur_seq_q_start_idx,
                    cur_head,
                    cur_block_m,
                    seq_len_extend,
                    acc,
                    l_i,
                    offs_m,
                    offs_dv,
                )

            if cfg.IS_WCA:
                tile_idx += self.total_programs
            else:
                tile_idx = 1


@composition
@aggregate
class ExtendAttnPingpong8WProgram:
    state: ExtendAttnProgram

    @gluon.constexpr_function
    def __init__(self, state):
        self.state = state

    # --------------------------------------------------------------------- #
    # 8w schedule-private prefix/extend loops
    # --------------------------------------------------------------------- #

    @gluon.jit
    def attn_fwd_inner_prefix_pingpong_8w(
        self,
        acc,
        l_i,
        m_i,
        q_dot,
        kv_start,
        cur_kv_head,
        seq_len_prefix,
        bank,
        qk_scale,
        xai_temperature_reg,
        q_abs_pos,
        SCALAR_MASK: gl.constexpr = False,
    ):
        """8-warp pingpong prefix: warp_pipeline_stage-scheduled async DMA loop.

        Runs compute (``compute*``) and memory (``memory*``) on
        alternating warp groups; requires NUM_STAGES>=2 physical LDS
        buffers for deterministic pipeline scheduling. Callers pass the
        full prefix length -- the helper processes the BLOCK_N-aligned
        bulk via the pipelined loop and then runs a single masked block
        covering any partial tail. On the WCA path
        ``seq_len_prefix`` is already BLOCK_N-aligned, so that tail
        branch is a runtime no-op.
        """
        cfg: gl.constexpr = self.cfg
        STREAMS: gl.constexpr = 2
        # warp_pipeline_stage requires >=2 physical LDS buffers. With
        # NS=1 the stage_idx collapses to 0 and iter N's DMA write to
        # smem[0] races iter N+1's relaxed read from smem[0] because
        # membarFilter skips barriers between BufferLoadToLocalOp and
        # syncedViaAsyncWait loads. Callers wanting NS=1 must route to
        # attn_fwd_inner_prefix_unpipelined.
        tl.static_assert(
            cfg.NUM_STAGES >= 2,
            "attn_fwd_inner_prefix_pingpong_8w requires NUM_STAGES>=2 for "
            "determinism (warp_pipeline_stage needs multiple LDS buffers).",
        )

        aligned_prefix_len = (seq_len_prefix // cfg.BLOCK_N) * cfg.BLOCK_N
        n_prefix_blocks = aligned_prefix_len // cfg.BLOCK_N

        kt_async_layout: gl.constexpr = bank.kt_async_layout
        v_async_layout: gl.constexpr = bank.v_async_layout
        kt_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=kt_async_layout)
        v_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=v_async_layout)
        kt_offs_n_pf = gl.arange(0, cfg.BLOCK_N, layout=kt_offs_n_layout)
        v_offs_n_pf = gl.arange(0, cfg.BLOCK_N, layout=v_offs_n_layout)

        kt_loader = AsyncKVLoader.for_prefix_k(
            bank.kt_smem,
            self.K_Buffer,
            cur_kv_head,
            self.kv_indices,
            kv_start,
            self.stride_buf_kbs,
            self.stride_buf_kh,
            cfg.BLOCK_N,
            cfg.BLOCK_DMODEL,
            kt_async_layout,
        )
        v_loader = AsyncKVLoader.for_prefix_v(
            bank.v_smem,
            self.V_Buffer,
            cur_kv_head,
            self.kv_indices,
            kv_start,
            self.stride_buf_vbs,
            self.stride_buf_vh,
            cfg.BLOCK_N,
            cfg.BLOCK_DV,
            v_async_layout,
        )

        for stage in gl.static_range(cfg.NUM_STAGES):
            pf_init_n = stage * cfg.BLOCK_N
            kt_loader.issue(pf_init_n, seq_len_prefix, slot=stage)
            v_loader.issue(pf_init_n, seq_len_prefix, slot=stage)

        pf_start_n = cfg.NUM_STAGES * cfg.BLOCK_N
        n_idx_kt_pf = pf_start_n + kt_offs_n_pf
        mask_n_kt_pf = n_idx_kt_pf < seq_len_prefix
        kv_locs_kt_pf = gl.load(
            self.kv_indices + kv_start + n_idx_kt_pf,
            mask=mask_n_kt_pf,
            other=0,
        ).to(tl.int32)
        n_idx_v_pf = pf_start_n + v_offs_n_pf
        mask_n_v_pf = n_idx_v_pf < seq_len_prefix
        kv_locs_v_pf = gl.load(
            self.kv_indices + kv_start + n_idx_v_pf,
            mask=mask_n_v_pf,
            other=0,
        ).to(tl.int32)

        # Each logical tile issues K then V. The wait counts leave just
        # enough groups in flight to consume K first and V after QK softmax.
        WAIT_K: gl.constexpr = STREAMS * cfg.NUM_STAGES - (STREAMS - 1)
        WAIT_V: gl.constexpr = STREAMS * cfg.NUM_STAGES - STREAMS

        # Scalar-mask fast path: replaces the per-element
        # `n < seq_len_prefix` vector mask on K/V DMAs with a single
        # scalar `seq_len_prefix > 0` check. Safe whenever
        # `n_prefix_blocks * BLOCK_N` bounds were already established at
        # prologue time (which is the case here).
        dma_mask = seq_len_prefix > 0

        cdna4_async.wait_group(WAIT_K)
        kt_dot = cdna4_async.load_shared_relaxed(
            bank.kt_smem.index(0),
            cfg.layouts.pfx_kt_dot_layout,
        )

        main_loop_end = n_prefix_blocks - cfg.NUM_STAGES
        for block_n in tl.range(0, main_loop_end, loop_unroll_factor=2):

            with warp_pipeline_stage("compute0", priority=0):
                stage_idx = (block_n % cfg.NUM_STAGES).to(tl.int32)
                start_n = (block_n * cfg.BLOCK_N).to(tl.int32)
                qk = gl.zeros(
                    [cfg.BLOCK_M, cfg.BLOCK_N],
                    dtype=gl.float32,
                    layout=cfg.layouts.mma_layout,
                )
                qk = do_mma(q_dot, kt_dot, qk)

            cdna4_async.wait_group(WAIT_V)

            with warp_pipeline_stage("memory0", priority=1):
                v_dot = cdna4_async.load_shared_relaxed(
                    bank.v_smem.index(stage_idx),
                    cfg.layouts.pfx_v_dot_layout,
                )
                if SCALAR_MASK:
                    kt_loader.issue_from_locs(
                        kv_locs_kt_pf,
                        dma_mask,
                        slot=stage_idx,
                        IS_SCALAR_MASK=True,
                    )
                else:
                    kt_loader.issue_from_locs(
                        kv_locs_kt_pf,
                        mask_n_kt_pf,
                        slot=stage_idx,
                    )

            with warp_pipeline_stage("compute1", priority=0):
                acc, l_i, m_i, p = self.compute_softmax_prefix(
                    acc,
                    l_i,
                    m_i,
                    qk,
                    start_n,
                    seq_len_prefix,
                    qk_scale,
                    xai_temperature_reg,
                    q_abs_pos,
                    cfg.BLOCK_N,
                    cfg.ENABLE_PREFIX_UNMASKED,
                )

            with warp_pipeline_stage("memory1", priority=1):
                if SCALAR_MASK:
                    v_loader.issue_from_locs(
                        kv_locs_v_pf,
                        dma_mask,
                        slot=stage_idx,
                        IS_SCALAR_MASK=True,
                    )
                else:
                    v_loader.issue_from_locs(
                        kv_locs_v_pf,
                        mask_n_v_pf,
                        slot=stage_idx,
                    )
                nf_start_n = ((block_n + cfg.NUM_STAGES + 1) * cfg.BLOCK_N).to(tl.int32)
                n_idx_kt_nf = nf_start_n + kt_offs_n_pf
                mask_n_kt_pf = n_idx_kt_nf < seq_len_prefix
                kv_locs_kt_pf = gl.load(
                    self.kv_indices + kv_start + n_idx_kt_nf,
                    mask=mask_n_kt_pf,
                    other=0,
                ).to(tl.int32)

            with warp_pipeline_stage("compute2", priority=0):
                p_cast = p.to(v_dot.dtype)
                p_dot_reg = gl.convert_layout(p_cast, cfg.layouts.pfx_p_dot_layout)
                acc = do_mma(p_dot_reg, v_dot, acc)

            cdna4_async.wait_group(WAIT_K)

            with warp_pipeline_stage("memory2", priority=1):
                next_stage_idx = ((block_n + 1) % cfg.NUM_STAGES).to(tl.int32)
                kt_dot = cdna4_async.load_shared_relaxed(
                    bank.kt_smem.index(next_stage_idx),
                    cfg.layouts.pfx_kt_dot_layout,
                )
                n_idx_v_nf = nf_start_n + v_offs_n_pf
                mask_n_v_pf = n_idx_v_nf < seq_len_prefix
                kv_locs_v_pf = gl.load(
                    self.kv_indices + kv_start + n_idx_v_nf,
                    mask=mask_n_v_pf,
                    other=0,
                ).to(tl.int32)

        for tail_i in gl.static_range(cfg.NUM_STAGES):
            cdna4_async.wait_group(STREAMS * (cfg.NUM_STAGES - tail_i) - (STREAMS - 1))
            stage_idx = ((main_loop_end + tail_i) % cfg.NUM_STAGES).to(tl.int32)
            start_n = (main_loop_end + tail_i) * cfg.BLOCK_N

            kt_dot_tail = cdna4_async.load_shared_relaxed(
                bank.kt_smem.index(stage_idx),
                cfg.layouts.pfx_kt_dot_layout,
            )
            qk = gl.zeros(
                [cfg.BLOCK_M, cfg.BLOCK_N],
                dtype=gl.float32,
                layout=cfg.layouts.mma_layout,
            )
            qk = do_mma(q_dot, kt_dot_tail, qk)

            acc, l_i, m_i, p = self.compute_softmax_prefix(
                acc,
                l_i,
                m_i,
                qk,
                start_n,
                seq_len_prefix,
                qk_scale,
                xai_temperature_reg,
                q_abs_pos,
                cfg.BLOCK_N,
                cfg.ENABLE_PREFIX_UNMASKED,
            )

            cdna4_async.wait_group(STREAMS * (cfg.NUM_STAGES - tail_i) - STREAMS)
            v_dot_tail = cdna4_async.load_shared_relaxed(
                bank.v_smem.index(stage_idx),
                cfg.layouts.pfx_v_dot_layout,
            )
            p_cast = p.to(v_dot_tail.dtype)
            p_dot_tail = gl.convert_layout(p_cast, cfg.layouts.pfx_p_dot_layout)
            acc = do_mma(p_dot_tail, v_dot_tail, acc)

        # Masked tail: one partial block covering
        # [aligned_prefix_len, seq_len_prefix). Runs *after* the
        # pingpong flush drained every async op, so smem slot 0
        # is free to reuse via the unpipelined helper. Branch is
        # False on the WCA path (seq_len_prefix is BLOCK_N-
        # aligned there).
        if seq_len_prefix > aligned_prefix_len:
            acc, l_i, m_i = self.attn_fwd_inner_prefix_unpipelined(
                acc,
                l_i,
                m_i,
                q_dot,
                kv_start,
                cur_kv_head,
                seq_len_prefix,
                bank,
                qk_scale,
                xai_temperature_reg,
                q_abs_pos,
                block_start=n_prefix_blocks,
            )

        return acc, l_i, m_i

    @gluon.jit
    def _attn_fwd_inner_extend_pingpong_8w_block_span(
        self,
        acc,
        l_i,
        m_i,
        q_dot,
        cur_seq_q_start_idx,
        cur_kv_head,
        cur_block_m,
        seq_len_extend,
        block_start,
        block_end,
        bank,
        qk_scale,
        xai_temperature_reg,
        mask_base_idx,
        mask_row_stride,
        mask_kv_col_offset,
        EXT_N: gl.constexpr,
        EXT_NS: gl.constexpr,
        MASK_STEPS: gl.constexpr,
        SKIP_BOUNDS_CHECK: gl.constexpr = False,
    ):
        """Run one half-open extend block span with the 8w pingpong loop.

        Processes ``[block_start, block_end)``. Private; used by
        ``attn_fwd_inner_extend_pingpong_8w``. ``EXT_N`` / ``EXT_NS``
        parametrise the block size and pipe depth (8w always calls this
        with ``cfg.BLOCK_N`` / ``cfg.NUM_STAGES`` today, but the param
        shape mirrors the 4w sw-pipeline block-span helper).
        """
        cfg: gl.constexpr = self.cfg
        STREAMS: gl.constexpr = 2
        # warp_pipeline_stage requires >=2 physical LDS buffers. With
        # NS=1 the stage_idx collapses to 0 and iter N's DMA write to
        # smem[0] races iter N+1's relaxed read from smem[0] because
        # membarFilter skips barriers between BufferLoadToLocalOp and
        # syncedViaAsyncWait loads.
        tl.static_assert(
            EXT_NS >= 2,
            "_attn_fwd_inner_extend_pingpong_8w_block_span requires EXT_NS>=2 for "
            "determinism (warp_pipeline_stage needs multiple LDS buffers).",
        )

        kt_loader = AsyncKVLoader.for_extend_k(
            bank.kt_smem,
            self.K_Extend,
            cur_seq_q_start_idx,
            cur_kv_head,
            self.stride_kbs,
            self.stride_kh,
            EXT_N,
            cfg.BLOCK_DMODEL,
            bank.kt_async_layout,
        )
        v_loader = AsyncKVLoader.for_extend_v(
            bank.v_smem,
            self.V_Extend,
            cur_seq_q_start_idx,
            cur_kv_head,
            self.stride_vbs,
            self.stride_vh,
            EXT_N,
            cfg.BLOCK_DV,
            bank.v_async_layout,
        )

        cdna4_async.wait_group(0)

        for stage in gl.static_range(EXT_NS):
            pf_start_n = (block_start + stage) * EXT_N
            if SKIP_BOUNDS_CHECK:
                kt_loader.issue_nomask(pf_start_n, seq_len_extend, slot=stage)
                v_loader.issue_nomask(pf_start_n, seq_len_extend, slot=stage)
            else:
                kt_loader.issue(pf_start_n, seq_len_extend, slot=stage)
                v_loader.issue(pf_start_n, seq_len_extend, slot=stage)

        # Initial wait exposes the first K tile; loop waits expose V and
        # the next K tile while preserving the pingpong ordering.
        WAIT_INIT: gl.constexpr = STREAMS * EXT_NS - (STREAMS - 1)
        WAIT_LOOP: gl.constexpr = STREAMS * EXT_NS - STREAMS
        cdna4_async.wait_group(WAIT_INIT)
        kt_dot = cdna4_async.load_shared_relaxed(
            bank.kt_smem.index(0),
            cfg.layouts.kt_dot_layout,
        )

        main_loop_end = block_end - EXT_NS
        for block_n in tl.range(block_start, main_loop_end, loop_unroll_factor=2):

            with warp_pipeline_stage("dot1", priority=0):
                stage_idx = ((block_n - block_start) % EXT_NS).to(tl.int32)
                start_n = (block_n * EXT_N).to(tl.int32)
                future_start_n = ((block_n + EXT_NS) * EXT_N).to(tl.int32)
                qk = gl.zeros(
                    [cfg.BLOCK_M, EXT_N],
                    dtype=gl.float32,
                    layout=cfg.layouts.mma_layout,
                )
                qk = do_mma(q_dot, kt_dot, qk)
                p, alpha, m_new = self.softmax_part0(
                    m_i,
                    qk,
                    start_n,
                    cur_block_m,
                    seq_len_extend,
                    qk_scale,
                    xai_temperature_reg,
                    mask_base_idx,
                    mask_row_stride,
                    mask_kv_col_offset,
                    EXT_N,
                    MASK_STEPS,
                )

            cdna4_async.wait_group(WAIT_LOOP)

            with warp_pipeline_stage("mem1", priority=1):
                v_dot = cdna4_async.load_shared_relaxed(
                    bank.v_smem.index(stage_idx),
                    cfg.layouts.v_dot_layout,
                )
                if SKIP_BOUNDS_CHECK:
                    kt_loader.issue_nomask(
                        future_start_n, seq_len_extend, slot=stage_idx
                    )
                else:
                    kt_loader.issue(future_start_n, seq_len_extend, slot=stage_idx)

            with warp_pipeline_stage("dot2a", priority=0):
                acc, l_i, m_i = self.softmax_part1(acc, l_i, p, alpha, m_new)

            with warp_pipeline_stage("dot2b", priority=0):
                p_cast = p.to(v_dot.dtype)
                p_dot_reg = gl.convert_layout(p_cast, cfg.layouts.p_dot_layout)
                acc = do_mma(p_dot_reg, v_dot, acc)

            cdna4_async.wait_group(WAIT_LOOP)

            with warp_pipeline_stage("mem2", priority=1):
                next_stage_idx = ((block_n + 1 - block_start) % EXT_NS).to(tl.int32)
                kt_dot = cdna4_async.load_shared_relaxed(
                    bank.kt_smem.index(next_stage_idx),
                    cfg.layouts.kt_dot_layout,
                )
                if SKIP_BOUNDS_CHECK:
                    v_loader.issue_nomask(
                        future_start_n, seq_len_extend, slot=stage_idx
                    )
                else:
                    v_loader.issue(future_start_n, seq_len_extend, slot=stage_idx)

        for tail_i in gl.static_range(EXT_NS):
            cdna4_async.wait_group(STREAMS * (EXT_NS - tail_i) - (STREAMS - 1))
            stage_idx = ((main_loop_end + tail_i - block_start) % EXT_NS).to(tl.int32)
            start_n = (main_loop_end + tail_i) * EXT_N

            kt_dot_tail = cdna4_async.load_shared_relaxed(
                bank.kt_smem.index(stage_idx),
                cfg.layouts.kt_dot_layout,
            )
            qk = gl.zeros(
                [cfg.BLOCK_M, EXT_N], dtype=gl.float32, layout=cfg.layouts.mma_layout
            )
            qk = do_mma(q_dot, kt_dot_tail, qk)

            p, alpha, m_new = self.softmax_part0(
                m_i,
                qk,
                start_n,
                cur_block_m,
                seq_len_extend,
                qk_scale,
                xai_temperature_reg,
                mask_base_idx,
                mask_row_stride,
                mask_kv_col_offset,
                EXT_N,
                MASK_STEPS,
            )

            cdna4_async.wait_group(STREAMS * (EXT_NS - tail_i) - STREAMS)
            v_dot_tail = cdna4_async.load_shared_relaxed(
                bank.v_smem.index(stage_idx),
                cfg.layouts.v_dot_layout,
            )
            acc, l_i, m_i = self.softmax_part1(acc, l_i, p, alpha, m_new)
            p_cast = p.to(v_dot_tail.dtype)
            p_dot_tail = gl.convert_layout(p_cast, cfg.layouts.p_dot_layout)
            acc = do_mma(p_dot_tail, v_dot_tail, acc)

        return acc, l_i, m_i

    @gluon.jit
    def attn_fwd_inner_extend_pingpong_8w(
        self,
        acc,
        l_i,
        m_i,
        q_dot,
        cur_seq_q_start_idx,
        cur_kv_head,
        cur_block_m,
        seq_len_extend,
        bulk_end,
        tail_start,
        tail_end,
        bank,
        qk_scale,
        xai_temperature_reg,
        mask_base_idx,
        mask_row_stride,
        mask_kv_col_offset,
        EXT_N: gl.constexpr,
        EXT_NS: gl.constexpr,
    ):
        """Fused unmasked-bulk + masked-tail extend loop (8w pingpong).

        Same shape as the 4w sw-pipeline fused wrapper: bulk is
        ``[0, bulk_end)``, tail is ``[tail_start, tail_end)``, and
        each phase picks between the warp-pipeline pingpong helper
        (when the block count >= EXT_NS) and the synchronous
        preload-all unpipelined helper (otherwise). The pingpong
        helper itself statically requires EXT_NS>=2 because
        warp_pipeline_stage needs >=2 physical LDS buffers; the
        unpipelined fallback is what lets tiny bulk/tail slices still
        execute under the 8-warp launch. 8w callers always pass
        ``cfg.BLOCK_N`` / ``cfg.NUM_STAGES`` (no FP8-shrunken block).
        """
        if bulk_end >= EXT_NS:
            acc, l_i, m_i = self._attn_fwd_inner_extend_pingpong_8w_block_span(
                acc,
                l_i,
                m_i,
                q_dot,
                cur_seq_q_start_idx,
                cur_kv_head,
                cur_block_m,
                seq_len_extend,
                0,
                bulk_end,
                bank,
                qk_scale,
                xai_temperature_reg,
                mask_base_idx,
                mask_row_stride,
                mask_kv_col_offset,
                EXT_N,
                EXT_NS,
                MASK_STEPS=False,
                SKIP_BOUNDS_CHECK=True,
            )
        elif bulk_end > 0:
            acc, l_i, m_i = self._attn_fwd_inner_extend_unpipelined_block_span(
                acc,
                l_i,
                m_i,
                q_dot,
                cur_seq_q_start_idx,
                cur_kv_head,
                cur_block_m,
                seq_len_extend,
                0,
                bulk_end,
                bank,
                qk_scale,
                xai_temperature_reg,
                mask_base_idx,
                mask_row_stride,
                mask_kv_col_offset,
                EXT_N,
                MASK_STEPS=False,
                SKIP_BOUNDS_CHECK=True,
            )
        remaining_blocks = tail_end - tail_start
        if remaining_blocks >= EXT_NS:
            acc, l_i, m_i = self._attn_fwd_inner_extend_pingpong_8w_block_span(
                acc,
                l_i,
                m_i,
                q_dot,
                cur_seq_q_start_idx,
                cur_kv_head,
                cur_block_m,
                seq_len_extend,
                tail_start,
                tail_end,
                bank,
                qk_scale,
                xai_temperature_reg,
                mask_base_idx,
                mask_row_stride,
                mask_kv_col_offset,
                EXT_N,
                EXT_NS,
                MASK_STEPS=True,
                SKIP_BOUNDS_CHECK=False,
            )
        elif remaining_blocks > 0:
            acc, l_i, m_i = self._attn_fwd_inner_extend_unpipelined_block_span(
                acc,
                l_i,
                m_i,
                q_dot,
                cur_seq_q_start_idx,
                cur_kv_head,
                cur_block_m,
                seq_len_extend,
                tail_start,
                tail_end,
                bank,
                qk_scale,
                xai_temperature_reg,
                mask_base_idx,
                mask_row_stride,
                mask_kv_col_offset,
                EXT_N,
                MASK_STEPS=True,
                SKIP_BOUNDS_CHECK=False,
            )
        return acc, l_i, m_i

    @gluon.jit
    def run(self):
        """Body of ``gluon_extend_attn_fwd_8w`` (NS>=2, NW>=8)."""
        cfg: gl.constexpr = self.cfg
        tl.static_assert(
            cfg.num_warps >= 8, "8w pingpong program requires num_warps>=8"
        )
        tl.static_assert(
            cfg.NUM_STAGES >= 2, "8w pingpong program requires NUM_STAGES>=2"
        )
        if cfg.IS_FP8:
            tl.static_assert(
                cfg.BLOCK_DMODEL != 256,
                "FP8 D=256 unsupported (MFMA_F8 gfx950 codegen failure)",
            )
            if not cfg.IS_WCA:
                tl.static_assert(
                    not (cfg.num_warps == 8 and cfg.BLOCK_DMODEL < 128),
                    "FP8 data-centric 8w DMA path incomplete for D<128; route to WCA.",
                )

        offs_m = gl.arange(0, cfg.BLOCK_M, layout=cfg.layouts.offs_m_layout)
        offs_d = gl.arange(0, cfg.BLOCK_DMODEL, layout=cfg.layouts.offs_d_layout)
        offs_dv = gl.arange(0, cfg.BLOCK_DV, layout=cfg.layouts.offs_d_layout)
        qk_scale = self.sm_scale * LOG2E

        if cfg.IS_WCA:
            tile_idx = gl.program_id(0)
        else:
            tile_idx = 0

        # Same scheduler contract as the 4w program above. The math body is
        # shared in spirit, but WCA's compact tile_idx space and persistent
        # stride loop are still active when SPLIT_K == 1.
        while tile_idx < (self.total_valid_tiles if cfg.IS_WCA else 1):
            if cfg.IS_WCA:
                (
                    cur_seq,
                    cur_head,
                    cur_block_m,
                    cur_kv_head,
                    cur_seq_q_start_idx,
                    seq_len_extend,
                    cur_seq_kv_start_idx,
                    seq_len_prefix,
                    is_valid_tile,
                    output_tile,
                    k_split_id,
                ) = self._schedule_wca(tile_idx)
            else:
                (
                    cur_seq,
                    cur_head,
                    cur_block_m,
                    cur_kv_head,
                    cur_seq_q_start_idx,
                    seq_len_extend,
                    cur_seq_kv_start_idx,
                    seq_len_prefix,
                    is_valid_tile,
                    output_tile,
                    k_split_id,
                ) = self._schedule_data_centric()

            mask_base_idx, mask_row_stride, mask_kv_col_offset = (
                self._compute_mask_state(cur_seq, seq_len_extend, seq_len_prefix)
            )
            q_dot, pfx_q = self._load_q(
                cur_seq_q_start_idx,
                cur_head,
                cur_block_m,
                seq_len_extend,
                offs_m,
                offs_d,
            )
            m_i, l_i, acc = self._init_softmax()
            q_abs_pos, xai_temperature_reg = self._compute_q_abs_pos_and_xai(
                seq_len_prefix, cur_block_m
            )

            pfx_kv_start, pfx_seq_len, pfx_q_abs_pos = self._compute_swa_skip(
                cur_seq_kv_start_idx,
                seq_len_prefix,
                cur_block_m,
                q_abs_pos,
            )
            orig_seq_len_extend = seq_len_extend
            pfx_kv_start, pfx_seq_len, pfx_q_abs_pos, seq_len_extend = (
                self._apply_splitk_prefix_partition(
                    pfx_kv_start,
                    pfx_seq_len,
                    pfx_q_abs_pos,
                    k_split_id,
                    seq_len_extend,
                )
            )

            # 8w pingpong always reuses BLOCK_N / NUM_STAGES for the extend phase.
            _EXT_N: gl.constexpr = cfg.BLOCK_N
            _EXT_NS: gl.constexpr = cfg.NUM_STAGES

            effective_end, n_extend_blocks, n_full_blocks, swa_skip_n_blocks = (
                self._compute_extend_bounds(seq_len_extend, cur_block_m, _EXT_N)
            )

            # Dot layouts live on cfg.layouts; build the async tile layouts and
            # prefix smem policy directly from the same facade.
            kt_offset_bases: gl.constexpr = ExtendAttentionLayouts.make_kt_offset_bases(
                cfg.BLOCK_DMODEL, cfg.BLOCK_N
            )
            v_offset_bases: gl.constexpr = ExtendAttentionLayouts.prefix_v_offset_bases(
                cfg.layouts, cfg.num_warps, cfg.BLOCK_DV, cfg.BLOCK_N
            )
            kt_async_layout: gl.constexpr = ExtendAttentionLayouts.prefix_kt_dll(
                cfg.layouts, cfg.num_warps, cfg.BLOCK_DMODEL, cfg.BLOCK_N
            )
            v_async_layout: gl.constexpr = ExtendAttentionLayouts.prefix_v_dll(
                cfg.layouts, cfg.num_warps, cfg.BLOCK_DMODEL, cfg.BLOCK_DV, cfg.BLOCK_N
            )
            kt_smem_layout: gl.constexpr = ExtendAttentionLayouts.prefix_kt_smem_layout(
                cfg.layouts, cfg.BLOCK_DMODEL, cfg.BLOCK_N, kt_offset_bases
            )
            v_smem_layout: gl.constexpr = ExtendAttentionLayouts.prefix_v_smem_layout(
                cfg.layouts, cfg.BLOCK_N, cfg.BLOCK_DV, v_offset_bases
            )

            if cfg.IS_FP8:
                # FP8 prefix is native FP8, but extend K/V are BF16; select the
                # BF16-extend V exception in the same layout namespace.
                ext_kt_async_layout: gl.constexpr = ExtendAttentionLayouts.make_kt_dll(
                    cfg.num_warps, cfg.BLOCK_DMODEL, cfg.BLOCK_N
                )
                ext_v_async_layout: gl.constexpr = (
                    ExtendAttentionLayouts.make_fp8_extend_v_dll(
                        cfg.num_warps, cfg.BLOCK_DV, cfg.BLOCK_N
                    )
                )
                ext_kt_smem_layout: gl.constexpr = (
                    ExtendAttentionLayouts.make_padded_smem(
                        [cfg.BLOCK_DMODEL, cfg.BLOCK_N],
                        kt_offset_bases,
                        [[512, 16]],
                    )
                )
                ext_v_smem_layout: gl.constexpr = (
                    ExtendAttentionLayouts.make_padded_smem(
                        [cfg.BLOCK_N, cfg.BLOCK_DV],
                        v_offset_bases,
                        [[512, 16]],
                    )
                )
            else:
                ext_kt_async_layout: gl.constexpr = kt_async_layout
                ext_v_async_layout: gl.constexpr = v_async_layout
                ext_kt_smem_layout: gl.constexpr = kt_smem_layout
                ext_v_smem_layout: gl.constexpr = v_smem_layout

            bank = KVSmemBank.initialize(
                cfg,
                kt_smem_layout,
                v_smem_layout,
                kt_async_layout,
                v_async_layout,
                zero_fill=(cfg.IS_FP8 or is_valid_tile),
            )

            # Prefix dispatch: pingpong when we have >= NUM_STAGES full blocks;
            # fall back to unpipelined otherwise (pingpong helper statically
            # requires NUM_STAGES>=2, so we can't call it for smaller prefixes).
            n_full_prefix = pfx_seq_len // cfg.BLOCK_N
            _use_scalar_mask: gl.constexpr = cfg.IS_WCA
            if n_full_prefix >= cfg.NUM_STAGES:
                acc, l_i, m_i = self.attn_fwd_inner_prefix_pingpong_8w(
                    acc,
                    l_i,
                    m_i,
                    pfx_q,
                    pfx_kv_start,
                    cur_kv_head,
                    pfx_seq_len,
                    bank,
                    qk_scale,
                    xai_temperature_reg,
                    pfx_q_abs_pos,
                    _use_scalar_mask,
                )
            elif pfx_seq_len > 0:
                acc, l_i, m_i = self.attn_fwd_inner_prefix_unpipelined(
                    acc,
                    l_i,
                    m_i,
                    pfx_q,
                    pfx_kv_start,
                    cur_kv_head,
                    pfx_seq_len,
                    bank,
                    qk_scale,
                    xai_temperature_reg,
                    pfx_q_abs_pos,
                )

            if cfg.IS_FP8:
                bank = bank.transition_to_extend(
                    cfg,
                    self.Q_Extend.dtype.element_ty,
                    ext_kt_smem_layout,
                    ext_v_smem_layout,
                    ext_kt_async_layout,
                    ext_v_async_layout,
                    _EXT_N,
                    _EXT_NS,
                )

            # Extend hot-loop (fused unmasked-bulk + masked-tail 8w pingpong).
            masked_start = tl.maximum(n_full_blocks, swa_skip_n_blocks)
            acc, l_i, m_i = self.attn_fwd_inner_extend_pingpong_8w(
                acc,
                l_i,
                m_i,
                q_dot,
                cur_seq_q_start_idx,
                cur_kv_head,
                cur_block_m,
                seq_len_extend,
                n_full_blocks,
                masked_start,
                n_extend_blocks,
                bank,
                qk_scale,
                xai_temperature_reg,
                mask_base_idx,
                mask_row_stride,
                mask_kv_col_offset,
                _EXT_N,
                _EXT_NS,
            )

            l_i = self._apply_sinks(cur_head, l_i, m_i)

            if cfg.IS_WCA and cfg.SPLIT_K > 1:
                self._splitk_partial_store_and_reduce(
                    output_tile,
                    k_split_id,
                    cur_seq_q_start_idx,
                    cur_head,
                    cur_block_m,
                    orig_seq_len_extend,
                    acc,
                    l_i,
                    m_i,
                    offs_m,
                    offs_dv,
                )
            else:
                self._normalize_and_store(
                    cur_seq_q_start_idx,
                    cur_head,
                    cur_block_m,
                    seq_len_extend,
                    acc,
                    l_i,
                    offs_m,
                    offs_dv,
                )

            if cfg.IS_WCA:
                tile_idx += self.total_programs
            else:
                tile_idx = 1
