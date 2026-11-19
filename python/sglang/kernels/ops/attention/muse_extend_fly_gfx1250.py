# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Muse Spark extend (prefill) attention on gfx1250, FlyDSL: aiter's a8w8 ``m32x8`` FMHA design
(aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_fwd_prefill_a8w8_m32x8.py) with P biased into the upper e4m3 range,
plus the fused FP8 K/V store + Q quantization for prefix-free extends (``muse_extend_attention_fly_gfx1250``).

MHA Forward Prefill kernel — ``m32x8`` design, gfx1250.

A clean FlyDSL kernel written in the high-level layout-algebra style
(tiled copy / tiled MMA + ``SharedAllocator``).

``m32x8`` names the threadgroup shape: **8 waves per threadgroup**, each wave
owning a **32-row** Q span (2 adjacent 16-row WMMA tiles). gfx1250 runs wave32,
so a threadgroup is ``8 * 32 = 256`` threads and ``BLOCK_M = 32 * 8 = 256`` Q
rows. (The leading ``32`` is per-wave Q rows; ``16`` is the WMMA M dimension.)

Layout support — two device kernels over one shared compute core (option B):
  - ``muse_extend_fly_m32x8_thd``  — varlen THD, driven by ``cu_seqlens``.
  - ``muse_extend_fly_m32x8_bshd`` — batched BSHD, uniform ``seq_len`` scalar
    (no ``cu_seqlens`` tensors → nothing transient to bake into a CUDA graph).
Both resolve their per-workgroup base offsets + sequence bounds, then call the
layout-agnostic ``_core_attention`` helper.

Scope — v1 (this file is intentionally config-agnostic in its name):
  - ``qk_hdim in {64, 128, 192, 256}`` (D_qk), ``v_hdim in {64, 128, 256}`` (D_v); they
    are independent. ``n_block`` is picked by ``pick_n_block``. v_hdim 256 runs at
    BLOCK_M 128 only: the BLOCK_M 256 O tile does not fit its LDS staging chunk
  - dtype: fp8/bf8 for Q/K/V (independently), bf16 for O
  - grouped-query attention (GQA): ``gqa = nheads_q // nheads_k``
  - causal and non-causal

``qk_hdim``, ``v_hdim`` and the dtype are compile-time (build-time) parameters
captured by the builder closure, so they never appear in the file name and can
be generalized later without changing the runtime kernel signatures.

Target: gfx1250, wave32, 8 waves per threadgroup (256 threads).
"""

import functools
from enum import IntEnum

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
import triton
import triton.language as tl
from aiter.jit.utils.chip_info import get_lds_capacity_bytes
from aiter.ops.flydsl.kernels import buffer_ops

# Single source of truth for gfx1250 Expert Scheduling Mode 2 (DEP_MODE=2). Lives
# in fmha_buffer_managers. Under mode 2 the LLVM setreg (via the
# amdgpu-expert-scheduling-mode hint, set in _ensure_*_kernel) makes LLVM insert all
# depctr covers itself for the plain intrinsics the kernel emits.
from aiter.ops.flydsl.kernels.fmha_gfx1250.fmha_buffer_managers import (
    ENABLE_SCHED_MODE2,
    KManager16bV1,
    KManager16bV2,
    ProducerCtx,
    QManager16bV1,
    QManager16bV2,
    VManager16bV1,
    VManager16bV2,
    _ir,
)
from aiter.ops.flydsl.kernels.fmha_gfx1250.fmha_utils import (
    DTYPE_MAP,
    LDS_CHUNK_BYTES,
    LDS_QO_BYTES,
    NUM_Q_TILES_CHOICES,
    NUM_WAVES,
    WMMA_M,
    _alloc_lds,
    _bare_barrier,
    _block_m,
    _concat,
    _copy_fence,
    _copy_wait,
    _epi_state,
    _keepalive,
    _kv_drain_depths,
    _lane_id,
    _load_seqlen_pair,
    _load_sink_logit,
    _m_tile_idx,
    _o_manager,
    _o_pack,
    _o_unpack,
    _packed_tile_indices,
    _pick_num_q_tiles,
    _ring_drive,
    _ring_num_prefetch,
    _tree_reduce_multi,
    _warp_id,
    _wg_kv_span,
    schedule_bank,
    supports_gqa_ratio,
)
from aiter.ops.flydsl.kernels.kernels_common import LOG2E
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
from flydsl.expr import arith, const_expr, gpu, rocdl

# Q/K/V staging managers (own their LDS swizzles + async copy schedules). They are
# self-contained: this kernel maintains its own arch constants below and passes the
# config each manager needs through its constructor.
from flydsl.expr.rocdl import tdm_ops
from flydsl.expr.typing import T
from flydsl.expr.utils.arith import _to_raw as _raw

# ============================================================================
# Threadgroup / arch constants
# ============================================================================

WAVE_SIZE = 32  # gfx1250 kernels run wave32
BLOCK_SIZE = WAVE_SIZE * NUM_WAVES  # 256 threads

# "m32x8": each wave owns WMMA_ROW_PER_WAVE adjacent 16-row (WMMA M) Q sub-tiles →
# BLOCK_M = 16 * 2 * 8 = 256 Q rows per threadgroup. Each wave's 2 tiles are
# contiguous: warp i owns rows [i*32, i*32+32) = tiles 2i, 2i+1.
WMMA_N = 16  # kv rows per WMMA tile (the S^T=K@Q^T output's n_block-direction axis)
WMMA_K = 32  # WMMA contraction depth (bf16 v_wmma_f32_16x16x32); d-tile width
# 8-bit WMMA contracts twice as deep for the same 8-VGPR operand: 16x16x64 carries
# 32 B/lane of fp8 where 16x16x32 carries 32 B/lane of bf16 (probe-confirmed on HW).
WMMA_K_B8 = 64
WMMA_ROW_PER_WAVE = 2  # Q WMMA tiles per wave (the "x2" step from m16x8 to m32x8)
BLOCK_M = WMMA_M * WMMA_ROW_PER_WAVE * NUM_WAVES  # 256
# 1 => BLOCK_M 128 ("m16x8"), 2 => BLOCK_M 256. Chosen per call by _pick_num_q_tiles.


# A wave owns WMMA_M * num_q_tiles contiguous packed (seq, q_head_in_group) rows, and Q/O
# describe that tile with ONE rectangular TDM box. That only works when a head group is
# never split across waves: either a wave holds whole groups, or it sits inside one.


# Causal work grows with the M index, so dispatch the heavy tiles first.


class WarpType(IntEnum):
    """Warp-specialization role (compile-time). gfx1250 pairs wave i with wave i+4 on
    one SIMD; the low half (waves 0..3) and high half (waves 4..7) run different
    main-loop preamble orderings so one wave drives memory while its SIMD-mate computes.

    SIMD parity (wave w -> SIMD w%4) is NOT a role: the swapped KV-half read order
    carries it as a runtime term instead, so the body stays 2-way.
    """

    LO = 0
    HI = 1

    @property
    def is_lo(self):
        return self is WarpType.LO


DEFAULT_QK_HDIM = 128
DEFAULT_V_HDIM = 128
DEFAULT_DTYPE = "fp8"
_B8_DTYPES = (fx.Float8E4M3FN, fx.Float8E5M2)
# ``q_dtype`` types Q, ``kv_dtype`` types K and V; the two are independent (the wmma
# takes a per-operand type) so fp8 Q against bf8 KV, or the reverse, is a legal build.
# P and O follow neither: softmax P is always fp8 -- the gemm sees UNNORMALIZED
# exp(S - m), whose row max is exactly 1.0, so the extra mantissa bit beats bf8's
# exponent range (terms under e4m3's floor are negligible against a sum >= 1). The O
# accumulator and store stay 16-bit.
P_DTYPE = fx.Float8E4M3FN
_TORCH_P_DTYPE = {
    fx.Float8E4M3FN: torch.float8_e4m3fn,
    fx.Float8E5M2: torch.float8_e5m2,
}[P_DTYPE]
O_DTYPE = fx.BFloat16
TORCH_O_DTYPE = torch.bfloat16
# Q/K/V inputs this kernel serves; 16-bit belongs to the a16w16 sibling.
_TORCH_IN_DTYPES = {torch.float8_e4m3fn: "fp8", torch.float8_e5m2: "bf8"}
# What THIS kernel serves. DTYPE_MAP is shared with the 16-bit sibling and names every
# dtype, so validation must gate on these, never on the map's keys.
SUPPORTED_DTYPES = tuple(_TORCH_IN_DTYPES.values())
SUPPORTED_TORCH_DTYPES = tuple(_TORCH_IN_DTYPES)
SUPPORTED_QK_HDIM = (64, 128, 192, 256)
SUPPORTED_V_HDIM = (64, 128, 256)

# KV sequence block (columns of one QK GEMM tile). Configurable; 64 for now.
N_BLOCK_CHOICES = (32, 64, 128, 256)
# ---- LDS chunk layout: 12 x 26 KB = 312 KB of the 320 KB budget. ----
# Chunk i sits at LDS_CHUNK_BYTES * i. Every K|V tile is split 2-way along n_block, the
# halves 6 chunks apart so they land in different 64 KB LDS segments:
#
#   K[0][0]   0..26  seg 0      K[0][1] 156..182  seg 2
#   V[0][0]  26..52  seg 0      V[0][1] 182..208  seg 2,3
#   K[1][0]  52..78  seg 0,1    K[1][1] 208..234  seg 3
#   V[1][0]  78..104 seg 1      V[1][1] 234..260  seg 3,4
#   K[2][0] 104..130 seg 1,2    K[2][1] 260..286  seg 4
#   V[2][0] 130..156 seg 2      V[2][1] 286..312  seg 4
#
# Q and O own no LDS -- they time-share KV chunks (see _q_wave_base / _o_wave_base).
KV_LDS_SPLITS = 2  # halves one K (or V) tile is split into along n_block
# Per-wave Q / O region inside a chunk pair. 17 KB covers 32 rows x 256 hdim + pad.

DEFAULT_N_BLOCK = 64
# Auto-picked n_block, widest first: a wider tile must still fit one chunk per split half
# and the register file (128 doubles the live fragment state, spilling above this hdim).
N_BLOCK_PREF = (128, 64)
N_BLOCK_WIDE_MAX_QK_HDIM = 128

# K|V LDS slots the main loop rotates through. 3 is the minimum for the software-pipelined
# body: it reads V(tile_idx-1) from slot 0 and K(tile_idx) from slot 1 while tile_idx+1 is
# written into slot 2. Two slots would need V written into the slot this body still reads,
# which also races the carried ring head.
N_KV_PP = 3

# Softmax runs in log2 units: LOG2E is folded into Q by the Q loader's bf16 multiply, so
# the inner loop is a plain exp2(S' - m) and m/LSE live in the log2 domain.

# Deferred oaccu rescale (FAv4, hk_mla spec §9.1.1): corr == 1 when the running max does
# not move, so keep m STALE while the row max stays within RESCALE_THRESHOLD natural
# logits of it and skip the full-width multiply. False (or threshold < 0) always rescales.
ENABLE_DEFER_RESCALE = True
# P = exp(S - m_stale) reaches exp(RESCALE_THRESHOLD), so the threshold is capped by
# P_DTYPE's finite max: e4m3 tops out at 448 (exp 6.10), e5m2 at 57344 (exp 10.96).
# Overflowing e4m3 converts to NaN while the denominator keeps the exact f32 sum, so
# the cap is derived from P_DTYPE rather than written down -- retyping P re-derives it.
# Muse: P is biased by 2**P_BIAS_LOG2 into the upper e4m3 range (the denominator sums the same biased P, so the
# bias cancels) and the running max is never left stale (threshold 0), so P <= 256 < 448: relL2 vs FP32 matches the
# Triton extend kernel at the 8K Ulysses shape (a 0.3 rescale threshold is slightly faster and less accurate;
# non-integer biases were less accurate).
P_BIAS_LOG2 = 8.0
RESCALE_THRESHOLD = 0.0

# Running-max seed: finite big-negative, not -inf, so a fully-masked row keeps m finite
# and softmax's (m_prev - m_new) / fma(s, .., -m) never do -inf arithmetic (NaN).
# exp2(big_neg - real) still underflows to 0. Masked scores themselves stay -inf.
BIG_NEG = -1.0e30

# Compile-time Q/K/V loader select. False = V1 (cluster_load_async), True = V2 (TDM,
# HW OOB, far fewer address VGPRs). The two differ only in transport, so this picks a
# class pair and no call site branches on it. O is selected separately by O_VARIANT.
USE_TDM_LOADER = True

# K/V producer specialization: LO issues every K copy, HI every V copy, each wave taking
# one dense n_block/KV_PRODUCER_WARPS row band, so a half's counter tracks one operand.
KV_PRODUCER_WARPS = NUM_WAVES // 2

# LO/HI anti-phase (FA3 ping-pong). Same tile stream, same barrier count, opposite phase:
#   LO body tile_idx:  gemm(tile_idx)                 | BAR | softmax(tile_idx)
#   HI body tile_idx:  softmax(tile_idx-1) + rescale  | BAR | gemm(tile_idx)
# so one half is in the WMMA stream while the other is in the softmax VALU stream. HI
# carries s_acc instead of P, seeded to -inf so its dead leading softmax is a no-op.
# O writer variant (decoupled from USE_TDM_LOADER): "v1" swizzled LDS + buffer_store (fastest so
# far), "v2" TDM store (padding ignored -> contiguous LDS -> bank conflict, slow), "v3" padded LDS +
# global_store_async_from_lds_b128.

# LDS->VGPR ring for the fused PV+QK WMMA stream (``_pv_qk_gemm``). Holds only RING loads
# live at 4 VGPR each, num_prefetch = RING - 2*LAG in flight, instead of bursting a whole
# tile before the first wmma. Keep RING a multiple of 4 -- an odd wrap flips slot parity,
# costing a cycle on half the wmma.
PVQK_RING = 24
PVQK_LAG = 2
# Ring-head loads issued at the END of a body (under the softmax VALU) and carried across
# the back edge, instead of at the top of the body that consumes them. Clamped to
# num_prefetch; 0 restores the un-carried head. Costs 4 VGPR per carried load.
PVQK_HEAD_CARRY = 20

# NOTE: the remaining tiling constants (chunk sizes, K/V write-tile + V swizzle
# granularity) live inside fmha_buffer_managers.py — they are intrinsic to the
# managers' LDS layouts, so the kernel no longer declares them here.


# ============================================================================
# Small device helpers
# ============================================================================


# ============================================================================
# Compute stages
# ============================================================================


_B8_WMMA = {
    (fx.Float8E4M3FN, fx.Float8E4M3FN): rocdl.wmma_f32_16x16x64_fp8_fp8,
    (fx.Float8E4M3FN, fx.Float8E5M2): rocdl.wmma_f32_16x16x64_fp8_bf8,
    (fx.Float8E5M2, fx.Float8E4M3FN): rocdl.wmma_f32_16x16x64_bf8_fp8,
    (fx.Float8E5M2, fx.Float8E5M2): rocdl.wmma_f32_16x16x64_bf8_bf8,
}


def _load_descale(ptr):
    """One per-tensor fp32 descale. Uniform and loop-invariant, so it hoists."""
    return fx.Float32(fx.ptr_load(fx.get_iter(ptr)))


def _wmma(a, b, c, *, a_dtype, b_dtype):
    """One WMMA: C[16x16 f32] = A @ B + C, operand reuse off, v8 f32 accumulator.

    ``16x16x64`` -- twice bf16's contraction depth for the same 8 VGPRs. A and B may mix
    FP8(E4M3FN) and BF8(E5M2) freely, one intrinsic per combination. Operands arrive as
    v8 i32 (the intrinsics reject float vectors, and nothing on the 8-bit path ever
    materializes an f8 vector type), so the caller names their element types.
    """
    v8f32 = fx.Vector.make_type(8, fx.Float32)
    wmma = _B8_WMMA[(a_dtype, b_dtype)]
    # modC defaults to WMMACModifier::none (== the old modC=0); omit it.
    return wmma(v8f32, _ir(a), _ir(b), _ir(c), reuseA=False, reuseB=False).result


_B8_PK8_CVT = {
    fx.Float8E4M3FN: rocdl.cvt_scalef32_pk8_fp8_f32,
    fx.Float8E5M2: rocdl.cvt_scalef32_pk8_bf8_f32,
}


def _convert_vectors(vec_lists, dtype):
    """Convert a list of lists of vectors to ``dtype``.

    ``.to()`` is no good for an 8-bit result: it emits ``arith.truncf``, which has no
    lowering pattern for an f8 vector, so the value stays f8-typed while the pipeline
    legalizes its users to i8 and the bridging cast dies unlegalized. The hardware
    convert sidesteps that and does all 8 elements in one instruction; scale is 1."""
    cvt = _B8_PK8_CVT[dtype]
    v2i32 = fx.Vector.make_type(2, fx.Int32)
    one = _ir(fx.Float32(1.0))
    return [[fx.Vector(cvt(v2i32, _ir(v), one)) for v in row] for row in vec_lists]


def _softmax(
    *,
    s_list,
    m_prev_list,
    d_prev_list,
    lane_idx,
    n_block,
    kv_swap_delta,
    kv_pos_base=None,
    q_max_list=None,
    q_min_list=None,
    kv_len=None,
    s_scale,
):
    """Online-softmax update for one KV tile, for ALL num_q_tiles q-WMMA-tiles this wave owns.

    Every ``*_list`` argument and return is length num_q_tiles (= num_q_tiles_per_wave). The num_q_tiles rows are
    independent (each owns its S, running m/d and mask bounds) but share the tile's K/V,
    so processing them together lets ``_tree_reduce_multi`` interleave their max/sum trees
    and dual-issue the combines, hiding each other's permlanex16 latency. ``s_list`` is
    already in log2 units (the loader folds softmax_scale*LOG2E into Q), so exp is exp2.

    Layout (from ``_pv_qk_gemm``): ``s_list[r]`` is ``num_kv_frags = n_block//WMMA_N`` v8-f32
    accumulators; this lane owns query ``q = warp*16 + l%16`` and, in tile ``kv_frag_idx``, kv rows
    ``kv_frag_idx*16 + (l//16)*8 + [0..8)``. The peer lane ``l^16`` holds the other 8-row half of
    the same q, so rows reduce locally over (kv_frag_idx, i) then across ``shuffle_xor(16)``.

    ``kv_swap_delta`` corrects the swapped half-order read: when the wave takes the tile's
    two halves in the other order, slot ``kv_frag_idx`` holds physical kv-tile ``kv_frag_idx ^ (num_kv_frags/2)``,
    so its absolute offset moves by +/- ``n_block/2``. A runtime Int32 (0 on even SIMDs)
    folded into two per-body mask bases rather than into the num_kv_frags per-slot constants.

    Masking (per element, sequence-relative ``kv_pos = kv_pos_base + (l//16)*8 + kv_frag_idx*16 +
    i``; a None bound is skipped): ``q_max_list[r]`` masks ``kv_pos > q_max`` (band upper
    edge ``q_seq + (kv_len-q_len) + window_right``, clamped to ``kv_len-1`` on the last
    tile to fold the tail); ``q_min_list[r]`` masks ``kv_pos < q_min`` (lower edge, minus
    ``window_left``); ``kv_len`` masks ``kv_pos >= kv_len`` and applies only when q_max is
    None (the standalone non-causal tail).

    Returns ``(p, m_new, d_new, corr)`` plus ``rescale_masks``. p is num_kv_frags v8 **f32** P^T =
    exp(S^T - m_new), NOT narrowed to the wmma element type -- the caller places that with
    ``_convert_vectors``. m_new is STALE (== m_prev) when that row's ballot did not fire (FAv4
    §9.1.1), with corr == 1 and d_new = corr*d_prev + rowsum(p). ``rescale_masks`` is the num_q_tiles
    raw ballots (None when deferral is compiled out), for the caller to fold into its one
    branch condition at the use site.
    """
    num_kv_frags = n_block // WMMA_N
    f32 = T.f32
    fast = arith.FastMathFlags.fast
    neg_inf = fx.Float32(float("-inf"))
    _defer = ENABLE_DEFER_RESCALE and RESCALE_THRESHOLD >= 0.0

    def fmax(a, b):
        return fx.maxnumf(a, b, fastmath=fast)

    def fadd(a, b):
        return a.addf(b, fastmath=fast)

    # fast-math WITHOUT reassoc: LLVM's Reassociate pass otherwise re-linearizes the
    # sum tree back into a serial chain (max survives — Reassociate ignores maxnum).
    _fast_math = arith.FastMathFlags
    _no_reassoc = (
        _fast_math.nnan
        | _fast_math.ninf
        | _fast_math.nsz
        | _fast_math.arcp
        | _fast_math.contract
        | _fast_math.afn
    )

    def fadd_t(a, b):
        return a.addf(b, fastmath=_no_reassoc)

    def fsub(a, b):
        with arith.fastmath(fast):
            return a - b

    def fmul(a, b):
        with arith.fastmath(fast):
            return a * b

    def fma_inf(a, b, c):
        # a*b + c with NO ninf fast-math: masked s is -inf and must stay -inf, and
        # -inf * (positive finite) + finite is still -inf.
        return fx.fma(a, b, c)

    def exp2(x):
        return fx.exp2(x, fastmath="afn")

    # permlanex16 selectors: identity cross-16 gather (nibbles 0..15) => lane l<->l^16.
    sel_lo, sel_hi = _raw(fx.Int32(0x76543210)), _raw(fx.Int32(0xFEDCBA98))

    def peer(v):  # cross-lane reduce partner: lane l <-> l^16 (the other kv half)
        return fx.Float32(
            rocdl.permlanex16(
                f32,
                _raw(v),
                _raw(v),
                sel_lo,
                sel_hi,
                fi=False,
                bound_control=False,
            )
        )

    kv_half_idx = lane_idx // fx.Int32(
        WMMA_M
    )  # 0/1: which 8-row kv half this lane owns

    # Mask base: swapped reads move the low num_kv_frags/2 slots up by n_block/2 and the high slots
    # down, so one base per half absorbs the whole correction and the per-slot term stays
    # the compile-time kv_frag_idx*WMMA_N + i.
    if kv_pos_base is not None:
        _kv_pos_half_base = kv_pos_base + kv_half_idx * fx.Int32(8)
        _kv_pos_halves = [
            _kv_pos_half_base + kv_swap_delta,
            _kv_pos_half_base - kv_swap_delta,
        ]

    num_q_tiles = len(s_list)
    q_max_list = q_max_list if q_max_list is not None else [None] * num_q_tiles
    q_min_list = q_min_list if q_min_list is not None else [None] * num_q_tiles

    # ---- Pass 1: masked S, flattened (kv_frag_idx, i) order. Built for every row first so the
    # max-trees below emit INTERLEAVED. ----
    s_masked_list = []
    for r in range(num_q_tiles):
        s = s_list[r]
        q_max, q_min = q_max_list[r], q_min_list[r]
        s_masked = []
        for kv_frag_idx in range(num_kv_frags):
            svec = fx.Vector(_ir(s[kv_frag_idx]))
            for i in range(8):
                sval = fx.Float32(svec[i])
                if q_max is not None or q_min is not None or kv_len is not None:
                    kv_pos = _kv_pos_halves[
                        kv_frag_idx >= num_kv_frags // 2
                    ] + fx.Int32(kv_frag_idx * WMMA_N + i)
                    if q_max is not None:
                        ubound = (
                            q_max
                            if kv_len is None
                            else fx.min(q_max, kv_len - fx.Int32(1))
                        )
                        sval = (kv_pos > ubound).select(neg_inf, sval)
                    if q_min is not None:
                        sval = (kv_pos < q_min).select(neg_inf, sval)
                    if kv_len is not None and q_max is None:
                        sval = (kv_pos >= kv_len).select(neg_inf, sval)
                s_masked.append(sval)
        s_masked_list.append(s_masked)

    # ---- Row max: balanced max-trees, interleaved across rows. ----
    max3 = lambda a, b, c: fmax(fmax(a, b), c)
    local_max_list = _tree_reduce_multi(s_masked_list, max3, fmax)

    # ---- Per row: peer reduce + deferred-rescale decision + corr. ----
    m_new_list, corr_list = [], []
    rescale_masks = None if not _defer else []
    for r in range(num_q_tiles):
        m_prev, q_min = m_prev_list[r], q_min_list[r]
        row_max = fmax(local_max_list[r], peer(local_max_list[r]))
        # S is left RAW through mask and max tree so the scale has a single consumer
        # (the exp argument) and folds into its subtract. Valid because the scale is
        # positive: max(c*S) == c*max(S). One multiply per row, not per element.
        row_max = fmul(row_max, s_scale)
        m_full = fmax(m_prev, row_max)

        # Deferred oaccu rescale: keep m STALE while the running max barely moves, so the
        # caller SKIPS the wide `o_acc *= corr`. The ballot promotes the test to
        # wave-uniform for that branch. `>` lowers to ordered OGT, so a fully-masked
        # lane's -inf - -inf = NaN compares false and never forces a rescale.
        if _defer:
            need = fsub(row_max, m_prev) > fx.Float32(RESCALE_THRESHOLD * LOG2E)
            # Select on the per-lane `need`, not the ballot: every lane owns its own q row
            # (the l<->l^16 pair agrees after the peer reduce), so staleness is per-lane
            # and the ballot is only the caller's branch condition. That leaves the branch
            # as the compare's only other consumer, so the fold sinks to the s_cbranch
            # instead of sitting between the max tree and the exp chain.
            m_new = need.select(m_full, m_prev)
            rescale_masks.append(fx.Int32(rocdl.ballot(fx.Int32.ir_type, need)))
        else:
            m_new = m_full

        # log2-domain, so exp2 takes the difference directly. m is seeded to BIG_NEG
        # (finite), so a fully masked row keeps m_new = BIG_NEG -> corr = 1 and
        # p = exp2(-inf) = 0, with no (-inf)-(-inf) anywhere and no clamp needed.
        corr = exp2(fsub(m_prev, m_new))
        m_new_list.append(m_new)
        corr_list.append(corr)

    # ---- Pass 2: p = exp2(S - m_new), every row's subs before any exp. The subs pair
    # into v_dual_sub_f32 against their row's shared m_new, and the exp run stays pure
    # TRANS -- the coexecution hazard is TRANS followed by non-TRANS VALU, so an
    # uninterrupted run needs none of the v_nop padding an interleaved one does. ----
    # exp2(c*S - m) as one fma: the scale's only consumer, so it contracts.
    neg_m = [fsub(fx.Float32(P_BIAS_LOG2), m_new_list[r]) for r in range(num_q_tiles)]
    # Score pairs through v_pk_fma_f32: one VALU op per two exp arguments.
    diff_list = []
    for r in range(num_q_tiles):
        sm = s_masked_list[r]
        sc2 = fx.Vector.from_elements([s_scale, s_scale], fx.Float32)
        nm2 = fx.Vector.from_elements([neg_m[r], neg_m[r]], fx.Float32)
        d = []
        for j in range(0, len(sm), 2):
            r2 = fx.Vector(
                fx.fma(
                    fx.Vector.from_elements([sm[j], sm[j + 1]], fx.Float32), sc2, nm2
                )
            )
            d += [fx.Float32(r2[0]), fx.Float32(r2[1])]
        diff_list.append(d)
    p_flat_list = [[exp2(d) for d in diff] for diff in diff_list]
    p_list = [
        [
            fx.Vector.from_elements(
                p_flat[kv_frag_idx * 8 : (kv_frag_idx + 1) * 8], fx.Float32
            )
            for kv_frag_idx in range(num_kv_frags)
        ]
        for p_flat in p_flat_list
    ]

    # ---- Row sum: interleaved balanced sum-trees. fadd_t drops reassoc so LLVM's
    # Reassociate does not re-linearize the tree into a serial chain. ----
    # Row sums on 2-vectors (v_pk_add_f32); the two halves combine at the end.
    pairs_list = [
        [
            fx.Vector.from_elements([pf[j], pf[j + 1]], fx.Float32)
            for j in range(0, len(pf), 2)
        ]
        for pf in p_flat_list
    ]
    vadd = lambda a, b: a + b
    sums2 = _tree_reduce_multi(pairs_list, lambda a, b, c: vadd(vadd(a, b), c), vadd)
    local_sum_list = [fadd_t(fx.Float32(v[0]), fx.Float32(v[1])) for v in sums2]

    d_new_list = []
    for r in range(num_q_tiles):
        d_new_list.append(
            fadd(
                fmul(corr_list[r], d_prev_list[r]),
                fadd(local_sum_list[r], peer(local_sum_list[r])),
            )
        )
    return p_list, m_new_list, d_new_list, corr_list, rescale_masks


def _pv_qk_gemm(
    *,
    v_emit,
    k_emit,
    p_list,
    q_frags_list,
    v_hdim,
    n_block,
    warp_type,
    o_acc_list=None,
    head=None,
    ring=PVQK_RING,
    ring_lag=PVQK_LAG,
    vld_per_frag=2,
    q_dtype=None,
    kv_dtype=None,
    phase="pvqk",
):
    """GEMM2(u-1) then GEMM1(u) driven by ONE ring over the concatenated V-then-K
    fragment streams, for all num_q_tiles q-WMMA-tiles this wave owns.

    V and K are shared across the q-tiles, so each fragment is shuffled once and fed into
    num_q_tiles independent WMMA chains. Consumption stays strictly sequential -- a full PV pass
    then a full QK pass -- so P dies before the first QK wmma and the two accumulator sets
    are never co-live. What the single ring buys is the refill window: the K loads for
    QK(u) issue ~NP/2 fragments before the transition, while PV's wmma stream still runs.

    GEMM2, O^T = V^T @ P^T, D[M=d, N=q]: **A = V^T** (src_a, transpose-loaded via
    ds_load_tr16_b128, two V-tiles (kv, kv+16) shuffled) and **B = P^T** (src_b, two
    softmax tiles p[2*k_tile_idx], p[2*k_tile_idx+1]). Contracts kv in ``num_contract_tiles = n_block//WMMA_K`` tiles and
    produces ``d_tiles = v_hdim//WMMA_M`` d-tiles on the M axis. Lane ``l`` element ``si``
    of tile ``d_tile_idx`` holds O[q = l%16, d = d_tile_idx*WMMA_M + (l//16)*8 + si] -- the OManager16b
    fragment layout.

    GEMM1, S^T = K @ Q^T: **K = A-operand**, **Q = B-operand**. Contracts d in
    ``num_k_tiles = qk_hdim//WMMA_K`` tiles and produces ``num_kv_frags = n_block//WMMA_N`` kv-tiles. Lane
    ``l`` element ``si`` holds S^T[kv = kv_tile*WMMA_N + (l//16)*8 + si, q = l%16] (kv on
    the C-row / M axis, q on the C-col / N axis; GPU-verified). Each K fragment is the two
    16-col halves of a d-tile shuffled into a v16 matching the Q frag layout.

    ``p_list``/``q_frags_list``/``o_acc_list`` are length-num_q_tiles lists: per q-tile its bf16 P^T
    B-operands, its num_k_tiles v16-bf16 Q fragments, and (or None) its running ``d_tiles`` v8-f32
    O accumulator, already rescaled by ``corr``, which PV accumulates onto online.
    ``v_emit(j)``/``k_emit(j)`` emit the j-th V transpose-load / K ds_load of the resident
    block in flat ``(d_tile_idx, k_tile_idx, half)`` and ``(kv, d_tile_idx, half)`` order; ``_ring_drive`` calls
    them on demand. ``head`` adopts the NP-deep prefetch the caller issued before softmax,
    so that latency still hides under the softmax VALU.

    Returns ``(out_list, s_acc_list)``: the updated O accumulators, and per q-tile num_kv_frags
    v8-f32 S^T accumulators.
    """
    num_q_tiles = len(p_list)
    d_tiles = v_hdim // WMMA_M
    # Contraction depth follows the V load width: each tile is one WMMA's worth of kv,
    # vld_per_frag transpose loads of 16 rows each (32 kv at 16-bit, 64 at 8-bit).
    num_contract_tiles = n_block // (vld_per_frag * WMMA_N)
    num_kv_frags = n_block // WMMA_N
    num_k_tiles = len(q_frags_list[0])
    # softmax tiles per PV B operand: one per 16 kv rows of the contraction
    num_p_parts = num_kv_frags // num_contract_tiles

    # ``phase`` drops one half of the stream when the caller knows it is dead (the
    # single-KV-tile core runs QK and PV as separate passes around softmax).
    do_pv = phase != "qk"
    do_qk = phase != "pv"
    num_v_frags = d_tiles * num_contract_tiles if do_pv else 0
    num_k_frags = num_kv_frags * num_k_tiles if do_qk else 0
    num_vld = vld_per_frag * num_v_frags

    out_list = [[None] * d_tiles for _ in range(num_q_tiles)]
    if not do_pv:
        out_list = [list(row) for row in o_acc_list]
    s_acc_list = [[None] * num_kv_frags for _ in range(num_q_tiles)]
    p_frags = {}  # (q_tile_idx, k_tile_idx) -> B operand, handed to _ring_drive on its last fragment

    def emit(j):
        return v_emit(j) if j < num_vld else k_emit(j - num_vld)

    def consume(i, *parts):
        if i < num_v_frags:
            d_tile_idx, k_tile_idx = divmod(i, num_contract_tiles)
            v_frag = _concat(parts)
            for q_tile_idx in range(num_q_tiles):
                acc = out_list[q_tile_idx][d_tile_idx]
                if acc is None:
                    acc = (
                        o_acc_list[q_tile_idx][d_tile_idx]
                        if o_acc_list is not None
                        else fx.Vector.filled(8, 0.0, fx.Float32)
                    )
                p = p_list[q_tile_idx]
                p_frag = _concat(
                    [p[num_p_parts * k_tile_idx + t] for t in range(num_p_parts)]
                )
                p_frags[(q_tile_idx, k_tile_idx)] = p_frag
                out_list[q_tile_idx][d_tile_idx] = _wmma(
                    v_frag, p_frag, acc, a_dtype=kv_dtype, b_dtype=P_DTYPE
                )
                rocdl.sched_barrier(0)
            return
        kv, d_tile_idx = divmod(i - num_v_frags, num_k_tiles)
        k_frag = _concat(parts)
        for q_tile_idx in range(num_q_tiles):
            acc = (
                s_acc_list[q_tile_idx][kv]
                if d_tile_idx > 0
                else fx.Vector.filled(8, 0.0, fx.Float32)
            )
            s_acc_list[q_tile_idx][kv] = _wmma(
                k_frag,
                q_frags_list[q_tile_idx][d_tile_idx],
                acc,
                a_dtype=kv_dtype,
                b_dtype=q_dtype,
            )
            rocdl.sched_barrier(0)

    def dies(i):
        # The PV half's B operands die on the final d-tile row, which is where the ring is
        # already refilling for QK. Release them through the ring so the refill sees one
        # WAR distance, not two.
        if i >= num_v_frags:
            return ()
        d_tile_idx, k_tile_idx = divmod(i, num_contract_tiles)
        if d_tile_idx != d_tiles - 1:
            return ()
        return [
            p_frags.pop((q_tile_idx, k_tile_idx)) for q_tile_idx in range(num_q_tiles)
        ]

    # Raise wave priority for the whole WMMA stream: under anti-phase the other half is
    # in its softmax VALU here, and the gemm half must win issue arbitration.
    rocdl.s_setprio(2)
    _ring_drive(
        widths=[vld_per_frag] * num_v_frags + [2] * num_k_frags,
        emit=emit,
        consume=consume,
        ring=ring,
        ring_lag=ring_lag,
        head=head,
        dies=dies,
    )
    rocdl.s_setprio(0 if warp_type.is_lo else 1)
    return out_list, s_acc_list


# ============================================================================
# Shared, layout-agnostic compute core
# ============================================================================


# Keys ``_attention_prologue`` shares with the compute cores' kwarg dict.
_PROLOGUE_KEYS = (
    "qk_hdim",
    "v_hdim",
    "n_block",
    "mask_left",
    "mask_right",
    "gqa_ratio",
    "num_q_tiles_per_wave",
    "ptr_Q",
    "ptr_K",
    "ptr_V",
    "softmax_scale",
    "stride_q_seq",
    "stride_k_seq",
    "stride_v_seq",
    "stride_q_head",
    "stride_k_head",
    "stride_v_head",
    "q_start",
    "q_len",
    "kv_start",
    "kv_len",
    "window_left",
    "window_right",
    "q_dtype",
    "kv_dtype",
)


def _build_managers(
    *, qk_hdim, v_hdim, n_block, gqa_ratio, num_q_tiles_per_wave, q_dtype, kv_dtype
):
    """The Q/K/V staging managers both cores use; USE_TDM_LOADER picks the transport.

    Pure object construction -- no IR is emitted until a manager method is called, so
    sharing this between the cores cannot move the generated code.
    """
    if USE_TDM_LOADER:
        return (
            QManager16bV2(
                qk_hdim=qk_hdim,
                gqa_ratio=gqa_ratio,
                num_waves=NUM_WAVES,
                q_tiles_per_wave=num_q_tiles_per_wave,
                elem_dtype=q_dtype,
            ),
            KManager16bV2(
                qk_hdim=qk_hdim,
                n_block=n_block,
                num_waves=NUM_WAVES,
                elem_dtype=kv_dtype,
            ),
            VManager16bV2(
                v_hdim=v_hdim,
                n_block=n_block,
                num_waves=NUM_WAVES,
                elem_dtype=kv_dtype,
            ),
        )
    return (
        QManager16bV1(
            qk_hdim=qk_hdim,
            gqa_ratio=gqa_ratio,
            num_waves=NUM_WAVES,
            q_tiles_per_wave=num_q_tiles_per_wave,
            elem_dtype=q_dtype,
        ),
        KManager16bV1(
            qk_hdim=qk_hdim,
            n_block=n_block,
            num_waves=NUM_WAVES,
            elem_dtype=kv_dtype,
        ),
        VManager16bV1(
            v_hdim=v_hdim,
            n_block=n_block,
            num_waves=NUM_WAVES,
            elem_dtype=kv_dtype,
        ),
    )


def _attention_prologue(
    *,
    qk_hdim,
    v_hdim,
    n_block,
    mask_left,
    mask_right,
    gqa_ratio,
    num_q_tiles_per_wave,
    ptr_Q,
    ptr_K,
    ptr_V,
    softmax_scale,
    stride_q_seq,
    stride_k_seq,
    stride_v_seq,
    stride_q_head,
    stride_k_head,
    stride_v_head,
    q_start,
    q_len,
    kv_start,
    kv_len,
    window_left,
    window_right,
    q_dtype,
    kv_dtype,
    warp_idx,  # runtime fx.Int32 wave index
    lds_base,  # LDS base (fx.Int32), allocated once by the caller (_alloc_lds)
):
    """Everything both halves do before the first body, traced once per build.

    The halves differ in four things only: which band they copy (LO K, HI V), which
    slot their second copy fills, how many copies fit in Q's global-load shadow, and
    HI's resting priority. All four ride two runtime warp-type branches here, both
    inside Q's global load; everything else -- the managers, the LDS map, Q's own
    load, and this WG's KV span -- is identical and reaches the binary once.
    """
    BLOCK_M = _block_m(num_q_tiles_per_wave)
    lane_idx = _lane_id()
    kv_head, q_head_idx, seq_idx = _packed_tile_indices(
        gqa_ratio, warp_idx, lane_idx, num_q_tiles_per_wave
    )

    # softmax_scale*LOG2E, folded into Q by the loader's bf16 multiply.
    # 8-bit Q cannot carry the scale (no f8 multiply), so softmax applies it to S.
    _q_scale = None
    _s_scale = softmax_scale * fx.Float32(LOG2E)

    # K/V staging: N_KV_PP slots of 2 LDS_CHUNK_BYTES chunks each, every tile split 2-way
    # along n_block into chunks 6 apart (different 64 KB segments). Q time-shares slot 1's
    # chunks, O the two slots the loop has finished with -- see the 12-chunk map up top.
    q_mgr, k_mgr, v_mgr = _build_managers(
        qk_hdim=qk_hdim,
        v_hdim=v_hdim,
        n_block=n_block,
        gqa_ratio=gqa_ratio,
        num_q_tiles_per_wave=num_q_tiles_per_wave,
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
    )
    k_blk_bytes = k_mgr.get_lds_size_in_byte()
    v_blk_bytes = v_mgr.get_lds_size_in_byte()
    assert k_blk_bytes % KV_LDS_SPLITS == 0 and v_blk_bytes % KV_LDS_SPLITS == 0
    # Chunk stride between a tile's two n_block halves (K[slot_idx][0] -> K[slot_idx][1]) and between
    # consecutive slots. Slot slot_idx occupies chunks 2pp, 2pp+1 low and 2pp+6, 2pp+7 high.
    _SPLIT_STRIDE = 6 * LDS_CHUNK_BYTES  # 156 KB
    slot_bytes = 2 * LDS_CHUNK_BYTES  # 52 KB: one slot's K|V pair in one half
    for _label, _bytes in (("K", k_blk_bytes), ("V", v_blk_bytes)):
        assert _bytes // KV_LDS_SPLITS <= LDS_CHUNK_BYTES, (
            f"{_label} split {_bytes // KV_LDS_SPLITS}B exceeds the {LDS_CHUNK_BYTES}B chunk "
            f"(qk_hdim={qk_hdim}, v_hdim={v_hdim}, n_block={n_block})"
        )
    assert 2 * N_KV_PP * slot_bytes <= get_lds_capacity_bytes("gfx1250"), (
        "12-chunk K|V layout over LDS capacity"
    )

    def _k_lds_buf(
        slot_idx,
    ):  # slot ``pp``'s low-half base == K[slot_idx][0] (int or fx.Int32; folds when const)
        if isinstance(slot_idx, int):
            slot_idx = fx.Int32(slot_idx)
        return lds_base + slot_idx * fx.Int32(slot_bytes)

    def _v_lds_buf(slot_idx):  # V[slot_idx][0] == the slot base one chunk in
        return _k_lds_buf(slot_idx) + fx.Int32(LDS_CHUNK_BYTES)

    # Logical slot -> physical chunk pair. Q covers all of physical slot 1 and the low
    # 17 KB of slot 2's K chunks, so physical slot 0 is the only Q-disjoint one; putting
    # the first tile (logical slot 1) there lets the prologue issue it before Q is read.
    _PHYSICAL_SLOT = [2, 0, 1]

    def _k_bufs_at(
        slot,
    ):  # K[.][0], K[.][1] of the slot whose low-half base is ``slot``
        return [slot, slot + fx.Int32(_SPLIT_STRIDE)]

    def _v_bufs_at(slot):
        v_start_desc = slot + fx.Int32(LDS_CHUNK_BYTES)
        return [v_start_desc, v_start_desc + fx.Int32(_SPLIT_STRIDE)]

    # READ side only -- the producer keeps writing split s to chunk s. Odd SIMDs take the
    # two bases in the other order, so the parities are never in the same 64 KB segment
    # set at the same point of a gemm. _split_bufs runs only in the prologue and its
    # results are carried as ds pointers, so this costs a few adds and nothing in-loop.
    assert KV_LDS_SPLITS == 2, "parity order needs a 2-way split"
    _is_odd_warp = warp_idx & fx.Int32(1)
    _read_order_lo = _is_odd_warp * fx.Int32(_SPLIT_STRIDE)
    _read_order = [_read_order_lo, fx.Int32(_SPLIT_STRIDE) - _read_order_lo]
    # Slot kv_frag_idx then holds physical kv-tile kv_frag_idx ^ (num_kv_frags/2); the mask is the only consumer
    # of the absolute index, and it takes the correction as +/- this delta.
    _kv_swap_delta = _is_odd_warp * fx.Int32(n_block // 2)

    def _split_bufs(b):
        return [b + o for o in _read_order]

    def _k_lds_bufs(slot_idx):
        return _split_bufs(_k_lds_buf(slot_idx))

    def _v_lds_bufs(slot_idx):
        return _split_bufs(_v_lds_buf(slot_idx))

    # ---- Q and O own no LDS: both time-share KV chunks in LDS_QO_BYTES per-wave slices.
    # Q sits in the K[1][0] / K[1][1] chunk pairs (52 KB and 208 KB, 4 waves x 17 KB each);
    # it is drained into VGPR and dead before the prologue issues a tile into slot 1 or 2
    # (s_wait_dscnt + barrier below). Wave w -> chunk B iff (w&1) ^ ((w>>2)&1), so the two
    # waves sharing a SIMD (w and w+4) always land in different chunks. ----
    for _label, _bytes in (("Q", q_mgr.warp_lds_size_in_byte()),):
        assert _bytes <= LDS_QO_BYTES, (
            f"{_label} per-wave {_bytes}B over the {LDS_QO_BYTES}B slice"
        )
    _q_chunk_sel = (warp_idx & fx.Int32(1)) ^ ((warp_idx >> fx.Int32(2)) & fx.Int32(1))
    q_lds_warp = (
        _k_lds_buf(1)
        + _q_chunk_sel * fx.Int32(_SPLIT_STRIDE)
        + (warp_idx >> fx.Int32(1)) * fx.Int32(LDS_QO_BYTES)
    )

    q_mgr.load_q_to_vgpr_part1(
        ptr_Q=ptr_Q,
        stride_q_seq=stride_q_seq,
        stride_q_head=stride_q_head,
        q_start=q_start,
        q_len=q_len,
        kv_head=kv_head,
        block_x=_m_tile_idx(),
        warp_idx=warp_idx,
        lane_idx=lane_idx,
        ptr_lds_warp=q_lds_warp,
    )

    # ---- This WG's KV tiles span relative kv [start_tile_idx*n_block, kv_len_wg); mask_left
    # moves start_tile_idx past whole tiles before the WG's min query's band start. The
    # defensive min() keeps start_tile_idx a valid buffer index even for an over-launched WG
    # whose whole band is empty (its per-element masks zero the work anyway).
    m_tile_idx, causal_off, kv_len_wg, num_tiles = _wg_kv_span(
        n_block=n_block,
        mask_right=mask_right,
        gqa_ratio=gqa_ratio,
        num_q_tiles_per_wave=num_q_tiles_per_wave,
        q_len=q_len,
        kv_len=kv_len,
        window_right=window_right,
    )
    last_tile_idx = num_tiles - fx.Int32(1)  # always carries the kv_len tail
    if mask_left:
        wg_min_seq = (m_tile_idx * fx.Int32(BLOCK_M)) // fx.Int32(gqa_ratio)
        kv_lo = fx.max(wg_min_seq + causal_off - window_left, fx.Int32(0))
        start_tile_idx = kv_lo // fx.Int32(n_block)
        start_tile_idx = fx.min(start_tile_idx, last_tile_idx)
    else:
        start_tile_idx = fx.Int32(0)

    def _tile_row0(t):  # clamped to the last tile, never guarded off the end
        return fx.min(t, last_tile_idx) * fx.Int32(n_block)

    # How many rows of tile t are in-bounds: n_block for every tile but the last, whose
    # tail is loop-invariant. Tiles past last_tile_idx read as the last one, matching
    # _tile_row0's clamp. A scalar select keeps this out of the VALU -- the min/max form
    # folds to v_med3_i32, which has no scalar counterpart, so a uniform value would go
    # out to a VGPR and come back through v_readfirstlane_b32 on every body.
    tail_valid = kv_len_wg - last_tile_idx * fx.Int32(n_block)

    def _kv_valid(t):
        return (t < last_tile_idx).select(fx.Int32(n_block), tail_valid)

    # ---- Prologue. All K/V address arithmetic is pure (no memory op until
    # ``.async_load()``), so it is hoisted into the Q global-load shadow and the copies
    # themselves issue last, leaving nothing between them and the prologue barrier.
    #
    # Slot rotation is local to this WG's tile stream, so start_tile_idx never enters the
    # placement. Body tile_idx reads K from slot 1 and V from slot 0, so the first body needs tile
    # start_tile_idx's K AND V in slot 1 (K for QK(start_tile_idx), V for the next body's PV) and
    # only FINITE data in slot 0's V region -- its PV is the dead leading one, p == 0, and
    # 0 * NaN would poison O. Loading start_tile_idx's V there is the cheapest such filler.
    start_row0 = start_tile_idx * fx.Int32(n_block)
    # This wave's slot among its half's producers; it owns one dense
    # n_block/KV_PRODUCER_WARPS row band and copies it alone.
    _producer_warp = warp_idx % fx.Int32(KV_PRODUCER_WARPS)
    _kv_ctx = ProducerCtx(
        producer_warp=_producer_warp,
        num_producer_warps=KV_PRODUCER_WARPS,
        lane_idx=lane_idx,
    )

    def _k_desc(slot, row0, valid):
        """K descriptor for one tile, addressed off the slot's low-half base. Pure until
        ``.async_load()``; which transport it carries (async vs TDM) is the manager's
        business, not this call site's."""
        return k_mgr.load_descriptor(
            ptr_lds=_k_bufs_at(slot),
            ptr_src=ptr_K,
            stride_seq=stride_k_seq,
            stride_head=stride_k_head,
            head=kv_head,
            row0=kv_start + row0,
            valid=valid,
            ctx=_kv_ctx,
        )

    def _v_desc(slot, row0, valid):
        return v_mgr.load_descriptor(
            ptr_lds=_v_bufs_at(slot),
            ptr_src=ptr_V,
            stride_seq=stride_v_seq,
            stride_head=stride_v_head,
            head=kv_head,
            row0=kv_start + row0,
            valid=valid,
            ctx=_kv_ctx,
        )

    # LO copies the K band, HI the V band, both of tile start into slot 1; HI owes one more,
    # V(start) into slot 0 for body start's dead PV. Built here because the address math is
    # pure; only the ``.async_load()``s go under the warp-type branch.
    _start_valid = _kv_valid(start_tile_idx)
    k_start_desc = _k_desc(_k_lds_buf(_PHYSICAL_SLOT[1]), start_row0, _start_valid)
    v_start_desc = _v_desc(_k_lds_buf(_PHYSICAL_SLOT[1]), start_row0, _start_valid)
    v_dummy_desc = _v_desc(_k_lds_buf(_PHYSICAL_SLOT[0]), start_row0, _start_valid)

    # Per-half counter usage, off the descriptors' declared counts. A counter this half
    # never touches gets -1 ("do not wait on it"), NOT 0.
    kv_counts = {
        WarpType.LO: (k_start_desc.tensorcnt or -1, k_start_desc.asynccnt or -1),
        WarpType.HI: (v_start_desc.tensorcnt or -1, v_start_desc.asynccnt or -1),
    }
    _drains = {w: _kv_drain_depths(*c) for w, c in kv_counts.items()}
    # Fully-drained depths name only WHICH counters are live, and both bands live on the
    # same one, so one prologue fence serves both halves.
    assert _drains[WarpType.LO] == _drains[WarpType.HI], "prologue fence is per-WG"
    _kv_drain = _drains[WarpType.LO]

    # Under V2 no prologue copy lands on Q's chunks, so all go out before Q is read and
    # overlap its latency. Under V1 Q's own stage shares asynccnt and part2 waits a depth
    # counted over Q's loads alone, so every copy must go after part2.
    if USE_TDM_LOADER:
        _early = {
            WarpType.LO: [k_start_desc],
            WarpType.HI: [v_start_desc, v_dummy_desc],
        }
        _late = {WarpType.LO: [], WarpType.HI: []}
    else:
        _early = {WarpType.LO: [], WarpType.HI: []}
        _late = {WarpType.LO: [k_start_desc], WarpType.HI: [v_start_desc, v_dummy_desc]}

    _is_lo = warp_idx < fx.Int32(NUM_WAVES // 2)
    if USE_TDM_LOADER:
        # The halves issue different numbers of copies here, so the drain immediate is
        # per-half and rides the same branch. Both sit inside Q's global load.
        @flyc.jit
        def _issue_early(is_lo):
            if is_lo:
                for _desc in _early[WarpType.LO]:
                    _desc.async_load()
                tdm_ops.tensor_wait(sum(d.tensorcnt for d in _early[WarpType.LO]))
            else:
                for _desc in _early[WarpType.HI]:
                    _desc.async_load()
                tdm_ops.tensor_wait(sum(d.tensorcnt for d in _early[WarpType.HI]))

        _issue_early(_is_lo)
    q_frags = q_mgr.load_q_to_vgpr_part2(scale=_q_scale)

    @flyc.jit
    def _issue_late(is_lo):
        if is_lo:
            for _desc in _late[WarpType.LO]:
                _desc.async_load()
        else:
            rocdl.s_setprio(1)
            for _desc in _late[WarpType.HI]:
                _desc.async_load()

    _issue_late(_is_lo)
    rocdl.sched_barrier(0)
    _copy_fence(*_kv_drain)

    return {
        "lane_idx": lane_idx,
        "kv_head": kv_head,
        "q_head_idx": q_head_idx,
        "seq_idx": seq_idx,
        "k_mgr": k_mgr,
        "v_mgr": v_mgr,
        "kv_ctx": _kv_ctx,
        "k_lds_buf": _k_lds_buf,
        "k_bufs_at": _k_bufs_at,
        "v_bufs_at": _v_bufs_at,
        "k_lds_bufs": _k_lds_bufs,
        "v_lds_bufs": _v_lds_bufs,
        "kv_swap_delta": _kv_swap_delta,
        "split_stride": _SPLIT_STRIDE,
        "physical_slot": _PHYSICAL_SLOT,
        "m_tile_idx": m_tile_idx,
        "causal_off": causal_off,
        "num_tiles": num_tiles,
        "last_tile_idx": last_tile_idx,
        "start_tile_idx": start_tile_idx,
        "tile_row0": _tile_row0,
        "kv_valid": _kv_valid,
        "kv_counts": kv_counts,
        "kv_drain": _kv_drain,
        "q_frags": q_frags,
    }


schedule_bank.initialized = False
schedule_bank.func = None


def _attention_core_multi_kv_tiles(
    *,
    qk_hdim,
    v_hdim,
    n_block,  # compile-time KV block width (columns of one QK GEMM tile)
    mask_left,  # compile-time: bound the left band edge (finite window_left)
    mask_right,  # compile-time: bound the right band edge (causal or finite window_right)
    return_lse,
    has_sink,  # compile-time: fold a per-head sink logit into the softmax denom
    gqa_ratio,  # compile-time GQA group size = nheads_q // nheads_kv
    num_q_tiles_per_wave,  # compile-time q-WMMA tiles per wave; BLOCK_M = 16*this*8
    ptr_O,
    ptr_Q,
    ptr_K,
    ptr_V,
    ptr_LSE,
    ptr_sink,  # [nheads_q] fp32 per-head sink logits; read only when has_sink
    softmax_scale,
    stride_q_seq,
    stride_k_seq,
    stride_v_seq,
    stride_o_seq,
    stride_q_head,
    stride_k_head,
    stride_v_head,
    stride_o_head,
    # LSE addressing (element strides + per-batch bound), resolved by the caller.
    # Only consumed when return_lse; the caller may pass anything otherwise.
    stride_lse_seq,
    stride_lse_head,
    lse_base_elems,  # first element offset of this batch's LSE slab
    lse_num_records_bytes,  # buffer-resource bound (below the 0x7FFFFFFF drop)
    # Per-batch token ranges (fx.Int32), resolved by the caller:
    q_start,  # first Q token index of this batch in the global tensor
    q_len,  # valid Q tokens in this batch
    kv_start,  # first K/V token index of this batch
    kv_len,  # valid K/V tokens in this batch
    # Sliding-window bounds (runtime fx.Int32, >= 0). window_left read only when
    # mask_left, window_right only when mask_right. Causal == mask_right, window_right=0.
    window_left,
    window_right,
    warp_idx,  # runtime fx.Int32 wave index
    warp_type,  # compile-time WarpType (LO/HI x SIMD parity)
    lds_base,  # LDS base (fx.Int32), allocated once by the caller (_alloc_lds)
    prologue,  # dict from _attention_prologue: managers, LDS map, KV span, Q fragments
    q_dtype,  # compile-time Q fragment type; K/V are kv_dtype, P is P_DTYPE, O is O_DTYPE
    kv_dtype,  # compile-time K/V fragment type (may differ from q_dtype; both 8-bit)
    v_descale=None,  # the epilogue's; here so one kwarg dict serves both
):
    """Layout-agnostic m32x8 compute, shared by the THD and BSHD kernel entries.

    Takes over from ``_attention_prologue`` (Q already in VGPR, the first tiles already
    in flight) and hands ``_attention_epilogue`` the finished softmax/O state. The caller
    dispatches on runtime ``warp_type``, tracing this body once per compile-time value;
    the two instantiations differ only in their main-loop phase ordering (LO drives the
    K load, HI shadows it), which is why only this middle part is duplicated.
    """
    BLOCK_M = _block_m(num_q_tiles_per_wave)
    lane_idx = prologue["lane_idx"]
    kv_head = prologue["kv_head"]
    q_head_idx = prologue["q_head_idx"]
    seq_idx = prologue["seq_idx"]
    k_mgr = prologue["k_mgr"]
    v_mgr = prologue["v_mgr"]
    _kv_ctx = prologue["kv_ctx"]
    _k_lds_buf = prologue["k_lds_buf"]
    _k_bufs_at = prologue["k_bufs_at"]
    _v_bufs_at = prologue["v_bufs_at"]
    _k_lds_bufs = prologue["k_lds_bufs"]
    _v_lds_bufs = prologue["v_lds_bufs"]
    _kv_swap_delta = prologue["kv_swap_delta"]
    _SPLIT_STRIDE = prologue["split_stride"]
    _PHYSICAL_SLOT = prologue["physical_slot"]
    m_tile_idx = prologue["m_tile_idx"]
    causal_off = prologue["causal_off"]
    num_tiles = prologue["num_tiles"]
    last_tile_idx = prologue["last_tile_idx"]
    start_tile_idx = prologue["start_tile_idx"]
    _tile_row0 = prologue["tile_row0"]
    _kv_valid = prologue["kv_valid"]
    _kv_drain = prologue["kv_drain"]
    q_frags = prologue["q_frags"]
    # 8-bit Q cannot carry the scale (no f8 multiply); softmax applies it to S.
    _s_scale = softmax_scale * fx.Float32(LOG2E)

    def _get_kv_desc(slot, row0, valid):
        """This half's BufferOpDescriptor for one tile: LO issues every K copy, HI every
        V copy. Pure until ``.async_load()``; which transport it carries (async vs TDM) is
        the manager's business, not this call site's."""
        if warp_type.is_lo:
            return k_mgr.load_descriptor(
                ptr_lds=_k_bufs_at(slot),
                ptr_src=ptr_K,
                stride_seq=stride_k_seq,
                stride_head=stride_k_head,
                head=kv_head,
                row0=kv_start + row0,
                valid=valid,
                ctx=_kv_ctx,
            )
        return v_mgr.load_descriptor(
            ptr_lds=_v_bufs_at(slot),
            ptr_src=ptr_V,
            stride_seq=stride_v_seq,
            stride_head=stride_v_head,
            head=kv_head,
            row0=kv_start + row0,
            valid=valid,
            ctx=_kv_ctx,
        )

    # ---- Loop init: the online-softmax seed and the O accumulators, all iter_args.
    #
    # Seeding m=-inf, d=0, O=0 makes the first tile's corr = exp2(m_prev-m_new) = 0 zero
    # the (already-zero) O before its PV adds in — the standard flash seed.
    #
    # Attention sink (compile-time) is one extra ``exp(sink)`` term in the softmax denom.
    # Fold it in by seeding m = sink[q_head]*LOG2E (m is log2-domain) and d = 1.0
    # (= exp(sink-sink)); the rescales carry that d seed to exactly exp(sink - m_final).
    # Without a sink the first tile's corr zeroes the d seed, so d=1 would equal d=0 --
    # the no-sink path keeps d=0 to stay byte-for-byte.
    d_tiles = v_hdim // WMMA_M
    num_q_tiles = num_q_tiles_per_wave
    num_kv_frags = n_block // WMMA_N
    # Per-q-tile carried state: [m, d, O, P_0 .. P_{num_kv_frags-1}]. The lagging half
    # carries the f32 s_acc in the P slots instead of bf16 P; the slot count is the same.
    # O travels the back edge as ONE tuple (see _o_pack); the P slots do not, because the
    # gemm ring produces and consumes them one at a time.
    _q_state_slots = 2 + 1 + num_kv_frags
    _is_lagging_softmax = not warp_type.is_lo
    if _is_lagging_softmax:
        # Barrier-count filler: re-pairs the two halves' barrier streams.
        _copy_fence(*_kv_drain)
    if has_sink:
        num_heads_q = gpu.grid_dim.y * fx.Int32(gqa_ratio)
        m_init = [
            _load_sink_logit(ptr_sink, q_head_idx[q_tile_idx], num_heads_q)
            * fx.Float32(LOG2E)
            for q_tile_idx in range(num_q_tiles)
        ]
        d_init = [fx.Float32(2.0**P_BIAS_LOG2) for _ in range(num_q_tiles)]
    else:
        m_init = [fx.Float32(BIG_NEG) for _ in range(num_q_tiles)]
        d_init = [fx.Float32(0.0) for _ in range(num_q_tiles)]
    _init = []
    for q_tile_idx in range(num_q_tiles):
        _init += (
            [
                _raw(m_init[q_tile_idx]),
                _raw(d_init[q_tile_idx]),
            ]
            + [_raw(fx.Vector.filled(8 * d_tiles, 0.0, fx.Float32))]
            + [
                _raw(
                    fx.Vector.filled(8, float("-inf"), fx.Float32)
                    if _is_lagging_softmax
                    else _convert_vectors(
                        [[fx.Vector.filled(8, 0.0, fx.Float32)]], P_DTYPE
                    )[0][0]
                )
                for _ in range(num_kv_frags)
            ]
        )

    # ---- One ds_load base-pointer set per slot plus the slot's byte base, carried as
    # iter_args and LEFT-ROTATED in the yield, so no runtime "% N_KV_PP" is evaluated. ----
    k_lds_ld = [
        k_mgr.ds_load_ptrs(ptr_lds=_k_lds_bufs(_PHYSICAL_SLOT[i]), lane_idx=lane_idx)
        for i in range(N_KV_PP)
    ]
    v_lds_ld = [
        v_mgr.ds_load_ptrs(ptr_lds=_v_lds_bufs(_PHYSICAL_SLOT[i]), lane_idx=lane_idx)
        for i in range(N_KV_PP)
    ]
    _num_k_bases = len(k_lds_ld[0])  # ds bases per K buffer (one per LDS split)
    _num_v_bases = len(v_lds_ld[0])
    _k_bases_idx = len(_init)
    for i in range(N_KV_PP):
        _init = _init + k_lds_ld[i]
    _v_bases_idx = len(_init)
    for i in range(N_KV_PP):
        _init = _init + v_lds_ld[i]
    _slot_bases_idx = len(_init)
    _init = _init + [_raw(_k_lds_buf(_PHYSICAL_SLOT[i])) for i in range(N_KV_PP)]

    # Carried ring head: _num_head_carry of the first body's V loads, issued here (slot 0 is resident
    # -- the prologue fence just drained it) and thereafter at the end of each body.
    _num_k_frags = k_mgr.num_ds_loads() // k_mgr.loads_per_frag
    _num_v_frags = v_mgr.num_ds_loads() // v_mgr.loads_per_frag
    _num_vld = v_mgr.num_ds_loads()
    _vld_group = v_mgr.loads_per_frag  # pin the head in whole V fragments
    _num_prefetch = _ring_num_prefetch(
        _num_vld + k_mgr.num_ds_loads(), PVQK_RING, PVQK_LAG
    )
    # The lagging half opens its body with softmax, so its ring head is issued there and
    # consumed after the barrier -- within one body, nothing to carry.
    # The carry is issued a body early, where only V is resident, so it cannot run past V.
    _num_head_carry = (
        0 if _is_lagging_softmax else min(PVQK_HEAD_CARRY, _num_prefetch, _num_vld)
    )
    assert _num_head_carry <= _num_vld, "carried head must be V loads only"
    _head_carry_idx = len(_init)
    _seed_head = []
    for j in range(_num_head_carry):
        # Same per-pair pin as ``_issue_head`` in the body.
        _seed_head.append(_raw(v_mgr.load_one_to_reg(v_lds_ld[0], j)))
        if len(_seed_head) % _vld_group == 0:
            rocdl.sched_barrier(0)
    _init = _init + _seed_head

    # ========================================================================
    # Main KV loop -- SOFTWARE-PIPELINED by one tile. The loop variable ``u`` is the
    # SOFTMAX tile, and body tile_idx runs
    #
    #     PV(tile_idx-1)  ->  QK(tile_idx)  ->  softmax(tile_idx)  ->  rescale O by corr(tile_idx)
    #
    # carrying P across the back edge. That puts the two gemms back to back (a shared
    # ring feeds both) and leaves exactly three live slots: V(tile_idx-1) in slot 0, K(tile_idx) in
    # slot 1, tile tile_idx+1's copy landing in slot 2. Slots are selected by the carried ds
    # pointers / bases, left-rotated at the end of each `main_loop`.
    #
    # tile_idx runs [start_tile_idx, num_tiles] -- ONE body more than there are tiles. Both extra
    # half-bodies are dead rather than peeled, an extra trace costing far more than an
    # extra iteration: body start_tile_idx's PV multiplies the seeded P == 0, and body
    # num_tiles' QK/softmax sits at kv_pos_base >= kv_len so every element masks to -inf
    # (m unchanged, corr == 1, P == 0, d unchanged).
    # ========================================================================
    def main_loop(tile_idx, state, *, mask_left, mask_right, kv_len):
        # mask_left/mask_right/kv_len shadow the closure flags: the caller splits the
        # tile stream into a mask-free clean region + boundary loops and passes None for
        # any edge this sub-loop provably doesn't cross. The split points are already
        # expressed in SOFTMAX tiles, so they carry over unchanged. The lagging half runs
        # softmax one tile behind its gemm, hence tile_idx-1.
        softmax_tile_idx = tile_idx - fx.Int32(1) if _is_lagging_softmax else tile_idx
        kv_tile_start = softmax_tile_idx * fx.Int32(
            n_block
        )  # softmax tile's first (batch-relative) kv row

        # Unpack: num_q_tiles independent per-q-tile (m, d, O, P) groups, then the shared ds state.
        m_prev = [
            fx.Float32(state[q_tile_idx * _q_state_slots + 0])
            for q_tile_idx in range(num_q_tiles)
        ]
        d_prev = [
            fx.Float32(state[q_tile_idx * _q_state_slots + 1])
            for q_tile_idx in range(num_q_tiles)
        ]
        o_acc = [
            _o_unpack(fx.Vector(state[q_tile_idx * _q_state_slots + 2]), d_tiles)
            for q_tile_idx in range(num_q_tiles)
        ]
        # Carried gemm/softmax hand-off: P(tile_idx-1) on the leading half, S(tile_idx-1) on the
        # lagging one (same slot count, different element type).
        carry_prev = [
            [
                fx.Vector(state[q_tile_idx * _q_state_slots + 3 + kv_frag_idx])
                for kv_frag_idx in range(num_kv_frags)
            ]
            for q_tile_idx in range(num_q_tiles)
        ]
        k_slots = [
            list(
                state[
                    _k_bases_idx + i * _num_k_bases : _k_bases_idx
                    + (i + 1) * _num_k_bases
                ]
            )
            for i in range(N_KV_PP)
        ]
        v_slots = [
            list(
                state[
                    _v_bases_idx + i * _num_v_bases : _v_bases_idx
                    + (i + 1) * _num_v_bases
                ]
            )
            for i in range(N_KV_PP)
        ]
        slot_of = [fx.Int32(state[_slot_bases_idx + i]) for i in range(N_KV_PP)]
        head_carry = [
            fx.Vector(state[_head_carry_idx + i]) for i in range(_num_head_carry)
        ]
        # The ring's LDS base pointers: banked with the fragments they feed.
        v_curr = [
            schedule_bank(p, bank=1, reason="ring_lds_addr") for p in v_slots[0]
        ]  # tile tile_idx-1: this body's PV
        k_curr = [
            schedule_bank(p, bank=1, reason="ring_lds_addr") for p in k_slots[1]
        ]  # tile tile_idx:   this body's QK

        # Tile prefetched by this body, CLAMPED to the last tile rather than guarded off
        # past the end. Every body then issues exactly one tile's copies, so the fence
        # count is uniform and the last iteration needs no peel -- cheaper than the extra
        # trace a peel costs, against ~2 dead L2-resident tile loads per workgroup. The
        # clamped re-load lands in the slot body num_tiles reads as its dead K, and O
        # stages past all the slots. Both halves copy tile tile_idx+1 into slot 2 -- K and V
        # occupy disjoint chunks of it, and the slot last held tile tile_idx-2, dead since body
        # tile_idx-2 -- so the distance is the same on both and the slot arithmetic has no half
        # to branch on.
        write_slot = slot_of[N_KV_PP - 1]
        prefetch_tile_idx = tile_idx + fx.Int32(1)
        prefetch_row0 = _tile_row0(prefetch_tile_idx)
        prefetch_valid = _kv_valid(prefetch_tile_idx)

        def _drain_barrier():
            # Both halves have exactly one copy outstanding and the next body reads it, so
            # both drain to 0: there is nothing older to leave in flight.
            #
            # dscnt is drained only to _num_head_carry: the leading half issues its ring head at the
            # tail of the previous body, so a full drain here would retire it right before
            # the gemm that wants it in flight. The gemm's own last ring fragment already
            # took dscnt to 0, so every read older than the head is retired regardless --
            # the WAR wall for the slot about to be written still holds.
            _copy_fence(*_kv_drain, num_dscnt=_num_head_carry)

        # This half's descriptor for tile ``pf``'s K (LO) or V (HI) into the oldest slot.
        # Built up front: pure, so its address VALU overlaps the drain; only the copy
        # itself must stay after the barrier.
        prefetch_desc = _get_kv_desc(write_slot, prefetch_row0, prefetch_valid)

        def _k_emit(j):
            return k_mgr.load_one_to_reg(k_curr, j)

        def _v_emit(j):
            return v_mgr.load_one_to_reg(v_curr, j)

        def _issue_head(emit, first, last, carried=()):
            # Pin every fragment's loads in place. The ring consumes one group per
            # fragment, but the scheduler scatters the 20-deep burst freely --
            # in the dumps a partner load drifted 12+ slots back, so the wmma waited on
            # most of the burst (s_wait_dscnt 0x9 / 0x7) instead of the ring's 0x12.
            # NOT _keepalive -- that is a USE, so it drags an s_wait_dscnt 0x0 in with it.
            out = list(carried)
            for j in range(first, last):
                out.append(emit(j))
                if len(out) % _vld_group == 0:
                    rocdl.sched_barrier(0)
            return out

        def _vk_emit(j):
            # The ring runs V then K; past the V loads the head continues into K, matching
            # ``_pv_qk_gemm``'s emit. Both slots are resident, so either is issuable here.
            return _v_emit(j) if j < _num_vld else _k_emit(j - _num_vld)

        def _pvqk_head():
            # The first _num_head_carry loads came from the previous body (issued under its softmax);
            # only the remainder is issued here.
            return _issue_head(_vk_emit, _num_head_carry, _num_prefetch, head_carry)

        def _softmax_phase(s_in, o_in):
            # This lane's query (tile q_tile_idx) attends [q_min, q_max] in batch-relative kv.
            # None bounds are skipped, so a clean-region tile passes all-None and does
            # zero per-element masking. kv_len is passed only on the right-boundary
            # sub-loop, which is also what neutralizes the dead trailing softmax. All num_q_tiles
            # q-tiles go in ONE call so their tree reductions emit interleaved.
            q_max_list = [
                seq_idx[q_tile_idx] + causal_off + window_right if mask_right else None
                for q_tile_idx in range(num_q_tiles)
            ]
            q_min_list = [
                seq_idx[q_tile_idx] + causal_off - window_left if mask_left else None
                for q_tile_idx in range(num_q_tiles)
            ]
            p_list, m_new_list, d_new_list, corr_list, rescale_masks = _softmax(
                s_list=s_in,
                m_prev_list=m_prev,
                d_prev_list=d_prev,
                lane_idx=lane_idx,
                n_block=n_block,
                kv_pos_base=kv_tile_start,
                kv_swap_delta=_kv_swap_delta,
                q_max_list=q_max_list,
                q_min_list=q_min_list,
                kv_len=kv_len,
                s_scale=_s_scale,
            )

            # Rescale each q-tile's running O by this tile's corr. The leading half
            # rescales at the END of the body and the next body's PV accumulates onto the
            # result; the lagging half rescales before its own PV. Same product either way.
            # Under deferral the wide multiply is gated behind a non-divergent scf.if that
            # fires only when a running max actually moved, with ONE branch covering all num_q_tiles
            # rows -- neither row moves in the common case, so the steady state is a single
            # not-taken s_cbranch_vccz, and a row that stayed stale has corr == 1 exactly,
            # making its rescale inside the taken branch the identity.
            corr_vecs = [
                fx.Vector.from_elements(
                    [corr_list[q_tile_idx]], fx.Float32
                ).broadcast_to(8)
                for q_tile_idx in range(num_q_tiles)
            ]
            o_vecs = [
                [
                    fx.Vector(_ir(o_in[q_tile_idx][d_tile_idx]))
                    for d_tile_idx in range(d_tiles)
                ]
                for q_tile_idx in range(num_q_tiles)
            ]
            if rescale_masks is None:
                o_resc_list = [
                    [ov * corr_vecs[q_tile_idx] for ov in o_vecs[q_tile_idx]]
                    for q_tile_idx in range(num_q_tiles)
                ]
            else:
                # Folded HERE, not in _softmax: the ballots are VALU-produced SGPRs, so
                # keeping the s_or/s_cmp at the use site leaves the whole row-max chain
                # between the v_cmp and the turnaround. OR the raw masks, not the per-row
                # booleans -- one s_or_b32 covers all num_q_tiles rows.
                mask_any = rescale_masks[0]
                for _mask in rescale_masks[1:]:
                    mask_any = mask_any | _mask

                @flyc.jit
                def _maybe_rescale_all(o_flat_in, corr_flat, do_rescale):
                    result = o_flat_in
                    if do_rescale:
                        result = [ov * cv for ov, cv in zip(o_flat_in, corr_flat)]
                    return result

                # One flat list: the num_q_tiles rows share the folded condition, so they rescale
                # (or pass through) together.
                o_flat = list(
                    _maybe_rescale_all(
                        [v for row in o_vecs for v in row],
                        [
                            corr_vecs[q_tile_idx]
                            for q_tile_idx in range(num_q_tiles)
                            for _ in range(d_tiles)
                        ],
                        mask_any != fx.Int32(0),
                    )
                )
                o_resc_list = [
                    o_flat[q_tile_idx * d_tiles : (q_tile_idx + 1) * d_tiles]
                    for q_tile_idx in range(num_q_tiles)
                ]
            return p_list, m_new_list, d_new_list, o_resc_list

        # Only the drain barrier (the one before tensor_load_to_lds) is required; the wall
        # is pure scheduling. Dropping it halves the in-loop barriers and wins everywhere
        # except non-window-masked qk_hdim 256 (n_block 64), which regresses slightly.
        # Compile-time constant, so both halves still agree on the barrier count.
        _keep_phase_wall = qk_hdim >= 256 and not (mask_left and mask_right)

        def _phase_barrier():
            # One half leaves the WMMA stream here as the other enters it; the
            # sched_barriers pin that split even when the s_barrier itself is gone.
            rocdl.sched_barrier(0)
            if _keep_phase_wall:
                _bare_barrier()
            rocdl.sched_barrier(0)

        # GEMM2(tile_idx-1) then GEMM1(tile_idx) on one ring: O += P^T(tile_idx-1) @ V(tile_idx-1), then
        # S^T = K(tile_idx) @ Q^T. sched_barrier fences the ring head out of the WMMA stream
        # (no wmma<-ds_load bubble); the ring itself issues the per-fragment s_wait_dscnt.
        if _is_lagging_softmax:
            prefetch_desc.async_load()
            p_f32, m_new_list, d_new_list, o_resc = _softmax_phase(carry_prev, o_acc)
            # Head out BEFORE P is narrowed: it is ds_loads with no dependence on P, so
            # issuing it first puts the conversion in its shadow rather than behind it.
            rocdl.sched_barrier(0)
            pvqk_head = _pvqk_head()
            rocdl.sched_barrier(0)
            p_list = _convert_vectors(p_f32, P_DTYPE)
            # Anchor the conversion in THIS block. Its only real use is past the barrier,
            # so MachineSink (which ignores sched_barrier) would otherwise sink every
            # v_cvt_pk_bf16_f32 into the gemm and interleave it with the WMMA.
            _keepalive([v for pt in p_list for v in pt])
            _phase_barrier()
            o_out, s_acc = _pv_qk_gemm(
                v_emit=_v_emit,
                k_emit=_k_emit,
                vld_per_frag=v_mgr.loads_per_frag,
                q_dtype=q_dtype,
                kv_dtype=kv_dtype,
                p_list=p_list,
                q_frags_list=q_frags,
                v_hdim=v_hdim,
                n_block=n_block,
                warp_type=warp_type,
                o_acc_list=o_resc,
                head=pvqk_head,
            )
            carry_next = s_acc
            _drain_barrier()
            head_next = []
        else:
            _drain_barrier()
            pvqk_head = _pvqk_head()
            prefetch_desc.async_load()
            rocdl.sched_barrier(0)
            o_acc, s_list = _pv_qk_gemm(
                v_emit=_v_emit,
                k_emit=_k_emit,
                vld_per_frag=v_mgr.loads_per_frag,
                q_dtype=q_dtype,
                kv_dtype=kv_dtype,
                p_list=carry_prev,
                q_frags_list=q_frags,
                v_hdim=v_hdim,
                n_block=n_block,
                warp_type=warp_type,
                o_acc_list=o_acc,
                head=pvqk_head,
            )
            _phase_barrier()
            carry_f32, m_new_list, d_new_list, o_out = _softmax_phase(s_list, o_acc)
            # Next body's ring head: V(tile_idx) from the slot this body read K from, resident
            # and already fenced. Mirrors the lagging half -- both issue the head just
            # before the barrier preceding the gemm that consumes it, ahead of the
            # narrowing for the same reason.
            rocdl.sched_barrier(0)
            head_next = _issue_head(
                lambda j: v_mgr.load_one_to_reg(v_slots[1], j), 0, _num_head_carry
            )
            rocdl.sched_barrier(0)
            carry_next = _convert_vectors(carry_f32, P_DTYPE)

        # Yield: num_q_tiles updated (m, d, O, P) groups, then the ds pointers and slot bases
        # left-rotated by one so slot 0 holds tile tile_idx (next body's PV) and the oldest
        # rotates into the write position.
        out = []
        for q_tile_idx in range(num_q_tiles):
            out += (
                [_raw(m_new_list[q_tile_idx]), _raw(d_new_list[q_tile_idx])]
                + [_raw(_o_pack(o_out[q_tile_idx]))]
                + [_raw(cv) for cv in carry_next[q_tile_idx]]
            )
        rotation = list(range(1, N_KV_PP)) + [0]
        for i in rotation:
            out += k_slots[i]
        for i in rotation:
            out += v_slots[i]
        out += [_raw(slot_of[i]) for i in rotation]
        out += [_raw(h) for h in head_next]
        return out

    # ---- Stream softmax tiles [start_tile_idx, num_tiles] through 3 sub-loops split by the
    # attention band, so interior tiles fully inside the band skip masking. clean_lo_idx /
    # clean_hi_idx are runtime split points, but each sub-loop's mask on/off is COMPILE-TIME.
    # The rotation state threads continuously through all three, leaving the slot
    # assignment intact, and the bounds are already softmax-tile indices, so the pipeline
    # shift only extends the last loop by one body (the dead-QK one).
    #   [start_tile_idx, clean_lo_idx)    left boundary   (emitted only when mask_left)
    #   [clean_lo_idx,   clean_hi_idx)    clean, no mask
    #   [clean_hi_idx,   num_tiles+1) right boundary + kv_len tail + the dead trailing body
    # clean_hi_idx = first tile that could need RIGHT masking = the WG's earliest query's
    # diagonal tile ((min q_max + 1)//n_block). Kept <= last_tile_idx so the tail tile stays in
    # the right loop, and >= start_tile_idx for a valid partition.
    if mask_right:
        wg_min_seq = (m_tile_idx * fx.Int32(BLOCK_M)) // fx.Int32(gqa_ratio)
        qmax_min = fx.max(wg_min_seq + causal_off + window_right, fx.Int32(0))
        clean_hi_idx = (qmax_min + fx.Int32(1)) // fx.Int32(n_block)
    else:
        clean_hi_idx = fx.Int32(num_tiles)
    clean_hi_idx = fx.max(fx.min(clean_hi_idx, last_tile_idx), start_tile_idx)

    # clean_lo_idx = first tile fully at/above the WG's latest query's window start
    # (ceildiv(max q_min, n_block)); clamped into [start_tile_idx, clean_hi_idx].
    if mask_left:
        wg_max_seq = fx.min(
            (m_tile_idx * fx.Int32(BLOCK_M) + fx.Int32(BLOCK_M - 1))
            // fx.Int32(gqa_ratio),
            q_len - fx.Int32(1),
        )
        qmin_max = fx.max(wg_max_seq + causal_off - window_left, fx.Int32(0))
        clean_lo_idx = (qmin_max + fx.Int32(n_block - 1)) // fx.Int32(n_block)
    else:
        clean_lo_idx = start_tile_idx
    clean_lo_idx = fx.min(fx.max(clean_lo_idx, start_tile_idx), clean_hi_idx)
    if mask_left:
        # The lagging half's softmax tile is tile_idx-1, so the clean loop must start one body
        # later or tile clean_lo_idx-1 would cross the left edge unmasked. The left loop
        # absorbs the extra body; over-masking a clean tile is a no-op. Only needed when
        # a left loop exists at all -- with mask_left off, clean_lo_idx == start_tile_idx and
        # shifting would drop the first body entirely.
        clean_lo_idx = fx.min(clean_lo_idx + fx.Int32(1), clean_hi_idx)

    @flyc.jit
    def _run_tiles(state, lo_i32, hi_i32, *, mask_left, mask_right, kv_len):
        final_state = state
        for tile, carried in range(fx.Index(lo_i32), fx.Index(hi_i32), 1, init=state):
            next_state = main_loop(
                fx.Int32(tile),
                list(carried),
                mask_left=mask_left,
                mask_right=mask_right,
                kv_len=kv_len,
            )
            final_state = yield next_state
        return final_state

    state = _init
    if mask_left:
        state = _run_tiles(
            state,
            start_tile_idx,
            clean_lo_idx,
            mask_left=mask_left,
            mask_right=mask_right,
            kv_len=None,
        )
    state = _run_tiles(
        state, clean_lo_idx, clean_hi_idx, mask_left=None, mask_right=None, kv_len=None
    )
    state = _run_tiles(
        state,
        clean_hi_idx,
        fx.Int32(num_tiles) + fx.Int32(1),
        mask_left=mask_left,
        mask_right=mask_right,
        kv_len=kv_len,
    )
    final = state
    if not _is_lagging_softmax:
        # Partner for the lagging half's prologue filler -- without it the two halves
        # disagree on the barrier count. It also orders this half's O stores against its
        # OWN last K reads: this half's last in-loop barrier is the PHASE one, which sits
        # BEFORE the gemm, so wave 1 could otherwise store O into final[0]'s K chunk while
        # wave 0 is still reading K out of it.
        _bare_barrier()

    _o_free_slots = [
        fx.Int32(final[_slot_bases_idx + 2]),
        fx.Int32(final[_slot_bases_idx + 0]),
    ]
    o_lds_warp = (
        ((warp_idx & fx.Int32(1)) > fx.Int32(0)).select(
            _o_free_slots[1], _o_free_slots[0]
        )
        + ((warp_idx >> fx.Int32(1)) & fx.Int32(1)) * fx.Int32(LDS_QO_BYTES)
        + (warp_idx >> fx.Int32(2)) * fx.Int32(_SPLIT_STRIDE)
    )
    # The trailing body's dead clamped tile copies are still in flight. HI's targets
    # final[1], a V chunk O never touches; LO's two K copies land in final[2] and final[1],
    # and final[2] IS an O chunk -- so the drain needs a rendezvous behind it, which
    # ``_attention_epilogue`` supplies, since a wave only retires its own copies and O's
    # chunk halves are shared by wave pairs.
    _copy_wait(*_kv_drain)
    return _epi_state(
        m_list=[
            fx.Float32(final[q_tile_idx * _q_state_slots + 0])
            for q_tile_idx in range(num_q_tiles)
        ],
        d_list=[
            fx.Float32(final[q_tile_idx * _q_state_slots + 1])
            for q_tile_idx in range(num_q_tiles)
        ],
        o_list=[
            _o_unpack(fx.Vector(final[q_tile_idx * _q_state_slots + 2]), d_tiles)
            for q_tile_idx in range(num_q_tiles)
        ],
        o_lds_warp=o_lds_warp,
    )


def _attention_core_one_kv_tile(
    *,
    qk_hdim,
    v_hdim,
    n_block,
    mask_left,
    mask_right,
    return_lse,
    has_sink,
    gqa_ratio,
    num_q_tiles_per_wave,
    ptr_O,
    ptr_Q,
    ptr_K,
    ptr_V,
    ptr_LSE,
    ptr_sink,
    softmax_scale,
    stride_q_seq,
    stride_k_seq,
    stride_v_seq,
    stride_o_seq,
    stride_q_head,
    stride_k_head,
    stride_v_head,
    stride_o_head,
    stride_lse_seq,
    stride_lse_head,
    lse_base_elems,
    lse_num_records_bytes,
    q_start,
    q_len,
    kv_start,
    kv_len,
    window_left,
    window_right,
    warp_idx,
    lds_base,
    q_dtype,
    kv_dtype,
    v_descale=None,  # the epilogue's; here so one kwarg dict serves both
):
    """Whole-attention compute for a workgroup whose band fits in ONE KV tile.

    Warp specialization buys nothing here: with a single tile there is no next tile to
    prefetch, so there is nothing for a producer half to run ahead on and no PV/QK
    ping-pong to anti-phase. All 8 waves run this one trace -- QK, softmax, PV -- and the
    only thing the wave's half still decides is which of K or V it copies from VRAM, a
    runtime branch rather than a second trace.

    Everything the pipelined core needs for a tile stream is gone: no loop, no slot
    rotation, no carried P/S hand-off, no clamped prefetch, no dead half-bodies. The one
    tile lands in physical slot 0 while Q sits in slot 1 and O stages through slots 1|2,
    so the tile is never overwritten and the whole body needs exactly ONE barrier -- the
    fence that publishes the tile.
    """
    lane_idx = _lane_id()
    kv_head, q_head_idx, seq_idx = _packed_tile_indices(
        gqa_ratio, warp_idx, lane_idx, num_q_tiles_per_wave
    )
    # 8-bit Q cannot carry the scale (no f8 multiply), so softmax applies it to S.
    _q_scale = None
    _s_scale = softmax_scale * fx.Float32(LOG2E)
    num_q_tiles = num_q_tiles_per_wave

    q_mgr, k_mgr, v_mgr = _build_managers(
        qk_hdim=qk_hdim,
        v_hdim=v_hdim,
        n_block=n_block,
        gqa_ratio=gqa_ratio,
        num_q_tiles_per_wave=num_q_tiles,
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
    )
    _SPLIT_STRIDE = 6 * LDS_CHUNK_BYTES
    slot_bytes = 2 * LDS_CHUNK_BYTES
    for _label, _bytes in (
        ("K", k_mgr.get_lds_size_in_byte()),
        ("V", v_mgr.get_lds_size_in_byte()),
    ):
        assert (
            _bytes % KV_LDS_SPLITS == 0 and _bytes // KV_LDS_SPLITS <= LDS_CHUNK_BYTES
        ), (
            f"{_label} split {_bytes // KV_LDS_SPLITS}B exceeds the {LDS_CHUNK_BYTES}B chunk "
            f"(qk_hdim={qk_hdim}, v_hdim={v_hdim}, n_block={n_block})"
        )
    assert q_mgr.warp_lds_size_in_byte() <= LDS_QO_BYTES

    # The tile is physical slot 0: K at chunks 0|6, V at 1|7.
    k_buf = lds_base
    v_buf = lds_base + fx.Int32(LDS_CHUNK_BYTES)
    assert KV_LDS_SPLITS == 2, "parity order needs a 2-way split"
    _is_odd_warp = warp_idx & fx.Int32(1)
    _read_order_lo = _is_odd_warp * fx.Int32(_SPLIT_STRIDE)
    _read_order = [_read_order_lo, fx.Int32(_SPLIT_STRIDE) - _read_order_lo]
    _kv_swap_delta = _is_odd_warp * fx.Int32(n_block // 2)

    def _split_bufs(b):
        return [b + o for o in _read_order]

    _q_chunk_sel = (warp_idx & fx.Int32(1)) ^ ((warp_idx >> fx.Int32(2)) & fx.Int32(1))
    q_lds_warp = (
        lds_base
        + fx.Int32(slot_bytes)
        + _q_chunk_sel * fx.Int32(_SPLIT_STRIDE)
        + (warp_idx >> fx.Int32(1)) * fx.Int32(LDS_QO_BYTES)
    )
    q_mgr.load_q_to_vgpr_part1(
        ptr_Q=ptr_Q,
        stride_q_seq=stride_q_seq,
        stride_q_head=stride_q_head,
        q_start=q_start,
        q_len=q_len,
        kv_head=kv_head,
        block_x=_m_tile_idx(),
        warp_idx=warp_idx,
        lane_idx=lane_idx,
        ptr_lds_warp=q_lds_warp,
    )

    _, causal_off, kv_len_wg, _ = _wg_kv_span(
        n_block=n_block,
        mask_right=mask_right,
        gqa_ratio=gqa_ratio,
        num_q_tiles_per_wave=num_q_tiles,
        q_len=q_len,
        kv_len=kv_len,
        window_right=window_right,
    )

    # Both descriptors are pure; only the copy one of them issues is warp-dependent.
    _kv_ctx = ProducerCtx(
        producer_warp=warp_idx % fx.Int32(KV_PRODUCER_WARPS),
        num_producer_warps=KV_PRODUCER_WARPS,
        lane_idx=lane_idx,
    )
    k_desc = k_mgr.load_descriptor(
        ptr_lds=[k_buf, k_buf + fx.Int32(_SPLIT_STRIDE)],
        ptr_src=ptr_K,
        stride_seq=stride_k_seq,
        stride_head=stride_k_head,
        head=kv_head,
        row0=kv_start,
        valid=kv_len_wg,
        ctx=_kv_ctx,
    )
    v_desc = v_mgr.load_descriptor(
        ptr_lds=[v_buf, v_buf + fx.Int32(_SPLIT_STRIDE)],
        ptr_src=ptr_V,
        stride_seq=stride_v_seq,
        stride_head=stride_v_head,
        head=kv_head,
        row0=kv_start,
        valid=kv_len_wg,
        ctx=_kv_ctx,
    )

    @flyc.jit
    def _issue_kv(is_lo):
        if is_lo:
            k_desc.async_load()
        else:
            v_desc.async_load()

    _is_lo = warp_idx < fx.Int32(NUM_WAVES // 2)
    _num_tdm_copies = max(k_desc.tensorcnt, v_desc.tensorcnt) or -1
    _num_async_copies = max(k_desc.asynccnt, v_desc.asynccnt) or -1
    if USE_TDM_LOADER:
        # The tile misses Q's chunks entirely, so it goes out before Q is read and its
        # global latency overlaps Q's. Waves differ in how many copies they issued, so
        # part2 drains only down to the count EVERY wave has in flight.
        _issue_kv(_is_lo)
        tdm_ops.tensor_wait(min(k_desc.tensorcnt, v_desc.tensorcnt))
        q_frags = q_mgr.load_q_to_vgpr_part2(scale=_q_scale)
    else:
        # V1 puts Q's own stage on asynccnt and part2 waits a depth counted over Q alone.
        q_frags = q_mgr.load_q_to_vgpr_part2(scale=_q_scale)
        _issue_kv(_is_lo)
    # The only barrier in the body: a wave's counters bound its own copies, so this is
    # what publishes the tile to the other seven.
    _copy_fence(*_kv_drain_depths(_num_tdm_copies, _num_async_copies))

    k_lds = k_mgr.ds_load_ptrs(ptr_lds=_split_bufs(k_buf), lane_idx=lane_idx)
    v_lds = v_mgr.ds_load_ptrs(ptr_lds=_split_bufs(v_buf), lane_idx=lane_idx)

    def _k_emit(j):
        return k_mgr.load_one_to_reg(k_lds, j)

    def _v_emit(j):
        return v_mgr.load_one_to_reg(v_lds, j)

    _, s_list = _pv_qk_gemm(
        v_emit=None,
        k_emit=_k_emit,
        vld_per_frag=v_mgr.loads_per_frag,
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
        p_list=[[] for _ in range(num_q_tiles)],
        q_frags_list=q_frags,
        v_hdim=v_hdim,
        n_block=n_block,
        warp_type=WarpType.LO,
        o_acc_list=[[] for _ in range(num_q_tiles)],
        phase="qk",
    )

    # PV's ring head goes out before softmax so its LDS latency hides under the VALU.
    _num_prefetch = _ring_num_prefetch(v_mgr.num_ds_loads(), PVQK_RING, PVQK_LAG)
    rocdl.sched_barrier(0)
    pv_head = []
    for j in range(_num_prefetch):
        pv_head.append(_v_emit(j))
        if len(pv_head) % v_mgr.loads_per_frag == 0:
            rocdl.sched_barrier(0)
    rocdl.sched_barrier(0)

    if has_sink:
        num_heads_q = gpu.grid_dim.y * fx.Int32(gqa_ratio)
        m_prev = [
            _load_sink_logit(ptr_sink, q_head_idx[q_tile_idx], num_heads_q)
            * fx.Float32(LOG2E)
            for q_tile_idx in range(num_q_tiles)
        ]
        d_prev = [fx.Float32(2.0**P_BIAS_LOG2) for _ in range(num_q_tiles)]
    else:
        m_prev = [fx.Float32(BIG_NEG) for _ in range(num_q_tiles)]
        d_prev = [fx.Float32(0.0) for _ in range(num_q_tiles)]
    # This tile is both the first and the last, so it always carries the kv_len tail and
    # O needs no rescale: the seed is 0 and corr*0 == 0 whatever the running max does.
    p_f32, m_new_list, d_new_list, _corr, _masks = _softmax(
        s_list=s_list,
        m_prev_list=m_prev,
        d_prev_list=d_prev,
        lane_idx=lane_idx,
        n_block=n_block,
        kv_pos_base=fx.Int32(0),
        kv_swap_delta=_kv_swap_delta,
        q_max_list=[
            seq_idx[q_tile_idx] + causal_off + window_right if mask_right else None
            for q_tile_idx in range(num_q_tiles)
        ],
        q_min_list=[
            seq_idx[q_tile_idx] + causal_off - window_left if mask_left else None
            for q_tile_idx in range(num_q_tiles)
        ],
        kv_len=kv_len,
        s_scale=_s_scale,
    )
    p_list = _convert_vectors(p_f32, P_DTYPE)
    _keepalive([v for pt in p_list for v in pt])
    o_list, _ = _pv_qk_gemm(
        v_emit=_v_emit,
        k_emit=None,
        vld_per_frag=v_mgr.loads_per_frag,
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
        p_list=p_list,
        q_frags_list=q_frags,
        v_hdim=v_hdim,
        n_block=n_block,
        warp_type=WarpType.LO,
        o_acc_list=None,
        head=pv_head,
        phase="pv",
    )

    # O stages through physical slots 1 and 2 -- Q's chunks, dead since the prologue, and
    # never touched by the tile in slot 0. Nothing is in flight and no wave can still be
    # reading them, so the epilogue needs neither a drain nor a rendezvous.
    o_lds_warp = (
        (_is_odd_warp > fx.Int32(0)).select(
            lds_base + fx.Int32(2 * slot_bytes), lds_base + fx.Int32(slot_bytes)
        )
        + ((warp_idx >> fx.Int32(1)) & fx.Int32(1)) * fx.Int32(LDS_QO_BYTES)
        + (warp_idx >> fx.Int32(2)) * fx.Int32(_SPLIT_STRIDE)
    )
    return _epi_state(
        m_list=m_new_list, d_list=d_new_list, o_list=o_list, o_lds_warp=o_lds_warp
    )


def _store_o_lse(
    *,
    v_descale,
    o_mgr,
    m_list,
    d_list,
    o_list,
    d_tiles,
    num_q_tiles,
    ptr_O,
    stride_o_seq,
    stride_o_head,
    q_start,
    q_len,
    kv_head,
    m_tile_idx,
    warp_idx,
    lane_idx,
    o_lds_warp,
    seq_idx,
    q_head_idx,
    return_lse,
    ptr_LSE,
    lse_base_elems,
    lse_num_records_bytes,
    stride_lse_seq,
    stride_lse_head,
):
    """Normalize, stage and store O (and optionally LSE) for this wave's num_q_tiles q-tiles.

    Shared tail of both compute cores; ``o_lds_warp`` is the caller's choice of which
    finished KV chunk this wave stages through."""
    for q_tile_idx in range(num_q_tiles):
        # Normalize this q-tile's O by its running denom d, then reshape+store to VRAM.
        # o_final[d_tile_idx] lane l elem si = sum_kv P[q,kv] V[kv, d_tile_idx*16+(l//16)*8+si]
        # (unnormalized); divide by the per-query denom d (peer-consistent across the
        # lane pair) to finish softmax. OManager16b masks rows with seq >= q_len.
        d_final = d_list[q_tile_idx]
        o_final = o_list[q_tile_idx]
        # Fully-masked row (d_final==0): 1/0=inf, o_final=0, 0*inf=NaN -> guard to O=0.
        inv_denom = (d_final > fx.Float32(0.0)).select(
            fx.Float32(1.0) / d_final, fx.Float32(0.0)
        )
        if v_descale is not None:
            inv_denom = inv_denom * v_descale
        inv_denom_vec = fx.Vector.from_elements([inv_denom], fx.Float32).broadcast_to(8)
        o_norm = [o_final[d_tile_idx] * inv_denom_vec for d_tile_idx in range(d_tiles)]
        if q_tile_idx > 0:
            rocdl.s_wait_dscnt(0)  # drain prev q-tile's O ring/DS ops before reuse
        o_mgr.store_o_to_vram(
            ptr_O=ptr_O,
            o_base_elems=fx.Int32(0),
            stride_o_seq=stride_o_seq,
            stride_o_head=stride_o_head,
            q_start=q_start,
            q_len=q_len,
            kv_head=kv_head,
            block_x=m_tile_idx,
            warp_idx=warp_idx,
            lane_idx=lane_idx,
            ptr_lds_warp=o_lds_warp,
            o_frags=o_norm,
            qtile=q_tile_idx,
        )

    # ---- LSE store (optional). LSE = (m_final + log2(d_final)) / LOG2E: m is carried in
    # log2 units of the scaled score and d is domain-free, so one multiply converts both,
    # matching torch.logsumexp(scale * Q @ K^T, dim=kv). The lane pair (l, l^16) holds
    # each query identically, so store once from the kv_half_idx==0 lanes, masked by
    # seq < q_len; buffer_store redirects mask-drops to byte 0x7FFFFFFF.
    if return_lse:
        khalf0 = (lane_idx // fx.Int32(WMMA_M)) == fx.Int32(0)
        lse_rsrc = buffer_ops.create_buffer_resource(
            ptr_LSE, num_records_bytes=lse_num_records_bytes
        )
        for q_tile_idx in range(num_q_tiles):
            m_final = m_list[q_tile_idx]
            d_final = d_list[q_tile_idx]
            # fx.log2 lowers to the HW v_log_f32 (base-2), matching m's log2 domain.
            lse_val = (
                m_final + fx.log2(d_final) - fx.Float32(P_BIAS_LOG2)
            ) * fx.Float32(1.0 / LOG2E)
            lse_mask = khalf0 & (seq_idx[q_tile_idx] < q_len)
            lse_off_el = (
                lse_base_elems
                + seq_idx[q_tile_idx] * stride_lse_seq
                + q_head_idx[q_tile_idx] * stride_lse_head
            )
            # Pre-mask the offset (OOB rows -> 0x7fffffff) and pass mask=None so the
            # store maps 1:1 to a single buffer_store with masking already SSA-visible.
            lse_off_masked = lse_mask.select(
                lse_off_el * fx.Int32(4), fx.Int32(0x7FFFFFFF)
            )
            buffer_ops.buffer_store(
                lse_val, lse_rsrc, lse_off_masked, mask=None, offset_is_bytes=True
            )


# Keys ``_attention_epilogue`` shares with the compute cores' kwarg dict.
_EPILOGUE_KEYS = (
    "v_hdim",
    "gqa_ratio",
    "num_q_tiles_per_wave",
    "return_lse",
    "ptr_O",
    "ptr_LSE",
    "stride_o_seq",
    "stride_o_head",
    "stride_lse_seq",
    "stride_lse_head",
    "lse_base_elems",
    "lse_num_records_bytes",
    "q_start",
    "q_len",
    "v_descale",
)


def _epi_state_zeros(*, num_q_tiles_per_wave, v_hdim):
    """Type exemplar for the warp-type scf.if. Both branches overwrite it, so the
    constants are dead; the rewriter needs the names bound before the branch."""
    d_tiles = v_hdim // WMMA_M
    num_q_tiles = num_q_tiles_per_wave
    return _epi_state(
        m_list=[fx.Float32(0.0) for _ in range(num_q_tiles)],
        d_list=[fx.Float32(0.0) for _ in range(num_q_tiles)],
        o_list=[
            [fx.Vector.filled(8, 0.0, fx.Float32) for _ in range(d_tiles)]
            for _ in range(num_q_tiles)
        ],
        o_lds_warp=fx.Int32(0),
    )


def _attention_epilogue(
    *,
    v_descale,
    epi_state,
    rendezvous,  # compile-time: the multi core leaves KV copies in O's chunk
    warp_idx,
    v_hdim,
    gqa_ratio,
    num_q_tiles_per_wave,
    return_lse,
    ptr_O,
    ptr_LSE,
    stride_o_seq,
    stride_o_head,
    stride_lse_seq,
    stride_lse_head,
    lse_base_elems,
    lse_num_records_bytes,
    q_start,
    q_len,
):
    """The O/LSE tail, traced once per build instead of once per warp type.

    Every core hands back the same flat state -- [m, d, O_0..O_{d_tiles-1}] per q-tile
    plus the LDS chunk this wave stages O through -- and everything else here is a pure
    function of the launch, so the warp-type scf.if can close before it."""
    d_tiles = v_hdim // WMMA_M
    num_q_tiles = num_q_tiles_per_wave
    _q_epi_slots = 2 + d_tiles
    lane_idx = _lane_id()
    kv_head, q_head_idx, seq_idx = _packed_tile_indices(
        gqa_ratio, warp_idx, lane_idx, num_q_tiles
    )
    if rendezvous:
        # O's chunk halves are shared by wave pairs and still hold the trailing dead KV
        # copies; each half drained only its own, so the pair meets before the ds_writes.
        _bare_barrier()
    _store_o_lse(
        v_descale=v_descale,
        o_mgr=_o_manager(
            v_hdim=v_hdim,
            gqa_ratio=gqa_ratio,
            num_q_tiles_per_wave=num_q_tiles,
            elem_dtype=O_DTYPE,
        ),
        m_list=[
            fx.Float32(epi_state[q_tile_idx * _q_epi_slots + 0])
            for q_tile_idx in range(num_q_tiles)
        ],
        d_list=[
            fx.Float32(epi_state[q_tile_idx * _q_epi_slots + 1])
            for q_tile_idx in range(num_q_tiles)
        ],
        o_list=[
            [
                fx.Vector(epi_state[q_tile_idx * _q_epi_slots + 2 + d_tile_idx])
                for d_tile_idx in range(d_tiles)
            ]
            for q_tile_idx in range(num_q_tiles)
        ],
        d_tiles=d_tiles,
        num_q_tiles=num_q_tiles,
        ptr_O=ptr_O,
        stride_o_seq=stride_o_seq,
        stride_o_head=stride_o_head,
        q_start=q_start,
        q_len=q_len,
        kv_head=kv_head,
        m_tile_idx=_m_tile_idx(),
        warp_idx=warp_idx,
        lane_idx=lane_idx,
        o_lds_warp=fx.Int32(epi_state[num_q_tiles * _q_epi_slots]),
        seq_idx=seq_idx,
        q_head_idx=q_head_idx,
        return_lse=return_lse,
        ptr_LSE=ptr_LSE,
        lse_base_elems=lse_base_elems,
        lse_num_records_bytes=lse_num_records_bytes,
        stride_lse_seq=stride_lse_seq,
        stride_lse_head=stride_lse_head,
    )


def _zero_fill_attention(
    *,
    v_hdim,
    gqa_ratio,
    num_q_tiles_per_wave,
    return_lse,
    has_sink,
    ptr_sink,
    ptr_O,
    ptr_LSE,
    stride_o_seq,
    stride_o_head,
    stride_lse_seq,
    stride_lse_head,
    lse_num_records_bytes,
    q_start,
    q_len,
):
    """q_len>0 with kv_len==0 (cross-attention): softmax over an empty KV set, so O=0 for
    this WG's valid query rows and LSE=-inf -- or LSE=sink[head] with a sink, exp(sink)
    being the only surviving term. Flat coalesced b128 write, no WMMA layout."""
    BLOCK_M = _block_m(num_q_tiles_per_wave)
    thread_idx = _warp_id() * fx.Int32(WAVE_SIZE) + _lane_id()
    kv_head = fx.Int32(gpu.block_id("y"))
    row0 = _m_tile_idx() * fx.Int32(BLOCK_M)
    g = fx.Int32(gqa_ratio)
    _chunk_elems = 8  # bf16 per b128 store
    num_chunks_per_row = v_hdim // _chunk_elems  # b128 chunks per O row

    # i64: an i32 product (large total_q * stride) can overflow negative, then the
    # descriptor sign-extends it to a huge bound, defeating the 0x7FFFFFFF OOB drop.
    o_num_records_bytes = (
        fx.Int64(q_start + q_len) * fx.Int64(stride_o_seq) * fx.Int64(2)
    )
    o_rsrc = buffer_ops.create_buffer_resource(
        ptr_O, num_records_bytes=o_num_records_bytes
    )
    zero_o = fx.Vector.filled(_chunk_elems, 0.0, O_DTYPE)
    for r in range(BLOCK_M * num_chunks_per_row // BLOCK_SIZE):
        chunk_idx = (
            fx.Int32(r * BLOCK_SIZE) + thread_idx
        )  # flat b128-chunk index this round
        packed_row_idx = row0 + chunk_idx // fx.Int32(num_chunks_per_row)
        d = (chunk_idx % fx.Int32(num_chunks_per_row)) * fx.Int32(_chunk_elems)
        seq = packed_row_idx // g
        head = kv_head * g + packed_row_idx % g
        off = (q_start + seq) * stride_o_seq + head * stride_o_head + d
        off_masked = (seq < q_len).select(off * fx.Int32(2), fx.Int32(0x7FFFFFFF))
        buffer_ops.buffer_store(
            zero_o, o_rsrc, off_masked, mask=None, offset_is_bytes=True
        )

    if return_lse:
        lse_rsrc = buffer_ops.create_buffer_resource(
            ptr_LSE, num_records_bytes=lse_num_records_bytes
        )
        packed_row_idx = row0 + thread_idx  # one LSE per packed row, one row per thread
        seq = packed_row_idx // g
        head = kv_head * g + packed_row_idx % g
        if has_sink:
            num_heads_q = gpu.grid_dim.y * g
            lse_val = _load_sink_logit(ptr_sink, head, num_heads_q)
        else:
            lse_val = fx.Float32(float("-inf"))
        off = (q_start + seq) * stride_lse_seq + head * stride_lse_head
        off_masked = (seq < q_len).select(off * fx.Int32(4), fx.Int32(0x7FFFFFFF))
        if BLOCK_M < BLOCK_SIZE:
            off_masked = (thread_idx < fx.Int32(BLOCK_M)).select(
                off_masked, fx.Int32(0x7FFFFFFF)
            )
        buffer_ops.buffer_store(
            lse_val, lse_rsrc, off_masked, mask=None, offset_is_bytes=True
        )


# ============================================================================
# Builder — one device kernel per (layout, config)
# ============================================================================


def kv_split_bytes(qk_hdim, v_hdim, n_block, kv_dtype):
    """Bytes one half of a 2-way-split K and V tile occupies, straight from the active
    managers -- n_block//KV_LDS_SPLITS rows of hdim*sizeof(elem) plus their per-row pad
    (16 B for K, 32 B for V)."""
    k_cls = KManager16bV2 if USE_TDM_LOADER else KManager16bV1
    v_cls = VManager16bV2 if USE_TDM_LOADER else VManager16bV1
    k_mgr = k_cls(
        qk_hdim=qk_hdim, n_block=n_block, num_waves=NUM_WAVES, elem_dtype=kv_dtype
    )
    v_mgr = v_cls(
        v_hdim=v_hdim, n_block=n_block, num_waves=NUM_WAVES, elem_dtype=kv_dtype
    )
    return (
        k_mgr.get_lds_size_in_byte() // KV_LDS_SPLITS,
        v_mgr.get_lds_size_in_byte() // KV_LDS_SPLITS,
    )


def pick_n_block(qk_hdim, v_hdim, kv_dtype):
    """Widest n_block from N_BLOCK_PREF whose split K and V tiles each still fit one
    LDS_CHUNK_BYTES chunk.

    The 16-bit kernel also caps wide n_block at N_BLOCK_WIDE_MAX_QK_HDIM, a capacity
    proxy that does not apply here: 8-bit halves the bytes, and n_block 64 is not merely
    narrow but invalid, since a contraction tile is 64 kv deep and each of the
    KV_LDS_SPLITS splits must hold whole tiles."""
    for candidate_n_block in N_BLOCK_PREF:
        if (
            max(kv_split_bytes(qk_hdim, v_hdim, candidate_n_block, kv_dtype))
            <= LDS_CHUNK_BYTES
        ):
            return candidate_n_block
    raise ValueError(
        f"no n_block in {N_BLOCK_PREF} fits the {LDS_CHUNK_BYTES}B chunk "
        f"(qk_hdim={qk_hdim}, v_hdim={v_hdim})"
    )


@functools.cache
def build_fmha_fwd_prefill_a8w8_m32x8(
    *,
    layout: str = "thd",
    qk_hdim: int = DEFAULT_QK_HDIM,
    v_hdim: int = DEFAULT_V_HDIM,
    n_block: int | None = None,
    q_dtype_str: str = DEFAULT_DTYPE,
    kv_dtype_str: str = DEFAULT_DTYPE,
    mask_left: bool = False,
    mask_right: bool = False,
    one_kv_tile: bool = False,
    return_lse: bool = False,
    has_sink: bool = False,
    has_descale: bool = False,
    gqa_ratio: int = 1,
    num_q_tiles_per_wave: int = WMMA_ROW_PER_WAVE,
):
    """Build the m32x8 device kernel for a given layout ("thd" varlen or "bshd" batched)
    plus config; every parameter here is compile-time and baked into the trace.
    ``gqa_ratio`` (= ``nheads_q // nheads_kv``) is among them so the per-lane ``% / //``
    fold to shift/and when it is a power of two.
    """
    assert layout in ("thd", "bshd"), f"layout must be thd|bshd, got {layout!r}"
    # qk_hdim (D_qk) is a WMMA_K multiple; v_hdim (D_v) a WMMA_M multiple. Independent.
    assert qk_hdim in SUPPORTED_QK_HDIM and v_hdim in SUPPORTED_V_HDIM, (
        f"supports qk_hdim in {SUPPORTED_QK_HDIM} x v_hdim in {SUPPORTED_V_HDIM}, got {qk_hdim}/{v_hdim}"
    )
    assert q_dtype_str in SUPPORTED_DTYPES and kv_dtype_str in SUPPORTED_DTYPES, (
        f"dtypes must be in {list(SUPPORTED_DTYPES)}, got {q_dtype_str!r}/{kv_dtype_str!r}"
    )
    Q_ELEM_DTYPE = DTYPE_MAP[q_dtype_str]
    KV_ELEM_DTYPE = DTYPE_MAP[kv_dtype_str]
    assert gqa_ratio >= 1, f"gqa_ratio must be >= 1, got {gqa_ratio}"
    assert num_q_tiles_per_wave in NUM_Q_TILES_CHOICES, (
        f"num_q_tiles_per_wave must be in {NUM_Q_TILES_CHOICES}, got {num_q_tiles_per_wave}"
    )
    if n_block is None:
        n_block = pick_n_block(qk_hdim, v_hdim, KV_ELEM_DTYPE)
    assert n_block in N_BLOCK_CHOICES, (
        f"n_block must be in {N_BLOCK_CHOICES}, got {n_block}"
    )

    QK_HDIM = qk_hdim
    V_HDIM = v_hdim
    N_BLOCK = int(n_block)
    MASK_LEFT = bool(mask_left)
    MASK_RIGHT = bool(mask_right)
    HAS_DESCALE = bool(has_descale)
    ONE_KV_TILE = bool(one_kv_tile)
    RET_LSE = bool(return_lse)
    HAS_SINK = bool(has_sink)
    GQA_RATIO = int(gqa_ratio)
    NUM_Q_TILES = int(num_q_tiles_per_wave)

    if layout == "thd":

        @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
        def muse_extend_fly_m32x8_thd(
            ptr_O: fx.Pointer,
            ptr_Q: fx.Pointer,
            ptr_K: fx.Pointer,
            ptr_V: fx.Pointer,
            ptr_LSE: fx.Pointer,
            ptr_sink: fx.Pointer,
            ptr_q_descale: fx.Pointer,
            ptr_k_descale: fx.Pointer,
            ptr_v_descale: fx.Pointer,
            ptr_cu_seqlens_q: fx.Pointer,
            ptr_cu_seqlens_k: fx.Pointer,
            softmax_scale: fx.Float32,
            stride_q_seq: fx.Int32,
            stride_k_seq: fx.Int32,
            stride_v_seq: fx.Int32,
            stride_o_seq: fx.Int32,
            stride_q_head: fx.Int32,
            stride_k_head: fx.Int32,
            stride_v_head: fx.Int32,
            stride_o_head: fx.Int32,
            stride_lse_seq: fx.Int32,
            stride_lse_head: fx.Int32,
            window_left: fx.Int32,
            window_right: fx.Int32,
            max_seqlen_q: fx.Int32,
            max_seqlen_k: fx.Int32,
        ):
            """Varlen THD entry: this batch's token ranges come from cu_seqlens
            (batch = grid.z)."""
            batch = fx.Int32(gpu.block_id("z"))
            q_start, q_end = _load_seqlen_pair(ptr_cu_seqlens_q, batch)
            kv_start, kv_end = _load_seqlen_pair(ptr_cu_seqlens_k, batch)
            q_len = q_end - q_start
            kv_len = kv_end - kv_start

            # LSE is [nheads_q, total_q]. Bound the buffer resource by the last element
            # this batch can touch over BOTH axes; a seq-only bound is exact only when
            # seq is the major axis.
            num_heads_q = gpu.grid_dim.y * fx.Int32(GQA_RATIO)
            lse_base_elems = q_start * stride_lse_seq
            lse_num_records_bytes = (
                fx.Int64(q_start + q_len - fx.Int32(1)) * fx.Int64(stride_lse_seq)
                + fx.Int64(num_heads_q - fx.Int32(1)) * fx.Int64(stride_lse_head)
                + fx.Int64(1)
            ) * fx.Int64(4)

            # Per-tensor descales, as gfx950 does: q/k scale the fp32 logits (so they
            # ride softmax_scale and reach the LSE through m), v scales O alone.
            if const_expr(HAS_DESCALE):
                _softmax_scale = softmax_scale * (
                    _load_descale(ptr_q_descale) * _load_descale(ptr_k_descale)
                )
                _v_descale = _load_descale(ptr_v_descale)
            else:
                _softmax_scale = softmax_scale
                _v_descale = None

            # An empty batch must not enter the core: kv_len==0 would write O/0 = NaN.
            if (q_len > fx.Int32(0)) & (kv_len > fx.Int32(0)):
                _core_kwargs = {
                    "qk_hdim": QK_HDIM,
                    "v_hdim": V_HDIM,
                    "n_block": N_BLOCK,
                    "mask_left": MASK_LEFT,
                    "mask_right": MASK_RIGHT,
                    "return_lse": RET_LSE,
                    "has_sink": HAS_SINK,
                    "gqa_ratio": GQA_RATIO,
                    "num_q_tiles_per_wave": NUM_Q_TILES,
                    "ptr_O": ptr_O,
                    "ptr_Q": ptr_Q,
                    "ptr_K": ptr_K,
                    "ptr_V": ptr_V,
                    "ptr_LSE": ptr_LSE,
                    "ptr_sink": ptr_sink,
                    "softmax_scale": _softmax_scale,
                    "v_descale": _v_descale,
                    "stride_q_seq": stride_q_seq,
                    "stride_k_seq": stride_k_seq,
                    "stride_v_seq": stride_v_seq,
                    "stride_o_seq": stride_o_seq,
                    "stride_q_head": stride_q_head,
                    "stride_k_head": stride_k_head,
                    "stride_v_head": stride_v_head,
                    "stride_o_head": stride_o_head,
                    "stride_lse_seq": stride_lse_seq,
                    "stride_lse_head": stride_lse_head,
                    "lse_base_elems": lse_base_elems,
                    "lse_num_records_bytes": lse_num_records_bytes,
                    "q_start": q_start,
                    "q_len": q_len,
                    "kv_start": kv_start,
                    "kv_len": kv_len,
                    "window_left": window_left,
                    "window_right": window_right,
                    "q_dtype": Q_ELEM_DTYPE,
                    "kv_dtype": KV_ELEM_DTYPE,
                }
                # Exactly one core reaches the binary: the host decides from the shape (bshd) or
                # from max_seqlen_k (thd) whether every WG sees a single KV tile.
                lds_base = _alloc_lds()
                warp_idx = _warp_id()

                _epi_state_out = _epi_state_zeros(
                    num_q_tiles_per_wave=NUM_Q_TILES, v_hdim=V_HDIM
                )
                if ONE_KV_TILE:
                    _epi_state_out = _attention_core_one_kv_tile(
                        warp_idx=warp_idx, lds_base=lds_base, **_core_kwargs
                    )
                else:
                    # One traced copy of Q's load and the first KV tiles, outside the
                    # warp-type branch; the two halves differ only under its own branches.
                    _prologue = _attention_prologue(
                        warp_idx=warp_idx,
                        lds_base=lds_base,
                        **{k: _core_kwargs[k] for k in _PROLOGUE_KEYS},
                    )
                    # Warp specialization: LO (waves 0..N/2-1) vs HI (N/2..N-1).
                    if warp_idx // fx.Int32(NUM_WAVES // 2) == fx.Int32(0):
                        _epi_state_out = _attention_core_multi_kv_tiles(
                            warp_idx=warp_idx,
                            warp_type=WarpType.LO,
                            lds_base=lds_base,
                            prologue=_prologue,
                            **_core_kwargs,
                        )
                    else:
                        _epi_state_out = _attention_core_multi_kv_tiles(
                            warp_idx=warp_idx,
                            warp_type=WarpType.HI,
                            lds_base=lds_base,
                            prologue=_prologue,
                            **_core_kwargs,
                        )
                # One traced copy of the O/LSE tail, outside the warp-type branch.
                _attention_epilogue(
                    epi_state=_epi_state_out,
                    rendezvous=not ONE_KV_TILE,
                    warp_idx=warp_idx,
                    **{k: _core_kwargs[k] for k in _EPILOGUE_KEYS},
                )
            elif q_len > fx.Int32(0):
                # Cross-attention tail: q_len>0 but kv_len==0 -> O=0, LSE=-inf (or sink).
                _zero_fill_attention(
                    v_hdim=V_HDIM,
                    gqa_ratio=GQA_RATIO,
                    num_q_tiles_per_wave=NUM_Q_TILES,
                    return_lse=RET_LSE,
                    has_sink=HAS_SINK,
                    ptr_sink=ptr_sink,
                    ptr_O=ptr_O,
                    ptr_LSE=ptr_LSE,
                    stride_o_seq=stride_o_seq,
                    stride_o_head=stride_o_head,
                    stride_lse_seq=stride_lse_seq,
                    stride_lse_head=stride_lse_head,
                    lse_num_records_bytes=lse_num_records_bytes,
                    q_start=q_start,
                    q_len=q_len,
                )

        return muse_extend_fly_m32x8_thd

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def muse_extend_fly_m32x8_bshd(
        ptr_O: fx.Pointer,
        ptr_Q: fx.Pointer,
        ptr_K: fx.Pointer,
        ptr_V: fx.Pointer,
        ptr_LSE: fx.Pointer,
        ptr_sink: fx.Pointer,
        ptr_q_descale: fx.Pointer,
        ptr_k_descale: fx.Pointer,
        ptr_v_descale: fx.Pointer,
        softmax_scale: fx.Float32,
        stride_q_seq: fx.Int32,
        stride_k_seq: fx.Int32,
        stride_v_seq: fx.Int32,
        stride_o_seq: fx.Int32,
        stride_q_head: fx.Int32,
        stride_k_head: fx.Int32,
        stride_v_head: fx.Int32,
        stride_o_head: fx.Int32,
        stride_lse_seq: fx.Int32,
        stride_lse_head: fx.Int32,
        stride_lse_batch: fx.Int32,
        window_left: fx.Int32,
        window_right: fx.Int32,
        seq_len_q: fx.Int32,
        seq_len_k: fx.Int32,
    ):
        """Batched BSHD entry: uniform ``seq_len_q``/``seq_len_k`` scalars replace
        cu_seqlens, so nothing is transient and this path is CUDA-graph safe. Token base
        is batch_idx * seq_len (batch = grid.z)."""
        batch = fx.Int32(gpu.block_id("z"))

        # LSE is [B, nheads_q, seq_q]: base = batch*stride_lse_batch; every valid
        # element offset is < base + stride_lse_batch (< the 0x7FFFFFFF drop).
        lse_base_elems = batch * stride_lse_batch
        lse_num_records_bytes = fx.Int64(lse_base_elems + stride_lse_batch) * fx.Int64(
            4
        )

        # Per-tensor descales, as in the thd entry.
        if const_expr(HAS_DESCALE):
            _softmax_scale = softmax_scale * (
                _load_descale(ptr_q_descale) * _load_descale(ptr_k_descale)
            )
            _v_descale = _load_descale(ptr_v_descale)
        else:
            _softmax_scale = softmax_scale
            _v_descale = None

        _core_kwargs = {
            "qk_hdim": QK_HDIM,
            "v_hdim": V_HDIM,
            "n_block": N_BLOCK,
            "mask_left": MASK_LEFT,
            "mask_right": MASK_RIGHT,
            "return_lse": RET_LSE,
            "has_sink": HAS_SINK,
            "gqa_ratio": GQA_RATIO,
            "num_q_tiles_per_wave": NUM_Q_TILES,
            "ptr_O": ptr_O,
            "ptr_Q": ptr_Q,
            "ptr_K": ptr_K,
            "ptr_V": ptr_V,
            "ptr_LSE": ptr_LSE,
            "ptr_sink": ptr_sink,
            "softmax_scale": _softmax_scale,
            "v_descale": _v_descale,
            "stride_q_seq": stride_q_seq,
            "stride_k_seq": stride_k_seq,
            "stride_v_seq": stride_v_seq,
            "stride_o_seq": stride_o_seq,
            "stride_q_head": stride_q_head,
            "stride_k_head": stride_k_head,
            "stride_v_head": stride_v_head,
            "stride_o_head": stride_o_head,
            "stride_lse_seq": stride_lse_seq,
            "stride_lse_head": stride_lse_head,
            "lse_base_elems": lse_base_elems,
            "lse_num_records_bytes": lse_num_records_bytes,
            "q_start": batch * seq_len_q,
            "q_len": seq_len_q,
            "kv_start": batch * seq_len_k,
            "kv_len": seq_len_k,
            "window_left": window_left,
            "window_right": window_right,
            "q_dtype": Q_ELEM_DTYPE,
            "kv_dtype": KV_ELEM_DTYPE,
        }
        # Exactly one core reaches the binary: the host decides from the shape (bshd) or
        # from max_seqlen_k (thd) whether every WG sees a single KV tile.
        lds_base = _alloc_lds()
        warp_idx = _warp_id()

        _epi_state_out = _epi_state_zeros(
            num_q_tiles_per_wave=NUM_Q_TILES, v_hdim=V_HDIM
        )
        if ONE_KV_TILE:
            _epi_state_out = _attention_core_one_kv_tile(
                warp_idx=warp_idx, lds_base=lds_base, **_core_kwargs
            )
        else:
            # One traced copy of Q's load and the first KV tiles, outside the
            # warp-type branch; the two halves differ only under its own branches.
            _prologue = _attention_prologue(
                warp_idx=warp_idx,
                lds_base=lds_base,
                **{k: _core_kwargs[k] for k in _PROLOGUE_KEYS},
            )
            # Warp specialization: LO (waves 0..N/2-1) vs HI (N/2..N-1).
            if warp_idx // fx.Int32(NUM_WAVES // 2) == fx.Int32(0):
                _epi_state_out = _attention_core_multi_kv_tiles(
                    warp_idx=warp_idx,
                    warp_type=WarpType.LO,
                    lds_base=lds_base,
                    prologue=_prologue,
                    **_core_kwargs,
                )
            else:
                _epi_state_out = _attention_core_multi_kv_tiles(
                    warp_idx=warp_idx,
                    warp_type=WarpType.HI,
                    lds_base=lds_base,
                    prologue=_prologue,
                    **_core_kwargs,
                )
        # One traced copy of the O/LSE tail, outside the warp-type branch.
        _attention_epilogue(
            epi_state=_epi_state_out,
            rendezvous=not ONE_KV_TILE,
            warp_idx=warp_idx,
            **{k: _core_kwargs[k] for k in _EPILOGUE_KEYS},
        )

    return muse_extend_fly_m32x8_bshd


# ============================================================================
# Launch wrappers + host entries
# ============================================================================

_launch_fns = {}  # {(layout, mask_left, mask_right, return_lse, has_sink, gqa_ratio): fn}


def _pick_one_kv_tile(max_seqlen_k, qk_hdim, v_hdim, kv_dtype_str):
    """True when the launch bounds every WG to a single KV tile."""
    n_block = pick_n_block(qk_hdim, v_hdim, DTYPE_MAP[kv_dtype_str])
    return int(max_seqlen_k) <= n_block


def _ensure_thd_kernel(
    mask_left: bool,
    mask_right: bool,
    one_kv_tile: bool,
    return_lse: bool,
    has_sink: bool,
    has_descale: bool,
    gqa_ratio: int,
    qk_hdim: int = DEFAULT_QK_HDIM,
    v_hdim: int = DEFAULT_V_HDIM,
    q_dtype_str: str = DEFAULT_DTYPE,
    kv_dtype_str: str = DEFAULT_DTYPE,
    num_q_tiles_per_wave: int = WMMA_ROW_PER_WAVE,
):
    key = (
        "thd",
        bool(mask_left),
        bool(mask_right),
        bool(return_lse),
        bool(has_sink),
        bool(has_descale),
        int(gqa_ratio),
        int(qk_hdim),
        int(v_hdim),
        str(q_dtype_str),
        str(kv_dtype_str),
        int(num_q_tiles_per_wave),
        bool(one_kv_tile),
    )
    if key in _launch_fns:
        return
    kernel = build_fmha_fwd_prefill_a8w8_m32x8(
        layout="thd",
        qk_hdim=qk_hdim,
        v_hdim=v_hdim,
        mask_left=mask_left,
        mask_right=mask_right,
        one_kv_tile=one_kv_tile,
        return_lse=return_lse,
        has_sink=has_sink,
        has_descale=has_descale,
        gqa_ratio=gqa_ratio,
        q_dtype_str=q_dtype_str,
        kv_dtype_str=kv_dtype_str,
        num_q_tiles_per_wave=num_q_tiles_per_wave,
    )
    block_m = _block_m(num_q_tiles_per_wave)

    @flyc.jit
    def _launch(
        ptr_O: fx.Pointer,
        ptr_Q: fx.Pointer,
        ptr_K: fx.Pointer,
        ptr_V: fx.Pointer,
        ptr_LSE: fx.Pointer,
        ptr_sink: fx.Pointer,
        ptr_q_descale: fx.Pointer,
        ptr_k_descale: fx.Pointer,
        ptr_v_descale: fx.Pointer,
        ptr_cu_seqlens_q: fx.Pointer,
        ptr_cu_seqlens_k: fx.Pointer,
        softmax_scale: fx.Float32,
        stride_q_seq: fx.Int32,
        stride_k_seq: fx.Int32,
        stride_v_seq: fx.Int32,
        stride_o_seq: fx.Int32,
        stride_q_head: fx.Int32,
        stride_k_head: fx.Int32,
        stride_v_head: fx.Int32,
        stride_o_head: fx.Int32,
        stride_lse_seq: fx.Int32,
        stride_lse_head: fx.Int32,
        window_left: fx.Int32,
        window_right: fx.Int32,
        max_seqlen_q: fx.Int32,
        max_seqlen_k: fx.Int32,
        num_heads_kv: fx.Int32,
        batch_size: fx.Int32,
        stream: fx.Stream,
    ):
        # 3D grid: x = tiles over (seq, q_head_in_group) per kv-head,
        #          y = kv_head, z = batch. block = 256 (8 waves x wave32).
        grid_x = fx.Index(
            fx.ceildiv(fx.Uint32(max_seqlen_q * gqa_ratio), fx.Uint32(block_m))
        )
        grid_y = fx.Index(num_heads_kv)
        grid_z = fx.Index(batch_size)

        launcher = kernel(
            ptr_O,
            ptr_Q,
            ptr_K,
            ptr_V,
            ptr_LSE,
            ptr_sink,
            ptr_q_descale,
            ptr_k_descale,
            ptr_v_descale,
            ptr_cu_seqlens_q,
            ptr_cu_seqlens_k,
            softmax_scale,
            stride_q_seq,
            stride_k_seq,
            stride_v_seq,
            stride_o_seq,
            stride_q_head,
            stride_k_head,
            stride_v_head,
            stride_o_head,
            stride_lse_seq,
            stride_lse_head,
            window_left,
            window_right,
            max_seqlen_q,
            max_seqlen_k,
        )
        launcher.launch(
            grid=(grid_x, grid_y, grid_z),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    _launch.compile_hints["llvm_options"] = {
        "amdgpu-expert-scheduling-mode": ENABLE_SCHED_MODE2,
        # "amdgpu-sched-strategy": "coexec",  # gfx1250 co-exec sched for warp specialization
    }
    _launch.compile_hints["waves_per_eu"] = 2
    _launch_fns[key] = _launch


def _ensure_bshd_kernel(
    mask_left: bool,
    mask_right: bool,
    one_kv_tile: bool,
    return_lse: bool,
    has_sink: bool,
    has_descale: bool,
    gqa_ratio: int,
    qk_hdim: int = DEFAULT_QK_HDIM,
    v_hdim: int = DEFAULT_V_HDIM,
    q_dtype_str: str = DEFAULT_DTYPE,
    kv_dtype_str: str = DEFAULT_DTYPE,
    num_q_tiles_per_wave: int = WMMA_ROW_PER_WAVE,
):
    key = (
        "bshd",
        bool(mask_left),
        bool(mask_right),
        bool(return_lse),
        bool(has_sink),
        bool(has_descale),
        int(gqa_ratio),
        int(qk_hdim),
        int(v_hdim),
        str(q_dtype_str),
        str(kv_dtype_str),
        int(num_q_tiles_per_wave),
        bool(one_kv_tile),
    )
    if key in _launch_fns:
        return
    kernel = build_fmha_fwd_prefill_a8w8_m32x8(
        layout="bshd",
        qk_hdim=qk_hdim,
        v_hdim=v_hdim,
        mask_left=mask_left,
        mask_right=mask_right,
        one_kv_tile=one_kv_tile,
        return_lse=return_lse,
        has_sink=has_sink,
        has_descale=has_descale,
        gqa_ratio=gqa_ratio,
        q_dtype_str=q_dtype_str,
        kv_dtype_str=kv_dtype_str,
        num_q_tiles_per_wave=num_q_tiles_per_wave,
    )
    block_m = _block_m(num_q_tiles_per_wave)

    @flyc.jit
    def _launch(
        ptr_O: fx.Pointer,
        ptr_Q: fx.Pointer,
        ptr_K: fx.Pointer,
        ptr_V: fx.Pointer,
        ptr_LSE: fx.Pointer,
        ptr_sink: fx.Pointer,
        ptr_q_descale: fx.Pointer,
        ptr_k_descale: fx.Pointer,
        ptr_v_descale: fx.Pointer,
        softmax_scale: fx.Float32,
        stride_q_seq: fx.Int32,
        stride_k_seq: fx.Int32,
        stride_v_seq: fx.Int32,
        stride_o_seq: fx.Int32,
        stride_q_head: fx.Int32,
        stride_k_head: fx.Int32,
        stride_v_head: fx.Int32,
        stride_o_head: fx.Int32,
        stride_lse_seq: fx.Int32,
        stride_lse_head: fx.Int32,
        stride_lse_batch: fx.Int32,
        window_left: fx.Int32,
        window_right: fx.Int32,
        seq_len_q: fx.Int32,
        seq_len_k: fx.Int32,
        num_heads_kv: fx.Int32,
        batch_size: fx.Int32,
        stream: fx.Stream,
    ):
        # 3D grid: x = tiles over (seq, q_head_in_group) per kv-head,
        #          y = kv_head, z = batch. block = 256 (8 waves x wave32).
        grid_x = fx.Index(
            fx.ceildiv(fx.Uint32(seq_len_q * gqa_ratio), fx.Uint32(block_m))
        )
        grid_y = fx.Index(num_heads_kv)
        grid_z = fx.Index(batch_size)

        launcher = kernel(
            ptr_O,
            ptr_Q,
            ptr_K,
            ptr_V,
            ptr_LSE,
            ptr_sink,
            ptr_q_descale,
            ptr_k_descale,
            ptr_v_descale,
            softmax_scale,
            stride_q_seq,
            stride_k_seq,
            stride_v_seq,
            stride_o_seq,
            stride_q_head,
            stride_k_head,
            stride_v_head,
            stride_o_head,
            stride_lse_seq,
            stride_lse_head,
            stride_lse_batch,
            window_left,
            window_right,
            seq_len_q,
            seq_len_k,
        )
        launcher.launch(
            grid=(grid_x, grid_y, grid_z),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    _launch.compile_hints["llvm_options"] = {
        "amdgpu-expert-scheduling-mode": ENABLE_SCHED_MODE2,
        # "amdgpu-sched-strategy": "coexec",  # gfx1250 co-exec sched for warp specialization
    }
    _launch.compile_hints["waves_per_eu"] = 2
    _launch_fns[key] = _launch


def flash_attn_a8w8_m32x8_supported(
    mode,
    q,
    k,
    v,
    *,
    dropout_p=0.0,
    window_size=(-1, -1),
    bias=None,
    alibi_slopes=None,
    sink=None,
    block_table=None,
    return_attn_probs=False,
) -> bool:
    """Can the m32x8 prefill kernel serve this configuration?

    Metadata only -- no allocation, compile or launch -- so a caller can gate before
    preparing buffers. The ARCH is the caller's business: dispatch on it first and call
    this only on gfx1250. ``mode`` is "varlen" or "batch"; both are served by the same
    kernel today, so it is accepted and ignored.

    Per-call buffer validation (a caller-provided ``out``) stays with the entry points.
    """
    del mode  # same gate for both layouts today
    qk_hdim, v_hdim = q.shape[-1], v.shape[-1]
    num_q, num_kv = q.shape[-2], k.shape[-2]
    sink_ok = sink is None or (
        torch.is_tensor(sink) and sink.dtype == torch.float32 and sink.shape == (num_q,)
    )
    return (
        qk_hdim in SUPPORTED_QK_HDIM
        and v_hdim in SUPPORTED_V_HDIM
        and k.shape[-1] == qk_hdim
        and q.dtype in _TORCH_IN_DTYPES
        and k.dtype in _TORCH_IN_DTYPES
        and v.dtype == k.dtype
        and num_kv > 0
        and num_q % num_kv == 0
        and supports_gqa_ratio(num_q // max(num_kv, 1), v_hdim)
        and dropout_p == 0.0
        and (len(window_size) < 3 or window_size[2] == 0)
        and block_table is None
        and bias is None
        and alibi_slopes is None
        and sink_ok
        and not return_attn_probs
    )


def flash_attn_varlen_a8w8_m32x8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    out=None,
    return_lse=False,
    sink=None,
    lse=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    num_q_tiles_per_wave=None,
):
    """Host entry — varlen THD, qk_hdim in {64,128,192,256} / v_hdim in {64,128,256}, fp8 or bf8.

    ``window_size`` is ``(left, right)`` sliding-window bounds, ``-1`` meaning infinite
    on that side; ``causal`` forces ``right=0``. Only finiteness is baked into the kernel
    (compile-time ``mask_left``/``mask_right``), so one variant serves any magnitude.
    ``sink`` is an optional ``[nheads_q]`` fp32 per-head logit in the scaled-score domain,
    its presence likewise compile-time. ``lse`` is an optional caller-provided
    ``[nheads_q, total_q]`` fp32 buffer, allocated here when ``return_lse`` and None.
    """
    assert q.dtype in SUPPORTED_TORCH_DTYPES and k.dtype in SUPPORTED_TORCH_DTYPES, (
        f"Expected fp8 or bf8, got {q.dtype}/{k.dtype}"
    )
    assert v.dtype == k.dtype, f"k/v dtype must match, got {k.dtype}/{v.dtype}"
    q_dtype_str = _TORCH_IN_DTYPES[q.dtype]
    kv_dtype_str = _TORCH_IN_DTYPES[k.dtype]
    qk_hdim = q.shape[-1]
    assert qk_hdim in SUPPORTED_QK_HDIM, (
        f"Expected qk_hdim in {SUPPORTED_QK_HDIM}, got {qk_hdim}"
    )
    v_hdim = v.shape[-1]
    assert v_hdim in SUPPORTED_V_HDIM, (
        f"Expected v_hdim in {SUPPORTED_V_HDIM}, got {v_hdim}"
    )

    total_q_tokens = q.shape[0]
    batch = cu_seqlens_q.shape[0] - 1
    nheads_q = q.shape[1]
    nheads_k = k.shape[1]
    assert nheads_q % nheads_k == 0, (
        f"nheads_q={nheads_q} must be a multiple of nheads_k={nheads_k}"
    )
    gqa = nheads_q // nheads_k

    has_sink = sink is not None
    if has_sink:
        assert sink.dtype == torch.float32, f"sink must be fp32, got {sink.dtype}"
        assert sink.dim() == 1 and sink.shape[0] == nheads_q, (
            f"sink must be [nheads_q={nheads_q}], got {tuple(sink.shape)}"
        )
    # ptr_sink is only read when has_sink; pass q as a valid placeholder otherwise.
    sink_ptr = sink if has_sink else q

    # Per-tensor fp32 descales, all three or none, as gfx950 requires. Presence is
    # compile-time; the pointers are always passed, with q as a placeholder when unused.
    _descales = (q_descale, k_descale, v_descale)
    has_descale = any(d is not None for d in _descales)
    if has_descale:
        assert all(d is not None for d in _descales), (
            "q/k/v_descale must be given together"
        )
        for _name, _d in zip(("q", "k", "v"), _descales):
            assert torch.is_tensor(_d) and _d.dtype == torch.float32, (
                f"{_name}_descale must be a float32 tensor, got {type(_d)}"
            )
            assert _d.numel() == 1, f"{_name}_descale must hold one element"
            assert _d.device == q.device, f"{_name}_descale must be on {q.device}"
    qd_ptr, kd_ptr, vd_ptr = _descales if has_descale else (q, q, q)

    if softmax_scale is None:
        softmax_scale = 1.0 / (q.shape[-1] ** 0.5)

    # Sliding window: causal forces right=0. Finiteness (>=0) is compile-time
    # (mask_left/mask_right); the magnitudes ride along as runtime Int32 args.
    win_left, win_right = int(window_size[0]), int(window_size[1])
    if causal:
        win_right = 0
    mask_left = win_left >= 0
    mask_right = win_right >= 0
    window_left = max(win_left, 0)
    window_right = max(win_right, 0)

    if out is None:
        out = torch.empty(
            (total_q_tokens, nheads_q, v_hdim),
            dtype=TORCH_O_DTYPE,
            device=q.device,
        )
    if return_lse:
        if lse is None:
            # [nheads_q, total_q] is the aiter varlen LSE convention, what CK and the
            # gfx1250 ASM varlen kernel both return. The kernel is stride-driven, so the
            # layout lives entirely in the two strides below.
            lse = torch.empty(
                (nheads_q, total_q_tokens), dtype=torch.float32, device=q.device
            )
        lse_ptr = lse
        stride_lse_seq = lse.stride(1)
        stride_lse_head = lse.stride(0)
    else:
        lse_ptr = q
        stride_lse_seq = 0
        stride_lse_head = 0

    # Q/K/V/O strides in ELEMENTS (TDM loaders consume them directly).
    stride_q_seq = q.stride(0)
    stride_k_seq = k.stride(0)
    stride_v_seq = v.stride(0)
    stride_o_seq = out.stride(0)
    stride_q_head = q.stride(1)
    stride_k_head = k.stride(1)
    stride_v_head = v.stride(1)
    stride_o_head = out.stride(1)

    if num_q_tiles_per_wave is None:
        num_q_tiles_per_wave = _pick_num_q_tiles(
            nheads_k, batch, max_seqlen_q, gqa, v_hdim
        )
    num_q_tiles_per_wave = int(num_q_tiles_per_wave)

    one_kv_tile = _pick_one_kv_tile(max_seqlen_k, qk_hdim, v_hdim, kv_dtype_str)

    _ensure_thd_kernel(
        mask_left,
        mask_right,
        one_kv_tile,
        bool(return_lse),
        has_sink,
        has_descale,
        gqa,
        qk_hdim=qk_hdim,
        v_hdim=v_hdim,
        q_dtype_str=q_dtype_str,
        kv_dtype_str=kv_dtype_str,
        num_q_tiles_per_wave=num_q_tiles_per_wave,
    )

    _run_compiled(
        _launch_fns[
            (
                "thd",
                mask_left,
                mask_right,
                bool(return_lse),
                has_sink,
                has_descale,
                gqa,
                qk_hdim,
                v_hdim,
                q_dtype_str,
                kv_dtype_str,
                num_q_tiles_per_wave,
                one_kv_tile,
            )
        ],
        out,
        q,
        k,
        v,
        lse_ptr,
        sink_ptr,
        qd_ptr,
        kd_ptr,
        vd_ptr,
        cu_seqlens_q,
        cu_seqlens_k,
        softmax_scale,
        stride_q_seq,
        stride_k_seq,
        stride_v_seq,
        stride_o_seq,
        stride_q_head,
        stride_k_head,
        stride_v_head,
        stride_o_head,
        stride_lse_seq,
        stride_lse_head,
        window_left,
        window_right,
        max_seqlen_q,
        max_seqlen_k,
        nheads_k,
        batch,
        torch.cuda.current_stream(),
    )

    if return_lse:
        return out, lse
    return out


def flash_attn_batch_a8w8_m32x8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    out=None,
    return_lse=False,
    sink=None,
    lse=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    num_q_tiles_per_wave=None,
):
    """Host entry — batched BSHD ``[B, S, H, D]``, qk_hdim in {64,128,192,256} / v_hdim in {64,128,256}, fp8 or bf8.

    Uses the dedicated BSHD kernel with a uniform ``seq_len`` scalar (no cu_seqlens), so
    there is nothing transient to bake into a CUDA graph. ``window_size``, ``causal``,
    ``sink`` and ``lse`` behave as in ``flash_attn_varlen_a8w8_m32x8``; the LSE buffer is
    ``[B, nheads_q, S_q]`` here.
    """
    assert q.dtype in SUPPORTED_TORCH_DTYPES and k.dtype in SUPPORTED_TORCH_DTYPES, (
        f"Expected fp8 or bf8, got {q.dtype}/{k.dtype}"
    )
    assert v.dtype == k.dtype, f"k/v dtype must match, got {k.dtype}/{v.dtype}"
    q_dtype_str = _TORCH_IN_DTYPES[q.dtype]
    kv_dtype_str = _TORCH_IN_DTYPES[k.dtype]
    assert q.dim() == 4, f"Expected 4D BSHD tensor, got rank {q.dim()}"
    qk_hdim = q.shape[-1]
    assert qk_hdim in SUPPORTED_QK_HDIM, (
        f"Expected qk_hdim in {SUPPORTED_QK_HDIM}, got {qk_hdim}"
    )
    v_hdim = v.shape[-1]
    assert v_hdim in SUPPORTED_V_HDIM, (
        f"Expected v_hdim in {SUPPORTED_V_HDIM}, got {v_hdim}"
    )

    batch, seq_len_q, nheads_q, _ = q.shape
    seq_len_k = k.shape[1]
    nheads_k = k.shape[2]
    assert nheads_q % nheads_k == 0, (
        f"nheads_q={nheads_q} must be a multiple of nheads_k={nheads_k}"
    )
    gqa = nheads_q // nheads_k

    has_sink = sink is not None
    if has_sink:
        assert sink.dtype == torch.float32, f"sink must be fp32, got {sink.dtype}"
        assert sink.dim() == 1 and sink.shape[0] == nheads_q, (
            f"sink must be [nheads_q={nheads_q}], got {tuple(sink.shape)}"
        )
    # ptr_sink is only read when has_sink; pass q as a valid placeholder otherwise.
    sink_ptr = sink if has_sink else q

    # Per-tensor fp32 descales, all three or none, as gfx950 requires. Presence is
    # compile-time; the pointers are always passed, with q as a placeholder when unused.
    _descales = (q_descale, k_descale, v_descale)
    has_descale = any(d is not None for d in _descales)
    if has_descale:
        assert all(d is not None for d in _descales), (
            "q/k/v_descale must be given together"
        )
        for _name, _d in zip(("q", "k", "v"), _descales):
            assert torch.is_tensor(_d) and _d.dtype == torch.float32, (
                f"{_name}_descale must be a float32 tensor, got {type(_d)}"
            )
            assert _d.numel() == 1, f"{_name}_descale must hold one element"
            assert _d.device == q.device, f"{_name}_descale must be on {q.device}"
    qd_ptr, kd_ptr, vd_ptr = _descales if has_descale else (q, q, q)

    if softmax_scale is None:
        softmax_scale = 1.0 / (q.shape[-1] ** 0.5)

    # Sliding window: causal forces right=0. Finiteness (>=0) is compile-time
    # (mask_left/mask_right); the magnitudes ride along as runtime Int32 args.
    win_left, win_right = int(window_size[0]), int(window_size[1])
    if causal:
        win_right = 0
    mask_left = win_left >= 0
    mask_right = win_right >= 0
    window_left = max(win_left, 0)
    window_right = max(win_right, 0)

    if out is None:
        out = torch.empty(
            (batch, seq_len_q, nheads_q, v_hdim),
            dtype=TORCH_O_DTYPE,
            device=q.device,
        )
    if return_lse:
        if lse is None:
            lse = torch.empty(
                (batch, nheads_q, seq_len_q), dtype=torch.float32, device=q.device
            )
        lse_ptr = lse
        stride_lse_seq = lse.stride(2)
        stride_lse_head = lse.stride(1)
        stride_lse_batch = lse.stride(0)
    else:
        lse_ptr = q
        stride_lse_seq = 0
        stride_lse_head = 0
        stride_lse_batch = 0

    # Empty tensor — skip the launch (host-known dims, no device sync). No queries means
    # no rows to write; no keys means an empty softmax set, so O=0 and LSE=-inf, or
    # LSE=sink[head] with a sink, exp(sink) being the only surviving term.
    if seq_len_q == 0 or seq_len_k == 0:
        if seq_len_q > 0 and seq_len_k == 0:
            out.zero_()
            if return_lse:
                if sink is not None:
                    lse.copy_(
                        sink.to(device=lse.device, dtype=lse.dtype)
                        .view(1, -1, 1)
                        .expand_as(lse)
                    )
                else:
                    lse.fill_(float("-inf"))
        return (out, lse) if return_lse else out

    # BSHD: seq is dim 1, head dim 2; the per-batch base is derived in-kernel as
    # batch_idx * seq_len. Strides in ELEMENTS (TDM loaders consume them directly).
    stride_q_seq = q.stride(1)
    stride_k_seq = k.stride(1)
    stride_v_seq = v.stride(1)
    stride_o_seq = out.stride(1)
    stride_q_head = q.stride(2)
    stride_k_head = k.stride(2)
    stride_v_head = v.stride(2)
    stride_o_head = out.stride(2)

    if num_q_tiles_per_wave is None:
        num_q_tiles_per_wave = _pick_num_q_tiles(
            nheads_k, batch, seq_len_q, gqa, v_hdim
        )
    num_q_tiles_per_wave = int(num_q_tiles_per_wave)

    one_kv_tile = _pick_one_kv_tile(seq_len_k, qk_hdim, v_hdim, kv_dtype_str)

    _ensure_bshd_kernel(
        mask_left,
        mask_right,
        one_kv_tile,
        bool(return_lse),
        has_sink,
        has_descale,
        gqa,
        qk_hdim=qk_hdim,
        v_hdim=v_hdim,
        q_dtype_str=q_dtype_str,
        kv_dtype_str=kv_dtype_str,
        num_q_tiles_per_wave=num_q_tiles_per_wave,
    )

    _run_compiled(
        _launch_fns[
            (
                "bshd",
                mask_left,
                mask_right,
                bool(return_lse),
                has_sink,
                has_descale,
                gqa,
                qk_hdim,
                v_hdim,
                q_dtype_str,
                kv_dtype_str,
                num_q_tiles_per_wave,
                one_kv_tile,
            )
        ],
        out,
        q,
        k,
        v,
        lse_ptr,
        sink_ptr,
        qd_ptr,
        kd_ptr,
        vd_ptr,
        softmax_scale,
        stride_q_seq,
        stride_k_seq,
        stride_v_seq,
        stride_o_seq,
        stride_q_head,
        stride_k_head,
        stride_v_head,
        stride_o_head,
        stride_lse_seq,
        stride_lse_head,
        stride_lse_batch,
        window_left,
        window_right,
        seq_len_q,
        seq_len_k,
        nheads_k,
        batch,
        torch.cuda.current_stream(),
    )

    if return_lse:
        return out, lse
    return out


@triton.jit
def _kv_store_q_absmax_kernel(
    Q,
    K,
    V,
    KC,
    VC,
    KO,
    VO,
    LOC,
    MAP,
    KS,
    VS,
    PART,
    n,
    q_stride,
    k_stride,
    v_stride,
    QW: tl.constexpr,
    BQW: tl.constexpr,
    KW: tl.constexpr,
    BLOCK_T: tl.constexpr,
    HAS_MAP: tl.constexpr,
):
    # Rows [pid * BLOCK_T, +BLOCK_T): Q's partial |max| (no atomics), FP8 K/V into the cache slots (reshape_and_cache
    # rounding) and into contiguous [tokens, KV heads * 64] copies for the kernel.
    pid = tl.program_id(0)
    t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    live = t < n
    c = tl.arange(0, BQW)
    q = tl.load(
        Q + t[:, None] * q_stride + c[None, :],
        mask=live[:, None] & (c < QW)[None, :],
        other=0.0,
    )
    tl.store(PART + pid, tl.max(tl.max(tl.abs(q.to(tl.float32)), axis=1), axis=0))
    ck = tl.arange(0, KW)
    k = tl.load(K + t[:, None] * k_stride + ck[None, :], mask=live[:, None], other=0.0)
    v = tl.load(V + t[:, None] * v_stride + ck[None, :], mask=live[:, None], other=0.0)
    k = (k / tl.load(KS)).to(KO.dtype.element_ty)
    v = (v / tl.load(VS)).to(VO.dtype.element_ty)
    dst = t.to(tl.int64)[:, None] * KW + ck[None, :]
    tl.store(KO + dst, k, mask=live[:, None])
    tl.store(VO + dst, v, mask=live[:, None])
    slot = tl.load(LOC + t, mask=live, other=-1).to(tl.int64)
    if HAS_MAP:
        slot = tl.load(MAP + slot, mask=slot >= 0, other=-1).to(tl.int64)
    cache = slot[:, None] * KW + ck[None, :]
    tl.store(KC + cache, k, mask=(slot >= 0)[:, None])
    tl.store(VC + cache, v, mask=(slot >= 0)[:, None])


@triton.jit
def _q_fp8_from_partials_kernel(
    Q,
    PART,
    OUT,
    DESCALE,
    n,
    n_part,
    q_stride,
    QW: tl.constexpr,
    BQW: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BP: tl.constexpr,
):
    # quantize_q_fp8's per-tensor scale and rounding, from the partial maxima.
    pid = tl.program_id(0)
    p = tl.arange(0, BP)
    amax = tl.max(tl.load(PART + p, mask=p < n_part, other=0.0), axis=0)
    descale = tl.maximum(amax * (1.0 / 448.0), 1e-12)
    if pid == 0:
        tl.store(DESCALE, descale)
    t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    c = tl.arange(0, BQW)
    mask = (t < n)[:, None] & (c < QW)[None, :]
    q = tl.load(Q + t[:, None] * q_stride + c[None, :], mask=mask, other=0.0).to(
        tl.float32
    )
    q = tl.math.div_rn(q, descale).to(tl.float8e4nv, fp_downcast_rounding="rtne")
    tl.store(OUT + t.to(tl.int64)[:, None] * QW + c[None, :], q, mask=mask)


def muse_extend_attention_fly_gfx1250(
    q,
    k,
    v,
    k_cache,
    v_cache,
    cache_loc,
    swa_mapping,
    k_descale,
    v_descale,
    out,
    cu_seqlens_q,
    max_q,
    scale,
    window=0,
    block_t=8,
    warps=4,
):
    """Prefix-free extend (every request's KV is exactly its query tokens) from this step's BF16 ``q [tokens, heads *
    64]`` and ``k`` / ``v [tokens, KV heads, 64]`` (rows may be strided): stores FP8 K/V at ``cache_loc`` (through
    ``swa_mapping`` when given) of the cache [slots, KV heads, 64], quantizes Q per tensor as ``quantize_q_fp8`` does,
    and runs the FlyDSL kernel on contiguous FP8 copies of K/V; ``window`` > 0 = sliding (key visible when
    q - window < key <= q). Writes ``out [tokens, heads, 64]`` BF16."""
    n, hk, d = k.shape
    q = q.reshape(n, -1)
    qw, kw = q.shape[1], hk * d
    assert (
        q.stride(1) == 1
        and k.stride(1) == v.stride(1) == d
        and k.stride(2) == v.stride(2) == 1
    )
    assert k_cache.is_contiguous() and v_cache.is_contiguous()
    k8 = torch.empty((n, hk, d), dtype=k_cache.dtype, device=q.device)
    v8 = torch.empty_like(k8)
    q8 = torch.empty((n, qw), dtype=torch.float8_e4m3fn, device=q.device)
    q_descale = torch.empty(1, dtype=torch.float32, device=q.device)
    n_part = triton.cdiv(n, block_t)
    part = torch.empty(n_part, dtype=torch.float32, device=q.device)
    bqw = triton.next_power_of_2(qw)
    _kv_store_q_absmax_kernel[(n_part,)](
        q,
        k,
        v,
        k_cache,
        v_cache,
        k8,
        v8,
        cache_loc,
        swa_mapping,
        k_descale,
        v_descale,
        part,
        n,
        q.stride(0),
        k.stride(0),
        v.stride(0),
        QW=qw,
        BQW=bqw,
        KW=kw,
        BLOCK_T=block_t,
        HAS_MAP=swa_mapping is not None,
        num_warps=warps,
    )
    _q_fp8_from_partials_kernel[(n_part,)](
        q,
        part,
        q8,
        q_descale,
        n,
        n_part,
        q.stride(0),
        QW=qw,
        BQW=bqw,
        BLOCK_T=block_t,
        BP=triton.next_power_of_2(n_part),
        num_warps=warps,
    )
    flash_attn_varlen_a8w8_m32x8(
        q8.view(n, qw // d, d),
        k8,
        v8,
        cu_seqlens_q,
        cu_seqlens_q,
        max_q,
        max_q,
        softmax_scale=scale,
        causal=True,
        window_size=(window - 1, 0) if window else (-1, -1),
        out=out,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    return out
