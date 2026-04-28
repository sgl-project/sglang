# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""SGLang-facing Gluon extend-attention dispatch for gfx950.

Public entry point: :func:`gluon_extend_attention_fwd`. The API mirrors
SGLang's Triton extend-attention backend for symmetric heads only
(``Lq == Lv``) at D in {64, 128, 256}, with BF16 or OCP FP8 KV cache.
DeepSeek MLA / mixed-dim heads live in a separate ``mla_prefill`` branch.

This module is intentionally a fast host entry point, not a pure wrapper:
it owns feature gating, shape-keyed dispatch heuristics, WCA/split-K
workspace reuse, and the direct Triton/Gluon ``kernel[grid](...)``
launches. The kernel bodies themselves are the schedule-specific wrappers
in this file:

* ``gluon_extend_attn_serial_fwd`` for serial NS=1 data-centric tiles.
* ``gluon_extend_attn_fwd_4w`` for the 4-warp software-pipeline schedule.
* ``gluon_extend_attn_fwd_8w`` for the 8-warp pingpong schedule.

Regular attention, sliding-window attention, and attention sinks all need
to stay on fast paths. The only deliberate exclusion is WCA/split-K for
SWA+sinks together, because that combination is unsafe in the WCA
body; it remains supported through the non-WCA data-centric launch.

Terminology: data-centric launches use a rectangular 3D grid where each
CTA maps directly to one ``(seq, head, block_m)`` tile. WCA launches a
compact 1D grid over an output-tile estimate; the in-kernel scan rejects
any overestimated slots, and each CTA may walk
``tile_idx += total_programs`` to cover more logical tiles. Therefore
``SPLIT_K == 1`` WCA means "no prefix partition/reduction", not "the
data-centric kernel".
"""

import functools
from enum import IntEnum
from typing import NamedTuple

import torch
import triton

from ._common import (
    ExtendAttentionLayouts,
    ExtendAttnConfig,
    ExtendAttnPingpong8WProgram,
    ExtendAttnProgram,
    ExtendAttnSerialProgram,
    ExtendAttnSwPipeline4WProgram,
    gl,
    gluon,
)

# ===---------------------------------------------------------------------===#
# Serial kernel (NS=1, NW in {2, 4})
# ===---------------------------------------------------------------------===#


@gluon.jit
def gluon_extend_attn_serial_fwd(
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
    sm_scale,
    kv_group_num,
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
    IS_CAUSAL: gl.constexpr,
    USE_CUSTOM_MASK: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
    IS_FP8: gl.constexpr = False,
    Sinks_present: gl.constexpr = False,
    LOGIT_CAP: gl.constexpr = 0.0,
    XAI_TEMPERATURE_LEN: gl.constexpr = -1,
    SLIDING_WINDOW_SIZE: gl.constexpr = -1,
    HAS_WINDOW_OFFSETS: gl.constexpr = False,
    v_scale=1.0,
    # XCD-aware PID remap metadata. The serial kernel is non-WCA and keeps
    # these as identity defaults; 4w/8w data-centric and WCA wrappers may
    # opt in through the dispatcher policy.
    XCD_REMAP: gl.constexpr = False,
    NUM_XCDS: gl.constexpr = 8,
    XCD_CHUNK: gl.constexpr = 1,
    XCD_MODE: gl.constexpr = 1,
):
    """Serial Gluon extend-attention kernel for gfx950 (NS=1, NW in {2, 4}).

    See ``ExtendAttnSerialProgram.run`` in ``_common.py`` for the body.
    Always launched as a 3D data-centric grid (one tile per CTA); there is no
    WCA variant.
    """
    num_warps: gl.constexpr = gl.num_warps()
    BLOCK_DV: gl.constexpr = BLOCK_DMODEL

    layouts: gl.constexpr = ExtendAttentionLayouts(
        IS_FP8,
        num_warps,
        Q_Extend.dtype.element_ty,
        K_Buffer.dtype.element_ty,
    )
    cfg: gl.constexpr = ExtendAttnConfig(
        layouts=layouts,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DV=BLOCK_DV,
        NUM_STAGES=1,
        EXT_BLOCK_N=0,
        EXT_NUM_STAGES=0,
        num_warps=num_warps,
        IS_CAUSAL=IS_CAUSAL,
        USE_CUSTOM_MASK=USE_CUSTOM_MASK,
        ENABLE_PREFIX_UNMASKED=False,
        LOGIT_CAP=LOGIT_CAP,
        XAI_TEMPERATURE_LEN=XAI_TEMPERATURE_LEN,
        SLIDING_WINDOW_SIZE=SLIDING_WINDOW_SIZE,
        HAS_WINDOW_OFFSETS=HAS_WINDOW_OFFSETS,
        IS_FP8=IS_FP8,
        HAS_SINK=Sinks_present,
        IS_WCA=False,
        SPLIT_K=1,
        XCD_REMAP=XCD_REMAP,
        NUM_XCDS=NUM_XCDS,
        XCD_CHUNK=XCD_CHUNK,
        XCD_MODE=XCD_MODE,
    )
    # Optional tensor slots use constexpr placeholders when the feature is off.
    # See ``ExtendAttnProgram`` for the union-typed aggregate fields.
    if Sinks_present:
        Sinks_arg = Sinks
    else:
        Sinks_arg = gl.constexpr(0)

    state = ExtendAttnProgram.initialize(
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
        Sinks_arg,
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
        0,
        0,
        0,
        0,
        0,
        0,
        0,  # WCA / split-K workspace (unused on the serial kernel)
    )
    pgm = ExtendAttnSerialProgram(state)
    pgm.run()


# ===---------------------------------------------------------------------===#
# 4-Warp sw-pipeline BF16 + FP8 KV Kernel (NS>=2, D>=128, NW<8)
# ===---------------------------------------------------------------------===#


@gluon.jit
def gluon_extend_attn_fwd_4w(
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
    sm_scale,
    kv_group_num,
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
    IS_CAUSAL: gl.constexpr,
    USE_CUSTOM_MASK: gl.constexpr,
    ENABLE_PREFIX_UNMASKED: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
    NUM_STAGES: gl.constexpr,
    Sinks,
    HAS_SINK: gl.constexpr,
    LOGIT_CAP: gl.constexpr,
    XAI_TEMPERATURE_LEN: gl.constexpr,
    SLIDING_WINDOW_SIZE: gl.constexpr,
    HAS_WINDOW_OFFSETS: gl.constexpr,
    v_scale,
    # Scheduling metadata: ``0`` (int) defaults are re-wrapped
    # to ``gl.constexpr(0)`` inside ``ExtendAttnProgram.__init__`` on the
    # identity path. The dispatcher passes real values for WCA/split-K and
    # opt-in data-centric XCD remap launches.
    num_heads=0,
    total_valid_tiles=0,
    total_programs=0,
    partial_out=0,
    partial_lse=0,
    tile_done=0,
    actual_batch_size=0,
    IS_WCA: gl.constexpr = False,
    SPLIT_K: gl.constexpr = 1,
    IS_FP8: gl.constexpr = False,
    EXT_BLOCK_N: gl.constexpr = 0,
    EXT_NUM_STAGES: gl.constexpr = 0,
    # XCD-aware PID remap (MI350X: 8 XCDs); identity defaults below,
    # dispatcher fills real values from the launch policy.
    XCD_REMAP: gl.constexpr = False,
    NUM_XCDS: gl.constexpr = 8,
    XCD_CHUNK: gl.constexpr = 1,
    XCD_MODE: gl.constexpr = 1,
):
    """4-warp software-pipelined Gluon extend-attention kernel for gfx950.

    See ``ExtendAttnSwPipeline4WProgram.run`` in ``_common.py`` for the body.
    Invariants: ``NUM_STAGES>=2``, ``BLOCK_DMODEL>=128``, and
    ``num_warps<8``.
    """
    num_warps: gl.constexpr = gl.num_warps()
    BLOCK_DV: gl.constexpr = BLOCK_DMODEL

    layouts: gl.constexpr = ExtendAttentionLayouts(
        IS_FP8,
        num_warps,
        Q_Extend.dtype.element_ty,
        K_Buffer.dtype.element_ty,
    )
    cfg: gl.constexpr = ExtendAttnConfig(
        layouts=layouts,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DV=BLOCK_DV,
        NUM_STAGES=NUM_STAGES,
        EXT_BLOCK_N=EXT_BLOCK_N,
        EXT_NUM_STAGES=EXT_NUM_STAGES,
        num_warps=num_warps,
        IS_CAUSAL=IS_CAUSAL,
        USE_CUSTOM_MASK=USE_CUSTOM_MASK,
        ENABLE_PREFIX_UNMASKED=ENABLE_PREFIX_UNMASKED,
        LOGIT_CAP=LOGIT_CAP,
        XAI_TEMPERATURE_LEN=XAI_TEMPERATURE_LEN,
        SLIDING_WINDOW_SIZE=SLIDING_WINDOW_SIZE,
        HAS_WINDOW_OFFSETS=HAS_WINDOW_OFFSETS,
        IS_FP8=IS_FP8,
        HAS_SINK=HAS_SINK,
        IS_WCA=IS_WCA,
        SPLIT_K=SPLIT_K,
        XCD_REMAP=XCD_REMAP,
        NUM_XCDS=NUM_XCDS,
        XCD_CHUNK=XCD_CHUNK,
        XCD_MODE=XCD_MODE,
    )
    # Optional tensor slots use constexpr placeholders when the feature is off.
    if HAS_SINK:
        Sinks_arg = Sinks
    else:
        Sinks_arg = gl.constexpr(0)

    state = ExtendAttnProgram.initialize(
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
        Sinks_arg,
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
    pgm = ExtendAttnSwPipeline4WProgram(state)
    pgm.run()


# ===---------------------------------------------------------------------===#
# 8-Warp Pingpong BF16 + FP8 KV Kernel (NS>=2, NW>=8)
# ===---------------------------------------------------------------------===#


@gluon.jit
def gluon_extend_attn_fwd_8w(
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
    sm_scale,
    kv_group_num,
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
    IS_CAUSAL: gl.constexpr,
    USE_CUSTOM_MASK: gl.constexpr,
    ENABLE_PREFIX_UNMASKED: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
    NUM_STAGES: gl.constexpr,
    Sinks,
    HAS_SINK: gl.constexpr,
    LOGIT_CAP: gl.constexpr,
    XAI_TEMPERATURE_LEN: gl.constexpr,
    SLIDING_WINDOW_SIZE: gl.constexpr,
    HAS_WINDOW_OFFSETS: gl.constexpr,
    v_scale,
    # Scheduling metadata: same convention as the 4w wrapper.
    num_heads=0,
    total_valid_tiles=0,
    total_programs=0,
    partial_out=0,
    partial_lse=0,
    tile_done=0,
    actual_batch_size=0,
    IS_WCA: gl.constexpr = False,
    SPLIT_K: gl.constexpr = 1,
    IS_FP8: gl.constexpr = False,
    EXT_BLOCK_N: gl.constexpr = 0,
    EXT_NUM_STAGES: gl.constexpr = 0,
    XCD_REMAP: gl.constexpr = False,
    NUM_XCDS: gl.constexpr = 8,
    XCD_CHUNK: gl.constexpr = 1,
    XCD_MODE: gl.constexpr = 1,
):
    """8-warp pingpong Gluon extend-attention kernel for gfx950.

    See ``ExtendAttnPingpong8WProgram.run`` in ``_common.py`` for the body.
    """
    num_warps: gl.constexpr = gl.num_warps()
    BLOCK_DV: gl.constexpr = BLOCK_DMODEL

    layouts: gl.constexpr = ExtendAttentionLayouts(
        IS_FP8,
        num_warps,
        Q_Extend.dtype.element_ty,
        K_Buffer.dtype.element_ty,
    )
    cfg: gl.constexpr = ExtendAttnConfig(
        layouts=layouts,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DV=BLOCK_DV,
        NUM_STAGES=NUM_STAGES,
        EXT_BLOCK_N=EXT_BLOCK_N,
        EXT_NUM_STAGES=EXT_NUM_STAGES,
        num_warps=num_warps,
        IS_CAUSAL=IS_CAUSAL,
        USE_CUSTOM_MASK=USE_CUSTOM_MASK,
        ENABLE_PREFIX_UNMASKED=ENABLE_PREFIX_UNMASKED,
        LOGIT_CAP=LOGIT_CAP,
        XAI_TEMPERATURE_LEN=XAI_TEMPERATURE_LEN,
        SLIDING_WINDOW_SIZE=SLIDING_WINDOW_SIZE,
        HAS_WINDOW_OFFSETS=HAS_WINDOW_OFFSETS,
        IS_FP8=IS_FP8,
        HAS_SINK=HAS_SINK,
        IS_WCA=IS_WCA,
        SPLIT_K=SPLIT_K,
        XCD_REMAP=XCD_REMAP,
        NUM_XCDS=NUM_XCDS,
        XCD_CHUNK=XCD_CHUNK,
        XCD_MODE=XCD_MODE,
    )
    if HAS_SINK:
        Sinks_arg = Sinks
    else:
        Sinks_arg = gl.constexpr(0)

    state = ExtendAttnProgram.initialize(
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
        Sinks_arg,
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
    pgm = ExtendAttnPingpong8WProgram(state)
    pgm.run()


# ===-----------------------------------------------------------------------===#
# Module state: XCD remap policy and host-side dispatch caches
# ===-----------------------------------------------------------------------===#

# XCD-aware PID remap (MI350X: 8 XCDs x 32 CUs x 4 MB L2-per-XCD).
# Selected WCA and D64 BF16 data-centric launches opt into first-class remap
# policies below; serial kernels stay on the identity mapping.
_NUM_XCDS = 8  # gfx950 / MI350X constant
_XCD_CHUNK = 8


class XcdRemapMode(IntEnum):
    OFF = 0
    CHUNKED = 1
    ATTENTION = 2


class XcdRemapConfig(NamedTuple):
    enabled: bool
    num_xcds: int
    chunk: int
    mode: XcdRemapMode


def _disabled_xcd_remap_config() -> XcdRemapConfig:
    return XcdRemapConfig(False, _NUM_XCDS, 1, XcdRemapMode.CHUNKED)


def _select_xcd_chunk(total_valid_tiles: int) -> int:
    return max(1, min(_XCD_CHUNK, max(1, int(total_valid_tiles) // _NUM_XCDS)))


def _select_data_centric_xcd_mode(
    *,
    block_dmodel: int,
    kv_is_fp8: bool,
    head_num: int,
    batch_size: int,
    total_valid_tiles: int,
    sliding_window_size: int,
) -> XcdRemapMode:
    """First-class XCD remap policy for the rectangular 3D data grid."""
    if int(block_dmodel) != 64 or kv_is_fp8 or int(batch_size) < 8:
        return XcdRemapMode.OFF

    data_blocks = int(total_valid_tiles) // max(1, int(batch_size) * int(head_num))
    if int(batch_size) >= 16 and data_blocks == 8 and int(sliding_window_size) <= 0:
        return XcdRemapMode.ATTENTION
    return XcdRemapMode.CHUNKED


def _select_wca_xcd_mode(
    *,
    split_k: int,
    block_dmodel: int,
    kv_is_fp8: bool,
    head_num: int,
    batch_size: int,
    total_extend_rows: int,
    sliding_window_size: int,
    logit_cap: float,
) -> XcdRemapMode:
    """First-class XCD remap policy for WCA/split-K launches."""
    long_qh32_non_swa = (
        int(head_num) == 32
        and int(sliding_window_size) <= 0
        and (
            (int(batch_size) >= 32 and int(total_extend_rows) < 4096)
            or (max(1, int(split_k)) > 1 and float(logit_cap) > 0.0)
        )
    )
    if int(block_dmodel) == 128 and not kv_is_fp8 and not long_qh32_non_swa:
        return XcdRemapMode.CHUNKED
    return XcdRemapMode.OFF


def _select_xcd_remap_config(
    *,
    is_wca: bool,
    is_serial: bool,
    split_k: int,
    total_valid_tiles: int,
    block_dmodel: int,
    kv_is_fp8: bool,
    head_num: int,
    batch_size: int,
    total_extend_rows: int,
    sliding_window_size: int,
    logit_cap: float,
) -> XcdRemapConfig:
    """Select the host-side XCD remap policy for a launch.

    The production heuristic is intentionally narrower than "all WCA":
    A/B data showed reliable wins on D128 BF16 WCA, but FP8 and D256 had
    mixed results, and qH=32 non-SWA B32 / split-logit shapes regressed.
    D64 BF16 data-centric launches use a separate rectangular-grid policy.
    """
    if total_valid_tiles <= 0:
        return _disabled_xcd_remap_config()
    if not is_wca:
        if is_serial:
            return _disabled_xcd_remap_config()
        # D64 BF16 data-centric A/B showed chunked is broadly positive for
        # B>=8. The decode-style contiguous remap only clearly wins for the
        # high-batch, 8-block, non-SWA shapes; leave smaller cases on chunked.
        mode = _select_data_centric_xcd_mode(
            block_dmodel=block_dmodel,
            kv_is_fp8=kv_is_fp8,
            head_num=head_num,
            batch_size=batch_size,
            total_valid_tiles=total_valid_tiles,
            sliding_window_size=sliding_window_size,
        )
        if mode == XcdRemapMode.OFF:
            return _disabled_xcd_remap_config()
        return XcdRemapConfig(
            True,
            _NUM_XCDS,
            _select_xcd_chunk(total_valid_tiles),
            mode,
        )

    mode = _select_wca_xcd_mode(
        split_k=split_k,
        block_dmodel=block_dmodel,
        kv_is_fp8=kv_is_fp8,
        head_num=head_num,
        batch_size=batch_size,
        total_extend_rows=total_extend_rows,
        sliding_window_size=sliding_window_size,
        logit_cap=logit_cap,
    )
    if mode == XcdRemapMode.OFF:
        return _disabled_xcd_remap_config()
    return XcdRemapConfig(
        True,
        _NUM_XCDS,
        _select_xcd_chunk(total_valid_tiles),
        mode,
    )


# ===-----------------------------------------------------------------------===#
# Route predicate helpers
# ===-----------------------------------------------------------------------===#


def _should_route_serial_data_centric_bf16(
    Lq: int,
    batch_size: int,
    max_len_extend: int,
    total_prefix_len: int | None,
) -> bool:
    """Preflight for BF16 shapes that should use the serial kernel.

    The serial kernel handles regular attention, SWA, sinks, and custom masks
    itself, so this route is feature-complete for BF16.
    """
    if Lq not in (64, 128):
        return False
    avg_pfx = total_prefix_len // max(batch_size, 1) if total_prefix_len else 0
    if Lq == 64:
        if batch_size == 1 and avg_pfx <= 1024 and max_len_extend <= 256:
            return True
        if 2 <= batch_size <= 7 and avg_pfx <= 1024 and max_len_extend <= 128:
            return True
        if batch_size >= 8 and avg_pfx <= 1024 and max_len_extend <= 64:
            return True
    else:  # Lq == 128
        if batch_size == 1 and avg_pfx <= 512 and max_len_extend <= 256:
            return True
        if 2 <= batch_size <= 7 and avg_pfx <= 512 and max_len_extend <= 128:
            return True
        if 8 <= batch_size <= 15 and avg_pfx <= 512 and max_len_extend <= 64:
            return True
    return False


# Tiny manual cache for a pure shape transform. ``Lq`` only takes a few
# production values, but this helper is on every dispatch path.
_QK_SPLIT_CACHE = {}


def _resolve_qk_split_dims(Lq: int):
    """Return BLOCK_DMODEL (next power of 2 >= Lq, >= 16)."""
    cached = _QK_SPLIT_CACHE.get(Lq)
    if cached is not None:
        return cached
    block_dmodel = max(triton.next_power_of_2(Lq), 16)
    _QK_SPLIT_CACHE[Lq] = block_dmodel
    return block_dmodel


# CU count is hardware state, not input-dependent. Most serving processes bind
# one GPU, so ``_single_device_num_CUs`` avoids even the per-device dict lookup
# after the first query while still handling multi-device tests correctly.
_cached_num_CUs = {}
_single_device_num_CUs = None


def _get_num_CUs(device):
    """Return CU count, cached for the common one-GPU-per-process serving case."""
    global _single_device_num_CUs
    if _single_device_num_CUs is not None:
        return _single_device_num_CUs
    idx = device.index if hasattr(device, "index") and device.index is not None else 0
    cached = _cached_num_CUs.get(idx)
    if cached is None:
        cached = torch.cuda.get_device_properties(device).multi_processor_count
        _cached_num_CUs[idx] = cached
    _single_device_num_CUs = cached
    return cached


# ===-----------------------------------------------------------------------===#
# WCA grid and split-K scheduling helpers
# ===-----------------------------------------------------------------------===#


def _select_wca_grid(total_valid_tiles: int, num_CUs: int) -> int:
    """Pick CTA count for the WCA kernel.

    When there are more tiles than CUs we cap at 2*CUs for good occupancy.
    When there are fewer tiles than CUs (decode, spec-decode) we use all
    available tiles -- split-K will eventually fill the remaining CUs.

    Example on MI350X (256 CUs), before split-K:
      * 250 compact WCA output tiles with a short prefix launches 250 CTAs.
        The remaining 6 CUs simply have no CTA for this kernel; no extra
        scheduler work or inter-CTA coordination happens.
      * 500 compact WCA output tiles launches 256 CTAs. CTA p handles
        tile_idx p, then p + 256. CTAs 0..243 process two output tiles
        each, while CTAs 244..255 process only their first tile because
        p + 256 is outside the 500-tile range.

    If the prefix is long enough and the compact output-tile count is below
    the CU count, ``_get_wca_grid_config`` may promote to split-K. For
    example, 250 output tiles with ``SPLIT_K=2`` becomes 500 logical
    tile-splits, still launched as 256 CTAs. Then tile_idx 0/1 are
    ``(output=0, split=0/1)``, tile_idx 256/257 are
    ``(output=128, split=0/1)``, and the last-arriving split for an
    output tile performs the in-kernel reduction.
    """
    if total_valid_tiles >= 2 * num_CUs:
        return 2 * num_CUs
    if total_valid_tiles >= num_CUs:
        return num_CUs
    return total_valid_tiles


def _select_k_splits(total_output_tiles, num_CUs):
    """Choose SPLIT_K for prefix partitioning across CTAs.

    Goal: fill the GPU when there are fewer output tiles than CUs.
    Each split multiplies the grid by SPLIT_K, so we pick the smallest
    power-of-two that brings CU utilization above ~75%.

    Example on MI350 (num_CUs=256, long-prefix decode batch):
        B=1, pfx=64k, ext=1, qH=32, kvH=8, BLOCK_M=64
        output tiles = B * qH * ceil(ext / BLOCK_M) = 1 * 32 * 1 = 32
        32 < 256, so return the smallest sk in {2, 4, 8} with
        32 * sk >= 256. 32*8 = 256 -> SPLIT_K = 8. Each of the 32
        output tiles is duplicated 8 ways across the prefix KV range,
        giving 32*8 = 256 CTAs, one per CU. The partial outputs are
        later reduced on the winning CTA (see the SPLIT_K epilogue
        inside the kernel).
    """
    if total_output_tiles >= num_CUs:
        return 1
    for sk in (2, 4, 8):
        if total_output_tiles * sk >= num_CUs:
            return sk
    return 8


_splitk_ws_out = None
_splitk_ws_lse = None
_splitk_ws_done = None


def _ensure_splitk_workspace(
    total_splits, total_output_tiles, BLOCK_M, BLOCK_DV, device
):
    """Return (partial_out, partial_lse, tile_done) workspace tensors,
    reusing cached allocations when possible.

    ``partial_out`` / ``partial_lse`` intentionally skip per-call
    zero-init: each split-K CTA writes to its own slot with ``po_mask ==
    r_m_mask``; the winner CTA reads with the same mask and supplies
    ``other=0.0`` / ``other=-inf`` for masked-off lanes. Initial /
    stale memory is therefore never observed, so zeroing these buffers
    would add two GPU launches plus roughly 6-10us of CPU ATen overhead
    per SPLIT_K>1 call. ``tile_done`` must still be zeroed because it is
    the atomic counter that drives the winner selection.
    """
    global _splitk_ws_out, _splitk_ws_lse, _splitk_ws_done
    if (
        _splitk_ws_out is not None
        and _splitk_ws_out.device == device
        and _splitk_ws_out.shape[0] >= total_splits
        and _splitk_ws_out.shape[1] >= BLOCK_M
        and _splitk_ws_out.shape[2] >= BLOCK_DV
    ):
        po = _splitk_ws_out[:total_splits, :BLOCK_M, :BLOCK_DV]
        pl = _splitk_ws_lse[:total_splits, :BLOCK_M]
    else:
        cap = max(total_splits, 2048)
        _splitk_ws_out = torch.empty(
            cap, BLOCK_M, BLOCK_DV, dtype=torch.float32, device=device
        )
        _splitk_ws_lse = torch.empty(cap, BLOCK_M, dtype=torch.float32, device=device)
        po = _splitk_ws_out[:total_splits]
        pl = _splitk_ws_lse[:total_splits]
    if (
        _splitk_ws_done is not None
        and _splitk_ws_done.device == device
        and _splitk_ws_done.shape[0] >= total_output_tiles
    ):
        td = _splitk_ws_done[:total_output_tiles]
    else:
        cap_td = max(total_output_tiles, 2048)
        _splitk_ws_done = torch.empty(cap_td, dtype=torch.int32, device=device)
        td = _splitk_ws_done[:total_output_tiles]
    td.zero_()
    return po, pl, td


def _prepare_wca_masks(
    q_extend,
    batch_size,
    custom_mask,
    mask_indptr,
    window_kv_offsets,
):
    """Normalize optional mask args without placeholder allocations."""
    use_custom_mask = custom_mask is not None
    has_window_offsets = window_kv_offsets is not None
    if not use_custom_mask:
        custom_mask = 0
        mask_indptr = 0
    if not has_window_offsets:
        window_kv_offsets = 0
    return (
        custom_mask,
        mask_indptr,
        window_kv_offsets,
        use_custom_mask,
        has_window_offsets,
    )


_FINALIZE_SKIP = (True,)


class WcaGridConfig(NamedTuple):
    skip: bool
    total_output_tiles: int
    total_valid_tiles: int
    total_programs: int
    split_k: int
    grid: tuple[int]


def _device_cache_index(device) -> int:
    return device.index if hasattr(device, "index") and device.index is not None else 0


@functools.lru_cache(maxsize=2048)
def _get_wca_grid_config(
    batch_size: int,
    head_num: int,
    total_extend_rows: int,
    BLOCK_M: int,
    BLOCK_N: int,
    BLOCK_DV: int,
    device_idx: int,
    num_CUs: int,
    total_prefix_len: int,
    SPLIT_K: int,
) -> WcaGridConfig:
    """Cached scalar WCA grid policy.

    Tensor workspaces are still allocated/sliced per call; this cache only
    stores shape-derived integers that repeat across layers.

    ``total_output_tiles`` is a compact upper bound for the WCA tile
    count, computed from total extension rows plus one tail allowance per
    sequence. It intentionally avoids the full rectangular
    ``batch * ceil(max_ext / BLOCK_M)`` grid; any remaining overestimate
    is rejected by ``_schedule_wca`` when its scan fails to find a real
    sequence tile. ``total_valid_tiles`` is this bound multiplied by
    ``split_k``. When callers pass ``SPLIT_K <= 1`` we may still promote
    to split-K for long-prefix decode-like shapes that underfill the 256
    CUs on MI350X.
    """
    total_output_tiles = (
        (total_extend_rows + batch_size * (BLOCK_M - 1)) // BLOCK_M
    ) * head_num
    if total_output_tiles == 0:
        return WcaGridConfig(True, 0, 0, 0, 1, (0,))

    split_k = max(1, int(SPLIT_K))
    if split_k <= 1:
        split_k = 1
        if total_output_tiles < num_CUs:
            avg_kv_len = total_prefix_len // max(1, batch_size)
            if avg_kv_len >= 4 * BLOCK_N:
                split_k = _select_k_splits(total_output_tiles, num_CUs)

    total_valid_tiles = total_output_tiles * split_k
    total_programs = _select_wca_grid(total_valid_tiles, num_CUs)
    return WcaGridConfig(
        False,
        total_output_tiles,
        total_valid_tiles,
        total_programs,
        split_k,
        (total_programs,),
    )


def _finalize_wca_grid(
    *,
    q_extend,
    batch_size,
    head_num,
    BLOCK_M,
    BLOCK_N,
    BLOCK_DV,
    device,
    total_prefix_len,
    SPLIT_K,
):
    """Shared WCA-grid finalization: compute the tight tile count,
    decide split-K, allocate (or reuse) the workspace.

    Returns a positional tuple::
        (skip, total_valid_tiles, total_programs,
         partial_out, partial_lse, tile_done, SPLIT_K, grid)

    The skip=True branch returns the cached ``_FINALIZE_SKIP``
    singleton ``(True,)`` so the caller's unpack works.
    """
    total_extend_rows = q_extend.shape[0]
    num_CUs = _get_num_CUs(device)
    prefix_len = 0 if total_prefix_len is None else int(total_prefix_len)
    grid_cfg = _get_wca_grid_config(
        batch_size,
        head_num,
        total_extend_rows,
        BLOCK_M,
        BLOCK_N,
        BLOCK_DV,
        _device_cache_index(device),
        num_CUs,
        prefix_len,
        SPLIT_K,
    )
    if grid_cfg.skip:
        return _FINALIZE_SKIP

    if grid_cfg.split_k > 1:
        total_splits = grid_cfg.total_valid_tiles
        partial_out, partial_lse, tile_done = _ensure_splitk_workspace(
            total_splits,
            grid_cfg.total_output_tiles,
            BLOCK_M,
            BLOCK_DV,
            device,
        )
    else:
        partial_out = partial_lse = tile_done = 0
    return (
        False,
        grid_cfg.total_valid_tiles,
        grid_cfg.total_programs,
        partial_out,
        partial_lse,
        tile_done,
        grid_cfg.split_k,
        grid_cfg.grid,
    )


# ===-----------------------------------------------------------------------===#
# Data-centric path heuristic configs
# ===-----------------------------------------------------------------------===#


class PrefixBucket(IntEnum):
    """Average prefix length bucket used by sync-free route heuristics."""

    NONE = 0  # no prefix / prefill from scratch
    TINY = 1  # avg < 512
    MODERATE = 2  # avg < 2048
    LARGE = 3  # avg < 8192
    HUGE = 4  # avg >= 8192


class HeuristicConfig(NamedTuple):
    """Kernel tile config selected before launch.

    ``pad_k`` / ``pad_v`` are currently only non-default for D=256 layouts;
    ``ext_*`` may differ from the prefix values on FP8 paths.
    """

    block_m: int
    block_n: int
    num_warps: int
    num_stages: int
    pad_k: int = 16
    pad_v: int = 16
    ext_block_n: int | None = None
    ext_num_stages: int | None = None

    def with_default_extend(self):
        return self._replace(
            ext_block_n=self.block_n if self.ext_block_n is None else self.ext_block_n,
            ext_num_stages=(
                self.num_stages if self.ext_num_stages is None else self.ext_num_stages
            ),
        )


# FP8 dispatch defaults for the rare D not in {64, 128} fallthrough. The
# main D=64 / D=128 FP8 paths have these baked in explicitly.
_FP8_DEFAULT_BN = 128
_FP8_DEFAULT_NS = 2
_FP8_DEFAULT_EXT_BN = 64
_FP8_DEFAULT_EXT_NS = 3


def _pfx_bucket(total_prefix_len, batch_size) -> PrefixBucket:
    """Bucket average per-sequence prefix length for dispatch."""
    if total_prefix_len is None or total_prefix_len <= 0 or batch_size <= 0:
        return PrefixBucket.NONE
    avg = total_prefix_len // batch_size
    if avg < 512:
        return PrefixBucket.TINY
    if avg < 2048:
        return PrefixBucket.MODERATE
    if avg < 8192:
        return PrefixBucket.LARGE
    return PrefixBucket.HUGE


def _prefix_proxy_from_bucket(prefix_bucket: PrefixBucket) -> int:
    """Representative avg-prefix proxy for sync-free cached config keys."""
    if prefix_bucket <= PrefixBucket.TINY:
        return 0
    if prefix_bucket == PrefixBucket.MODERATE:
        return 1024
    if prefix_bucket == PrefixBucket.LARGE:
        return 2048
    return 8192


def _select_d256_heuristic_config(
    batch_size: int,
    max_len_extend: int,
    min_len_extend: int | None,
    total_prefix_len: int | None,
    total_extend_len: int | None,
    prefix_bucket: PrefixBucket = PrefixBucket.NONE,
) -> HeuristicConfig:
    """D=256 BF16 launch policy (Gemma-class models).

    Returns a full ``HeuristicConfig``. D=256 BF16 uses ``BN=32``; the
    tuple also carries ``pad_k``/``pad_v`` because those layout pads differ
    from the D64/D128 default wiring. NS=1 is never emitted because the
    prefix-pipelined kernel asserts NS>=2 for determinism.
    """
    min_ext = max_len_extend if min_len_extend is None else min_len_extend
    total_ext = (
        batch_size * max_len_extend if total_extend_len is None else total_extend_len
    )

    # Without exact totals, fall back to the bucket proxy so the tree doesn't
    # treat every shape as no-prefix and mis-route decode batches to the
    # extend-dominated config.
    if total_prefix_len is not None and total_prefix_len > 0:
        avg_pfx = total_prefix_len // max(1, batch_size)
        total_tokens = max(1, total_prefix_len + total_ext)
        prefix_frac = total_prefix_len / total_tokens
    else:
        avg_pfx = _prefix_proxy_from_bucket(prefix_bucket)
        total_est = max(1, avg_pfx + max_len_extend)
        prefix_frac = avg_pfx / total_est
    ext_ratio = max_len_extend / max(1, min_ext)

    def cfg(block_m: int, num_warps: int, num_stages: int) -> HeuristicConfig:
        return HeuristicConfig(
            block_m, 32, num_warps, num_stages, 16, 16, 32, num_stages
        )

    if batch_size >= 4 and max_len_extend >= 256:
        return cfg(128, 8, 2)
    if batch_size == 1 and max_len_extend >= 2048:
        return cfg(128, 8, 2)

    if max_len_extend >= 768:
        if batch_size <= 2:
            return cfg(64, 4, 2)
        return cfg(128, 8, 3)

    # Extend-dominated (prefix_frac small): BM=128 only earns its keep at
    # B>=3 with at least one BM row of per-seq work; small-ext decode batches
    # stay on BM=64 to keep pad ratio under control.
    if prefix_frac <= 0.55:
        if batch_size <= 2 or max_len_extend <= 64:
            return cfg(64, 4, 2)
        return cfg(128, 8, 3)

    if batch_size <= 4:
        if max_len_extend <= 128 and avg_pfx >= 2048:
            return cfg(64, 4, 4)
        return cfg(64, 4, 2)

    if batch_size >= 8 and max_len_extend >= 128 and avg_pfx >= 2048:
        return cfg(128, 8, 3)

    if max_len_extend <= 64 and avg_pfx >= 2048:
        return cfg(64, 4, 2)

    if ext_ratio <= 2.5:
        return cfg(64, 4, 2)

    return cfg(64, 8, 4)


def _select_d64_bf16_data_centric_config(
    batch_size,
    max_len_extend,
    pfx_bucket,
    head_num,
):
    """Pick the normal D=64 BF16 data-centric config before refinements."""
    total_ext = batch_size * max_len_extend
    if (
        batch_size >= 8
        and max_len_extend <= 128
        and pfx_bucket >= PrefixBucket.MODERATE
    ):
        return HeuristicConfig(128, 128, 4, 4).with_default_extend()
    if batch_size >= 16 and max_len_extend <= 32:
        return HeuristicConfig(64, 64, 4, 4).with_default_extend()
    if batch_size >= 16:
        if max_len_extend >= 512:
            return HeuristicConfig(256, 64, 8, 2).with_default_extend()
        return HeuristicConfig(64, 64, 4, 4).with_default_extend()
    if batch_size >= 4:
        if total_ext >= 2048 or max_len_extend >= 512:
            return HeuristicConfig(256, 64, 8, 2).with_default_extend()
        if batch_size <= 7:
            return HeuristicConfig(128, 64, 8, 4).with_default_extend()
        return HeuristicConfig(64, 64, 4, 4).with_default_extend()

    # B<=3: the BM crossover depends on H_q because lower H already
    # under-subscribes the wave grid.
    h = head_num or 0
    if batch_size == 1 and h >= 64 and max_len_extend >= 1600:
        return HeuristicConfig(256, 64, 8, 2).with_default_extend()
    if batch_size == 1 and h == 32 and max_len_extend >= 1500:
        return HeuristicConfig(256, 64, 8, 2).with_default_extend()
    if max_len_extend >= 2048:
        return HeuristicConfig(256, 64, 8, 2).with_default_extend()
    return HeuristicConfig(128, 64, 8, 4).with_default_extend()


def _select_d128_bf16_data_centric_config(
    batch_size,
    max_len_extend,
    pfx_bucket,
    head_num,
):
    """Pick the normal D=128 BF16 data-centric config before refinements."""
    total_ext = batch_size * max_len_extend
    if batch_size >= 16 and max_len_extend <= 16:
        return HeuristicConfig(16, 64, 4, 2).with_default_extend()
    if batch_size >= 16 and max_len_extend <= 64:
        return HeuristicConfig(64, 64, 4, 2).with_default_extend()
    if pfx_bucket >= PrefixBucket.LARGE and max_len_extend >= 4096:
        return HeuristicConfig(128, 64, 8, 2).with_default_extend()
    if pfx_bucket >= PrefixBucket.MODERATE and max_len_extend >= 2048:
        return HeuristicConfig(128, 64, 8, 2).with_default_extend()
    if batch_size == 1:
        h = head_num or 0
        if h >= 64 and max_len_extend >= 2000:
            return HeuristicConfig(128, 64, 8, 2).with_default_extend()
        if h >= 64 and max_len_extend >= 1024:
            return HeuristicConfig(128, 64, 8, 4).with_default_extend()
        return HeuristicConfig(64, 64, 4, 2).with_default_extend()
    if batch_size <= 4:
        return HeuristicConfig(64, 64, 4, 2).with_default_extend()
    if total_ext >= 32768:
        return HeuristicConfig(128, 64, 8, 2).with_default_extend()
    return HeuristicConfig(64, 64, 4, 2).with_default_extend()


def _select_bf16_data_centric_base_config(
    Lq,
    batch_size,
    max_len_extend,
    prefix_bucket,
    head_num,
    *,
    min_len_extend=None,
    total_prefix_len=None,
    total_extend_len=None,
) -> HeuristicConfig:
    """Single BF16 data-centric base selector for fast and full paths."""
    if Lq == 256:
        return _select_d256_heuristic_config(
            batch_size,
            max_len_extend,
            min_len_extend,
            total_prefix_len,
            total_extend_len,
            prefix_bucket,
        )
    if Lq == 64:
        return _select_d64_bf16_data_centric_config(
            batch_size,
            max_len_extend,
            prefix_bucket,
            head_num,
        )
    return _select_d128_bf16_data_centric_config(
        batch_size,
        max_len_extend,
        prefix_bucket,
        head_num,
    )


def _apply_data_centric_policy_refinements(
    cfg: HeuristicConfig,
    Lq,
    batch_size,
    max_len_extend,
    prefix_bucket,
    is_fp8,
    sliding_window_size,
) -> HeuristicConfig:
    """Ordered refinement pipeline for data-centric launch configs.

    Order matters: start from the normal BF16 base policy, then apply targeted
    D64 BF16, FP8, SWA, and correctness refinements.
    """
    if (
        not is_fp8
        and Lq == 64
        and sliding_window_size <= 0
        and batch_size >= 16
        and max_len_extend == 256
    ):
        # Large-B uniform full-attn at ext==256: lift BM so the grid fully
        # covers the available CUs.
        cfg = HeuristicConfig(256, 64, 8, 2).with_default_extend()

    if is_fp8:
        if Lq == 128:
            cfg = HeuristicConfig(128, 128, 8, 2, 16, 16, 128, 2)
            if batch_size == 1 and max_len_extend <= 256:
                cfg = cfg._replace(block_m=64)
            elif (
                batch_size >= 32
                and max_len_extend <= 8
                and prefix_bucket <= PrefixBucket.MODERATE
            ):
                # Spec-verify / draft-extend continuous batches: BM=64 NW=4
                # avoids pad-heavy Q tiles.
                cfg = cfg._replace(block_m=64, num_warps=4)
            elif batch_size >= 16 and max_len_extend <= 64:
                # Decode-continuation: max_ext < BM leaves Q tiles mostly
                # padding at BM=128. Drop to the serial kernel.
                cfg = cfg._replace(
                    block_m=64, num_warps=4, num_stages=1, ext_num_stages=1
                )
            elif (
                batch_size >= 8
                and max_len_extend <= 64
                and prefix_bucket >= PrefixBucket.LARGE
            ):
                # Long prefix at moderate batch: keep NS=2 to pipeline the
                # prefix phase, but above B=15 the DMA cost outpaces per-tile
                # work so go fully serial.
                cfg = cfg._replace(block_m=64, num_warps=4)
                if batch_size > 15:
                    cfg = cfg._replace(num_stages=1, ext_num_stages=1)
        elif Lq == 64:
            cfg = HeuristicConfig(128, 128, 4, 1, 16, 16, 128, 1)
            # NS stays at 1: the D=64 FP8 data-centric body is NW=4 only and
            # the kernel asserts NS=1 for BLOCK_DMODEL<128. Drop BM on
            # pad-heavy tiles.
            if (
                max_len_extend <= 8
                or (batch_size >= 16 and max_len_extend <= 128)
                or (
                    batch_size >= 8
                    and max_len_extend <= 64
                    and prefix_bucket >= PrefixBucket.LARGE
                )
            ):
                cfg = cfg._replace(block_m=64)
        else:
            # Lq not in {64, 128} is filtered upstream for FP8; this fallback
            # only fires if an internal caller bypasses the public guards.
            cfg = cfg._replace(
                block_n=32 if Lq >= 256 else _FP8_DEFAULT_BN,
                num_warps=min(cfg.num_warps, 4) if Lq < 128 else cfg.num_warps,
                num_stages=_FP8_DEFAULT_NS,
                ext_block_n=_FP8_DEFAULT_EXT_BN,
                ext_num_stages=_FP8_DEFAULT_EXT_NS,
            )

    if (
        not is_fp8
        and Lq == 64
        and sliding_window_size > 0
        and sliding_window_size < max_len_extend
    ):
        # The per-tile key band is ~(BM + sw) wide regardless of tile count,
        # so the BM=256 plain-causal winner over-fetches under SWA.
        total_ext = batch_size * max_len_extend
        if max_len_extend >= 1024 and total_ext >= 2048:
            cfg = HeuristicConfig(64, 64, 2, 2).with_default_extend()
        elif max_len_extend > 16 * sliding_window_size:
            cfg = cfg._replace(
                block_m=min(cfg.block_m, 128),
                num_warps=4,
                num_stages=2,
                ext_num_stages=2,
            )
        elif cfg.block_m > 128:
            cfg = cfg._replace(block_m=128, num_stages=4, ext_num_stages=4)

    block_dmodel = _resolve_qk_split_dims(Lq)
    # Routing invariant: the 4w sw-pipeline kernel asserts D>=128. D<128
    # NW<=4 must emit NS=1 so the serial kernel takes the launch instead.
    if cfg.num_warps <= 4 and block_dmodel < 128:
        cfg = cfg._replace(num_stages=1, ext_num_stages=1)

    # Correctness clamps shared by fastpath and full-dispatch data-centric:
    #  * D=64 NW=8 BM=256 is nondeterministic at NS!=2.
    #  * D>=256 NW=8 NS=1 races on the DMA ring.
    if (
        block_dmodel == 64
        and cfg.num_warps == 8
        and cfg.block_m == 256
        and cfg.num_stages != 2
    ):
        cfg = cfg._replace(num_stages=2, ext_num_stages=2)
    if block_dmodel >= 256 and cfg.num_warps == 8 and cfg.num_stages == 1:
        cfg = cfg._replace(num_stages=2, ext_num_stages=2)

    return cfg.with_default_extend()


def _select_data_centric_heuristic_config(
    Lq,
    batch_size,
    max_len_extend,
    prefix_bucket,
    is_fp8,
    sliding_window_size=-1,
    head_num=None,
    *,
    min_len_extend=None,
    total_prefix_len=None,
    total_extend_len=None,
) -> HeuristicConfig:
    """Select the data-centric tile policy for both fast and full paths.

    The normal BF16 base policy is shared. FP8, SWA, and correctness-specific
    refinements are applied afterward so the route stays easy to audit.
    """
    cfg = _select_bf16_data_centric_base_config(
        Lq,
        batch_size,
        max_len_extend,
        PrefixBucket(prefix_bucket),
        head_num,
        min_len_extend=min_len_extend,
        total_prefix_len=total_prefix_len,
        total_extend_len=total_extend_len,
    )
    return _apply_data_centric_policy_refinements(
        cfg,
        Lq,
        batch_size,
        max_len_extend,
        PrefixBucket(prefix_bucket),
        is_fp8,
        sliding_window_size,
    )


@functools.lru_cache(maxsize=2048)
def _get_data_centric_heuristic_config(
    Lq,
    batch_size,
    max_len_extend,
    pfx_bucket,
    is_fp8,
    sliding_window_size=-1,
    head_num=None,
):
    """Cached sync-free data-centric heuristic config.

    The cache key remains the serving-safe tuple available without a GPU sync:
    ``(Lq, batch_size, max_len_extend, pfx_bucket, is_fp8,
    sliding_window_size, head_num)``. Full dispatch can call
    ``_select_data_centric_heuristic_config`` directly with exact length hints.
    On cache hits, Python returns the stored ``HeuristicConfig`` before this
    function body runs, so the selector tree is skipped entirely.
    """
    return _select_data_centric_heuristic_config(
        Lq,
        batch_size,
        max_len_extend,
        PrefixBucket(pfx_bucket),
        is_fp8,
        sliding_window_size=sliding_window_size,
        head_num=head_num,
        min_len_extend=max_len_extend,
        total_prefix_len=None,
        total_extend_len=batch_size * max_len_extend,
    )


@functools.lru_cache(maxsize=2048)
def _get_exact_data_centric_heuristic_config(
    Lq,
    batch_size,
    max_len_extend,
    min_len_extend,
    total_prefix_len,
    total_extend_len,
    pfx_bucket,
    is_fp8,
    sliding_window_size=-1,
    head_num=None,
    use_custom_mask=False,
):
    """Cached exact-hint data-centric config for full/custom-mask dispatch.

    These LRU caches store only Python scalar launch decisions. They never own
    tensor memory and are safe to reuse across layers because all dynamic data
    still flows through the real kernel arguments.

    ``use_custom_mask`` is currently a future-proof key bit: the tile selector
    does not inspect mask data, but mask-aware policies can diverge later
    without invalidating older cache entries.
    """
    return _select_data_centric_heuristic_config(
        Lq,
        batch_size,
        max_len_extend,
        PrefixBucket(pfx_bucket),
        is_fp8,
        sliding_window_size=sliding_window_size,
        head_num=head_num,
        min_len_extend=min_len_extend,
        total_prefix_len=total_prefix_len,
        total_extend_len=total_extend_len,
    )


# ===-----------------------------------------------------------------------===#
# Unified launch helpers
# ===-----------------------------------------------------------------------===#


def _is_serial_schedule(num_stages, num_warps, block_dmodel, kv_is_fp8):
    """Host-side predicate for the NS=1 serial kernel wrapper."""
    return (
        num_stages == 1
        and num_warps in (2, 4)
        and block_dmodel in (64, 128, 256)
        and not (kv_is_fp8 and block_dmodel == 256)
    )


def _launch_attention_grid(
    q_extend,
    k_extend,
    v_extend,
    o_extend,
    k_buffer,
    v_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    mask,
    mask_indptr,
    window_kv_offsets,
    *,
    use_custom_mask,
    is_causal,
    max_len_extend,
    k_scale,
    v_scale,
    sm_scale,
    logit_cap,
    sliding_window_size,
    sinks,
    xai_temperature_len,
    Lq,
    head_num,
    batch_size,
    kv_is_fp8,
    has_window_offsets,
    grid,
    block_m,
    block_n,
    block_dmodel,
    num_stages,
    num_warps,
    ext_block_n=None,
    ext_num_stages=None,
    enable_prefix_unmasked=True,
    is_wca=False,
    split_k=1,
    total_valid_tiles=0,
    total_programs=0,
    partial_out=0,
    partial_lse=0,
    tile_done=0,
    actual_batch_size=0,
    total_prefix_len_hint=None,
    total_extend_len_hint=None,
    min_len_extend_hint=None,
):
    """Single host-side launch body for all extend-attention grids.

    Route helpers above this function decide *what* to run: serial vs 4w/8w,
    WCA vs data-centric, split-K workspace, FP8 extend block shape. This body
    owns the shared mechanics: stride extraction, schedule-wrapper selection,
    XCD metadata, and the explicit ``kernel[grid](...)`` call.
    """
    sm_scale_local = (sm_scale if sm_scale is not None else Lq**-0.5) * k_scale
    kv_group_num = head_num // k_extend.shape[1]
    use_serial_kernel = _is_serial_schedule(
        num_stages,
        num_warps,
        block_dmodel,
        kv_is_fp8,
    )
    xcd_config = _select_xcd_remap_config(
        is_wca=is_wca,
        is_serial=use_serial_kernel,
        split_k=split_k,
        total_valid_tiles=total_valid_tiles,
        block_dmodel=block_dmodel,
        kv_is_fp8=kv_is_fp8,
        head_num=head_num,
        batch_size=batch_size,
        total_extend_rows=q_extend.shape[0],
        sliding_window_size=sliding_window_size,
        logit_cap=logit_cap,
    )
    q_s0, q_s1 = q_extend.stride(0), q_extend.stride(1)
    k_s0, k_s1 = k_extend.stride(0), k_extend.stride(1)
    v_s0, v_s1 = v_extend.stride(0), v_extend.stride(1)
    o_s0, o_s1 = o_extend.stride(0), o_extend.stride(1)
    kb_s0, kb_s1 = k_buffer.stride(0), k_buffer.stride(1)
    vb_s0, vb_s1 = v_buffer.stride(0), v_buffer.stride(1)

    if use_serial_kernel:
        serial_extra = {"IS_FP8": True} if kv_is_fp8 else {}
        gluon_extend_attn_serial_fwd[grid](
            q_extend,
            k_extend,
            v_extend,
            o_extend,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            mask,
            mask_indptr,
            window_kv_offsets,
            sinks,
            sm_scale_local,
            kv_group_num,
            q_s0,
            q_s1,
            k_s0,
            k_s1,
            v_s0,
            v_s1,
            o_s0,
            o_s1,
            kb_s0,
            kb_s1,
            vb_s0,
            vb_s1,
            IS_CAUSAL=is_causal,
            USE_CUSTOM_MASK=use_custom_mask,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_DMODEL=block_dmodel,
            Sinks_present=sinks is not None,
            LOGIT_CAP=logit_cap,
            XAI_TEMPERATURE_LEN=xai_temperature_len,
            SLIDING_WINDOW_SIZE=sliding_window_size,
            HAS_WINDOW_OFFSETS=has_window_offsets,
            v_scale=v_scale,
            **serial_extra,
            num_warps=num_warps,
            num_stages=1,
            waves_per_eu=2,
            matrix_instr_nonkdim=32,
        )
        return

    schedule_kernel = (
        gluon_extend_attn_fwd_8w if num_warps >= 8 else gluon_extend_attn_fwd_4w
    )
    kernel_extra = {}
    if kv_is_fp8:
        kernel_extra["IS_FP8"] = True
        kernel_extra["EXT_BLOCK_N"] = (
            ext_block_n if ext_block_n is not None else block_n
        )
        kernel_extra["EXT_NUM_STAGES"] = (
            ext_num_stages if ext_num_stages is not None else num_stages
        )

    schedule_kernel[grid](
        q_extend,
        k_extend,
        v_extend,
        o_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        mask,
        mask_indptr,
        window_kv_offsets,
        sm_scale_local,
        kv_group_num,
        q_s0,
        q_s1,
        k_s0,
        k_s1,
        v_s0,
        v_s1,
        o_s0,
        o_s1,
        kb_s0,
        kb_s1,
        vb_s0,
        vb_s1,
        IS_CAUSAL=is_causal,
        USE_CUSTOM_MASK=use_custom_mask,
        ENABLE_PREFIX_UNMASKED=enable_prefix_unmasked,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_DMODEL=block_dmodel,
        NUM_STAGES=num_stages,
        **kernel_extra,
        Sinks=sinks,
        HAS_SINK=sinks is not None,
        LOGIT_CAP=logit_cap,
        XAI_TEMPERATURE_LEN=xai_temperature_len,
        SLIDING_WINDOW_SIZE=sliding_window_size,
        HAS_WINDOW_OFFSETS=has_window_offsets,
        v_scale=1.0 if split_k > 1 else v_scale,
        num_heads=head_num,
        total_valid_tiles=total_valid_tiles,
        total_programs=total_programs,
        partial_out=partial_out,
        partial_lse=partial_lse,
        tile_done=tile_done,
        actual_batch_size=actual_batch_size,
        IS_WCA=is_wca,
        SPLIT_K=split_k,
        XCD_REMAP=xcd_config.enabled,
        NUM_XCDS=xcd_config.num_xcds,
        XCD_CHUNK=xcd_config.chunk,
        XCD_MODE=int(xcd_config.mode),
        num_warps=num_warps,
        num_stages=1,
        waves_per_eu=2,
        matrix_instr_nonkdim=32,
    )


def _select_wca_heuristic_config(
    *,
    Lq,
    block_dmodel,
    batch_size,
    max_len_extend,
    min_len_extend,
    total_prefix_len,
    total_extend_rows,
    kv_is_fp8,
    split_k,
    forced_config=None,
) -> HeuristicConfig:
    """Pick the WCA/split-K tile config before workspace finalization."""
    if forced_config is not None:
        return forced_config.with_default_extend()

    if kv_is_fp8:
        block_n = 128 if Lq == 128 else (32 if block_dmodel >= 256 else 64)
        if split_k > 1:
            return HeuristicConfig(
                128,
                block_n,
                8,
                _FP8_DEFAULT_NS,
                16,
                16,
                _FP8_DEFAULT_EXT_BN,
                _FP8_DEFAULT_EXT_NS,
            )
        if block_dmodel >= 256:
            block_m, num_warps = 64, 4
        elif max_len_extend <= 128:
            block_m, num_warps = 128, 8
        elif batch_size <= 4:
            block_m, num_warps = 128, 8
        elif min_len_extend >= 64 and max_len_extend >= 256:
            block_m, num_warps = 256, 8
        else:
            block_m, num_warps = 128, 8
        return HeuristicConfig(
            block_m,
            block_n,
            num_warps,
            _FP8_DEFAULT_NS,
            16,
            16,
            _FP8_DEFAULT_EXT_BN,
            _FP8_DEFAULT_EXT_NS,
        )

    block_n = 32 if block_dmodel >= 256 else 64
    total_pfx_est = max(0, int(total_prefix_len or 0))
    het_ratio = max_len_extend / max(min_len_extend, 1)
    pfx_dominated_big_b = (
        batch_size >= 8
        and total_pfx_est >= batch_size * 1024
        and 32 <= max_len_extend <= 512
    )
    lq128_ext_dominated = Lq == 128 and total_pfx_est < 4 * total_extend_rows
    use_small_tile = not pfx_dominated_big_b and (
        max_len_extend <= 128
        or (max_len_extend <= 256 and het_ratio >= 2.0)
        or (min_len_extend < 64 and max_len_extend <= 512 and batch_size <= 4)
        or lq128_ext_dominated
    )

    if block_dmodel >= 256:
        block_m, num_warps = 64, 4
    elif use_small_tile:
        block_m, num_warps = 64, 4
    elif batch_size <= 4:
        block_m, num_warps = 128, 8
    else:
        # BM=256 hits an LLVM iota_range assertion in the WCA kernel on some
        # shapes; BM=128 is the largest safe M-tile.
        block_m, num_warps = 128, 8

    if block_m == 64 and num_warps == 4:
        num_stages = 2
    elif block_dmodel >= 256:
        num_stages = 2
    else:
        num_stages = 4
    if block_m == 128 and num_warps == 8 and num_stages == 2:
        num_stages = 3

    return HeuristicConfig(
        block_m, block_n, num_warps, num_stages
    ).with_default_extend()


def _heuristic_config_cache_key(cfg: HeuristicConfig | None):
    if cfg is None:
        return None
    cfg = cfg.with_default_extend()
    return (
        cfg.block_m,
        cfg.block_n,
        cfg.num_warps,
        cfg.num_stages,
        cfg.pad_k,
        cfg.pad_v,
        cfg.ext_block_n,
        cfg.ext_num_stages,
    )


@functools.lru_cache(maxsize=2048)
def _get_wca_heuristic_config(
    Lq,
    block_dmodel,
    batch_size,
    max_len_extend,
    min_len_extend,
    total_prefix_len,
    total_extend_rows,
    kv_is_fp8,
    split_k,
    forced_config_key,
) -> HeuristicConfig:
    """Cached WCA/split-K tile config keyed on scalar launch facts."""
    forced_config = (
        HeuristicConfig(*forced_config_key).with_default_extend()
        if forced_config_key is not None
        else None
    )
    return _select_wca_heuristic_config(
        Lq=Lq,
        block_dmodel=block_dmodel,
        batch_size=batch_size,
        max_len_extend=max_len_extend,
        min_len_extend=min_len_extend,
        total_prefix_len=total_prefix_len,
        total_extend_rows=total_extend_rows,
        kv_is_fp8=kv_is_fp8,
        split_k=split_k,
        forced_config=forced_config,
    )


def _launch_wca(
    q_extend,
    k_extend,
    v_extend,
    o_extend,
    k_buffer,
    v_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    custom_mask,
    is_causal,
    mask_indptr,
    max_len_extend,
    k_scale,
    v_scale,
    sm_scale,
    logit_cap,
    sliding_window_size,
    sinks,
    window_kv_offsets,
    xai_temperature_len,
    kv_is_fp8,
    total_prefix_len,
    min_len_extend,
    *,
    split_k=1,
    forced_config=None,
    enable_prefix_unmasked=True,
):
    """Launch WCA or split-K through the unified 4w/8w launch body.

    ``split_k == 1`` still selects ``IS_WCA=True``: the kernel gets a 1D
    compact tile space and the in-kernel strided tile walk. Only the
    split-K prefix partition, partial workspace, and final reduction are
    disabled.
    """
    q_shape = q_extend.shape
    Lq = q_shape[-1]
    block_dmodel = _resolve_qk_split_dims(Lq)
    assert block_dmodel >= 128, (
        f"_launch_wca: D<128 has no pipelined home; got {block_dmodel}. "
        "Route D<128 through data-centric."
    )

    batch_size = qo_indptr.shape[0] - 1
    head_num = q_shape[1]
    if min_len_extend is None:
        min_len_extend = max_len_extend
    total_prefix_len_policy = (
        int(total_prefix_len) if total_prefix_len is not None else kv_indices.numel()
    )

    mask, mask_indptr, window_kv_offsets, use_custom_mask, has_window_offsets = (
        _prepare_wca_masks(
            q_extend,
            batch_size,
            custom_mask,
            mask_indptr,
            window_kv_offsets,
        )
    )
    cfg = _get_wca_heuristic_config(
        Lq,
        block_dmodel,
        batch_size,
        max_len_extend,
        min_len_extend,
        total_prefix_len_policy,
        q_shape[0],
        kv_is_fp8,
        split_k,
        _heuristic_config_cache_key(forced_config),
    )

    grid_state = _finalize_wca_grid(
        q_extend=q_extend,
        batch_size=batch_size,
        head_num=head_num,
        BLOCK_M=cfg.block_m,
        BLOCK_N=cfg.block_n,
        BLOCK_DV=block_dmodel,
        device=q_extend.device,
        total_prefix_len=total_prefix_len_policy,
        SPLIT_K=split_k,
    )
    if grid_state[0]:
        return
    (
        _skip,
        total_valid_tiles,
        total_programs,
        partial_out,
        partial_lse,
        tile_done,
        split_k,
        grid,
    ) = grid_state

    _launch_attention_grid(
        q_extend,
        k_extend,
        v_extend,
        o_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        mask,
        mask_indptr,
        window_kv_offsets,
        use_custom_mask=use_custom_mask,
        is_causal=is_causal,
        max_len_extend=max_len_extend,
        k_scale=k_scale,
        v_scale=v_scale,
        sm_scale=sm_scale,
        logit_cap=logit_cap,
        sliding_window_size=sliding_window_size,
        sinks=sinks,
        xai_temperature_len=xai_temperature_len,
        Lq=Lq,
        head_num=head_num,
        batch_size=batch_size,
        kv_is_fp8=kv_is_fp8,
        has_window_offsets=has_window_offsets,
        grid=grid,
        block_m=cfg.block_m,
        block_n=cfg.block_n,
        block_dmodel=block_dmodel,
        num_stages=cfg.num_stages,
        num_warps=cfg.num_warps,
        ext_block_n=cfg.ext_block_n,
        ext_num_stages=cfg.ext_num_stages,
        enable_prefix_unmasked=enable_prefix_unmasked,
        is_wca=True,
        split_k=split_k,
        total_valid_tiles=total_valid_tiles,
        total_programs=total_programs,
        partial_out=partial_out,
        partial_lse=partial_lse,
        tile_done=tile_done,
        actual_batch_size=batch_size,
        total_prefix_len_hint=total_prefix_len_policy,
        min_len_extend_hint=min_len_extend,
    )


def _launch_splitk(
    q_extend,
    k_extend,
    v_extend,
    o_extend,
    k_buffer,
    v_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    custom_mask,
    mask_indptr,
    window_kv_offsets,
    sm_scale,
    k_scale,
    v_scale,
    logit_cap,
    Lq,
    is_causal,
    max_len_extend,
    min_len_extend,
    sinks,
    xai_temperature_len,
    sliding_window_size,
    BLOCK_M,
    BLOCK_N,
    num_warps,
    NUM_STAGES,
    total_prefix_len=None,
    kv_is_fp8=False,
):
    """Route through WCA, adding split-K only when it improves CU fill."""
    head_num = q_extend.shape[1]
    batch_size = qo_indptr.shape[0] - 1
    total_output_tiles = (
        batch_size * head_num * ((max_len_extend + BLOCK_M - 1) // BLOCK_M)
    )
    if total_output_tiles == 0:
        return

    num_CUs = _get_num_CUs(q_extend.device)
    total_prefix_len_policy = (
        int(total_prefix_len) if total_prefix_len is not None else kv_indices.numel()
    )
    avg_kv_len = total_prefix_len_policy // max(1, batch_size)
    if not (total_output_tiles < num_CUs and avg_kv_len >= 4 * BLOCK_N):
        _launch_wca(
            q_extend,
            k_extend,
            v_extend,
            o_extend,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            custom_mask,
            is_causal,
            mask_indptr,
            max_len_extend,
            k_scale,
            v_scale,
            sm_scale,
            logit_cap,
            sliding_window_size,
            sinks,
            window_kv_offsets,
            xai_temperature_len,
            kv_is_fp8,
            total_prefix_len_policy,
            min_len_extend,
        )
        return

    split_k = _select_k_splits(total_output_tiles, num_CUs)
    block_dmodel = _resolve_qk_split_dims(Lq)
    if kv_is_fp8:
        forced = HeuristicConfig(
            128,
            128 if Lq <= 128 else (32 if block_dmodel >= 256 else 64),
            8,
            _FP8_DEFAULT_NS,
            16,
            16,
            _FP8_DEFAULT_EXT_BN,
            _FP8_DEFAULT_EXT_NS,
        )
    elif block_dmodel < 256:
        forced = HeuristicConfig(128, BLOCK_N, 8, 4).with_default_extend()
    else:
        forced = HeuristicConfig(
            BLOCK_M, BLOCK_N, num_warps, NUM_STAGES
        ).with_default_extend()

    _launch_wca(
        q_extend,
        k_extend,
        v_extend,
        o_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        custom_mask,
        is_causal,
        mask_indptr,
        max_len_extend,
        k_scale,
        v_scale,
        sm_scale,
        logit_cap,
        sliding_window_size,
        sinks,
        window_kv_offsets,
        xai_temperature_len,
        kv_is_fp8,
        total_prefix_len_policy,
        min_len_extend,
        split_k=split_k,
        forced_config=forced,
        enable_prefix_unmasked=is_causal,
    )


def _launch_data_centric_grid(
    q_extend,
    k_extend,
    v_extend,
    o_extend,
    k_buffer,
    v_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    is_causal,
    max_len_extend,
    k_scale,
    v_scale,
    sm_scale,
    logit_cap,
    sliding_window_size,
    sinks,
    xai_temperature_len,
    Lq,
    head_num,
    batch_size,
    kv_is_fp8,
    *,
    BLOCK_M,
    BLOCK_N,
    BLOCK_DMODEL,
    NUM_STAGES,
    num_warps,
    custom_mask_arg=None,
    mask_indptr_arg=None,
    window_kv_offsets_arg=None,
    ext_block_n=None,
    ext_num_stages=None,
    total_prefix_len_hint=None,
    total_extend_len_hint=None,
    min_len_extend_hint=None,
):
    """Normalize masks and launch the 3D data-centric grid.

    Data-centric CTAs do not walk a compact tile list. The grid is
    rectangular ``(batch, heads, ceil(max_ext / BLOCK_M))``; invalid tail
    tiles are masked inside the kernel.
    """
    use_custom_mask = custom_mask_arg is not None
    if use_custom_mask:
        mask = custom_mask_arg
        mask_indptr = mask_indptr_arg
    else:
        mask = 0
        mask_indptr = 0
    has_window_offsets = window_kv_offsets_arg is not None
    window_kv_offsets = window_kv_offsets_arg if has_window_offsets else 0

    use_serial = _is_serial_schedule(NUM_STAGES, num_warps, BLOCK_DMODEL, kv_is_fp8)
    grid = (batch_size, head_num, (max_len_extend + BLOCK_M - 1) // BLOCK_M)
    data_xcd_candidate = (
        not use_serial
        and int(BLOCK_DMODEL) == 64
        and not kv_is_fp8
        and int(batch_size) >= 8
    )
    data_total_programs = grid[0] * grid[1] * grid[2] if data_xcd_candidate else 0
    _launch_attention_grid(
        q_extend,
        k_extend,
        v_extend,
        o_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        mask,
        mask_indptr,
        window_kv_offsets,
        use_custom_mask=use_custom_mask,
        is_causal=is_causal,
        max_len_extend=max_len_extend,
        k_scale=k_scale,
        v_scale=v_scale,
        sm_scale=sm_scale,
        logit_cap=logit_cap,
        sliding_window_size=sliding_window_size,
        sinks=sinks,
        xai_temperature_len=xai_temperature_len,
        Lq=Lq,
        head_num=head_num,
        batch_size=batch_size,
        kv_is_fp8=kv_is_fp8,
        has_window_offsets=has_window_offsets,
        grid=grid,
        block_m=BLOCK_M,
        block_n=BLOCK_N,
        block_dmodel=BLOCK_DMODEL,
        num_stages=NUM_STAGES,
        num_warps=num_warps,
        ext_block_n=ext_block_n,
        ext_num_stages=ext_num_stages,
        enable_prefix_unmasked=is_causal,
        total_valid_tiles=data_total_programs,
        actual_batch_size=batch_size if data_xcd_candidate else 0,
        total_prefix_len_hint=total_prefix_len_hint,
        total_extend_len_hint=total_extend_len_hint,
        min_len_extend_hint=min_len_extend_hint,
    )


def gluon_extend_attention_fwd(
    q_extend,
    k_extend,
    v_extend,
    o_extend,
    k_buffer,
    v_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    custom_mask,
    is_causal,
    mask_indptr,
    max_len_extend,
    k_scale=1.0,
    v_scale=1.0,
    sm_scale=None,
    logit_cap=0.0,
    # Accepted for API parity with the Triton backend; Gluon always
    # skips the prefix custom-mask path.
    skip_prefix_custom_mask=True,  # noqa: ARG001
    sliding_window_size=-1,
    sinks=None,
    window_kv_offsets=None,
    xai_temperature_len=-1,
    min_len_extend=None,
    total_prefix_len=None,
    total_extend_len=None,
):
    _q_shape = q_extend.shape
    Lq = _q_shape[-1]
    Lv = v_extend.shape[-1]
    _kb_dtype = k_buffer.dtype
    _kv_is_fp8 = _kb_dtype is torch.float8_e4m3fn or _kb_dtype is torch.float8_e4m3fnuz
    if _kv_is_fp8:
        # gfx950 MFMA expects OCP FP8 (bias=7); FNUZ (bias=8) silently
        # doubles values on this hardware.
        if _kb_dtype is torch.float8_e4m3fnuz:
            raise ValueError(
                "Gluon FP8 extend on gfx950 requires OCP FP8 (torch.float8_e4m3fn). "
                "Got torch.float8_e4m3fnuz which produces 2-4x numerical errors "
                "because MFMA hardware on gfx950 uses bias=7 but FNUZ uses bias=8. "
                "Re-quantize KV buffers with torch.float8_e4m3fn, or use "
                "SGLang's fp8_kernel.fp8_dtype which auto-selects the right format."
            )
        # FP8 KV + custom_mask on D<=128 lands on the 8-warp pipelined
        # fallback which has IMA at NS>=2 and nondeterminism at NS=1.
        if custom_mask is not None and Lq <= 128:
            raise NotImplementedError(
                "Gluon FP8 KV + custom_mask is not supported on D<=128. Use "
                "BF16 KV for spec-decode verify."
            )
        # FP8 KV + D=256 fails Triton IR lowering (MFMA_F8 emits
        # unrealized_conversion_cast the backend can't materialize).
        if Lq == 256:
            raise NotImplementedError(
                "Gluon FP8 KV extend is not supported at head-dim 256. Use "
                "BF16 KV (Gemma and other D=256 models ship BF16)."
            )
    if Lq != Lv:
        raise ValueError(
            f"Gluon extend attention only supports symmetric heads (Lq == Lv), "
            f"got Lq={Lq}, Lv={Lv}. Use mla_prefill/ for mixed-dim DeepSeek MLA."
        )
    batch_size = qo_indptr.shape[0] - 1
    head_num = _q_shape[1]

    # Route order is intentionally explicit: cheap BF16 serial preflight,
    # sync-free WCA eligibility, cached data-centric config, then the full
    # dispatch fallback for custom masks / heterogeneous cases. Here
    # "fastpath" means the host can make a route/config choice from cheap
    # shape metadata (B, max extension length, total prefix hint/bucket) without
    # materializing exact per-sequence reductions or mask-specific state. It is
    # not a separate kernel family; it just avoids slow Python/GPU-sync work
    # before launching the same serial, data-centric, WCA, or split-K kernels.
    # WCA and split-K stay separate because they allocate workspace and launch a
    # 1D work-centric grid rather than the 3D data-centric grid.
    _total_pfx_est_pre = (
        total_prefix_len if total_prefix_len is not None else kv_indices.numel()
    )

    # BF16 serial-kernel preflight: cheap (Lq, B, ext, pfx) gate before
    # the heavier _is_ragged setup. The serial kernel itself handles
    # SWA and custom masks (see ``gluon_extend_attn_serial_fwd`` in
    # this file); FP8 KV is excluded here.
    if not _kv_is_fp8 and _should_route_serial_data_centric_bf16(
        Lq, batch_size, max_len_extend, _total_pfx_est_pre
    ):
        _BLOCK_DMODEL = _resolve_qk_split_dims(Lq)
        _launch_data_centric_grid(
            q_extend,
            k_extend,
            v_extend,
            o_extend,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            is_causal,
            max_len_extend,
            k_scale,
            v_scale,
            sm_scale,
            logit_cap,
            sliding_window_size,
            sinks,
            xai_temperature_len,
            Lq,
            head_num,
            batch_size,
            _kv_is_fp8,
            BLOCK_M=64,
            BLOCK_N=64,
            BLOCK_DMODEL=_BLOCK_DMODEL,
            NUM_STAGES=1,
            num_warps=4,
            custom_mask_arg=custom_mask,
            mask_indptr_arg=mask_indptr,
            window_kv_offsets_arg=window_kv_offsets,
            total_prefix_len_hint=_total_pfx_est_pre,
            total_extend_len_hint=total_extend_len,
            min_len_extend_hint=min_len_extend,
        )
        return

    # Sync-free uniform detection: q_extend.shape[0] == sum(ext_i),
    # so if it equals B * max_ext every seq has length max_ext.
    _total_extend_rows = _q_shape[0]
    _uniform_by_shape = _total_extend_rows == batch_size * max_len_extend

    # WCA (and split-K, which shares the same kernel body) does
    # not support SWA + sinks. The combination produces NaN on prefix-
    # dominated shapes and trips an LLVM iota_range assertion during compile
    # on others. The non-WCA data-centric kernels handle SWA+sinks fine, so
    # the fastpath block stays reachable -- only the WCA / split-K dispatch
    # points are gated off for these shapes.
    _has_sw_sinks = sliding_window_size > 0 and sinks is not None
    _can_use_fastpath = custom_mask is None
    _can_route_wca = _can_use_fastpath and not _has_sw_sinks
    _is_uniform = (
        batch_size <= 1 or min_len_extend == max_len_extend or _uniform_by_shape
    )
    _is_ragged_ext = _can_route_wca and not _is_uniform and batch_size >= 2
    # Small-ext + big-prefix-skew (e.g. spec-decode): looks uniform but
    # the longest-prefix CTA dominates, so WCA reclaims these.
    _prefix_bucket_fastpath = _pfx_bucket(_total_pfx_est_pre, batch_size)
    _is_ragged_pfx = (
        _can_route_wca
        and batch_size >= 4
        and max_len_extend <= 128
        and _total_pfx_est_pre >= batch_size * 2048
        # FP8 small-extend carve-outs: data-centric (split-K or NS=1) is faster
        # than WCA on these buckets. The _need_wca gate below
        # still sends long-prefix B<=4 back to WCA when useful.
        and not (_kv_is_fp8 and Lq == 128 and max_len_extend <= 64)
        and not (_kv_is_fp8 and Lq == 128 and batch_size <= 4 and max_len_extend >= 128)
        and not (
            _kv_is_fp8
            and Lq == 64
            and batch_size >= 8
            and max_len_extend <= 64
            and _total_pfx_est_pre >= batch_size * 8192
        )
    )
    _is_ragged = _is_ragged_ext or _is_ragged_pfx

    if _kv_is_fp8 and not _is_uniform and max_len_extend <= 64:
        _is_ragged = False

    _is_d64_tiny_prefix_small_ext = (
        _can_use_fastpath
        and Lq == 64
        and not _kv_is_fp8
        and _prefix_bucket_fastpath == PrefixBucket.TINY
        and max_len_extend <= 64
        and batch_size <= 8
    )

    # WCA scope: only D=128 BF16/FP8 has a pipelined WCA home.
    # D<128 kernels assert D>=128 (4w sw-pipe and FP8 8w pingpong), and
    # D=256 WCA hits LDS pressure issues; both fall through to data-centric.
    # D=128 B<=4 never satisfies the WCA clauses below, so skip.
    _skip_wca_check = (
        Lq == 128 and not _kv_is_fp8 and not _is_ragged_pfx and batch_size <= 4
    )
    if _is_ragged and Lq == 128 and not _skip_wca_check:
        _total_ext = _total_extend_rows
        _grid_est = batch_size * max_len_extend
        _waste_frac = 1.0 - _total_ext / max(1, _grid_est)
        _total_pfx_est = _total_pfx_est_pre
        if _kv_is_fp8:
            # FP8 WCA stays limited to prefix-driven shapes.
            _use_wca = _is_ragged_pfx
        else:
            _use_wca = (
                _is_ragged_pfx
                or (max_len_extend >= 1024 and _waste_frac > 0.05 and batch_size >= 5)
                or (batch_size >= 8 and _total_pfx_est >= batch_size * 1024)
                or (batch_size >= 8 and max_len_extend >= 768 and _waste_frac >= 0.4)
                or (max_len_extend >= 768 and _waste_frac >= 0.2 and batch_size >= 5)
                or (batch_size >= 16 and _waste_frac >= 0.2)
            )
        if _use_wca:
            _launch_wca(
                q_extend,
                k_extend,
                v_extend,
                o_extend,
                k_buffer,
                v_buffer,
                qo_indptr,
                kv_indptr,
                kv_indices,
                custom_mask,
                is_causal,
                mask_indptr,
                max_len_extend,
                k_scale,
                v_scale,
                sm_scale,
                logit_cap,
                sliding_window_size,
                sinks,
                window_kv_offsets,
                xai_temperature_len,
                _kv_is_fp8,
                _total_pfx_est_pre,
                min_len_extend,
            )
            return

    if _can_use_fastpath:
        _BLOCK_DMODEL = _resolve_qk_split_dims(Lq)

        # D=64 BF16 tiny-prefix small-extend: NS=1 serial data-centric kernel.
        if _is_d64_tiny_prefix_small_ext:
            _launch_data_centric_grid(
                q_extend,
                k_extend,
                v_extend,
                o_extend,
                k_buffer,
                v_buffer,
                qo_indptr,
                kv_indptr,
                kv_indices,
                is_causal,
                max_len_extend,
                k_scale,
                v_scale,
                sm_scale,
                logit_cap,
                sliding_window_size,
                sinks,
                xai_temperature_len,
                Lq,
                head_num,
                batch_size,
                _kv_is_fp8,
                BLOCK_M=64,
                BLOCK_N=64,
                BLOCK_DMODEL=_BLOCK_DMODEL,
                NUM_STAGES=1,
                num_warps=4,
                window_kv_offsets_arg=window_kv_offsets,
                total_prefix_len_hint=_total_pfx_est_pre,
                total_extend_len_hint=total_extend_len,
                min_len_extend_hint=min_len_extend,
            )
            return

        # D=128 long-prefix decode-like shapes route to WCA.
        # (D=64 has no pipelined WCA home; D=256 is handled by
        # the dedicated block below.)
        if Lq == 128:
            _BM_est = 128
            _n_m_est = (max_len_extend + _BM_est - 1) // _BM_est
            _total_tiles_est = batch_size * head_num * _n_m_est
            _num_CUs = _get_num_CUs(q_extend.device)
            _total_ext = batch_size * max_len_extend
            _total_pfx = total_prefix_len if total_prefix_len is not None else 0
            _avg_pfx = _total_pfx // max(1, batch_size)
            if _kv_is_fp8:
                _need_wca = _total_tiles_est < _num_CUs and (
                    (_avg_pfx >= 4096 and max_len_extend <= 64)
                    or (_avg_pfx >= 4096 and max_len_extend <= 128 and batch_size >= 5)
                    or (_avg_pfx >= 16384 and max_len_extend <= 256)
                )
            else:
                _need_wca = (
                    _avg_pfx >= 4096
                    and (batch_size >= 4 or _avg_pfx >= 16384)
                    and _total_tiles_est < _num_CUs
                )
            if _need_wca and _can_route_wca:
                _launch_wca(
                    q_extend,
                    k_extend,
                    v_extend,
                    o_extend,
                    k_buffer,
                    v_buffer,
                    qo_indptr,
                    kv_indptr,
                    kv_indices,
                    custom_mask,
                    is_causal,
                    mask_indptr,
                    max_len_extend,
                    k_scale,
                    v_scale,
                    sm_scale,
                    logit_cap,
                    sliding_window_size,
                    sinks,
                    window_kv_offsets,
                    xai_temperature_len,
                    _kv_is_fp8,
                    _total_pfx_est_pre,
                    min_len_extend,
                )
                return

        # D=256 BF16 prefix-aware WCA routing. FP8 D=256 is unsupported
        # upstream (the entry guard already raised for that case).
        if Lq == 256 and not _kv_is_fp8:
            _total_pfx_256 = total_prefix_len if total_prefix_len is not None else 0
            _avg_pfx_256 = _total_pfx_256 // max(1, batch_size)
            _n_m_256 = (max_len_extend + 63) // 64
            _tiles_256 = batch_size * head_num * _n_m_256
            _need_wca_256 = (
                _avg_pfx_256 >= 4096
                and (batch_size >= 8 or (batch_size <= 2 and _avg_pfx_256 >= 16384))
                and _tiles_256 < _get_num_CUs(q_extend.device)
            )
            if _need_wca_256 and _can_route_wca:
                _launch_wca(
                    q_extend,
                    k_extend,
                    v_extend,
                    o_extend,
                    k_buffer,
                    v_buffer,
                    qo_indptr,
                    kv_indptr,
                    kv_indices,
                    custom_mask,
                    is_causal,
                    mask_indptr,
                    max_len_extend,
                    k_scale,
                    v_scale,
                    sm_scale,
                    logit_cap,
                    sliding_window_size,
                    sinks,
                    window_kv_offsets,
                    xai_temperature_len,
                    _kv_is_fp8,
                    _total_pfx_est_pre,
                    min_len_extend,
                )
                return

        _cfg = _get_data_centric_heuristic_config(
            Lq,
            batch_size,
            max_len_extend,
            _prefix_bucket_fastpath,
            _kv_is_fp8,
            sliding_window_size=sliding_window_size,
            head_num=head_num,
        )

        _launch_data_centric_grid(
            q_extend,
            k_extend,
            v_extend,
            o_extend,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            is_causal,
            max_len_extend,
            k_scale,
            v_scale,
            sm_scale,
            logit_cap,
            sliding_window_size,
            sinks,
            xai_temperature_len,
            Lq,
            head_num,
            batch_size,
            _kv_is_fp8,
            BLOCK_M=_cfg.block_m,
            BLOCK_N=_cfg.block_n,
            BLOCK_DMODEL=_BLOCK_DMODEL,
            NUM_STAGES=_cfg.num_stages,
            num_warps=_cfg.num_warps,
            ext_block_n=_cfg.ext_block_n,
            ext_num_stages=_cfg.ext_num_stages,
            window_kv_offsets_arg=window_kv_offsets,
            total_prefix_len_hint=total_prefix_len,
            total_extend_len_hint=total_extend_len,
            min_len_extend_hint=min_len_extend,
        )
        return

    if custom_mask is not None:
        # Custom masks cannot use WCA/split-K today, but the data-centric tile
        # choice is shape-derived and cacheable with exact length hints.
        BLOCK_DMODEL = _resolve_qk_split_dims(Lq)
        if total_prefix_len is None:
            total_prefix_len = kv_indices.numel()
        if total_extend_len is None:
            total_extend_len = q_extend.shape[0]
        if min_len_extend is None:
            min_len_extend = max_len_extend
        _prefix_bucket_custom_mask = _pfx_bucket(total_prefix_len, batch_size)
        _cfg = _get_exact_data_centric_heuristic_config(
            Lq,
            batch_size,
            max_len_extend,
            min_len_extend,
            total_prefix_len,
            total_extend_len,
            _prefix_bucket_custom_mask,
            _kv_is_fp8,
            sliding_window_size,
            head_num,
            True,
        )
        _launch_data_centric_grid(
            q_extend,
            k_extend,
            v_extend,
            o_extend,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            is_causal,
            max_len_extend,
            k_scale,
            v_scale,
            sm_scale,
            logit_cap,
            sliding_window_size,
            sinks,
            xai_temperature_len,
            Lq,
            head_num,
            batch_size,
            _kv_is_fp8,
            BLOCK_M=_cfg.block_m,
            BLOCK_N=_cfg.block_n,
            BLOCK_DMODEL=BLOCK_DMODEL,
            NUM_STAGES=_cfg.num_stages,
            num_warps=_cfg.num_warps,
            custom_mask_arg=custom_mask,
            mask_indptr_arg=mask_indptr,
            window_kv_offsets_arg=window_kv_offsets,
            ext_block_n=_cfg.ext_block_n,
            ext_num_stages=_cfg.ext_num_stages,
            total_prefix_len_hint=total_prefix_len,
            total_extend_len_hint=total_extend_len,
            min_len_extend_hint=min_len_extend,
        )
        return

    # Full dispatch path: heterogeneous batches and exact-hint fallbacks.
    BLOCK_DMODEL = _resolve_qk_split_dims(Lq)

    # min_len_extend is materialized lazily below -- computing it eagerly
    # costs a ~15us GPU sync on heterogeneous batches.

    # Split-K delegates to the WCA body, so SWA+sinks shapes
    # intentionally skip it and fall through to the data-centric path below.
    _use_splitk = (
        (Lq > 64) and (Lq <= 128) and (custom_mask is None) and not _has_sw_sinks
    )

    if _use_splitk:
        _BN = 32 if BLOCK_DMODEL >= 256 else 64
        if min_len_extend is None:
            min_len_extend = max_len_extend
        _launch_splitk(
            q_extend,
            k_extend,
            v_extend,
            o_extend,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            custom_mask,
            mask_indptr,
            window_kv_offsets,
            sm_scale,
            k_scale,
            v_scale,
            logit_cap,
            Lq,
            is_causal,
            max_len_extend,
            min_len_extend,
            sinks,
            xai_temperature_len,
            sliding_window_size,
            BLOCK_M=64,
            BLOCK_N=_BN,
            num_warps=4,
            NUM_STAGES=2,
            total_prefix_len=total_prefix_len,
            kv_is_fp8=_kv_is_fp8,
        )
        return

    # Use the same data-centric selector as the fast path. Full dispatch can
    # provide exact length hints when available; missing hints fall back to
    # cheap CPU-side proxies instead of materializing GPU reductions here.
    if total_prefix_len is None:
        total_prefix_len = kv_indices.numel()
    if total_extend_len is None:
        total_extend_len = q_extend.shape[0]
    if min_len_extend is None:
        min_len_extend = max_len_extend
    _prefix_bucket_full_dispatch = _pfx_bucket(total_prefix_len, batch_size)
    _cfg = _get_exact_data_centric_heuristic_config(
        Lq,
        batch_size,
        max_len_extend,
        min_len_extend,
        total_prefix_len,
        total_extend_len,
        _prefix_bucket_full_dispatch,
        _kv_is_fp8,
        sliding_window_size,
        head_num,
        False,
    )

    _launch_data_centric_grid(
        q_extend,
        k_extend,
        v_extend,
        o_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        is_causal,
        max_len_extend,
        k_scale,
        v_scale,
        sm_scale,
        logit_cap,
        sliding_window_size,
        sinks,
        xai_temperature_len,
        Lq,
        head_num,
        batch_size,
        _kv_is_fp8,
        BLOCK_M=_cfg.block_m,
        BLOCK_N=_cfg.block_n,
        BLOCK_DMODEL=BLOCK_DMODEL,
        NUM_STAGES=_cfg.num_stages,
        num_warps=_cfg.num_warps,
        custom_mask_arg=custom_mask,
        mask_indptr_arg=mask_indptr,
        window_kv_offsets_arg=window_kv_offsets,
        ext_block_n=_cfg.ext_block_n,
        ext_num_stages=_cfg.ext_num_stages,
        total_prefix_len_hint=total_prefix_len,
        total_extend_len_hint=total_extend_len,
        min_len_extend_hint=min_len_extend,
    )
