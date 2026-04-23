# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon extend-attention dispatch for gfx950 (MI350X / CDNA4).

Public entry point: :func:`gluon_extend_attention_fwd`. Supports symmetric
heads only (Lq == Lv) at D in {64, 128, 256} with BF16 or FP8 KV cache.
Mixed-dim / DeepSeek MLA heads live on a separate ``mla_prefill`` branch
and are out of scope here.

Dispatch picks between four kernel bodies (BF16 basic, BF16 WCA persistent,
FP8 basic, FP8 WCA persistent). All kernels are launched via the standard
Triton ``kernel[grid](...)`` JIT entry point. Triton's own compile cache
avoids recompilation on warm launches; a small shape-keyed tile-config
cache avoids rerunning the heuristic tree on every call, and WCA metadata
(split-K workspace, dummy mask buffers) is computed once and reused across
the attention layers of a single forward pass. The persistent kernel walks
``qo_indptr`` directly via an inline O(B) scan to resolve
``tile_idx -> (seq, head, block_m)`` -- no cumsum tensor is needed.

Env vars
--------
- ``SGLANG_GLUON_FP8_KV_FORCE_BF16=1`` (user-facing safety rail): route
  FP8 KV through the BF16 kernels by casting per call. Used when a new
  FP8 config hits a numerical issue in production.
"""

import functools
import logging
import math
import os

import torch
import triton

from ._kernel_bf16_gfx950 import (
    gluon_extend_attn_fwd as _gluon_extend_attn_fwd_symmetric,
)
from ._kernel_fp8_gfx950 import (
    gluon_extend_attn_fwd as _gluon_extend_attn_fwd_symmetric_fp8,
)
# Unified BF16 kernel. Same JITFunction covers the basic path
# (IS_PERSISTENT=False) and the WCA persistent / split-K paths
# (IS_PERSISTENT=True).
_bf16_kernel = _gluon_extend_attn_fwd_symmetric
# Unified FP8 kernel. Same gating as BF16: IS_PERSISTENT toggles between
# the basic and the WCA persistent / split-K bodies.
_gluon_extend_attn_fwd_fp8_basic = _gluon_extend_attn_fwd_symmetric_fp8

_dummy_cm = None
_dummy_mi = None
_dummy_mi_size = 0
_dummy_wkvo = None
_dummy_wkvo_size = 0
# Singleton passed to the kernel when HAS_SINK=False. The kernel body
# never loads from it and DCE eliminates the path.
_dummy_sinks = None
# Singletons passed to the unified BF16 kernel when IS_PERSISTENT=False.
# Triton specializes CompiledKernel on pointer element dtype, so they must
# match the live tensor dtypes on the persistent path (fp32 / int32). The
# kernel body gates every access on `if IS_PERSISTENT:` so these are never
# dereferenced in the basic path; allocating 1 element is sufficient.
_dummy_partial_out = None
_dummy_partial_lse = None
_dummy_tile_done = None

_LOGGED_FP8_KV_MODE = False
logger = logging.getLogger(__name__)


def _ensure_dummies(device, mi_size, wkvo_size):
    """Lazy-init module-level singleton dummy tensors on first use."""
    global _dummy_cm, _dummy_mi, _dummy_mi_size, _dummy_wkvo, _dummy_wkvo_size
    global _dummy_sinks
    global _dummy_partial_out, _dummy_partial_lse, _dummy_tile_done
    if _dummy_cm is None:
        _dummy_cm = torch.empty(0, dtype=torch.uint8, device=device)
    if _dummy_mi is None or _dummy_mi_size < mi_size:
        _dummy_mi = torch.zeros(mi_size, dtype=torch.int64, device=device)
        _dummy_mi_size = mi_size
    if _dummy_wkvo is None or _dummy_wkvo_size < wkvo_size:
        _dummy_wkvo = torch.zeros(wkvo_size, dtype=torch.int32, device=device)
        _dummy_wkvo_size = wkvo_size
    if _dummy_sinks is None:
        _dummy_sinks = torch.zeros(1, dtype=torch.bfloat16, device=device)
    if _dummy_partial_out is None:
        _dummy_partial_out = torch.empty(1, dtype=torch.float32, device=device)
    if _dummy_partial_lse is None:
        _dummy_partial_lse = torch.empty(1, dtype=torch.float32, device=device)
    if _dummy_tile_done is None:
        _dummy_tile_done = torch.zeros(1, dtype=torch.int32, device=device)


# User-facing safety rail: route FP8 KV through the BF16 kernels (pays a
# per-call dtype cast). Referenced in user-visible error messages.
_FP8_KV_FORCE_BF16 = int(os.getenv("SGLANG_GLUON_FP8_KV_FORCE_BF16", "0")) != 0


# ===-----------------------------------------------------------------------===#
# Persistent / split-K launch helpers
# ===-----------------------------------------------------------------------===#
#
# ``_launch_persistent`` and ``_launch_splitk`` wrap the unified BF16 kernel
# (``_gluon_extend_attn_fwd_symmetric``, IS_PERSISTENT=True path) with
# tile-scheduling and split-K workspace management. Kernels are launched via
# standard ``kernel[grid](...)``; Triton's own compile cache absorbs the JIT
# cost on warm calls.


_QK_SPLIT_CACHE = {}


def _resolve_qk_split_dims(Lq: int):
    """Return BLOCK_DMODEL (next power of 2 >= Lq, >= 16).

    The production dispatcher only routes power-of-2 Lq (64/128/256), so
    BLOCK_DMODEL == Lq and no separate ACTUAL_BLOCK_DMODEL is needed.
    """
    cached = _QK_SPLIT_CACHE.get(Lq)
    if cached is not None:
        return cached
    block_dmodel = max(triton.next_power_of_2(Lq), 16)
    _QK_SPLIT_CACHE[Lq] = block_dmodel
    return block_dmodel


_cached_num_CUs = {}


def _get_num_CUs(device):
    idx = device.index if hasattr(device, 'index') and device.index is not None else 0
    if idx not in _cached_num_CUs:
        _cached_num_CUs[idx] = torch.cuda.get_device_properties(device).multi_processor_count
    return _cached_num_CUs[idx]


# Cached dummies for the no-custom-mask / no-sliding-window persistent path.
# Reused across launches; the zeroed mask_indptr buffer grows monotonically
# and any prefix is valid (the kernel reads index 0 for a "no mask" batch).
_dummy_cm_lh = None
_dummy_mi_lh = None
_dummy_wkvo_lh = None
_dummy_mi_cap = 0
_dummy_wkvo_cap = 0


def _ensure_dummy_mask_tensors(device, q_total, batch_size):
    """Return (custom_mask, mask_indptr, window_kv_offsets) dummy tensors
    for the persistent path when USE_CUSTOM_MASK=False."""
    global _dummy_cm_lh, _dummy_mi_lh, _dummy_wkvo_lh
    global _dummy_mi_cap, _dummy_wkvo_cap
    if _dummy_cm_lh is None or _dummy_cm_lh.device != device:
        _dummy_cm_lh = torch.empty(0, dtype=torch.uint8, device=device)
    needed_mi = q_total + 1
    if _dummy_mi_lh is None or _dummy_mi_lh.device != device or _dummy_mi_cap < needed_mi:
        _dummy_mi_cap = max(needed_mi, 1024)
        _dummy_mi_lh = torch.zeros(_dummy_mi_cap, dtype=torch.int64, device=device)
    if _dummy_wkvo_lh is None or _dummy_wkvo_lh.device != device or _dummy_wkvo_cap < batch_size:
        _dummy_wkvo_cap = max(batch_size, 256)
        _dummy_wkvo_lh = torch.zeros(_dummy_wkvo_cap, dtype=torch.int32, device=device)
    return _dummy_cm_lh, _dummy_mi_lh[:needed_mi], _dummy_wkvo_lh[:batch_size]


def _select_persistent_grid(total_valid_tiles: int, num_CUs: int) -> int:
    """Pick CTA count for the persistent kernel.

    When there are more tiles than CUs we cap at 2*CUs for good occupancy.
    When there are fewer tiles than CUs (decode, spec-decode) we use all
    available tiles -- split-K will eventually fill the remaining CUs.
    """
    if total_valid_tiles >= 2 * num_CUs:
        return 2 * num_CUs
    if total_valid_tiles >= num_CUs:
        return num_CUs
    return total_valid_tiles


def _select_k_splits(total_output_tiles, num_CUs, min_prefix_blocks=4):
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


_splitk_dummy = None


def _ensure_splitk_dummy(device):
    global _splitk_dummy
    if _splitk_dummy is None or _splitk_dummy.device != device:
        _splitk_dummy = torch.empty(1, dtype=torch.float32, device=device)
    return _splitk_dummy


_splitk_ws_out = None
_splitk_ws_lse = None
_splitk_ws_done = None


def _ensure_splitk_workspace(total_splits, total_output_tiles, BLOCK_M, BLOCK_DV, device):
    """Return (partial_out, partial_lse, tile_done) workspace tensors,
    re-using cached allocations when possible.
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
        _splitk_ws_out = torch.empty(cap, BLOCK_M, BLOCK_DV, dtype=torch.float32, device=device)
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
    po.zero_()
    pl.fill_(float("-inf"))
    td.zero_()
    return po, pl, td


def _launch_persistent(
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
    sliding_window_size=-1,
    sinks=None,
    window_kv_offsets=None,
    xai_temperature_len=-1,
    enable_prefix_unmasked=True,
    min_len_extend=None,
    total_prefix_len=None,
    skip_prefix_custom_mask=True,  # accepted for FP8-symmetric dispatch; BF16 kernel hardcodes this True
):
    Lq = q_extend.shape[-1]
    BLOCK_DMODEL = _resolve_qk_split_dims(Lq)

    USE_CUSTOM_MASK = custom_mask is not None
    _batch_size_prelim = qo_indptr.shape[0] - 1
    if not USE_CUSTOM_MASK:
        custom_mask, mask_indptr, _wkvo_dummy = _ensure_dummy_mask_tensors(
            q_extend.device, q_extend.shape[0], _batch_size_prelim,
        )
        if window_kv_offsets is None:
            window_kv_offsets = _wkvo_dummy
    elif window_kv_offsets is None:
        _, _, window_kv_offsets = _ensure_dummy_mask_tensors(
            q_extend.device, q_extend.shape[0], _batch_size_prelim,
        )
    assert q_extend.shape[1] % k_extend.shape[1] == 0

    BLOCK_N = 32 if BLOCK_DMODEL >= 256 else 64
    batch_size = qo_indptr.shape[0] - 1

    if min_len_extend is None:
        # Conservative fallback; the GPU-computed min would cost ~6ms of
        # CPU sync on heterogeneous batches. Tile selection below only
        # uses min_len_extend to shrink BM on skewed batches, so max is
        # safe and slightly pessimistic.
        min_len_extend = max_len_extend
    head_num = q_extend.shape[1]

    # Tile-config auto-select. Two regimes on top of the D>=256 and
    # default BM=128 NW=8 paths:
    #   * small-tile (BM=64 NW=4 NS=2): per-seq work is small or the
    #     batch is ragged enough that BM=128 wastes compute on masked
    #     tokens, OR Lq=128 ext-dominated chat-mix. Exception: big-B chat
    #     with a substantial prefix -- each tile iterates the full
    #     per-seq prefix regardless of BM, so larger BM amortizes the
    #     scan over more query rows.
    _het_ratio = max_len_extend / max(min_len_extend, 1)
    _total_pfx_est = (
        total_prefix_len if total_prefix_len is not None
        else kv_indices.numel()
    )
    _total_ext = q_extend.shape[0]
    _avg_ext = _total_ext / max(1, batch_size)
    _pfx_dominated_big_b = (
        batch_size >= 8
        and _total_pfx_est >= batch_size * 1024
        and 32 <= max_len_extend <= 512
    )
    _lq128_ext_dominated = (
        Lq == 128 and _total_pfx_est < 4 * _total_ext
    )
    _use_small_tile = (
        not _pfx_dominated_big_b
        and (
            max_len_extend <= 128
            or (max_len_extend <= 256 and _het_ratio >= 2.0)
            or (min_len_extend < 64 and max_len_extend <= 512 and batch_size <= 4)
            or _lq128_ext_dominated
        )
    )

    if BLOCK_DMODEL >= 256:
        BLOCK_M = 64
        num_warps = 4
    elif _use_small_tile:
        BLOCK_M = 64
        num_warps = 4
    elif batch_size <= 4:
        BLOCK_M = 128
        num_warps = 8
    else:
        # BM=256 hits an LLVM iota_range assertion in the persistent kernel
        # on some shapes; BM=128 is the largest safe M-tile (and matches
        # what CK picks on these shapes).
        BLOCK_M = 128
        num_warps = 8

    if BLOCK_DMODEL >= 256:
        NUM_STAGES = 1
    elif BLOCK_M == 64 and num_warps == 4:
        NUM_STAGES = 2
    elif BLOCK_M == 64:
        NUM_STAGES = 1
    else:
        NUM_STAGES = 4

    if BLOCK_M == 128 and num_warps == 8 and NUM_STAGES == 2:
        NUM_STAGES = 3

    sm_scale = sm_scale or 1.0 / math.sqrt(Lq)
    sm_scale = sm_scale * k_scale
    kv_group_num = q_extend.shape[1] // k_extend.shape[1]

    device = q_extend.device
    n_m_tiles = (max_len_extend + BLOCK_M - 1) // BLOCK_M
    total_output_tiles_upper = batch_size * head_num * n_m_tiles
    if total_output_tiles_upper == 0:
        return

    # Tight sync-free upper bound on sum(ceil(ext_i / BM)). For
    # heterogeneous batches this is 3-4x smaller than the loose bound
    # ``batch * ceil(max_ext/BM)`` -- the kernel's inline WCA scan
    # tolerates the over-provisioned slots with a cheap validity check.
    total_extend_rows = q_extend.shape[0]
    total_output_tiles = (
        (total_extend_rows + batch_size * (BLOCK_M - 1)) // BLOCK_M
    ) * head_num

    num_CUs = _get_num_CUs(device)

    # Auto-splitk here is only correct at the BM=128 NW=8 NS=4 "big-tile"
    # config used by _launch_splitk; other tile shapes produce NaNs from
    # the multi-split reduction. Callers who need split-K at other configs
    # must go through _launch_splitk directly.
    SPLIT_K = 1
    if (
        total_output_tiles < num_CUs
        and BLOCK_M == 128
        and num_warps == 8
        and NUM_STAGES == 4
    ):
        if total_prefix_len is not None:
            avg_kv_len = total_prefix_len // max(1, batch_size)
        else:
            avg_kv_len = int((kv_indptr[-1] - kv_indptr[0]).item()) // max(1, batch_size)
        if avg_kv_len >= 4 * BLOCK_N:
            SPLIT_K = _select_k_splits(total_output_tiles, num_CUs)

    if SPLIT_K > 1:
        total_splits = total_output_tiles * SPLIT_K
        partial_out, partial_lse, tile_done = _ensure_splitk_workspace(
            total_splits, total_output_tiles, BLOCK_M, BLOCK_DMODEL, device,
        )
        total_valid_tiles = total_splits
    else:
        partial_out = _ensure_splitk_dummy(device)
        partial_lse = _ensure_splitk_dummy(device)
        tile_done = _ensure_splitk_dummy(device)
        total_valid_tiles = total_output_tiles

    total_programs = _select_persistent_grid(total_valid_tiles, num_CUs)
    grid = (total_programs,)

    _enable_prefix_unmasked = enable_prefix_unmasked
    _has_sink = sinks is not None
    _v_scale_final = 1.0 if SPLIT_K > 1 else v_scale

    _bf16_kernel[grid](
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
        kv_group_num,
        q_extend.stride(0), q_extend.stride(1),
        k_extend.stride(0), k_extend.stride(1),
        v_extend.stride(0), v_extend.stride(1),
        o_extend.stride(0), o_extend.stride(1),
        k_buffer.stride(0), k_buffer.stride(1),
        v_buffer.stride(0), v_buffer.stride(1),
        IS_CAUSAL=is_causal,
        USE_CUSTOM_MASK=USE_CUSTOM_MASK,
        ENABLE_PREFIX_UNMASKED=_enable_prefix_unmasked,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        NUM_STAGES=NUM_STAGES,
        Sinks=sinks,
        HAS_SINK=_has_sink,
        LOGIT_CAP=logit_cap,
        XAI_TEMPERATURE_LEN=xai_temperature_len,
        SLIDING_WINDOW_SIZE=sliding_window_size,
        v_scale=_v_scale_final,
        num_heads=head_num,
        n_m_tiles=n_m_tiles,
        total_valid_tiles=total_valid_tiles,
        total_programs=total_programs,
        partial_out=partial_out,
        partial_lse=partial_lse,
        tile_done=tile_done,
        actual_batch_size=batch_size,
        IS_PERSISTENT=True,
        SPLIT_K=SPLIT_K,
        num_warps=num_warps,
        num_stages=1,
        waves_per_eu=2,
        matrix_instr_nonkdim=32,
    )


def _launch_splitk(
    q_extend, k_extend, v_extend, o_extend,
    k_buffer, v_buffer,
    qo_indptr, kv_indptr, kv_indices,
    custom_mask, mask_indptr, window_kv_offsets,
    sm_scale, k_scale, v_scale, logit_cap,
    Lq, Lv, is_causal, max_len_extend, min_len_extend,
    sinks, xai_temperature_len, sliding_window_size,
    BLOCK_M, BLOCK_N, num_warps, NUM_STAGES,
    total_prefix_len=None,
):
    """Split-K persistent kernel: partitions prefix across CTAs, then reduces."""
    head_num = q_extend.shape[1]
    device = q_extend.device
    batch_size = qo_indptr.shape[0] - 1

    n_m_tiles = (max_len_extend + BLOCK_M - 1) // BLOCK_M
    total_output_tiles = batch_size * head_num * n_m_tiles
    if total_output_tiles == 0:
        return

    num_CUs = _get_num_CUs(device)
    if total_prefix_len is not None:
        avg_kv_len = total_prefix_len // max(1, batch_size)
    else:
        avg_kv_len = int((kv_indptr[-1] - kv_indptr[0]).item()) // max(1, batch_size)
    need_real_splitk = (
        total_output_tiles < num_CUs and avg_kv_len >= 4 * BLOCK_N
    )

    if not need_real_splitk:
        _launch_persistent(
            q_extend, k_extend, v_extend, o_extend,
            k_buffer, v_buffer,
            qo_indptr, kv_indptr, kv_indices,
            custom_mask, is_causal, mask_indptr,
            max_len_extend,
            k_scale=k_scale, v_scale=v_scale, sm_scale=sm_scale,
            logit_cap=logit_cap,
            min_len_extend=min_len_extend,
            sinks=sinks,
            xai_temperature_len=xai_temperature_len,
            sliding_window_size=sliding_window_size,
            window_kv_offsets=window_kv_offsets,
            total_prefix_len=total_prefix_len,
        )
        return

    BLOCK_DMODEL = _resolve_qk_split_dims(Lq)

    SPLIT_K = _select_k_splits(total_output_tiles, num_CUs)

    if BLOCK_DMODEL < 256:
        BLOCK_M = 128
        num_warps = 8
        NUM_STAGES = 4

    USE_CUSTOM_MASK = custom_mask is not None
    _bs_splitk = qo_indptr.shape[0] - 1
    if not USE_CUSTOM_MASK:
        custom_mask, mask_indptr, _wkvo_dummy = _ensure_dummy_mask_tensors(
            device, q_extend.shape[0], _bs_splitk,
        )
        if window_kv_offsets is None:
            window_kv_offsets = _wkvo_dummy
    elif window_kv_offsets is None:
        _, _, window_kv_offsets = _ensure_dummy_mask_tensors(
            device, q_extend.shape[0], _bs_splitk,
        )
    enable_prefix_unmasked = is_causal

    sm_scale = sm_scale or 1.0 / math.sqrt(Lq)
    sm_scale = sm_scale * k_scale
    kv_group_num = q_extend.shape[1] // k_extend.shape[1]

    # n_m_tiles may have shifted if we forced a different BLOCK_M above.
    n_m_tiles = (max_len_extend + BLOCK_M - 1) // BLOCK_M
    # Tight sync-free upper bound on total_output_tiles (see _launch_persistent).
    total_output_tiles = (
        (q_extend.shape[0] + batch_size * (BLOCK_M - 1)) // BLOCK_M
    ) * head_num

    total_splits = total_output_tiles * SPLIT_K
    partial_out, partial_lse, tile_done = _ensure_splitk_workspace(
        total_splits, total_output_tiles, BLOCK_M, BLOCK_DMODEL, device,
    )

    total_valid_tiles = total_output_tiles * SPLIT_K
    total_programs = min(total_valid_tiles, 2 * num_CUs)
    grid = (total_programs,)

    _has_sink = sinks is not None
    _v_scale_final = 1.0 if SPLIT_K > 1 else v_scale

    _bf16_kernel[grid](
        q_extend, k_extend, v_extend, o_extend,
        k_buffer, v_buffer,
        qo_indptr, kv_indptr, kv_indices,
        custom_mask, mask_indptr, window_kv_offsets,
        sm_scale, kv_group_num,
        q_extend.stride(0), q_extend.stride(1),
        k_extend.stride(0), k_extend.stride(1),
        v_extend.stride(0), v_extend.stride(1),
        o_extend.stride(0), o_extend.stride(1),
        k_buffer.stride(0), k_buffer.stride(1),
        v_buffer.stride(0), v_buffer.stride(1),
        IS_CAUSAL=is_causal,
        USE_CUSTOM_MASK=USE_CUSTOM_MASK,
        ENABLE_PREFIX_UNMASKED=enable_prefix_unmasked,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        NUM_STAGES=NUM_STAGES,
        Sinks=sinks, HAS_SINK=_has_sink,
        LOGIT_CAP=logit_cap,
        XAI_TEMPERATURE_LEN=xai_temperature_len,
        SLIDING_WINDOW_SIZE=sliding_window_size,
        v_scale=_v_scale_final,
        num_heads=head_num, n_m_tiles=n_m_tiles,
        total_valid_tiles=total_valid_tiles, total_programs=total_programs,
        partial_out=partial_out, partial_lse=partial_lse,
        tile_done=tile_done,
        actual_batch_size=batch_size,
        IS_PERSISTENT=True, SPLIT_K=SPLIT_K,
        num_warps=num_warps, num_stages=1,
        waves_per_eu=2,
        matrix_instr_nonkdim=32,
    )


# ===-----------------------------------------------------------------------===#
# Basic-path tile-config heuristics
# ===-----------------------------------------------------------------------===#


def _select_d256_dispatch(
    batch_size: int,
    max_len_extend: int,
    min_len_extend: int,
    total_prefix_len: int,
    total_extend_len: int,
    avg_pfx_proxy: int = 0,
):
    """D=256 launch policy (Gemma-style models).

    Narrow oracle-tuned overrides for B>=4 ext>=256 and B=1 ext>=2048 fire
    first; everything else falls through to the prior tree. ``avg_pfx_proxy``
    is a coarse proxy for avg prefix length supplied by the caller when
    ``total_prefix_len`` is not threaded through.

    NS=1 is avoided even where it would win: the prefix-pipelined kernel
    hard-asserts NS>=2 for determinism, so NS=2 is used throughout at a
    3-6% cost (within BF16 noise) on shapes where NS=1 would have won.
    """
    total_tokens = max(1, total_prefix_len + total_extend_len)
    prefix_frac = total_prefix_len / total_tokens
    ext_ratio = max_len_extend / max(1, min_len_extend)
    avg_pfx = total_prefix_len // max(1, batch_size)

    if batch_size >= 4 and max_len_extend >= 256:
        return 128, 8, 2, 16, 16
    if batch_size == 1 and max_len_extend >= 2048:
        return 128, 8, 2, 16, 16

    if max_len_extend >= 768:
        if batch_size <= 2:
            return 64, 4, 2, 16, 16
        return 128, 8, 3, 16, 16

    if prefix_frac <= 0.55:
        if batch_size <= 2:
            return 64, 4, 2, 16, 16
        return 128, 8, 3, 16, 16

    if batch_size <= 4:
        if max_len_extend <= 128 and avg_pfx >= 2048:
            return 64, 4, 4, 16, 16
        return 64, 4, 2, 16, 16

    if max_len_extend <= 128 and avg_pfx >= 2048:
        return 64, 8, 4, 16, 16

    if ext_ratio <= 2.5:
        return 64, 4, 2, 16, 16

    return 64, 8, 4, 16, 16


# FP8 dispatch defaults for the rare D not in {64, 128} fallthrough. The
# main D=64 / D=128 FP8 paths have these baked in explicitly.
_FP8_DEFAULT_BN = 128
_FP8_DEFAULT_NS = 2
_FP8_DEFAULT_EXT_BN = 64
_FP8_DEFAULT_EXT_NS = 3


def _pfx_bucket(total_prefix_len, batch_size):
    """Bucket average per-sequence prefix length for dispatch.

    0 = no prefix (prefill from scratch)
    1 = tiny prefix      (avg < 512)
    2 = moderate prefix  (avg < 2048)
    3 = large prefix     (avg < 8192)
    4 = huge prefix      (avg >= 8192)
    """
    if total_prefix_len is None or total_prefix_len <= 0 or batch_size <= 0:
        return 0
    avg = total_prefix_len // batch_size
    if avg < 512:
        return 1
    if avg < 2048:
        return 2
    if avg < 8192:
        return 3
    return 4


def _dispatch_d64_bf16(batch_size, max_len_extend, pfx_bucket, head_num):
    """Pick (BM, NW, NS) for the D=64 BF16 basic kernel.

    Per-shape winners from the AITER/CK grid (see bench_fp8_vs_ck.py).
    B=1 uses ``head_num`` to choose BM — the win from BM=256 depends on
    how the grid ``(1, H, ceil(S/BM))`` fills waves across 256 CUs.
    """
    _total_ext = batch_size * max_len_extend
    if batch_size >= 8 and max_len_extend <= 128 and pfx_bucket >= 2:
        return 128, 128, 4, 4
    if batch_size >= 16 and max_len_extend <= 32:
        return 64, 64, 4, 4
    if batch_size >= 16:
        if max_len_extend >= 512:
            return 256, 64, 8, 2
        return 64, 64, 4, 4
    if batch_size >= 4:
        if _total_ext >= 2048 or max_len_extend >= 512:
            return 256, 64, 8, 2
        if batch_size <= 7:
            return 128, 64, 8, 4
        return 64, 64, 4, 4

    # B=1..3. H_q>=64 crosses over to BM=256 at S>=1600; H_q=32 at S>=1500
    # (lower H already under-subscribes so BM=256's bigger per-tile work
    # dominates earlier). B=2,3 keep the conservative S>=2048 boundary.
    _h = head_num or 0
    if batch_size == 1 and _h >= 64 and max_len_extend >= 1600:
        return 256, 64, 8, 2
    if batch_size == 1 and _h == 32 and max_len_extend >= 1500:
        return 256, 64, 8, 2
    if max_len_extend >= 2048:
        return 256, 64, 8, 2
    return 128, 64, 8, 4


def _dispatch_d128_bf16(batch_size, max_len_extend, pfx_bucket, head_num):
    """Pick (BM, NW, NS) for the D=128 BF16 basic kernel.

    NS=1 is unsafe on the prefix-pipelined path (DMA-ring races) and
    benches within +-3% of NS=2, so NS>=2 throughout. BM=256 NW=8 NS=2
    is avoided at D=128: a latent reduction-order issue produces
    max_err ~5e-3 to 1.3e-2 on ~0.1% of outputs. BM=128 NS=2 at H>=64
    captures most of the win; H=32 stays on BM=64 NW=4.
    """
    _total_ext = batch_size * max_len_extend
    if batch_size >= 16 and max_len_extend <= 16:
        return 16, 64, 4, 2
    if batch_size >= 16 and max_len_extend <= 64:
        return 64, 64, 4, 2
    if pfx_bucket >= 3 and max_len_extend >= 4096:
        return 128, 64, 8, 2
    if pfx_bucket >= 2 and max_len_extend >= 2048:
        return 128, 64, 8, 2
    if batch_size == 1:
        _h = head_num or 0
        if _h >= 64 and max_len_extend >= 2000:
            return 128, 64, 8, 2
        if _h >= 64 and max_len_extend >= 1024:
            return 128, 64, 8, 4
        return 64, 64, 4, 2
    if batch_size <= 4:
        return 64, 64, 4, 2
    if _total_ext >= 32768:
        return 128, 64, 8, 2
    return 64, 64, 4, 2


def _d64_bf16_batched_overrides(BM, BN, NW, NS, batch_size, max_len_extend,
                                sliding_window_size, total_ext):
    """Post-process the D=64 BF16 config for batched edge cases.

    EXT_BN/EXT_NS are derived from the post-override (BM, BN, NW, NS)
    by the caller, not here, so the BM=256 NS=2→NS=4 promotion propagates
    correctly.
    """
    # B>=16 ext==256 full-attn: BM=64 undersubscribes 256 CUs by 4x,
    # route to BM=256.
    if (
        sliding_window_size <= 0
        and batch_size >= 16 and max_len_extend == 256
    ):
        BM, BN, NW, NS = 256, 64, 8, 2

    # BM=256 NW=8 gets an extra stage on batched prefill shapes with
    # occupancy headroom. Skip for B=1 — single-seq ShareGPT already sits
    # at the occupancy floor and the extra stage only eats LDS/regs.
    if (
        BM == 256 and NW == 8 and NS == 2
        and max_len_extend >= 1024
        and batch_size != 1
        and (max_len_extend <= 8192 or total_ext <= 32768)
    ):
        NS = 4
    return BM, BN, NW, NS


def _apply_fp8_overrides(Lq, batch_size, max_len_extend, pfx_bucket,
                         BM, BN, NW, NS, total_ext):
    """FP8 basic dispatch. Invariants:

    * NS>=2 on the 8-warp pipelined path (NS=1 is bit-nondeterministic;
      NS>=3 OOMs LDS at BN=128).
    * D<128 only has the NW=4 NS=1 serial path; pipelined layouts are
      not wired up. BM>128 at NW=4 blows register pressure.
    """
    if Lq == 128:
        BM, BN, NW, NS = 128, 128, 8, 2
        EXT_BN, EXT_NS = 128, 2
        if batch_size == 1 and max_len_extend <= 256:
            BM = 64
        elif (
            batch_size >= 32
            and max_len_extend <= 8
            and pfx_bucket <= 2
        ):
            # Decode / spec-verify / draft-extend on short-prefix
            # continuous batches: BM=64 NW=4 avoids pad-heavy tiles.
            BM, NW = 64, 4
        elif batch_size >= 16 and max_len_extend <= 64:
            # Small-batch decode-continuation bucket. With BM=128, Q tiles
            # waste 50-87% of rows when max_ext < BM, which is exactly the
            # pattern ShareGPT radix-on hammers (B in {16,32}, ext in
            # {16,32,64}, pfx 512-2k). Pure-serial NW=4 NS=1 at BM=64
            # closes the gap — sweep over 10 weak-bucket shapes moves
            # this path from 0.66x-1.02x to 1.25x-1.55x vs Triton.
            # Kernel body has the NW<8 serial branch guarded by
            # `USE_PINGPONG = num_warps >= 8`; NS=1 skips the DMA pipeline.
            BM, NW, NS = 64, 4, 1
            EXT_NS = 1
        elif (
            batch_size >= 8
            and max_len_extend <= 64
            and pfx_bucket >= 3
        ):
            # Long-prefix short-extend at moderate batch (B=8 with pfx>=2k
            # avg per seq). Same pad-waste argument as the B>=16 clause
            # but the prefix is long enough that tile-utilisation on the
            # prefix phase also matters. Sweep shows B=8 pfx=2k ext=64
            # goes 0.64x -> 1.14x with this demotion; shorter prefixes
            # at B=8 still win with the 8-warp default so we don't
            # demote them.
            BM, NW, NS = 64, 4, 1
            EXT_NS = 1
        return BM, BN, NW, NS, EXT_BN, EXT_NS

    if Lq == 64:
        BM, BN, NW, NS = 128, 128, 4, 1
        EXT_BN, EXT_NS = 128, 1
        # BM=64 wins wherever tiles are pad-heavy.
        if max_len_extend <= 8 or (
            batch_size >= 16 and max_len_extend <= 128
        ):
            BM = 64
        return BM, BN, NW, NS, EXT_BN, EXT_NS

    # Lq == 256 (and other unusual D that land here) — serial fallback.
    BN = 32 if Lq >= 256 else _FP8_DEFAULT_BN
    NS = _FP8_DEFAULT_NS
    NW = min(NW, 4) if Lq < 128 else NW
    EXT_BN = _FP8_DEFAULT_EXT_BN
    EXT_NS = _FP8_DEFAULT_EXT_NS
    if Lq >= 256 and NW <= 4:
        EXT_NS = 1 if total_ext >= 512 else 2
    return BM, BN, NW, NS, EXT_BN, EXT_NS


def _apply_d64_swa_overrides(BM, BN, NW, NS, EXT_BN, EXT_NS,
                             batch_size, max_len_extend, sliding_window_size):
    """BF16 D=64 sliding-window override.

    The per-tile key band is ~(BM + sw) wide regardless of tile count, so
    BM=256 (base-tree win for plain causal) wastes work. Smaller BM splits
    the static window across more CTAs and hides memory latency. Gates
    tuned on D=64 H=32 kvH=4 sw=127.
    """
    total_ext = batch_size * max_len_extend
    if max_len_extend >= 1024 and total_ext >= 2048:
        BM, BN, NW, NS = 64, 64, 2, 2
        EXT_BN, EXT_NS = BN, NS
    elif max_len_extend > 16 * sliding_window_size:
        if BM > 128:
            BM = 128
        NW, NS = 4, 2
        EXT_NS = NS
    elif BM > 128:
        BM, NS = 128, 4
        EXT_NS = NS
    return BM, BN, NW, NS, EXT_BN, EXT_NS


@functools.lru_cache(maxsize=2048)
def _get_basic_dispatch_config(
    Lq, batch_size, max_len_extend, pfx_bucket, is_fp8, sliding_window_size=-1,
    head_num=None,
):
    """Route to the per-(Lq, dtype) helper and apply SWA / batched overrides.

    Returns (BLOCK_M, BLOCK_N, num_warps, NUM_STAGES, PAD_K, PAD_V,
    EXT_BLOCK_N, EXT_NUM_STAGES) for the basic (non-persistent) kernel.
    Memoized on the full argument tuple -- the heuristic tree is a step
    function with ~6 configs per (Lq, is_fp8) pair, so the live cache
    stays small and hit rate is near-100% in production.
    """
    _total_ext = batch_size * max_len_extend
    _PAD_K, _PAD_V = 16, 16

    if Lq == 256:
        _avg_pfx_proxy = (
            0 if pfx_bucket <= 1
            else 1024 if pfx_bucket == 2
            else 2048 if pfx_bucket == 3
            else 8192
        )
        _BM, _NW, _NS, _PAD_K, _PAD_V = _select_d256_dispatch(
            batch_size, max_len_extend, max_len_extend, 0, _total_ext,
            avg_pfx_proxy=_avg_pfx_proxy,
        )
        _BN = 32
        _EXT_BN, _EXT_NS = _BN, _NS
    elif Lq == 64:
        _BM, _BN, _NW, _NS = _dispatch_d64_bf16(
            batch_size, max_len_extend, pfx_bucket, head_num
        )
        _BM, _BN, _NW, _NS = _d64_bf16_batched_overrides(
            _BM, _BN, _NW, _NS, batch_size, max_len_extend,
            sliding_window_size, _total_ext,
        )
        _EXT_BN, _EXT_NS = _BN, _NS
    else:
        _BM, _BN, _NW, _NS = _dispatch_d128_bf16(
            batch_size, max_len_extend, pfx_bucket, head_num
        )
        _EXT_BN, _EXT_NS = _BN, _NS

    if is_fp8:
        _BM, _BN, _NW, _NS, _EXT_BN, _EXT_NS = _apply_fp8_overrides(
            Lq, batch_size, max_len_extend, pfx_bucket,
            _BM, _BN, _NW, _NS, _total_ext,
        )

    if (
        not is_fp8
        and Lq == 64
        and sliding_window_size > 0
        and sliding_window_size < max_len_extend
    ):
        _BM, _BN, _NW, _NS, _EXT_BN, _EXT_NS = _apply_d64_swa_overrides(
            _BM, _BN, _NW, _NS, _EXT_BN, _EXT_NS,
            batch_size, max_len_extend, sliding_window_size,
        )

    return _BM, _BN, _NW, _NS, _PAD_K, _PAD_V, _EXT_BN, _EXT_NS


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
    skip_prefix_custom_mask=True,
    sliding_window_size=-1,
    sinks=None,
    window_kv_offsets=None,
    xai_temperature_len=-1,
    min_len_extend=None,
    total_prefix_len=None,
    total_extend_len=None,
):
    global _LOGGED_FP8_KV_MODE
    # Cache shape tuples once; downstream accesses use tuple indexing.
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
                "BF16 KV for spec-decode verify, or set "
                "SGLANG_GLUON_FP8_KV_FORCE_BF16=1."
            )
        # FP8 KV + D=256 fails Triton IR lowering (MFMA_F8 emits
        # unrealized_conversion_cast the backend can't materialize).
        if Lq == 256:
            raise NotImplementedError(
                "Gluon FP8 KV extend is not supported at head-dim 256. Use "
                "BF16 KV (Gemma and other D=256 models ship BF16), or set "
                "SGLANG_GLUON_FP8_KV_FORCE_BF16=1."
            )
        if _FP8_KV_FORCE_BF16:
            bridge_dtype = q_extend.dtype
            k_buffer = k_buffer.to(bridge_dtype)
            v_buffer = v_buffer.to(bridge_dtype)
            _kv_is_fp8 = False
        if not _LOGGED_FP8_KV_MODE:
            if _kv_is_fp8:
                logger.info(
                    "Gluon FP8 KV path active: native fp8 symmetric kernels enabled."
                )
            else:
                logger.warning(
                    "Gluon FP8 KV bridge enabled: casting fp8 KV buffers to non-fp8 kernels."
                )
            _LOGGED_FP8_KV_MODE = True
    if Lq != Lv:
        raise ValueError(
            f"Gluon extend attention only supports symmetric heads (Lq == Lv), "
            f"got Lq={Lq}, Lv={Lv}. Use mla_prefill/ for mixed-dim DeepSeek MLA."
        )
    _kernel_fn = (
        _gluon_extend_attn_fwd_fp8_basic if _kv_is_fp8
        else _gluon_extend_attn_fwd_symmetric
    )
    batch_size = qo_indptr.shape[0] - 1
    head_num = _q_shape[1]

    # Uniform detection without hitting the GPU: `q_extend.shape[0]`
    # already equals sum(ext_i); if it equals B * max_ext, every seq
    # has length max_ext.
    _total_extend_rows = _q_shape[0]
    _uniform_by_shape = (_total_extend_rows == batch_size * max_len_extend)
    _uniform_like = (batch_size <= 1 or _uniform_by_shape)

    _has_sink = sinks is not None

    # WCA eligibility: no custom mask. Custom-mask workloads always fall
    # back to the basic per-batch grid.
    _can_route_wca = custom_mask is None
    _is_uniform = (batch_size <= 1 or min_len_extend == max_len_extend
                   or _uniform_by_shape)
    _is_ragged_ext = _can_route_wca and not _is_uniform and batch_size >= 2
    # Small-ext + big-prefix-skew (e.g. spec-decode) — looks uniform but
    # the longest-prefix CTA dominates; WCA persistent regains ~1.7x.
    _total_pfx_est_pre = (
        total_prefix_len if total_prefix_len is not None
        else kv_indices.numel()
    )
    _is_ragged_pfx = (
        _can_route_wca
        and batch_size >= 4
        and max_len_extend <= 128
        and _total_pfx_est_pre >= batch_size * 2048
        # FP8 D=128 short-extend stays on basic after the small-bucket
        # retune (1.4-1.5x vs Triton) instead of WCA persistent (0.64-
        # 0.66x vs Triton on the same shapes). Genuine pfx-skew would
        # still benefit from WCA but we can't cheaply distinguish it from
        # uniform at dispatch time.
        and not (_kv_is_fp8 and Lq == 128 and max_len_extend <= 64)
    )
    _is_ragged = _is_ragged_ext or _is_ragged_pfx

    if _kv_is_fp8 and not _is_uniform and max_len_extend <= 64:
        _is_ragged = False

    # D=128 B<=4: every WCA clause requires B>=5 so short-circuit.
    # D=256 stays on basic (tuned tree is sufficient and WCA BM=256 has
    # LDS issues).
    _skip_wca_check = (
        Lq == 128
        and not _kv_is_fp8
        and not _is_ragged_pfx
        and batch_size <= 4
    )
    if _is_ragged and Lq <= 128 and not _skip_wca_check:
        _total_ext = _total_extend_rows
        _grid_est = batch_size * max_len_extend
        _waste_frac = 1.0 - _total_ext / max(1, _grid_est)
        _total_pfx_est = _total_pfx_est_pre
        if Lq == 128:
            # D=128: WCA only reliably dominates at B>=5 (basic avoids
            # ~25us of WCA Python-side setup at smaller B).
            _use_wca = (
                _is_ragged_pfx
                or (max_len_extend >= 1024 and _waste_frac > 0.05 and batch_size >= 5)
                or (batch_size >= 8 and _total_pfx_est >= batch_size * 1024)
                or (batch_size >= 8 and max_len_extend >= 768 and _waste_frac >= 0.4)
                or (max_len_extend >= 768 and _waste_frac >= 0.2 and batch_size >= 5)
                or (batch_size >= 16 and _waste_frac >= 0.2)
            )
        else:
            ext_ratio = (
                max_len_extend / max(1, min_len_extend)
                if min_len_extend else float("inf")
            )
            _use_wca = (
                (batch_size >= 8 and _total_pfx_est >= batch_size * 1024
                 and max_len_extend >= 32)
                or (ext_ratio > 4.0 and max_len_extend >= 256)
                or (ext_ratio > 20.0 and max_len_extend >= 64
                    and (batch_size >= 5 or _total_pfx_est >= 512))
            )
        if _use_wca:
            # Launcher picks the BM/NW tile config internally (small-tile for
            # ext-dominated or ragged batches, big-tile for prefix-dominated
            # chat-mix).
            _wca_fn = _launch_persistent_fp8 if _kv_is_fp8 else _launch_persistent
            _wca_fn(
                q_extend, k_extend, v_extend, o_extend,
                k_buffer, v_buffer,
                qo_indptr, kv_indptr, kv_indices,
                custom_mask, is_causal, mask_indptr, max_len_extend,
                k_scale=k_scale, v_scale=v_scale, sm_scale=sm_scale,
                logit_cap=logit_cap,
                skip_prefix_custom_mask=skip_prefix_custom_mask,
                sliding_window_size=sliding_window_size,
                sinks=sinks, window_kv_offsets=window_kv_offsets,
                xai_temperature_len=xai_temperature_len,
                min_len_extend=min_len_extend,
                total_prefix_len=total_prefix_len,
            )
            return

    if _can_route_wca:
        _BLOCK_DMODEL = _resolve_qk_split_dims(Lq)

        if Lq <= 128:
            _BM_est = 128
            _n_m_est = (max_len_extend + _BM_est - 1) // _BM_est
            _total_tiles_est = batch_size * head_num * _n_m_est
            _num_CUs = _get_num_CUs(q_extend.device)
            _total_ext = batch_size * max_len_extend
            _total_pfx = total_prefix_len if total_prefix_len is not None else 0
            _avg_pfx = _total_pfx // max(1, batch_size)
            if _kv_is_fp8:
                if Lq == 128:
                    _need_persistent = (
                        _total_tiles_est < _num_CUs
                        and (
                            (_avg_pfx >= 4096 and max_len_extend <= 128)
                            or (_avg_pfx >= 16384 and max_len_extend <= 256)
                        )
                    )
                elif Lq == 64:
                    _need_persistent = (
                        _total_tiles_est < _num_CUs
                        and (
                            (_avg_pfx >= 4096 and max_len_extend <= 256)
                            or (_avg_pfx >= 16384 and max_len_extend <= 512)
                        )
                    )
                else:
                    _need_persistent = (
                        Lq < 256
                        and _total_tiles_est < _num_CUs
                        and _avg_pfx >= 4096
                        and batch_size * max_len_extend <= 2048
                    )
            else:
                if Lq == 128:
                    _need_persistent = (
                        _avg_pfx >= 4096
                        and (batch_size >= 4 or _avg_pfx >= 16384)
                        and _total_tiles_est < _num_CUs
                    )
                elif Lq == 64:
                    _need_persistent = (
                        _total_tiles_est < _num_CUs
                        and (
                            (max_len_extend >= 128
                             and (batch_size >= 8 or max_len_extend >= 1024))
                            or (_avg_pfx >= 4096
                                and (batch_size >= 4 or _avg_pfx >= 16384))
                        )
                    )
                else:
                    _need_persistent = (
                        _total_tiles_est < _num_CUs
                        and (
                            (max_len_extend >= 128
                             and (batch_size >= 8 or max_len_extend >= 1024))
                            or (_avg_pfx >= 4096
                                and (batch_size >= 4 or _avg_pfx >= 16384))
                        )
                    )
            if _need_persistent and _can_route_wca:
                if min_len_extend is None:
                    # Avoid the 6ms CPU sync that computing min_len_extend
                    # would cost; falling back to max is conservative.
                    min_len_extend = max_len_extend
                # Both launchers auto-select tiles for prefix-dominated
                # chat-mix (BF16 via _use_small_tile+_lq128_ext_dominated,
                # FP8 via _fp8_prefix_dominated).
                _wca_fn = _launch_persistent_fp8 if _kv_is_fp8 else _launch_persistent
                _wca_fn(
                    q_extend, k_extend, v_extend, o_extend,
                    k_buffer, v_buffer,
                    qo_indptr, kv_indptr, kv_indices,
                    custom_mask, is_causal, mask_indptr, max_len_extend,
                    k_scale=k_scale, v_scale=v_scale, sm_scale=sm_scale,
                    logit_cap=logit_cap,
                    sliding_window_size=sliding_window_size,
                    sinks=sinks, window_kv_offsets=window_kv_offsets,
                    xai_temperature_len=xai_temperature_len,
                    min_len_extend=min_len_extend,
                    total_prefix_len=total_prefix_len,
                )
                return

        _sm = (sm_scale if sm_scale is not None else Lq**-0.5) * k_scale
        _kv_gn = head_num // k_extend.shape[1]
        _wkvo = window_kv_offsets
        if _wkvo is None or _dummy_cm is None:
            _ensure_dummies(q_extend.device, q_extend.shape[0] + 1, batch_size)
        if _wkvo is None:
            _wkvo = _dummy_wkvo[:batch_size]

        # D=256 prefix-aware WCA routing. FP8 D=256 persistent overflows
        # LDS (204KB > 160KB) and hits LLVM codegen bugs, so BF16 only.
        if Lq == 256 and not _kv_is_fp8:
            _total_pfx_256 = total_prefix_len if total_prefix_len is not None else 0
            _avg_pfx_256 = _total_pfx_256 // max(1, batch_size)
            _n_m_256 = (max_len_extend + 63) // 64
            _tiles_256 = batch_size * head_num * _n_m_256
            _need_persistent_256 = (
                _avg_pfx_256 >= 4096
                and (batch_size >= 4 or _avg_pfx_256 >= 16384)
                and _tiles_256 < _get_num_CUs(q_extend.device)
            )
            if _need_persistent_256 and _can_route_wca:
                if min_len_extend is None:
                    min_len_extend = max_len_extend
                _wca_fn = _launch_persistent_fp8 if _kv_is_fp8 else _launch_persistent
                _wca_fn(
                    q_extend, k_extend, v_extend, o_extend,
                    k_buffer, v_buffer,
                    qo_indptr, kv_indptr, kv_indices,
                    custom_mask, is_causal, mask_indptr, max_len_extend,
                    k_scale=k_scale, v_scale=v_scale, sm_scale=sm_scale,
                    logit_cap=logit_cap,
                    sliding_window_size=sliding_window_size,
                    sinks=sinks, window_kv_offsets=window_kv_offsets,
                    xai_temperature_len=xai_temperature_len,
                    min_len_extend=min_len_extend,
                    total_prefix_len=total_prefix_len,
                )
                return

        # Cached dispatch config lookup (O(0.1us) after warmup).
        _pfx_b_full = _pfx_bucket(total_prefix_len, batch_size)
        _BM, _BN, _NW, _NS, _PAD_K, _PAD_V, _EXT_BN, _EXT_NS = (
            _get_basic_dispatch_config(
                Lq, batch_size, max_len_extend, _pfx_b_full, _kv_is_fp8,
                sliding_window_size=sliding_window_size,
                head_num=head_num,
            )
        )

        # Cache strides once (avoid 12 separate Python->C++ calls).
        _q_s0, _q_s1 = q_extend.stride(0), q_extend.stride(1)
        _k_s0, _k_s1 = k_extend.stride(0), k_extend.stride(1)
        _v_s0, _v_s1 = v_extend.stride(0), v_extend.stride(1)
        _o_s0, _o_s1 = o_extend.stride(0), o_extend.stride(1)
        _kb_s0, _kb_s1 = k_buffer.stride(0), k_buffer.stride(1)
        _vb_s0, _vb_s1 = v_buffer.stride(0), v_buffer.stride(1)

        _kernel_extra = {
            "EXT_BLOCK_N": _EXT_BN, "EXT_NUM_STAGES": _EXT_NS,
            "SKIP_PREFIX_CUSTOM_MASK": True,
            "ENABLE_MASK_SPLIT": Lq < 256,
            "BLOCK_DV": _BLOCK_DMODEL,
            "ASYNC_PAD_K": _PAD_K, "ASYNC_PAD_V": _PAD_V,
        } if _kv_is_fp8 else {}

        _grid = (batch_size, head_num, (max_len_extend + _BM - 1) // _BM)
        # Both BF16 and FP8 unified kernels take the persistent-superset
        # signature. IS_PERSISTENT=False collapses the outer tile loop to a
        # single pass and DCE eliminates every access to the persistent-only
        # runtime args, so integer zeros for scalars and 1-element module
        # dummies for the pointer workspaces are sufficient. Persistent args
        # go in via kwargs because the signature interleaves constexprs
        # between the stride block and the persistent-only runtime block.
        _kernel_fn[_grid](
            q_extend, k_extend, v_extend, o_extend,
            k_buffer, v_buffer,
            qo_indptr, kv_indptr, kv_indices,
            _dummy_cm, _dummy_mi[: q_extend.shape[0] + 1], _wkvo,
            _sm, _kv_gn,
            _q_s0, _q_s1, _k_s0, _k_s1, _v_s0, _v_s1,
            _o_s0, _o_s1, _kb_s0, _kb_s1, _vb_s0, _vb_s1,
            IS_CAUSAL=is_causal,
            USE_CUSTOM_MASK=False,
            ENABLE_PREFIX_UNMASKED=is_causal,
            BLOCK_M=_BM, BLOCK_N=_BN,
            BLOCK_DMODEL=_BLOCK_DMODEL,
            NUM_STAGES=_NS,
            **_kernel_extra,
            Sinks=sinks, HAS_SINK=sinks is not None,
            LOGIT_CAP=logit_cap,
            XAI_TEMPERATURE_LEN=xai_temperature_len,
            SLIDING_WINDOW_SIZE=sliding_window_size,
            v_scale=v_scale,
            num_heads=0, n_m_tiles=0,
            total_valid_tiles=0, total_programs=0,
            partial_out=_dummy_partial_out, partial_lse=_dummy_partial_lse,
            tile_done=_dummy_tile_done,
            actual_batch_size=0,
            IS_PERSISTENT=False, SPLIT_K=1,
            num_warps=_NW, num_stages=1, waves_per_eu=2,
            matrix_instr_nonkdim=32,
        )
        return

    # Full dispatch path: heterogeneous batches, custom_mask, test overrides.
    BLOCK_DMODEL = _resolve_qk_split_dims(Lq)

    # min_len_extend is only needed below on the D>=256 / persistent / splitk
    # branches. Computing it eagerly would cost a ~15us GPU sync; compute
    # lazily at the point of use.

    _ensure_dummies(q_extend.device, q_extend.shape[0] + 1, batch_size)

    num_CUs = _get_num_CUs(q_extend.device)

    _BM_est = 64 if Lq < 256 else 128
    n_m_tiles_est = (max_len_extend + _BM_est - 1) // _BM_est
    total_tiles_est = batch_size * head_num * n_m_tiles_est

    # Default routing: splitk for D<=128 causal (no custom_mask), basic
    # otherwise. D>=256 always falls through to the basic path below.
    use_splitk = (Lq <= 128) and (custom_mask is None)

    if use_splitk:
        _BN = 32 if BLOCK_DMODEL >= 256 else 64
        if min_len_extend is None:
            min_len_extend = max_len_extend
        _sk_fn = _launch_splitk_fp8 if _kv_is_fp8 else _launch_splitk
        _sk_fn(
            q_extend, k_extend, v_extend, o_extend,
            k_buffer, v_buffer,
            qo_indptr, kv_indptr, kv_indices,
            custom_mask, mask_indptr,
            window_kv_offsets if window_kv_offsets is not None else _dummy_wkvo[:batch_size],
            sm_scale, k_scale, v_scale, logit_cap,
            Lq, Lv, is_causal, max_len_extend, min_len_extend,
            sinks, xai_temperature_len, sliding_window_size,
            BLOCK_M=64,
            BLOCK_N=_BN,
            num_warps=4,
            NUM_STAGES=2,
            total_prefix_len=total_prefix_len,
        )
        return

    enable_prefix_unmasked = is_causal

    USE_CUSTOM_MASK = custom_mask is not None
    if not USE_CUSTOM_MASK:
        custom_mask = _dummy_cm
        mask_indptr = _dummy_mi[: q_extend.shape[0] + 1]
    if window_kv_offsets is None:
        window_kv_offsets = _dummy_wkvo[:batch_size]

    if BLOCK_DMODEL >= 256:
        BLOCK_N = 32
    else:
        BLOCK_N = 64
    if _kv_is_fp8 and BLOCK_DMODEL < 256:
        BLOCK_N = 128
    EXT_BLOCK_N = BLOCK_N
    NUM_STAGES = 1

    if BLOCK_DMODEL >= 256:
        # Caller should pass total_{prefix,extend}_len; fall back to
        # CPU-side proxies instead of a GPU reduction on miss.
        if total_prefix_len is None or total_extend_len is None:
            if total_prefix_len is None:
                total_prefix_len = kv_indices.numel()
            if total_extend_len is None:
                total_extend_len = q_extend.shape[0]
        if min_len_extend is None:
            min_len_extend = max_len_extend
        BLOCK_M, num_warps, NUM_STAGES, _, _ = _select_d256_dispatch(
            batch_size,
            max_len_extend,
            min_len_extend,
            total_prefix_len,
            total_extend_len,
        )
    elif BLOCK_DMODEL == 64:
        _total_ext = batch_size * max_len_extend
        if batch_size >= 16:
            if max_len_extend <= 64:
                BLOCK_M, num_warps = 64, 4
            elif max_len_extend <= 256:
                BLOCK_M, num_warps = 64, 4
            elif max_len_extend <= 512:
                BLOCK_M, num_warps = 256, 4
            else:
                BLOCK_M, num_warps = 256, 8
        elif batch_size >= 4:
            if max_len_extend <= 64:
                BLOCK_M, num_warps = 64, 4
            elif _total_ext >= 2048 or max_len_extend >= 512:
                BLOCK_M, num_warps = 256, 8
            else:
                BLOCK_M, num_warps = 64, 4
        else:
            if max_len_extend >= 2048:
                BLOCK_M, num_warps = 256, 8
            else:
                BLOCK_M, num_warps = 128, 8
    elif BLOCK_DMODEL == 128:
        _total_ext_full = batch_size * max_len_extend
        if batch_size >= 16 and max_len_extend <= 16:
            BLOCK_M, num_warps = 16, 4
        elif batch_size >= 16 and max_len_extend <= 64:
            BLOCK_M, num_warps = 64, 4
        elif batch_size == 1:
            BLOCK_M, num_warps = 64, 4
        elif _total_ext_full >= 32768:
            BLOCK_M, num_warps = 256, 8
        else:
            BLOCK_M, num_warps = 64, 4
    else:
        BLOCK_M = 128
        num_warps = 8

    if BLOCK_DMODEL >= 256:
        pass  # NUM_STAGES already set by _select_d256_dispatch
    elif BLOCK_DMODEL == 64:
        if BLOCK_M == 256 and num_warps == 8:
            NUM_STAGES = 2
        else:
            NUM_STAGES = 4
    elif BLOCK_DMODEL == 128:
        if BLOCK_M >= 256:
            NUM_STAGES = 2
        elif num_warps == 8 and BLOCK_M == 128:
            NUM_STAGES = 4
        elif num_warps == 4:
            NUM_STAGES = 2
        else:
            NUM_STAGES = 2
    elif BLOCK_M == 64:
        NUM_STAGES = 1
    else:
        NUM_STAGES = 4

    # FP8 never reaches this BF16 basic path: the dispatcher either splits
    # (Lq<=128, no custom_mask), routes to _launch_basic_fp8 on custom_mask
    # (via the FP8-specific full path), or raises on D=256 / D<=128+custom_mask.
    EXT_NUM_STAGES = NUM_STAGES

    # Correctness guards (D=64 NW=8 BM=256 is nondeterministic at NS!=2;
    # D>=256 NW=8 NS=1 races on the DMA ring).
    if BLOCK_DMODEL == 64 and num_warps == 8:
        if BLOCK_M in (64, 128) and NUM_STAGES == 1:
            NUM_STAGES = 2
        if BLOCK_M == 256 and NUM_STAGES in (1, 3, 4):
            NUM_STAGES = 2
    if BLOCK_DMODEL >= 256 and num_warps == 8 and NUM_STAGES == 1:
        NUM_STAGES = 2

    sm_scale = sm_scale or 1.0 / math.sqrt(Lq)
    sm_scale = sm_scale * k_scale
    kv_group_num = q_extend.shape[1] // k_extend.shape[1]

    # Cache strides and splitk dummy once.
    _q_s0, _q_s1 = q_extend.stride(0), q_extend.stride(1)
    _k_s0, _k_s1 = k_extend.stride(0), k_extend.stride(1)
    _v_s0, _v_s1 = v_extend.stride(0), v_extend.stride(1)
    _o_s0, _o_s1 = o_extend.stride(0), o_extend.stride(1)
    _kb_s0, _kb_s1 = k_buffer.stride(0), k_buffer.stride(1)
    _vb_s0, _vb_s1 = v_buffer.stride(0), v_buffer.stride(1)

    _kernel_extra_full = {
        "EXT_BLOCK_N": EXT_BLOCK_N, "EXT_NUM_STAGES": EXT_NUM_STAGES,
        "SKIP_PREFIX_CUSTOM_MASK": skip_prefix_custom_mask,
        "ENABLE_MASK_SPLIT": Lq < 256,
        "BLOCK_DV": BLOCK_DMODEL,
        "ASYNC_PAD_K": 16, "ASYNC_PAD_V": 16,
    } if _kv_is_fp8 else {}

    # Both BF16 and FP8 unified kernels take the persistent-superset
    # signature; IS_PERSISTENT=False collapses to the basic 3D-grid body.
    # Persistent runtime args go in as kwargs since the signature
    # interleaves constexprs between the stride block and the persistent-
    # only runtime block.
    _grid_full = (batch_size, head_num, triton.cdiv(max_len_extend, BLOCK_M))
    _kernel_fn[_grid_full](
        q_extend, k_extend, v_extend, o_extend,
        k_buffer, v_buffer,
        qo_indptr, kv_indptr, kv_indices,
        custom_mask, mask_indptr, window_kv_offsets,
        sm_scale, kv_group_num,
        _q_s0, _q_s1, _k_s0, _k_s1, _v_s0, _v_s1,
        _o_s0, _o_s1, _kb_s0, _kb_s1, _vb_s0, _vb_s1,
        IS_CAUSAL=is_causal,
        USE_CUSTOM_MASK=USE_CUSTOM_MASK,
        ENABLE_PREFIX_UNMASKED=enable_prefix_unmasked,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        NUM_STAGES=NUM_STAGES,
        **_kernel_extra_full,
        Sinks=sinks, HAS_SINK=sinks is not None,
        LOGIT_CAP=logit_cap,
        XAI_TEMPERATURE_LEN=xai_temperature_len,
        SLIDING_WINDOW_SIZE=sliding_window_size,
        v_scale=v_scale,
        num_heads=0, n_m_tiles=0,
        total_valid_tiles=0, total_programs=0,
        partial_out=_dummy_partial_out, partial_lse=_dummy_partial_lse,
        tile_done=_dummy_tile_done,
        actual_batch_size=0,
        IS_PERSISTENT=False, SPLIT_K=1,
        num_warps=num_warps, num_stages=1,
        waves_per_eu=2,
        matrix_instr_nonkdim=32,
    )


# FP8 persistent / split-K launchers.


def _launch_persistent_fp8(
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
    skip_prefix_custom_mask=True,
    sliding_window_size=-1,
    sinks=None,
    window_kv_offsets=None,
    xai_temperature_len=-1,
    enable_mask_split=True,
    enable_prefix_unmasked=True,
    min_len_extend=None,
    SPLIT_K=1,
    total_prefix_len=None,
):
    """Launch the FP8 persistent-CTA kernel, managing split-K workspace
    and the dummy mask tensors when ``USE_CUSTOM_MASK`` is False."""
    Lq = q_extend.shape[-1]
    Lv = v_extend.shape[-1]
    BLOCK_DMODEL = _resolve_qk_split_dims(Lq)

    USE_CUSTOM_MASK = custom_mask is not None
    SKIP_PREFIX_CUSTOM_MASK = skip_prefix_custom_mask
    _bs_fp8 = qo_indptr.shape[0] - 1
    if not USE_CUSTOM_MASK:
        custom_mask, mask_indptr, _wkvo_dummy_fp8 = _ensure_dummy_mask_tensors(
            q_extend.device, q_extend.shape[0], _bs_fp8,
        )
        if window_kv_offsets is None:
            window_kv_offsets = _wkvo_dummy_fp8
    elif window_kv_offsets is None:
        _, _, window_kv_offsets = _ensure_dummy_mask_tensors(
            q_extend.device, q_extend.shape[0], _bs_fp8,
        )
    assert q_extend.shape[1] % k_extend.shape[1] == 0

    BLOCK_DV = BLOCK_DMODEL
    batch_size = qo_indptr.shape[0] - 1

    if min_len_extend is None:
        min_len_extend = max_len_extend
    head_num = q_extend.shape[1]

    # Prefix-dominated chat-mix detection (Lq=128 only in practice, since
    # D>=256 FP8 is unsupported): when the per-seq prefix is large enough
    # that each tile iterates a long prefix, BM=64 NW=4 NS=1 BN=128 beats
    # the default BM>=128 NW=8 by ~1.5-2x (no waste on short extends,
    # better LDS/CU utilization per tile).
    _total_pfx_fp8 = total_prefix_len if total_prefix_len is not None else 0
    _avg_pfx_fp8 = _total_pfx_fp8 // max(1, batch_size)
    _fp8_prefix_dominated = (
        Lq == 128
        and _avg_pfx_fp8 >= 4096
        and (batch_size >= 4 or _avg_pfx_fp8 >= 16384)
    )

    if SPLIT_K > 1:
        # Split-K reduction is only correct at the BM=128 NW=8 big-tile
        # config; pre-selected by ``_launch_splitk_fp8`` before it dispatches
        # here with SPLIT_K>1.
        BLOCK_M = 128
        BLOCK_N = 128 if Lq <= 128 else (32 if BLOCK_DMODEL >= 256 else 64)
        num_warps = 8
        NUM_STAGES = _FP8_DEFAULT_NS
    elif _fp8_prefix_dominated:
        BLOCK_M = 64
        BLOCK_N = 128
        num_warps = 4
        NUM_STAGES = 1
    else:
        BLOCK_N = 128 if Lq <= 128 else (32 if BLOCK_DMODEL >= 256 else 64)
        if max(BLOCK_DMODEL, BLOCK_DV) >= 256:
            BLOCK_M = 64
            num_warps = 4
        elif max_len_extend <= 128:
            BLOCK_M = 128
            num_warps = 8
        elif batch_size <= 4:
            BLOCK_M = 128
            num_warps = 8
        elif BLOCK_DMODEL >= 128 and min_len_extend >= 64 and max_len_extend >= 256:
            BLOCK_M = 256
            num_warps = 8
        else:
            BLOCK_M = 128
            num_warps = 8

        if Lq < 128:
            num_warps = min(num_warps, 4)

        NUM_STAGES = _FP8_DEFAULT_NS

    EXT_BLOCK_N = _FP8_DEFAULT_EXT_BN
    EXT_NUM_STAGES = _FP8_DEFAULT_EXT_NS

    ASYNC_PAD_K = 8 if BLOCK_DMODEL >= 256 else 16
    ASYNC_PAD_V = 32 if BLOCK_DV >= 256 else 16

    sm_scale = sm_scale or 1.0 / math.sqrt(Lq)
    sm_scale = sm_scale * k_scale
    kv_group_num = q_extend.shape[1] // k_extend.shape[1]

    device = q_extend.device
    n_m_tiles = (max_len_extend + BLOCK_M - 1) // BLOCK_M

    # Tight sync-free upper bound on sum(ceil(ext_i / BM)). For heterogeneous
    # batches this is 3-4x smaller than ``batch * ceil(max_ext/BM)``; the
    # kernel's inline WCA scan tolerates the over-provisioned slots via a
    # cheap validity check. Mirrors the BF16 ``_launch_persistent`` above.
    total_extend_rows = q_extend.shape[0]
    total_output_tiles = (
        (total_extend_rows + batch_size * (BLOCK_M - 1)) // BLOCK_M
    ) * head_num

    if total_output_tiles == 0:
        return

    num_CUs = _get_num_CUs(device)

    if SPLIT_K <= 1:
        SPLIT_K = 1
        if total_output_tiles < num_CUs:
            if total_prefix_len is not None:
                avg_kv_len = total_prefix_len // max(1, batch_size)
            else:
                avg_kv_len = int((kv_indptr[-1] - kv_indptr[0]).item()) // max(1, batch_size)
            if avg_kv_len >= 4 * BLOCK_N:
                SPLIT_K = _select_k_splits(total_output_tiles, num_CUs)

    if SPLIT_K > 1:
        total_splits = total_output_tiles * SPLIT_K
        partial_out, partial_lse, tile_done = _ensure_splitk_workspace(
            total_splits, total_output_tiles, BLOCK_M, BLOCK_DV, device,
        )
        total_valid_tiles = total_splits
    else:
        partial_out = _ensure_splitk_dummy(device)
        partial_lse = _ensure_splitk_dummy(device)
        tile_done = _ensure_splitk_dummy(device)
        total_valid_tiles = total_output_tiles

    total_programs = _select_persistent_grid(total_valid_tiles, num_CUs)
    grid = (total_programs,)

    _gluon_extend_attn_fwd_symmetric_fp8[grid](
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
        kv_group_num,
        q_extend.stride(0),
        q_extend.stride(1),
        k_extend.stride(0),
        k_extend.stride(1),
        v_extend.stride(0),
        v_extend.stride(1),
        o_extend.stride(0),
        o_extend.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        IS_CAUSAL=is_causal,
        USE_CUSTOM_MASK=USE_CUSTOM_MASK,
        SKIP_PREFIX_CUSTOM_MASK=SKIP_PREFIX_CUSTOM_MASK,
        ENABLE_PREFIX_UNMASKED=enable_prefix_unmasked,
        ENABLE_MASK_SPLIT=enable_mask_split,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DV=BLOCK_DV,
        NUM_STAGES=NUM_STAGES,
        EXT_BLOCK_N=EXT_BLOCK_N,
        EXT_NUM_STAGES=EXT_NUM_STAGES,
        ASYNC_PAD_K=ASYNC_PAD_K,
        ASYNC_PAD_V=ASYNC_PAD_V,
        Sinks=sinks,
        HAS_SINK=sinks is not None,
        LOGIT_CAP=logit_cap,
        XAI_TEMPERATURE_LEN=xai_temperature_len,
        SLIDING_WINDOW_SIZE=sliding_window_size,
        v_scale=1.0 if SPLIT_K > 1 else v_scale,
        num_heads=head_num,
        n_m_tiles=n_m_tiles,
        total_valid_tiles=total_valid_tiles,
        total_programs=total_programs,
        partial_out=partial_out,
        partial_lse=partial_lse,
        tile_done=tile_done,
        actual_batch_size=batch_size,
        IS_PERSISTENT=True,
        SPLIT_K=SPLIT_K,
        num_warps=num_warps,
        num_stages=1,
        waves_per_eu=2,
        matrix_instr_nonkdim=32,
    )


def _launch_splitk_fp8(
    q_extend, k_extend, v_extend, o_extend,
    k_buffer, v_buffer,
    qo_indptr, kv_indptr, kv_indices,
    custom_mask, mask_indptr, window_kv_offsets,
    sm_scale, k_scale, v_scale, logit_cap,
    Lq, Lv, is_causal, max_len_extend, min_len_extend,
    sinks, xai_temperature_len, sliding_window_size,
    BLOCK_M, BLOCK_N, num_warps, NUM_STAGES,
    total_prefix_len=None,
):
    """Pick ``SPLIT_K`` and delegate to ``_launch_persistent_fp8``; falls
    back to SPLIT_K=1 when the tiles-per-CU / pfx-per-BN heuristics
    don't justify a multi-split reduction.

    The pre-selected ``BLOCK_M`` / ``num_warps`` / ``NUM_STAGES`` are
    inputs to the SPLIT_K=1 tile estimate; the persistent launcher
    re-picks tiles for SPLIT_K>1 (it must use BM=128 NW=8 NS=default).
    """
    head_num = q_extend.shape[1]
    device = q_extend.device
    batch_size = qo_indptr.shape[0] - 1

    n_m_tiles = (max_len_extend + BLOCK_M - 1) // BLOCK_M
    total_output_tiles = batch_size * head_num * n_m_tiles
    if total_output_tiles == 0:
        return

    num_CUs = _get_num_CUs(device)
    if total_prefix_len is not None:
        avg_kv_len = total_prefix_len // max(1, batch_size)
    else:
        avg_kv_len = int((kv_indptr[-1] - kv_indptr[0]).item()) // max(1, batch_size)
    need_real_splitk = (
        total_output_tiles < num_CUs and avg_kv_len >= 4 * BLOCK_N
    )

    if not need_real_splitk:
        _launch_persistent_fp8(
            q_extend, k_extend, v_extend, o_extend,
            k_buffer, v_buffer,
            qo_indptr, kv_indptr, kv_indices,
            custom_mask, is_causal, mask_indptr,
            max_len_extend,
            k_scale=k_scale, v_scale=v_scale, sm_scale=sm_scale,
            logit_cap=logit_cap,
            min_len_extend=min_len_extend,
            sinks=sinks,
            xai_temperature_len=xai_temperature_len,
            sliding_window_size=sliding_window_size,
            window_kv_offsets=window_kv_offsets,
            total_prefix_len=total_prefix_len,
        )
        return

    SPLIT_K = _select_k_splits(total_output_tiles, num_CUs)
    enable_prefix_unmasked = is_causal
    enable_mask_split = (custom_mask is None) and is_causal

    _launch_persistent_fp8(
        q_extend, k_extend, v_extend, o_extend,
        k_buffer, v_buffer,
        qo_indptr, kv_indptr, kv_indices,
        custom_mask, is_causal, mask_indptr,
        max_len_extend,
        k_scale=k_scale, v_scale=v_scale, sm_scale=sm_scale,
        logit_cap=logit_cap,
        min_len_extend=min_len_extend,
        sinks=sinks,
        xai_temperature_len=xai_temperature_len,
        sliding_window_size=sliding_window_size,
        window_kv_offsets=window_kv_offsets,
        enable_mask_split=enable_mask_split,
        enable_prefix_unmasked=enable_prefix_unmasked,
        SPLIT_K=SPLIT_K,
        total_prefix_len=total_prefix_len,
    )
