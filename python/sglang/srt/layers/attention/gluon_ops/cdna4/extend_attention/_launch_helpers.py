# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Python-side launch helpers for persistent CTA and split-K attention kernels.

Provides _launch_persistent and _launch_splitk which wrap the unified
gluon_extend_attn_fwd kernel with appropriate tile scheduling and workspace
management.
"""

import math
import os

import torch
import triton

from ._bf16_extend_persistent_gfx950 import (
    gluon_extend_attn_fwd_persistent as _bf16_kernel,
)

_CACHED_ENV_UNIFY_CAUSAL_PATH = None


def _unify_causal_path_env():
    """Single-block causal loop (n_full_blocks=0). Env-cached once per process."""
    global _CACHED_ENV_UNIFY_CAUSAL_PATH
    if _CACHED_ENV_UNIFY_CAUSAL_PATH is None:
        _CACHED_ENV_UNIFY_CAUSAL_PATH = (
            int(os.getenv("SGLANG_GLUON_UNIFY_CAUSAL_PATH", "0")) != 0
        )
    return _CACHED_ENV_UNIFY_CAUSAL_PATH


# ===-----------------------------------------------------------------------===#
# Fast-runner cache for the persistent kernel (shared between _launch_persistent
# and _launch_splitk). Both end up calling `_bf16_kernel` with the same
# constexpr bundle, so a single cache covers the heterogeneous-batch fast path.
#
# Same idea as the `_make_fast_runner` in extend_attention_gfx950.py for the
# basic kernel: skip Triton's ~40us JITFunction.run specialization on cache
# hits by invoking HIPLauncher directly with pre-bound constexprs.
# ===-----------------------------------------------------------------------===#

_persistent_fast_cache = {}


def _make_persistent_fast_runner(compiled_kernel, frozen_kw: dict):
    """Build a direct-HIPLauncher closure for `gluon_extend_attn_fwd_persistent`.

    `frozen_kw` must contain every constexpr in the kernel signature:
        IS_CAUSAL, USE_CUSTOM_MASK, ENABLE_PREFIX_UNMASKED,
        BLOCK_M, BLOCK_N, BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL, NUM_STAGES,
        HAS_SINK, LOGIT_CAP, XAI_TEMPERATURE_LEN, SLIDING_WINDOW_SIZE,
        IS_PERSISTENT, SPLIT_K, MAX_BATCH_LOG2.
    kv_group_num and sm_scale flow in at call time so a single cached runner
    serves multiple GQA configs sharing the same constexpr bundle.
    """
    from triton.runtime import driver as _triton_driver
    from triton import knobs as _triton_knobs

    compiled_kernel._init_handles()
    _hip_launcher = compiled_kernel.run
    _fn_handle = compiled_kernel.function
    _packed_md = compiled_kernel.packed_metadata
    _active = _triton_driver.active
    _get_dev = _active.get_current_device
    _get_stream = _active.get_current_stream
    _enter_hook = _triton_knobs.runtime.launch_enter_hook
    _exit_hook = _triton_knobs.runtime.launch_exit_hook

    IS_CAUSAL = frozen_kw['IS_CAUSAL']
    USE_CUSTOM_MASK = frozen_kw['USE_CUSTOM_MASK']
    ENABLE_PREFIX_UNMASKED = frozen_kw['ENABLE_PREFIX_UNMASKED']
    BLOCK_M = frozen_kw['BLOCK_M']
    BLOCK_N = frozen_kw['BLOCK_N']
    BLOCK_DMODEL = frozen_kw['BLOCK_DMODEL']
    ACTUAL_BLOCK_DMODEL = frozen_kw['ACTUAL_BLOCK_DMODEL']
    NUM_STAGES = frozen_kw['NUM_STAGES']
    HAS_SINK = frozen_kw['HAS_SINK']
    LOGIT_CAP = frozen_kw['LOGIT_CAP']
    XAI_TEMPERATURE_LEN = frozen_kw['XAI_TEMPERATURE_LEN']
    SLIDING_WINDOW_SIZE = frozen_kw['SLIDING_WINDOW_SIZE']
    IS_PERSISTENT = frozen_kw['IS_PERSISTENT']
    SPLIT_K = frozen_kw['SPLIT_K']
    MAX_BATCH_LOG2 = frozen_kw['MAX_BATCH_LOG2']
    TILE_MAP_MODE = frozen_kw['TILE_MAP_MODE']

    def _fast_run(
        q, k, v, o, kb, vb,
        qo_i, kv_i, kv_x,
        mask, mi, wkvo,
        sm_scale, kv_gn, strides,
        sinks, v_scale,
        num_heads, n_m_tiles,
        total_valid_tiles, total_programs,
        partial_out, partial_lse, tile_done,
        cum_tiles_per_batch, actual_batch_size,
        grid,
    ):
        dev = _get_dev()
        stream = _get_stream(dev)
        _hip_launcher(
            grid[0], grid[1] if len(grid) > 1 else 1, grid[2] if len(grid) > 2 else 1,
            stream, _fn_handle, _packed_md, None,
            _enter_hook, _exit_hook,
            q, k, v, o, kb, vb,
            qo_i, kv_i, kv_x,
            mask, mi, wkvo,
            sm_scale, kv_gn,
            strides[0], strides[1], strides[2], strides[3],
            strides[4], strides[5], strides[6], strides[7],
            strides[8], strides[9], strides[10], strides[11],
            IS_CAUSAL, USE_CUSTOM_MASK, ENABLE_PREFIX_UNMASKED,
            BLOCK_M, BLOCK_N, BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL,
            NUM_STAGES,
            sinks, HAS_SINK,
            LOGIT_CAP, XAI_TEMPERATURE_LEN, SLIDING_WINDOW_SIZE,
            v_scale,
            num_heads, n_m_tiles,
            total_valid_tiles, total_programs,
            partial_out, partial_lse, tile_done,
            cum_tiles_per_batch, actual_batch_size,
            IS_PERSISTENT, SPLIT_K, MAX_BATCH_LOG2, TILE_MAP_MODE,
        )

    return _fast_run


def _persistent_cache_key(Lq, frozen_kw: dict, dtypes: tuple = ()) -> tuple:
    """Key that uniquely identifies a compiled persistent/split-K kernel.

    `dtypes` is a tuple of torch.dtype values for the indptr/indices
    tensors whose pointer element types the Triton AOT compile bakes into
    SASS (qo_indptr, kv_indptr, kv_indices). Different dtype combos MUST
    map to separate CompiledKernel entries — otherwise a cached int64
    kernel launched on int32 data reads 8-byte strides from a 4-byte
    buffer and asynchronously faults with HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION.
    See the matching fix in extend_attention_gfx950.py::_try_config_cache.
    """
    return (
        Lq,
        frozen_kw['IS_CAUSAL'],
        frozen_kw['USE_CUSTOM_MASK'],
        frozen_kw['ENABLE_PREFIX_UNMASKED'],
        frozen_kw['BLOCK_M'],
        frozen_kw['BLOCK_N'],
        frozen_kw['BLOCK_DMODEL'],
        frozen_kw['ACTUAL_BLOCK_DMODEL'],
        frozen_kw['NUM_STAGES'],
        frozen_kw['HAS_SINK'],
        frozen_kw['LOGIT_CAP'],
        frozen_kw['XAI_TEMPERATURE_LEN'],
        frozen_kw['SLIDING_WINDOW_SIZE'],
        frozen_kw['IS_PERSISTENT'],
        frozen_kw['SPLIT_K'],
        frozen_kw['MAX_BATCH_LOG2'],
        frozen_kw['TILE_MAP_MODE'],
        frozen_kw['_num_warps'],
        dtypes,
    )


# Default MAX_BATCH_LOG2. 8 -> supports batch_size up to 256, 8 static
# iterations of binary search per tile. Fixed as constexpr so all launches
# share the same compiled kernel.
_DEFAULT_MAX_BATCH_LOG2 = 8


# Module-level override for the tile-map mode. Set from benchmarks / tests
# to route all persistent launches through mode 0 (cum_tiles binary search)
# or mode 1 (WCA inline linear scan) regardless of the per-call setting.
# Respected by _launch_persistent / _launch_splitk when truthy (0 or 1).
_GLOBAL_TILE_MAP_MODE = None


def set_global_tile_map_mode(mode):
    """Override the TILE_MAP_MODE constexpr for all subsequent persistent
    launches. Pass None to restore the per-call default (0 = cum_tiles).
    """
    global _GLOBAL_TILE_MAP_MODE
    _GLOBAL_TILE_MAP_MODE = None if mode is None else int(mode)


def _effective_tile_map_mode(explicit: int) -> int:
    if _GLOBAL_TILE_MAP_MODE is not None:
        return _GLOBAL_TILE_MAP_MODE
    return int(explicit)


# ===-----------------------------------------------------------------------===#
# Utilities
# ===-----------------------------------------------------------------------===#

_QK_SPLIT_CACHE = {}


def _resolve_qk_split_dims(Lq: int):
    """Return (BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL) for symmetric heads."""
    cached = _QK_SPLIT_CACHE.get(Lq)
    if cached is not None:
        return cached
    block_dmodel = max(triton.next_power_of_2(Lq), 16)
    result = (block_dmodel, Lq)
    _QK_SPLIT_CACHE[Lq] = result
    return result


_cached_num_CUs = {}


def _get_num_CUs(device):
    idx = device.index if hasattr(device, 'index') and device.index is not None else 0
    if idx not in _cached_num_CUs:
        _cached_num_CUs[idx] = torch.cuda.get_device_properties(device).multi_processor_count
    return _cached_num_CUs[idx]


# Cached dummy tensors for the no-custom-mask / no-sliding-window case.
# Avoids per-call torch.empty/zeros allocations in the persistent/splitk
# launchers. The mask_indptr dummy is a single zeroed int64 buffer that grows
# monotonically; the kernel reads mask_indptr[batch_id] which is 0 for a
# "no mask" batch, so any prefix of a zeroed buffer is valid.
_dummy_cm_lh = None
_dummy_mi_lh = None
_dummy_wkvo_lh = None
_dummy_mi_cap = 0
_dummy_wkvo_cap = 0


def _ensure_dummy_mask_tensors(device, q_total, batch_size):
    """Return (custom_mask, mask_indptr, window_kv_offsets) dummy tensors
    sized at least to (0, q_total+1, batch_size). Reused across calls."""
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
    """
    if total_output_tiles >= num_CUs:
        return 1
    for sk in (2, 4, 8):
        if total_output_tiles * sk >= num_CUs:
            return sk
    return 8


# ===-----------------------------------------------------------------------===#
# Split-K Workspace Management
# ===-----------------------------------------------------------------------===#

_splitk_dummy = None


def _ensure_splitk_dummy(device):
    global _splitk_dummy
    if _splitk_dummy is None or _splitk_dummy.device != device:
        _splitk_dummy = torch.empty(1, dtype=torch.float32, device=device)
    return _splitk_dummy


_splitk_ws_out = None
_splitk_ws_lse = None
_splitk_ws_done = None

# ===-----------------------------------------------------------------------===#
# Variable-tile-per-batch scheduling helpers
# ===-----------------------------------------------------------------------===#

_cum_tiles_buf = None
_cum_tiles_cap = 0


def _ensure_cum_tiles(qo_indptr, BLOCK_M, num_heads, batch_size):
    """Compute cum_tiles_per_batch = [0, ceil(e0/BM), ceil(e0+e1/BM), ...]
    on GPU without CPU sync. Returns (cum_tiles_tensor, total_output_tiles_tensor).

    The kernel's persistent path binary-searches cum_tiles_per_batch to
    find the (seq, head, block_m) for each tile_idx, enabling dense
    packing of valid tiles across CTAs (no wasted CTA slots on over-
    provisioned n_m_tiles). See _bf16_extend_persistent_gfx950.py lines
    ~155-175 for the kernel-side scheduling logic.

    Note: we store total_output_tiles on device (Python caller passes
    this to the kernel's total_valid_tiles arg after multiplying by
    SPLIT_K if applicable). Using device tensor avoids a sync and is
    acceptable because the kernel reads it once per CTA.
    """
    global _cum_tiles_buf, _cum_tiles_cap
    device = qo_indptr.device
    needed = batch_size + 1
    if (
        _cum_tiles_buf is None
        or _cum_tiles_buf.device != device
        or _cum_tiles_cap < needed
    ):
        _cum_tiles_cap = max(needed, 256)
        _cum_tiles_buf = torch.zeros(_cum_tiles_cap, dtype=torch.int32, device=device)
    cum_tiles = _cum_tiles_buf[:needed]
    ext_lens = (qo_indptr[1:batch_size + 1] - qo_indptr[:batch_size]).to(torch.int32)
    n_tiles_per_batch = (ext_lens + (BLOCK_M - 1)) // BLOCK_M
    cum_tiles[0] = 0
    torch.cumsum(n_tiles_per_batch, 0, dtype=torch.int32, out=cum_tiles[1:])
    return cum_tiles


def _wca_total_seq_tiles(qo_indptr, BLOCK_M, batch_size):
    """Return the exact number of (seq * tile) pairs sum(ceil(ext/BM)).

    Performs ONE CPU sync -- that's the cost of the pure-WCA path (vs
    the cum_tiles path which avoids syncs but pays a per-tile binary
    search). Caller multiplies by num_heads [* SPLIT_K] to get the
    total_valid_tiles scalar passed to the kernel.
    """
    ext_lens = qo_indptr[1:batch_size + 1] - qo_indptr[:batch_size]
    n_tiles = (ext_lens + (BLOCK_M - 1)) // BLOCK_M
    return int(n_tiles.sum().item())


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


# ===-----------------------------------------------------------------------===#
# Persistent Launch
# ===-----------------------------------------------------------------------===#


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
    skip_prefix_custom_mask=True,
    sliding_window_size=-1,
    sinks=None,
    window_kv_offsets=None,
    xai_temperature_len=-1,
    enable_mask_split=True,
    enable_prefix_unmasked=True,
    _force_block_m=None,
    _force_num_warps=None,
    _force_num_stages=None,
    _force_waves_per_eu=None,
    _force_block_n=None,
    _ck_v_preload=False,
    min_len_extend=None,
    total_prefix_len=None,
    _force_tile_map_mode=1,  # WCA inline scan (fast, no CPU sync)
):
    Lq = q_extend.shape[-1]
    Lv = v_extend.shape[-1]
    BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL = _resolve_qk_split_dims(Lq)

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
    if _force_block_n is not None:
        BLOCK_N = _force_block_n
    batch_size = qo_indptr.shape[0] - 1

    if min_len_extend is None:
        # Cheap conservative fallback: treat the batch as uniform (ratio=1).
        # The actual GPU min would cost ~6ms of CPU sync on heterogeneous
        # batches — this is the single largest source of first-call latency
        # when SGLang doesn't thread through min_len_extend. Tile-selection
        # downstream only uses min_len_extend to *shrink* BM for extremely
        # skewed batches (ratio>=2 with max<=256); defaulting to max is
        # conservative (slightly larger BM) but avoids the sync.
        min_len_extend = max_len_extend
    head_num = q_extend.shape[1]

    # Small-tile regime: when per-seq work is small OR the batch is heterogeneous
    # enough that BM=128/256 wastes compute on masked tokens. Basic-path kernel
    # beats persistent with BM=128/256 by 1.5-2x on these shapes; matching BM=64
    # nw=4 ns=2 here closes the gap (confirmed by bench_wca_vs_cum_tiles.py).
    #
    # EXCEPTION: big-B batches with a substantial prefix (multi-turn chat, long-
    # context decode). Each tile iterates its seq's full prefix regardless of
    # BM, so larger BM amortizes that scan over more query rows. Random-perf
    # bench showed BM=128 NW=8 NS=4 is 2-3x faster than BM=64 for
    # big-B-chat shapes (B>=8, max_ext<=256, avg_pfx >= 1k per seq).
    _het_ratio = max_len_extend / max(min_len_extend, 1)
    # Cheap no-sync total-pfx: kv_indices is sized to total_prefix_len by
    # construction in every caller path. Threaded-in value preferred; the
    # numel fallback is correct and costs no CPU sync.
    _total_pfx_est = (
        total_prefix_len if total_prefix_len is not None
        else kv_indices.numel()
    )
    # Exclude spec-decode (max_ext<=8) from this override: for ext=1..8 the
    # BM=64 tile avoids wasting compute on masked rows beyond min_ext, and
    # random-perf bench showed BM=128 regressed spec-decode 20-70% there.
    # The bigger-BM win only holds when there are real query rows to amortize
    # prefix streaming over (max_ext >= 32 comfortably covers chat-mix max
    # ~200-500 which benefits).
    _pfx_dominated_big_b = (
        batch_size >= 8
        and _total_pfx_est >= batch_size * 1024
        and 32 <= max_len_extend <= 512
    )
    _use_small_tile = (
        not _pfx_dominated_big_b
        and (
            max_len_extend <= 128
            or (max_len_extend <= 256 and _het_ratio >= 2.0)
            or (min_len_extend < 64 and max_len_extend <= 512 and batch_size <= 4)
        )
    )

    if _force_block_m is not None and _force_num_warps is not None:
        BLOCK_M = _force_block_m
        num_warps = _force_num_warps
    elif BLOCK_DMODEL >= 256:
        BLOCK_M = 64
        num_warps = 4
    elif _use_small_tile:
        BLOCK_M = 64
        num_warps = 4
    elif batch_size <= 4:
        BLOCK_M = 128
        num_warps = 8
    else:
        # BM=256 was observed to trigger an LLVM iota_range assertion in the
        # persistent kernel at some shapes (e.g. B=32 uniform max=256) in the
        # current Triton snapshot. BM=128 is the largest safe M-tile for the
        # persistent path; it is also the tile size CK uses on equivalent
        # shapes, so we are not leaving meaningful throughput on the table.
        BLOCK_M = 128
        num_warps = 8

    if _force_num_stages is not None:
        NUM_STAGES = _force_num_stages
    elif BLOCK_DMODEL >= 256:
        NUM_STAGES = 1
    elif BLOCK_M == 64 and num_warps == 4:
        NUM_STAGES = 2  # matches basic-path dispatch; ns=1 was a regression here
    elif BLOCK_M == 64:
        NUM_STAGES = 1
    else:
        NUM_STAGES = 4

    if (
        BLOCK_M == 128
        and num_warps == 8
        and NUM_STAGES == 2
        and _force_num_stages is None
    ):
        NUM_STAGES = 3

    sm_scale = sm_scale or 1.0 / math.sqrt(Lq)
    sm_scale = sm_scale * k_scale
    kv_group_num = q_extend.shape[1] // k_extend.shape[1]

    device = q_extend.device
    n_m_tiles = (max_len_extend + BLOCK_M - 1) // BLOCK_M
    # Uniform upper bound -- used for SPLIT_K heuristic and as a fallback
    # grid sizer. For mode 0 (cum_tiles binary search) the kernel honors
    # the exact count via cum_tiles_per_batch, so over-provisioning the
    # grid is safe. For mode 1 (WCA) we decide upper-bound vs exact count
    # based on batch skew: uniform/near-uniform -> upper bound (sync-free),
    # highly-skewed -> exact (pays ~15us CPU sync but avoids many invalid
    # CTAs doing WCA scans to find no work).
    total_output_tiles_upper = batch_size * head_num * n_m_tiles
    if total_output_tiles_upper == 0:
        return

    TILE_MAP_MODE = _effective_tile_map_mode(_force_tile_map_mode)
    if TILE_MAP_MODE == 1:
        # The WCA kernel's inline scan tolerates over-provisioned slots
        # (is_valid_tile=False). For near-uniform batches the upper bound
        # is essentially exact, so skip the 15-30us .item() sync. Only
        # pay the sync when the batch is skewed enough that empty-slot
        # scanning would waste meaningful CU cycles.
        #
        # Uniformity proxy: min_len_extend / max_len_extend. We already
        # computed min_len_extend (defaulted to max when SGLang didn't
        # thread it, so uniform-looking by default). Threshold 0.5 keeps
        # the sync for batches where >=50% of CTAs would scan empty slots.
        _uniformity = min_len_extend / max(1, max_len_extend)
        if _uniformity >= 0.5:
            total_output_tiles = total_output_tiles_upper
            cum_tiles_per_batch = _ensure_splitk_dummy(device)
        else:
            _total_seq_tiles = _wca_total_seq_tiles(qo_indptr, BLOCK_M, batch_size)
            total_output_tiles = _total_seq_tiles * head_num
            cum_tiles_per_batch = _ensure_splitk_dummy(device)
    else:
        total_output_tiles = total_output_tiles_upper
        cum_tiles_per_batch = _ensure_cum_tiles(qo_indptr, BLOCK_M, head_num, batch_size)

    num_CUs = _get_num_CUs(device)

    # Auto-splitk inside the persistent launcher is only correct at BM=128
    # nw=8 NS=4 (the "big tile" configuration used by _launch_splitk). At other
    # tile configs (e.g. BM=64 nw=4 NS=2) the multi-split reduction kernel
    # produces NaNs because the accumulator layout diverges from what
    # `_ensure_splitk_workspace` assumes. Callers who need split-K MUST go
    # through _launch_splitk, which hardcodes the canonical big-tile config.
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
    _frozen_kw = {
        'IS_CAUSAL': is_causal,
        'USE_CUSTOM_MASK': USE_CUSTOM_MASK,
        'ENABLE_PREFIX_UNMASKED': _enable_prefix_unmasked,
        'BLOCK_M': BLOCK_M,
        'BLOCK_N': BLOCK_N,
        'BLOCK_DMODEL': BLOCK_DMODEL,
        'ACTUAL_BLOCK_DMODEL': ACTUAL_BLOCK_DMODEL,
        'NUM_STAGES': NUM_STAGES,
        'HAS_SINK': _has_sink,
        'LOGIT_CAP': logit_cap,
        'XAI_TEMPERATURE_LEN': xai_temperature_len,
        'SLIDING_WINDOW_SIZE': sliding_window_size,
        'IS_PERSISTENT': True,
        'SPLIT_K': SPLIT_K,
        'MAX_BATCH_LOG2': _DEFAULT_MAX_BATCH_LOG2,
        'TILE_MAP_MODE': TILE_MAP_MODE,
        '_num_warps': num_warps,
    }
    _strides = (
        q_extend.stride(0), q_extend.stride(1),
        k_extend.stride(0), k_extend.stride(1),
        v_extend.stride(0), v_extend.stride(1),
        o_extend.stride(0), o_extend.stride(1),
        k_buffer.stride(0), k_buffer.stride(1),
        v_buffer.stride(0), v_buffer.stride(1),
    )
    _dtypes = (qo_indptr.dtype, kv_indptr.dtype, kv_indices.dtype)
    _cache_key = _persistent_cache_key(Lq, _frozen_kw, _dtypes)
    _runner = _persistent_fast_cache.get(_cache_key)

    if _runner is not None:
        _runner(
            q_extend, k_extend, v_extend, o_extend, k_buffer, v_buffer,
            qo_indptr, kv_indptr, kv_indices,
            custom_mask, mask_indptr, window_kv_offsets,
            sm_scale, kv_group_num, _strides,
            sinks, _v_scale_final,
            head_num, n_m_tiles,
            total_valid_tiles, total_programs,
            partial_out, partial_lse, tile_done,
            cum_tiles_per_batch, batch_size,
            grid,
        )
        return

    _compiled = _bf16_kernel.run(
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
        *_strides,
        IS_CAUSAL=is_causal,
        USE_CUSTOM_MASK=USE_CUSTOM_MASK,
        ENABLE_PREFIX_UNMASKED=_enable_prefix_unmasked,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        ACTUAL_BLOCK_DMODEL=ACTUAL_BLOCK_DMODEL,
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
        cum_tiles_per_batch=cum_tiles_per_batch,
        actual_batch_size=batch_size,
        IS_PERSISTENT=True,
        SPLIT_K=SPLIT_K,
        MAX_BATCH_LOG2=_DEFAULT_MAX_BATCH_LOG2,
        TILE_MAP_MODE=TILE_MAP_MODE,
        grid=grid,
        num_warps=num_warps,
        num_stages=1,
        waves_per_eu=_force_waves_per_eu if _force_waves_per_eu is not None else 2,
        matrix_instr_nonkdim=32,
        warmup=False,
    )
    if _cache_key not in _persistent_fast_cache:
        _persistent_fast_cache[_cache_key] = _make_persistent_fast_runner(
            _compiled, _frozen_kw,
        )


# ===-----------------------------------------------------------------------===#
# Split-K Launch
# ===-----------------------------------------------------------------------===#


def _launch_splitk(
    q_extend, k_extend, v_extend, o_extend,
    k_buffer, v_buffer,
    qo_indptr, kv_indptr, kv_indices,
    custom_mask, mask_indptr, window_kv_offsets,
    sm_scale, k_scale, v_scale, logit_cap,
    Lq, Lv, is_causal, max_len_extend, min_len_extend,
    sinks, xai_temperature_len, sliding_window_size,
    BLOCK_M, BLOCK_N, num_warps, NUM_STAGES,
    _force_waves_per_eu=None,
    total_prefix_len=None,
    _force_tile_map_mode=1,  # WCA inline scan (fast, no CPU sync)
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
            _force_tile_map_mode=_force_tile_map_mode,
        )
        return

    BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL = _resolve_qk_split_dims(Lq)

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
    total_output_tiles_upper = batch_size * head_num * n_m_tiles

    TILE_MAP_MODE = _effective_tile_map_mode(_force_tile_map_mode)
    if TILE_MAP_MODE == 1:
        _total_seq_tiles = _wca_total_seq_tiles(qo_indptr, BLOCK_M, batch_size)
        total_output_tiles = _total_seq_tiles * head_num
        cum_tiles_per_batch = _ensure_splitk_dummy(device)
    else:
        total_output_tiles = total_output_tiles_upper
        cum_tiles_per_batch = _ensure_cum_tiles(qo_indptr, BLOCK_M, head_num, batch_size)

    total_splits = total_output_tiles * SPLIT_K
    partial_out, partial_lse, tile_done = _ensure_splitk_workspace(
        total_splits, total_output_tiles, BLOCK_M, BLOCK_DMODEL, device,
    )

    total_valid_tiles = total_output_tiles * SPLIT_K
    total_programs = min(total_valid_tiles, 2 * num_CUs)
    grid = (total_programs,)

    _has_sink = sinks is not None
    _v_scale_final = 1.0 if SPLIT_K > 1 else v_scale
    _frozen_kw = {
        'IS_CAUSAL': is_causal,
        'USE_CUSTOM_MASK': USE_CUSTOM_MASK,
        'ENABLE_PREFIX_UNMASKED': enable_prefix_unmasked,
        'BLOCK_M': BLOCK_M,
        'BLOCK_N': BLOCK_N,
        'BLOCK_DMODEL': BLOCK_DMODEL,
        'ACTUAL_BLOCK_DMODEL': ACTUAL_BLOCK_DMODEL,
        'NUM_STAGES': NUM_STAGES,
        'HAS_SINK': _has_sink,
        'LOGIT_CAP': logit_cap,
        'XAI_TEMPERATURE_LEN': xai_temperature_len,
        'SLIDING_WINDOW_SIZE': sliding_window_size,
        'IS_PERSISTENT': True,
        'SPLIT_K': SPLIT_K,
        'MAX_BATCH_LOG2': _DEFAULT_MAX_BATCH_LOG2,
        'TILE_MAP_MODE': TILE_MAP_MODE,
        '_num_warps': num_warps,
    }
    _strides = (
        q_extend.stride(0), q_extend.stride(1),
        k_extend.stride(0), k_extend.stride(1),
        v_extend.stride(0), v_extend.stride(1),
        o_extend.stride(0), o_extend.stride(1),
        k_buffer.stride(0), k_buffer.stride(1),
        v_buffer.stride(0), v_buffer.stride(1),
    )
    _dtypes = (qo_indptr.dtype, kv_indptr.dtype, kv_indices.dtype)
    _cache_key = _persistent_cache_key(Lq, _frozen_kw, _dtypes)
    _runner = _persistent_fast_cache.get(_cache_key)

    if _runner is not None:
        _runner(
            q_extend, k_extend, v_extend, o_extend, k_buffer, v_buffer,
            qo_indptr, kv_indptr, kv_indices,
            custom_mask, mask_indptr, window_kv_offsets,
            sm_scale, kv_group_num, _strides,
            sinks, _v_scale_final,
            head_num, n_m_tiles,
            total_valid_tiles, total_programs,
            partial_out, partial_lse, tile_done,
            cum_tiles_per_batch, batch_size,
            grid,
        )
        return

    _compiled = _bf16_kernel.run(
        q_extend, k_extend, v_extend, o_extend,
        k_buffer, v_buffer,
        qo_indptr, kv_indptr, kv_indices,
        custom_mask, mask_indptr, window_kv_offsets,
        sm_scale, kv_group_num,
        *_strides,
        IS_CAUSAL=is_causal,
        USE_CUSTOM_MASK=USE_CUSTOM_MASK,
        ENABLE_PREFIX_UNMASKED=enable_prefix_unmasked,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL=ACTUAL_BLOCK_DMODEL,
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
        cum_tiles_per_batch=cum_tiles_per_batch,
        actual_batch_size=batch_size,
        IS_PERSISTENT=True, SPLIT_K=SPLIT_K,
        MAX_BATCH_LOG2=_DEFAULT_MAX_BATCH_LOG2,
        TILE_MAP_MODE=TILE_MAP_MODE,
        grid=grid,
        num_warps=num_warps, num_stages=1,
        waves_per_eu=_force_waves_per_eu if _force_waves_per_eu is not None else 2,
        matrix_instr_nonkdim=32,
        warmup=False,
    )
    if _cache_key not in _persistent_fast_cache:
        _persistent_fast_cache[_cache_key] = _make_persistent_fast_runner(
            _compiled, _frozen_kw,
        )
