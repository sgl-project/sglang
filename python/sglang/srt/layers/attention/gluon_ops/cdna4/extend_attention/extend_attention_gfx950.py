# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon extend-attention dispatch for gfx950 (MI350X / CDNA4).

Symmetric heads only (Lq == Lv): D64, D128, D256.
Mixed-dim / DeepSeek MLA heads (Lq != Lv) are handled by mla_prefill/.

The public entry point is gluon_extend_attention_fwd().

Known tech debt (deferred to fast-follow refactor):
- BLOCK_DPE / ACTUAL_BLOCK_DPE constexpr scaffolding is threaded through
  every kernel call site but is always 0 on this (symmetric) branch.
  It exists so the kernels stay source-compatible with the MLA prefill
  branch where DPE > 0. Removing it here is safe but mechanically large
  (~30 call sites * 4 kernel files + _common.py helpers); do it in a
  dedicated refactor PR after this branch merges.
- _bf16_extend_gfx950.py and _bf16_extend_persistent_gfx950.py share
  >90% of their body and can be collapsed once the persistent-grid
  scheduling is factored out. Same for the two FP8 kernels.
"""

import functools
import logging
import math
import os

import torch
import triton

from ._bf16_extend_gfx950 import (
    gluon_extend_attn_fwd as _gluon_extend_attn_fwd_symmetric,
)
from ._launch_helpers import (
    _select_persistent_grid,
    _launch_persistent,
    _ensure_splitk_dummy,
    _ensure_splitk_workspace,
    _select_k_splits,
    _launch_splitk,
    _get_num_CUs,
    _resolve_qk_split_dims,
    _ensure_dummy_mask_tensors,
)
from ._fp8_kv_extend_symmetric_gfx950 import (
    gluon_extend_attn_fwd as _gluon_extend_attn_fwd_symmetric_fp8,
)
from ._fp8_kv_extend_basic_gfx950 import (
    gluon_extend_attn_fwd_basic as _gluon_extend_attn_fwd_fp8_basic,
)

_gluon_extend_attn_fwd_symmetric_d128_experimental = None
_launch_d128_decode1_wca_experimental = None
try:
    from ._gluon_kernel_symmetric_d128_experimental import (
        gluon_extend_attn_fwd as _gluon_extend_attn_fwd_symmetric_d128_experimental,
        launch_d128_decode1_wca as _launch_d128_decode1_wca_experimental,
    )
except ImportError:
    pass

_dummy_cm = None
_dummy_mi = None
_dummy_mi_size = 0
_dummy_wkvo = None
_dummy_wkvo_size = 0
# Sinks tensor is consumed only when HAS_SINK=True. For HAS_SINK=False we
# still need *some* tensor to pass through the HIPLauncher (arg list is
# fixed; the kernel body never loads from it under HAS_SINK=False). Keep a
# single 1-element bf16 tensor around; dtype matches the common sinks
# dtype for GPT-OSS/Gemma and the kernel body's unused-load branch is DCE'd.
_dummy_sinks = None

_LOGGED_FP8_KV_MODE = False
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config-keyed dispatch cache.
#
# Problem: max_len_extend and batch_size change every scheduling step in
# real serving, so a cache keyed on exact (batch_size, max_len_extend) will
# NEVER hit in production.
#
# Fix: key the cache on the CONFIG OUTPUT of _get_basic_dispatch_config(),
# not the input.  The dispatch logic is a step function with ~6 distinct
# configs per (Lq, is_fp8) pair.  Two calls that produce the same
# (BM, BN, NW, NS, ...) tuple share one cache entry regardless of the
# exact batch_size/max_len_extend that produced it.
#
# Per-call, we recompute only:
#   grid = (batch_size, head_num, ceildiv(max_len_extend, BM))
# which is one integer division (~10ns).
# ---------------------------------------------------------------------------
# config_key -> either:
#   ("fast", fast_run_closure, sm, strides, BM)    # BF16: direct CompiledKernel invoke
#   ("legacy", kernel_fn.run, sm, kv_gn, strides, frozen_kwargs, BM)  # FP8: via JITFunction.run
# The "fast" path bypasses ~55us/call of JITFunction.run specialization overhead
# (see profile_fast_path.py) by capturing the CompiledKernel from the first call
# and invoking it directly via CompiledKernel[grid](...).
_config_cache = {}

# Dispatch-path counters. Incremented on every call so we can prove whether
# the fast-path cache is actually hitting at E2E time vs silently routing
# through WCA / slow first-call JIT / full dispatch. Flip SGLANG_GLUON_TRACE=1
# to get a WARN log every N calls summarising the breakdown; otherwise the
# counters are free.
_dispatch_counters = {
    "total": 0,
    "early_cache_hit": 0,       # uniform_like fast-path hit
    "wca_persistent": 0,        # ragged -> WCA persistent
    "wca_persistent_d256": 0,   # D=256 WCA
    "late_cache_hit": 0,        # het/ragged late cache hit (non uniform_like)
    "basic_slow_first": 0,      # first-time JIT + install
    "full_dispatch": 0,         # custom mask / force_* / full path
    "wrapper_fallback": 0,      # wrapper-level triton_fallback (outside this fn)
}


def _bump_dispatch(key):
    _dispatch_counters[key] = _dispatch_counters.get(key, 0) + 1
    _dispatch_counters["total"] += 1
    _maybe_trace_dispatch()


_CACHED_ENV_TRACE = None
_TRACE_LAST_TOTAL = 0


def _trace_enabled():
    global _CACHED_ENV_TRACE
    if _CACHED_ENV_TRACE is None:
        _CACHED_ENV_TRACE = int(os.getenv("SGLANG_GLUON_TRACE", "0"))
    return _CACHED_ENV_TRACE


def _maybe_trace_dispatch():
    interval = _trace_enabled()
    if interval <= 0:
        return
    global _TRACE_LAST_TOTAL
    total = _dispatch_counters["total"]
    if total - _TRACE_LAST_TOTAL < interval:
        return
    _TRACE_LAST_TOTAL = total
    parts = [f"total={total}"]
    for k, v in _dispatch_counters.items():
        if k == "total" or v == 0:
            continue
        parts.append(f"{k}={v}")
    logger.warning("gluon extend dispatch: " + " ".join(parts))


def get_dispatch_counters():
    """Snapshot of per-path dispatch counts. Useful for tests/benches that
    want to assert the fast path is hitting vs silently falling back."""
    return dict(_dispatch_counters)


def reset_dispatch_counters():
    for k in list(_dispatch_counters):
        _dispatch_counters[k] = 0
    global _TRACE_LAST_TOTAL
    _TRACE_LAST_TOTAL = 0


def _ensure_dummies(device, mi_size, wkvo_size):
    """Lazy-init module-level singleton dummy tensors on first use."""
    global _dummy_cm, _dummy_mi, _dummy_mi_size, _dummy_wkvo, _dummy_wkvo_size
    global _dummy_sinks
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


_CACHED_ENV_FP8_KV_FORCE_BF16 = None
_CACHED_ENV_D128_EXPERIMENTAL = None
_CACHED_ENV_UNIFY_CAUSAL_PATH = None
_CACHED_ENV_DISABLE_CFG_CACHE = None


def _cfg_cache_disabled():
    """Debug escape hatch: bypass the _config_cache fast path entirely.

    When SGLANG_GLUON_DISABLE_CFG_CACHE=1, every extend call goes through
    the full dispatch (JITFunction.run) instead of the cached CompiledKernel
    direct-invoke path. Used to isolate whether fast-path caching is
    responsible for production crashes / correctness issues.
    """
    global _CACHED_ENV_DISABLE_CFG_CACHE
    if _CACHED_ENV_DISABLE_CFG_CACHE is None:
        _CACHED_ENV_DISABLE_CFG_CACHE = (
            int(os.getenv("SGLANG_GLUON_DISABLE_CFG_CACHE", "0")) != 0
        )
    return _CACHED_ENV_DISABLE_CFG_CACHE


def _use_fp8_kv_bf16_bridge():
    """Whether to force fp8 KV -> bf16-kernel bridge path.

    When enabled, fp8 prefix KV cache tensors are cast to bf16 and dispatched
    through the bf16 Gluon kernels. This is intended as a global fallback /
    perf-regression escape hatch during fp8 rollout.
    """
    global _CACHED_ENV_FP8_KV_FORCE_BF16
    if _CACHED_ENV_FP8_KV_FORCE_BF16 is None:
        _CACHED_ENV_FP8_KV_FORCE_BF16 = (
            int(os.getenv("SGLANG_GLUON_FP8_KV_FORCE_BF16", "0")) != 0
        )
    return _CACHED_ENV_FP8_KV_FORCE_BF16


def _use_d128_experimental_kernel():
    """Whether to route D128 symmetric path to the experimental kernel fork."""
    if _gluon_extend_attn_fwd_symmetric_d128_experimental is None:
        return False
    global _CACHED_ENV_D128_EXPERIMENTAL
    if _CACHED_ENV_D128_EXPERIMENTAL is None:
        _CACHED_ENV_D128_EXPERIMENTAL = (
            int(os.getenv("SGLANG_GLUON_D128_EXPERIMENTAL", "0")) != 0
        )
    return _CACHED_ENV_D128_EXPERIMENTAL


def _use_unify_causal_path():
    """Whether to force the causal masked path to process all blocks in a
    single pipelined loop (n_full_blocks=0).

    The split full/masked path pays a pipeline drain + barrier cost when
    transitioning between the unmasked and masked sections. For causal
    attention, processing everything as masked (mask is a no-op below the
    diagonal) lets the pipeline run uninterrupted. Matches the CK-style
    experimental kernel's approach.
    """
    global _CACHED_ENV_UNIFY_CAUSAL_PATH
    if _CACHED_ENV_UNIFY_CAUSAL_PATH is None:
        _CACHED_ENV_UNIFY_CAUSAL_PATH = (
            int(os.getenv("SGLANG_GLUON_UNIFY_CAUSAL_PATH", "0")) != 0
        )
    return _CACHED_ENV_UNIFY_CAUSAL_PATH


_cached_num_xcds = {}


def _make_fast_runner(compiled_kernel, frozen_kw: dict, kv_gn: int):
    """Build a closure that bypasses Triton's JITFunction.run specialization
    (~55 us/call, see profile_fast_path.py) AND `compiled_kernel[grid]` closure
    indirection by invoking the HIPLauncher directly.

    Once we know which dispatch-config applies, all constexprs are fixed.
    The only things that change per call are the 12 tensors, `sm_scale`,
    the 12 stride ints, and the 3-int grid. Everything else is baked into
    the closure.

    Call chain is now: `gluon_extend_attention_fwd` -> `_fast_run` ->
    `HIPLauncher.__call__` -> `hip_utils.launch`. We skip Triton's
    `JITFunction.run` and the intermediate `runner` closure from
    `CompiledKernel.__getitem__`.

    Signature for the returned callable:
        fast_run(q, k, v, o, kb, vb, qo_i, kv_i, kv_x,
                 mask, mi, wkvo, sm_scale, strides_tuple, grid)

    Assumes `knobs.runtime.launch_enter_hook` / `launch_exit_hook` are None
    at closure construction time (the common case with no profiler). If hooks
    get installed later, the cached closure won't see them; callers that need
    profiler integration should clear `_config_cache`.
    """
    from triton.runtime import driver as _triton_driver  # local import: keep cold-path overhead out of top-level imports
    from triton import knobs as _triton_knobs

    compiled_kernel._init_handles()
    _hip_launcher = compiled_kernel.run  # returns HIPLauncher.__call__ (memoized)
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
    # Sinks tensor is a RUNTIME arg (pointer differs per layer for GPT-OSS).
    # HAS_SINK is a compile-time constant baked into the specialized kernel,
    # so swapping the Sinks pointer across calls is safe as long as the
    # cache key discriminates HAS_SINK=True vs HAS_SINK=False (it does).
    HAS_SINK = frozen_kw['HAS_SINK']
    LOGIT_CAP = frozen_kw['LOGIT_CAP']
    XAI_TEMPERATURE_LEN = frozen_kw['XAI_TEMPERATURE_LEN']
    SLIDING_WINDOW_SIZE = frozen_kw['SLIDING_WINDOW_SIZE']
    v_scale = frozen_kw['v_scale']

    def _fast_run(q, k, v, o, kb, vb, qo_i, kv_i, kv_x,
                  mask, mi, wkvo, sinks, sm_scale, strides, grid):
        dev = _get_dev()
        stream = _get_stream(dev)
        _hip_launcher(
            grid[0], grid[1], grid[2], stream, _fn_handle, _packed_md, None,
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
        )

    return _fast_run


def _get_num_xcds(device):
    idx = device.index if hasattr(device, 'index') and device.index is not None else 0
    if idx not in _cached_num_xcds:
        num_CUs = torch.cuda.get_device_properties(device).multi_processor_count
        # MI350X: 256 CUs / 8 XCDs = 32 CUs/XCD
        # MI300X: 304 CUs / 8 XCDs = 38 CUs/XCD
        _cached_num_xcds[idx] = max(1, num_CUs // 38)
    return _cached_num_xcds[idx]


# _resolve_qk_split_dims is imported from _launch_helpers


def _select_d256_dispatch(
    batch_size: int,
    max_len_extend: int,
    min_len_extend: int,
    total_prefix_len: int,
    total_extend_len: int,
    avg_pfx_proxy: int = 0,
):
    """D=256 launch policy — prior tree with Apr 2026 oracle overrides.

    Overrides vs prior tree:
      - B>=4 ext>=256:           BM128 BN32 NW8 NS1  (oracle +31-43% over NS=3)
      - B=1 ext in [2048,4096):  BM128 BN32 NW8 NS1  (oracle +6-13%)
      - B=1 ext>=4096 small pfx: BM128 BN32 NW8 NS2  (v3: pipeline hides
                                                      long-ext latency better
                                                      when there's no prefix
                                                      to amortize it over)
      - B=1 ext>=4096 large pfx: BM128 BN32 NW8 NS1  (prefix-heavy; NS=1 fine)

    Everything else keeps the prior selector's choice. The prior tree
    was ignoring `total_prefix_len` (always called with 0) in basic
    dispatch, which meant its `prefix_frac` branch fired for all
    ext<768 shapes; retaining that behavior avoids regressing shapes
    the oracle did not sample.

    `avg_pfx_proxy` is an approximation of avg prefix length supplied by
    the caller when the raw `total_prefix_len` is not threaded through.
    """
    total_tokens = max(1, total_prefix_len + total_extend_len)
    prefix_frac = total_prefix_len / total_tokens
    ext_ratio = max_len_extend / max(1, min_len_extend)
    avg_pfx = total_prefix_len // max(1, batch_size)
    avg_pfx_hint = max(avg_pfx, avg_pfx_proxy)

    # Oracle-proven overrides (take precedence, narrowly scoped).
    if batch_size >= 4 and max_len_extend >= 256:
        return 128, 8, 1, 16, 16
    if batch_size == 1 and max_len_extend >= 4096:
        # Long-extend B=1: NS choice depends on prefix.
        # Small prefix -> NS=2 (pipeline hides long ext-loop latency).
        # Large prefix -> NS=1 (prefix work dominates, NS=1 is fine).
        if avg_pfx_hint < 2048:
            return 128, 8, 2, 16, 16
        return 128, 8, 1, 16, 16
    if batch_size == 1 and max_len_extend >= 2048:
        return 128, 8, 1, 16, 16

    # Prior tree (unchanged logic, for shapes outside oracle domain).
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


# ===-----------------------------------------------------------------------===#
# Cached dispatch config (AITER-style per-shape lookup)
# ===-----------------------------------------------------------------------===#

_CACHED_FP8_ENV = None


def _get_fp8_env():
    """Cache FP8 tuning env reads once per process."""
    global _CACHED_FP8_ENV
    if _CACHED_FP8_ENV is None:
        _CACHED_FP8_ENV = {
            'bn': int(os.environ.get('_GLUON_FP8_BN', '128')),
            'ns': int(os.environ.get('_GLUON_FP8_NS', '2')),
            'ext_bn': int(os.environ.get('_GLUON_FP8_EXT_BN', '64')),
            'ext_ns': int(os.environ.get('_GLUON_FP8_EXT_NS', '3')),
        }
    return _CACHED_FP8_ENV


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


@functools.lru_cache(maxsize=2048)
def _get_basic_dispatch_config(
    Lq, batch_size, max_len_extend, pfx_bucket, is_fp8, sliding_window_size=-1,
):
    """Return (BLOCK_M, BLOCK_N, num_warps, NUM_STAGES, PAD_K, PAD_V,
              EXT_BLOCK_N, EXT_NUM_STAGES) for the basic (non-persistent) path.

    Cached per shape signature. pfx_bucket buckets avg prefix length (see
    _pfx_bucket) so the cache stays small while exposing the prefix-vs-no-
    prefix distinction the oracle identified as first-order.

    sliding_window_size lets us pick configs tuned for SWA shapes. When SWA
    is active with sw < max_len_extend, each tile's useful KV range is at
    most ~(BM + sw) columns, so a smaller BM + fewer warps tends to beat the
    large-BM configs we'd pick for plain prefill at the same seqlen.
    """
    if Lq == 256:
        _BN = 32
        _total_ext = batch_size * max_len_extend
        # Mirror prior behavior: pass 0 as total_prefix_len so the selector's
        # fallback tree sees prefix_frac=0 (same as pre-retune code). The
        # narrow oracle-proven overrides inside _select_d256_dispatch fire
        # before that fallback and do not depend on total_prefix_len.
        #
        # Supply a coarse avg-prefix proxy derived from pfx_bucket so the
        # v3 B=1 ext>=4096 small-vs-large-prefix split can take effect.
        # Thresholds mirror _pfx_bucket: bucket<=1 -> avg<512 -> proxy 0;
        # bucket==2 -> avg in [512,2048) -> proxy ~1024 (< 2048 cutoff);
        # bucket>=3 -> avg>=2048 -> proxy 2048+.
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
    elif Lq == 64:
        # D=64 BF16 dispatch (prior tree + narrow oracle-proven overrides).
        #
        # Override: B>=8 ext<=128 with prefix -> BM128 BN128 NW4 NS4
        #           (oracle: +42-50% over prior).
        # All other shapes fall through to the prior tree, which the
        # Apr 2026 sweep confirmed is competitive (many shapes already
        # beat CK on the prior tree).
        _PAD_K, _PAD_V = 16, 16
        _total_ext = batch_size * max_len_extend
        _BN = 64
        if batch_size >= 8 and max_len_extend <= 128 and pfx_bucket >= 2:
            _BM, _BN, _NW, _NS = 128, 128, 4, 4
        elif batch_size >= 16 and max_len_extend <= 32:
            _BM, _NW, _NS = 64, 4, 4
        elif batch_size >= 16:
            _BM, _NW, _NS = (256, 8, 2) if max_len_extend >= 512 else (64, 4, 4)
        elif batch_size >= 4:
            if _total_ext >= 2048 or max_len_extend >= 512:
                _BM, _NW, _NS = 256, 8, 2
            elif batch_size <= 7:
                _BM, _NW, _NS = 128, 8, 4
            else:
                _BM, _NW, _NS = 64, 4, 4
        else:
            _BM, _NW, _NS = (256, 8, 2) if max_len_extend >= 2048 else (128, 8, 4)

        # D=64 BF16 full-attention tune: B=16 with max_ext=256 misdispatches
        # to BM=64_4_4 (tile count 16*256/64 = 64 tiles -> undersubscribes
        # 256 CUs by 4x). Verified by /tmp/tune_losing_b16.py:
        #   B=16 p=1024 ext=256: 256_8_2 @ 0.114ms vs default 64_4_4 @ 0.190ms
        #                        (1.67x vs 1.40x CK -> switch to 1.19x WIN)
        #   B=16 p=2048 ext=256: 256_8_2 @ 0.165ms vs default 64_4_4 @ 0.349ms
        #                        (2.12x vs 1.52x CK -> switch to 1.39x WIN)
        # The fix: route B>=16 ext==256 (full) to BM=256 NW=8 NS=2 directly.
        # B=8 ext==256 already picks BM=256 NW=8 NS=2 via the base tree.
        if (
            sliding_window_size <= 0
            and batch_size >= 16 and max_len_extend == 256
        ):
            _BM, _BN, _NW, _NS = 256, 64, 8, 2
            _EXT_BN, _EXT_NS = 64, 2

        # D=64 BF16 full-attention tune: NS=4 beats NS=2 on BM=256 across
        # the prefill shape grid. Verified by /tmp/bench_full_long.py and
        # /tmp/bench_fa_big_batch.py:
        #   B=1 L=4096: NS=4 saves 3.3%, L=8192: 4.9%, L=16384: tied
        #   B=2 L=4096: NS=4 saves 8.5%, L=16384: saves 0.8%
        #   B=4 L=4096: NS=4 saves 5.3%, L=16384: NS=2 wins 0.7%
        #   B=8 L=1024: NS=4 saves 9%, L=8192: saves 1.5%
        #   B=16 L=1024: NS=4 saves 7%, L=2048: 4%, L=4096: 2.5%
        #
        # The cross-over is governed by per-tile pipeline depth (max_ext)
        # AND total-ext (CU occupancy pressure):
        #   - per-seq ext <= 8K: always NS=4 wins.
        #   - per-seq ext = 16K: NS=4 wins at B<=2, ties at B=1, loses at
        #     B>=4 because the grid already saturates all 256 CUs.
        # Gate on (max_ext <= 8192) OR (total_ext <= 32768) so we only
        # apply NS=4 when tile occupancy has room for the extra stage.
        # Require max_ext >= 1024 so small-extend spec-decode shapes
        # don't pay for the extra pipeline startup.
        if (
            _BM == 256 and _NW == 8 and _NS == 2
            and max_len_extend >= 1024
            and (max_len_extend <= 8192 or _total_ext <= 32768)
        ):
            _NS = 4
    else:
        # D=128 BF16 dispatch (prior tree + narrow oracle-proven overrides).
        #
        # Overrides (all vs prior dispatch BM=64 NW=4 NS=2):
        #   B>=2 ext>=2048 pfx>=moderate    -> BM128 NW8 NS1  (+10-15%)
        #   B=1  ext>=2048 pfx>=moderate    -> BM128 NW8 NS1  (+10-17%)
        #   B=1  ext>=4096 pfx>=large       -> BM128 NW8 NS2  (+11%)
        #
        # Note: the oracle's decode-like override (B>=8 ext<=128 NS=4) did
        # not generalize — in-the-wild bench shows NS=4 is a wash or regress
        # vs the NS=2 default at those shapes, likely because the oracle's
        # "default" baseline picked a slower config. We keep the prior tree
        # for small-extend high-batch shapes.
        _BN = 64
        _PAD_K, _PAD_V = 16, 16
        _total_ext = batch_size * max_len_extend
        if batch_size >= 16 and max_len_extend <= 16:
            _BM, _NW, _NS = 16, 4, 2
        elif batch_size >= 16 and max_len_extend <= 64:
            _BM, _NW, _NS = 64, 4, 2
        elif pfx_bucket >= 3 and max_len_extend >= 4096:
            _BM, _NW, _NS = 128, 8, 2
        elif pfx_bucket >= 2 and max_len_extend >= 2048:
            # NOTE: NS=1 here is bit-nondeterministic on the 8-warp DMA path
            # (attn_fwd_inner_prefix_pipelined). For NS=1 the prefix pipeline's
            # stage_idx collapses to 0, so the warp-pipeline-stage read of
            # v_smem[0] (memory0 cluster) and the DMA write back into v_smem[0]
            # (memory1 cluster) can rotate across iterations at the ring-pipeline
            # boundary and race. NS=2 keeps the ring two-deep, restores
            # determinism, and benches within +-3% of NS=1 on D=128 BF16 prefix
            # (E=2048,4096 x pfx=512..4096 sweep, 100-run median):
            #   E=2048 pfx=1024: NS=1 135us  NS=2 137us  (-1.0%)
            #   E=4096 pfx=1024: NS=1 424us  NS=2 385us  (+10.1%)
            #   B=8 E=2048 pfx=1024: NS=1 1017us NS=2 1047us (-2.8%)
            _BM, _NW, _NS = 128, 8, 2
        elif batch_size == 1 and max_len_extend <= 256:
            _BM, _NW, _NS = 64, 4, 2
        elif batch_size == 1:
            _BM, _NW, _NS = 64, 4, 2
        elif batch_size <= 4:
            _BM, _NW, _NS = 64, 4, 2
        elif _total_ext >= 32768:
            _BM, _NW, _NS = 256, 8, 2
        else:
            _BM, _NW, _NS = 64, 4, 2

    if is_fp8:
        _PAD_K, _PAD_V = 16, 16
        _total_ext = batch_size * max_len_extend
        if Lq == 128:
            # D=128 FP8 basic: NS=2 is mandatory for determinism on the
            # 8-warp DMA path. NS=1 is bit-nondeterministic because the
            # warp_pipeline_stage clusters in attn_fwd_inner_extend_pipelined
            # / attn_fwd_inner_prefix_pipelined rotate across iterations and
            # with a single physical buffer (NS=1), iter N's DMA write to
            # smem[0] races with iter N+1's relaxed read from smem[0].
            # `membarFilter` in the backend explicitly skips barriers between
            # BufferLoadToLocalOp + LocalLoadOp marked syncedViaAsyncWait,
            # so the compiler never inserts a barrier at the cluster
            # boundary. NS=2 gives the pipeliner two slots to rotate
            # through, which is the minimum that avoids the race.
            #
            # Perf: NS=2 is within noise (0-6%) of NS=1 across the whole
            # extend/prefix grid. Verified via /tmp/fp8_ns_sweep.py with
            # properly-plumbed _force_num_stages:
            #   B=1 E=2048 pfx=0:   NS=1 0.157ms / NS=2 0.157ms (1.00x)
            #   B=1 E=8192 pfx=0:   NS=1 1.19ms  / NS=2 1.25ms  (1.05x)
            #   B=4 E=2048 pfx=0:   NS=1 0.39ms  / NS=2 0.41ms  (1.05x)
            #   B=8 E=1024 pfx=0:   NS=1 0.26ms  / NS=2 0.27ms  (1.04x)
            # Prior tune notes claiming NS=1 is 2x faster were invalid:
            # _force_num_stages wasn't plumbed to the basic path, so the
            # comparison was NS=1 vs NS=1.
            # NS>=3 OOMs LDS (per-stage K+V is ~64KB, NS=3 would need ~192KB > 160KB/CU).
            _BM, _BN, _NW, _NS = 128, 128, 8, 2
            _EXT_BN, _EXT_NS = 128, 2
            if batch_size == 1 and max_len_extend <= 256:
                _BM = 64
        elif Lq == 64:
            # D=64 FP8 basic only supports NW=4 NS=1 (the 4-warp DMA
            # path assumes BLOCK_DMODEL >= 128 for its layouts; the
            # 8-warp path is likewise incomplete for D<128). For
            # pipelined / multi-stage D=64 FP8, callers should route
            # to the persistent (symmetric) kernel which gates its
            # DMA path on BLOCK_DMODEL >= 128 and has a separate
            # D<128 layout branch. See _fp8_kv_extend_symmetric_gfx950.py
            # lines 277, 1527 for the working persistent path.
            #
            # BM tune (Apr 2026 /tmp/tune_fp8_extend.py): BM=128 NW=4 NS=1
            # is consistently best across all extend lengths. BM=256
            # is 2-3x slower due to 4-warp register pressure blowing up
            # when DQ registers span 256 rows. Keep BM=128 throughout.
            #   B=1 E=2048:  BM=128 @ 0.068ms vs BM=256 @ 0.250ms (3.68x)
            #   B=1 E=8192:  BM=128 @ 0.613ms vs BM=256 @ 1.987ms (3.24x)
            #   B=1 E=32768: BM=128 @ 9.160ms vs BM=256 @ 26.527ms (2.90x)
            _BM, _BN, _NW, _NS = 128, 128, 4, 1
            _EXT_BN, _EXT_NS = 128, 1
            if batch_size >= 16 and max_len_extend <= 128:
                # Small-extend high-batch: BM=64 spreads CTAs better
                _BM = 64
        else:
            fp8 = _get_fp8_env()
            _BN = 32 if Lq >= 256 else fp8['bn']
            _NS = fp8['ns']
            _NW = min(_NW, 4) if Lq < 128 else _NW
            _EXT_BN = fp8['ext_bn']
            _EXT_NS = fp8['ext_ns']
            if Lq >= 256 and _NW <= 4:
                _EXT_NS = 1 if _total_ext >= 512 else 2
    else:
        _EXT_BN = _BN
        _EXT_NS = _NS

    # SWA-aware BM override for BF16 D=64. When SWA is effective (sw < ext),
    # the causal+SWA band per query tile is ~(BM + sw) keys wide, regardless
    # of how many tiles you make. The default dispatch picks BM=256 for big
    # batches (B>=4) because tile-count per CU is the binding constraint for
    # plain causal, but for SWA the binding constraint is different: smaller
    # BM splits the (static) window work across more CTAs, hiding memory
    # latency better on MI350.
    #
    # Strategy (verified apples-to-apples with /tmp/bench_swa_patched.py on
    # GPT-OSS per-rank D=64 H=32 kvH=4 sw=127 across all 9 E2E shapes):
    #   - ext <= 2*sw: SWA barely binds; keep base config.
    #   - 2*sw < ext <= 16*sw: cap BM to 128, bump NS to 4. At B=4 L=1024
    #     this recovers ~7% over default (BM=128_8_4 @ 41us vs
    #     BM=128_8_2 @ 44us). At B=1,2 with BM already 128 this is a no-op.
    #   - ext > 16*sw (very long seqs, e.g. 2048+ with sw=127): cap BM to
    #     128 AND switch to NW=4 NS=2 (short-per-tile pipeline). At B=4
    #     L=4096 this recovers 2.5% (150us vs 154us default).
    #
    # Empirically BM=128_4_2 beats BM=64_4_2 for this whole range by
    # 0-2.5% (BM=64 doubles tile count but each tile is too cheap to
    # amortize the scheduler overhead on MI350).
    if (
        not is_fp8
        and Lq == 64
        and sliding_window_size > 0
        and sliding_window_size < max_len_extend
    ):
        # Long-extend SWA: 64_2_2 is a consistent 20-35% win over the
        # default BM=128 configs across the whole (B, L) grid. Verified
        # by /tmp/tune_pf_swa.py apples-to-apples (same monkey-patched
        # fast-path so kernel-only time, not Python dispatch):
        #   pf B=1 L=2048-8192: 17-25% gain
        #   pf B=2 L=1024-8192: 17-31% gain
        #   pf B=4 L=1024-8192: 16-33% gain
        #   pf B=8 L=1024-8192: 26-34% gain
        #   pf B=16 L=1024-4096: 31-36% gain
        # Exception: pf_B1_L1024 loses 1us (5%) switching to 64_2_2
        # because total_ext=1024 can't saturate 64-tile grid enough.
        # Gate: total_ext >= 2048 AND max_len_extend >= 1024.
        #
        # Theoretical reason: SWA causal band per query is ~sw + BM
        # keys wide. For sw=128 BM=64: ~192 KV per tile (vs 256 for
        # BM=128). Smaller BM doubles tile count and lets the
        # scheduler hide KV load latency across more in-flight waves.
        # NW=2 NS=2 is the minimum pipeline that still covers memory
        # latency; fewer warps means more concurrent WG's per CU.
        _total_ext_swa = batch_size * max_len_extend
        if (
            max_len_extend >= 1024
            and _total_ext_swa >= 2048
        ):
            _BM, _BN, _NW, _NS = 64, 64, 2, 2
            _EXT_BN, _EXT_NS = _BN, _NS
        elif max_len_extend > 16 * sliding_window_size:
            if _BM > 128:
                _BM = 128
            _NW, _NS = 4, 2
            _EXT_NS = _NS
        elif _BM > 128:
            # ext <= 16*sw: keep NW, raise NS to 4 to match the BM=128
            # config the base dispatch would have picked at smaller B.
            _BM, _NS = 128, 4
            # Keep extend-phase NS aligned with prefix-phase NS, since
            # during initial prefill (the dominant E2E case) the extend
            # loop is where all the SWA work happens.
            _EXT_NS = _NS

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
    _force_block_m=None,
    _force_num_warps=None,
    _force_num_stages=None,
    _force_waves_per_eu=None,
    _force_use_persistent=None,
    _force_use_splitk=None,
    _force_block_n=None,
    _ck_v_preload=False,
    _mask_split_ext_threshold=1024,
    min_len_extend=None,
    total_prefix_len=None,
    total_extend_len=None,
    use_rfidx_prefix=False,
):
    global _LOGGED_FP8_KV_MODE
    Lq = q_extend.shape[-1]
    Lv = v_extend.shape[-1]
    _kv_is_fp8 = k_buffer.dtype in (torch.float8_e4m3fnuz, torch.float8_e4m3fn)
    _kv_was_fp8 = _kv_is_fp8
    # gfx950 MFMA expects OCP FP8 (float8_e4m3fn). The kernel's do_mma()
    # bit-casts between FP8 flavors, which silently doubles values if the
    # source is FNUZ (bias 8) but MFMA interprets bits as OCP (bias 7).
    # This produces ~2-4x numerical errors with no compile/runtime failure.
    # SGLang already resolves to e4m3fn on gfx950 via fp8_kernel.is_fp8_fnuz,
    # so this guard only trips when external callers pass FNUZ by mistake.
    if _kv_is_fp8 and k_buffer.dtype == torch.float8_e4m3fnuz:
        raise ValueError(
            "Gluon FP8 extend on gfx950 requires OCP FP8 (torch.float8_e4m3fn). "
            "Got torch.float8_e4m3fnuz which produces 2-4x numerical errors "
            "because MFMA hardware on gfx950 uses bias=7 but FNUZ uses bias=8. "
            "Re-quantize KV buffers with torch.float8_e4m3fn, or use "
            "SGLang's fp8_kernel.fp8_dtype which auto-selects the right format."
        )
    # Global escape hatch: bridge fp8 KV to bf16 kernels when requested.
    _force_bridge = _kv_is_fp8 and _use_fp8_kv_bf16_bridge()
    if _force_bridge:
        # Keep fp8 KV cache externally, but bridge to non-fp8 Gluon kernels by
        # casting prefix KV on entry to the active compute dtype.
        bridge_dtype = q_extend.dtype
        k_buffer = k_buffer.to(bridge_dtype)
        v_buffer = v_buffer.to(bridge_dtype)
        _kv_is_fp8 = False
    if Lq != Lv:
        raise ValueError(
            f"Gluon extend attention only supports symmetric heads (Lq == Lv), "
            f"got Lq={Lq}, Lv={Lv}. Use mla_prefill/ for mixed-dim DeepSeek MLA."
        )
    # Correctness guard: custom_mask + FP8 KV on D<=128 hits the
    # full-dispatch basic fallback which routes to the 8-warp pipelined
    # path. NS=1 there violates the determinism invariant (warp_pipeline_stage
    # needs >=2 LDS buffers), and NS=2 in the fallback launcher has a
    # separate illegal-memory-access bug (the persistent custom_mask path
    # is also affected). Rather than silently emit wrong output or crash,
    # reject the combination early with a clear message.
    if _kv_is_fp8 and custom_mask is not None and Lq <= 128:
        raise NotImplementedError(
            "Gluon FP8 KV + custom_mask is not supported yet on D<=128 "
            "(needed for spec decode verify). The 8-warp pipelined path "
            "requires NUM_STAGES>=2 for determinism and the fallback "
            "launcher has a known IMA at NS>=2. Workarounds: "
            "(1) use BF16 KV cache for spec decode, or "
            "(2) set GLUON_FP8_KV_TO_BF16_BRIDGE=1 to cast prefix KV to "
            "BF16 inside Gluon (small perf cost, correct output)."
        )
    if _kv_was_fp8 and not _LOGGED_FP8_KV_MODE:
        if _kv_is_fp8:
            logger.info(
                "Gluon FP8 KV path active: native fp8 symmetric kernels enabled."
            )
        else:
            logger.warning(
                "Gluon FP8 KV bridge enabled: casting fp8 KV buffers to non-fp8 kernels."
            )
        _LOGGED_FP8_KV_MODE = True
    if _kv_is_fp8:
        _kernel_fn = _gluon_extend_attn_fwd_fp8_basic
    else:
        if Lq == 128 and _use_d128_experimental_kernel():
            _kernel_fn = _gluon_extend_attn_fwd_symmetric_d128_experimental
        else:
            _kernel_fn = _gluon_extend_attn_fwd_symmetric
    batch_size = qo_indptr.shape[0] - 1
    head_num = q_extend.shape[1]

    # Experimental D128 ext>=1 split/fix-up path:
    # - independent D128 kernel fork
    # - WCA-style partial-state fix-up with host merge + locks
    if (
        Lq == 128
        and Lv == 128
        and _use_d128_experimental_kernel()
        and (not _kv_is_fp8)
        and custom_mask is None
        and mask_indptr is None
        and is_causal
        and sliding_window_size <= 0
    ):
        _used = _launch_d128_decode1_wca_experimental(
            q_extend,
            k_extend,
            v_extend,
            o_extend,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            sm_scale=sm_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        if _used:
            return

    # Zero-cost uniform detection: if sum of extend lengths equals
    # batch_size * max_len_extend, every entry must equal max_len_extend
    # (monotone sum bound). q_extend.shape[0] IS that sum. This lets the
    # cache fast-path engage even when callers forget to pass min_len_extend,
    # which is a common pattern in benches and some serving paths.
    _uniform_by_shape = (q_extend.shape[0] == batch_size * max_len_extend)
    _uniform_like = (batch_size <= 1 or _uniform_by_shape)

    # -- Config-keyed dispatch cache --
    # Key on the CONFIG OUTPUT, not the raw (batch_size, max_len_extend) input.
    # The dispatch logic is a step function with ~6 configs per model, so this
    # cache has near-100% hit rate after workup regardless of workload mix.
    # Grid is recomputed per-call (one integer division).
    # The fast config cache only serves the basic (non-persistent, non-splitk)
    # code path. Treat `_force_use_persistent=False` and `_force_use_splitk=False`
    # as compatible with the fast cache (caller explicitly asking for basic).
    #
    # IMPORTANT: Gate the basic-path cache on `_uniform_like`. The basic kernel
    # dispatches a grid of (B, H, ceil(max_ext/BM)) tiles, so a heterogeneous
    # batch with a large max/min skew makes most tiles do no useful work --
    # WCA persistent beats basic by 10-100x on those shapes. For non-uniform
    # batches, fall through to the `_is_ragged` WCA routing below. The
    # persistent path has its own `_persistent_fast_cache` so het callers
    # still get a cache-hit launch.
    _basic_forced_only = (
        (_force_use_persistent is False or _force_use_persistent is None)
        and (_force_use_splitk is False or _force_use_splitk is None)
    )
    # Detect "small-ext + big-pfx-skew" cases (e.g. spec-decode B=8 ext=[7]*8
    # with pfx ranging 100..8192). `_uniform_like` is True (uniform ext),
    # but the basic path launches 1 CTA per seq/head with widely varying
    # per-seq prefix lengths: the longest-pfx CTA dominates kernel runtime
    # and WCA persistent wins ~1.77x. We need to exclude these from the
    # basic cache so the ragged-routing block below can send them to WCA.
    # Gated to D=128 because the WCA ragged-routing block below only fires
    # for Lq==128; excluding D=64 would just force those calls into an
    # uncached path that JITs every dispatch.
    _cache_exclude_pfx_skew = (
        Lq == 128
        and batch_size >= 4
        and max_len_extend <= 128
        and (
            total_prefix_len if total_prefix_len is not None
            else kv_indices.numel()
        ) >= batch_size * 2048
        and (_force_use_persistent is not False)
        and (_force_use_splitk is not False)
    )
    # `_cfg_cache_eligible` is the *structural* gate: no force_* overrides,
    # no custom mask, no sinks/window/preload hooks. The actual decision
    # between basic-fast-path and WCA-persistent is deferred: we check the
    # cache even for heterogeneous batches since many of them (low ext_ratio
    # on D=64 at moderate max_ext) stay on the basic path and benefit from
    # the same CompiledKernel closure a uniform batch installs. The WCA
    # routing block below still fires FIRST for het shapes where WCA wins
    # (see `_is_ragged` computation).
    # Fast-path cache is eligible even when `sinks` or `window_kv_offsets` are
    # supplied: `_fast_run` now accepts both as per-call tensor args
    # (HAS_SINK / SLIDING_WINDOW_SIZE are compile-time constexprs already in
    # the cache key, so distinct variants don't collide).
    _cfg_cache_eligible = (
        _force_block_m is None
        and _force_block_n is None
        and not _ck_v_preload
        and _basic_forced_only
        and custom_mask is None
        and _force_num_warps is None
        and _force_num_stages is None
        and _force_waves_per_eu is None
        and not _cache_exclude_pfx_skew
        and not _cfg_cache_disabled()
    )
    _has_sink = sinks is not None

    def _try_config_cache():
        """Compute the cache key for the current shape and return the entry
        if present. Called both for uniform-like batches (early fast path)
        and heterogeneous batches that bypass WCA routing (late fast path).
        """
        _pfx_b_local = _pfx_bucket(total_prefix_len, batch_size)
        _cfg_local = _get_basic_dispatch_config(
            Lq, batch_size, max_len_extend, _pfx_b_local, _kv_is_fp8,
            sliding_window_size=sliding_window_size,
        )
        # (head_num, k_head_num) must be in the key: the fast-runner closure
        # bakes kv_group_num, and strides are also model-dependent. Without
        # this, processes serving multiple models (or tests cycling through
        # specs) would alias different GQA configs to the same cache entry
        # and launch the cached kernel with the wrong kv_group_num.
        # `_has_sink` added so HAS_SINK-specialized kernels don't alias the
        # HAS_SINK=False variant (kernel bodies differ).
        # (qo/kv_indptr/kv_indices dtypes) are included because Triton
        # specializes CompiledKernel on each tensor arg's dtype — the pointer
        # element type is baked into SASS. SGLang's triton_backend hands us
        # int64/int32/int64 while AITER hands us int32/int32/int64; each
        # combination needs its own cached kernel (no cross-dtype aliasing).
        return _config_cache.get(
            (_cfg_local, Lq, _kv_is_fp8,
             is_causal, logit_cap, sliding_window_size, xai_temperature_len,
             head_num, k_extend.shape[1], _has_sink,
             qo_indptr.dtype, kv_indptr.dtype, kv_indices.dtype)
        )

    def _run_config_cache_entry(_cc_entry):
        """Execute a cache hit. Returns nothing; caller must `return` after."""
        _mi_n = q_extend.shape[0] + 1
        _wkvo_n = max(1, batch_size)
        if _dummy_mi_size < _mi_n or _dummy_wkvo_size < _wkvo_n or _dummy_sinks is None:
            _ensure_dummies(q_extend.device, _mi_n, _wkvo_n)
        # Route real tensor pointers through to the kernel. When caller
        # didn't supply sinks / window_kv_offsets, pass the dummy; the kernel
        # body ignores them under HAS_SINK=False / SLIDING_WINDOW_SIZE<0.
        _sinks_arg = sinks if sinks is not None else _dummy_sinks
        _wkvo_arg = (
            window_kv_offsets
            if window_kv_offsets is not None
            else _dummy_wkvo[:_wkvo_n]
        )
        _tag = _cc_entry[0]
        if _tag == "fast":
            # Fast path: CompiledKernel direct-call, ~8us launch vs ~55us via JITFunction.run.
            _, _cc_run, _cc_sm, _cc_strides, _cc_BM = _cc_entry
            _cc_run(
                q_extend, k_extend, v_extend, o_extend,
                k_buffer, v_buffer,
                qo_indptr, kv_indptr, kv_indices,
                _dummy_cm, _dummy_mi[:_mi_n],
                _wkvo_arg, _sinks_arg,
                _cc_sm, _cc_strides,
                (batch_size, head_num,
                 (max_len_extend + _cc_BM - 1) // _cc_BM),
            )
        else:
            # Legacy path (FP8): still goes through JITFunction.run with baked kwargs.
            # FP8 still bakes Sinks/WindowKvOffsets in frozen_kw, so per-call
            # dispatch must mutate those fields before the launch (rare path,
            # FP8 KV is small surface area and not used by GPT-OSS).
            _, _cc_run, _cc_sm, _cc_gn, _cc_strides, _cc_kw, _cc_BM = _cc_entry
            if "Sinks" in _cc_kw:
                _cc_kw = {**_cc_kw, "Sinks": _sinks_arg}
            _cc_run(
                q_extend, k_extend, v_extend, o_extend,
                k_buffer, v_buffer,
                qo_indptr, kv_indptr, kv_indices,
                _dummy_cm, _dummy_mi[:_mi_n],
                _wkvo_arg,
                _cc_sm, _cc_gn, *_cc_strides,
                grid=(batch_size, head_num,
                      (max_len_extend + _cc_BM - 1) // _cc_BM),
                **_cc_kw,
            )

    if _uniform_like and _cfg_cache_eligible:
        _cc_entry = _try_config_cache()
        if _cc_entry is not None:
            _bump_dispatch("early_cache_hit")
            _run_config_cache_entry(_cc_entry)
            return
        # Cache miss: fall through to normal path, which will populate the cache.

    # -- Fast path: no custom mask, no test overrides --
    # For uniform/B=1: launches with hardcoded constants.
    # For ragged batches with significant imbalance: routes to WCA persistent.
    # Treat `_force_use_*=False` as "explicitly ask for basic, no persistent or
    # splitk" which is compatible with the basic fast path.
    _is_fast_eligible = (
        _force_block_m is None
        and _force_block_n is None
        and not _ck_v_preload
        and _basic_forced_only
        and custom_mask is None
    )
    _is_uniform = (batch_size <= 1 or min_len_extend == max_len_extend
                   or _uniform_by_shape)
    # WCA auto-routing requires the caller to NOT have explicitly asked for
    # basic (i.e. force-disabled persistent). `_is_fast_eligible` now accepts
    # `_force_use_*=False` (so the basic config cache can fire), but we must
    # not auto-route to WCA when the caller has explicitly asked for basic.
    _ragged_routable = (
        _is_fast_eligible
        and (_force_use_persistent is not False)
        and (_force_use_splitk is not False)
    )
    _is_ragged_ext = _ragged_routable and not _is_uniform and batch_size >= 2
    # Extra path: "small-ext pfx-skew" cases like spec-decode batches
    # (ext=[7]*8 with pfx ranging 100..8192). Basic launches (ceil(max_ext/BM),
    # B*H) = (1, B*H) tiles, each serially streaming its seq's full prefix.
    # With no per-tile rebalancing and widely varying per-seq prefix lengths,
    # the longest-prefix CTA dominates kernel runtime. WCA persistent
    # rebalances across CUs and regains ~1.7x on these shapes.
    _total_pfx_est_pre = (
        total_prefix_len if total_prefix_len is not None
        else kv_indices.numel()
    )
    _is_ragged_pfx = (
        _ragged_routable
        and batch_size >= 4
        and max_len_extend <= 128
        and _total_pfx_est_pre >= batch_size * 2048
    )
    _is_ragged = _is_ragged_ext or _is_ragged_pfx

    # Route ragged batches to WCA persistent when extend imbalance is significant.
    # Only for D64/D128; D256 uses the full dispatch path.
    if _is_ragged and Lq <= 128:
        # Zero-cost waste-fraction: fraction of basic-path grid that does no
        # useful work (empty rows beyond each seq's extend length). Benchmarks
        # show basic path beats WCA when waste_frac is small AND tiles are
        # light; WCA wins decisively when tiles are heavy OR the batch is big.
        _total_ext = q_extend.shape[0]
        _grid_est = batch_size * max_len_extend
        _waste_frac = 1.0 - _total_ext / max(1, _grid_est)
        # Approx total prefix (kv_indices is sized to total_prefix_len by
        # construction in every caller path; fallback to threaded-in value if
        # present). Zero-cost (no sync). Reuse the value computed pre-gate.
        _total_pfx_est = _total_pfx_est_pre
        if Lq == 128:
            # D=128 BF16 routing, calibrated against bench_real_het.py:
            #   A. any heterogeneous batch with max_ext >= 1024: WCA almost
            #      always wins since per-tile cost is large.
            #   B. batch_size >= 8 with significant total prefix work: the
            #      basic path's per-seq prefix-scan imbalance dominates; WCA
            #      redistributes the tile graph. Catches chunk-mix / spec-
            #      decode / realistic cases.
            #   C. batch_size >= 8 + max_ext in [768, 1024) + high waste:
            #      chunked prefill without prefix but big enough tiles for
            #      WCA to amortize its scheduler cost.
            #   D. moderate max (>=768) with meaningful skew.
            #   E. very large batch with any meaningful skew: B=32 chat-mix
            #      with max=128 is 21x slower on basic.
            #   F. pfx-skew-ragged (from above): small ext, big total pfx, B>=4.
            # NOTE: B=8 short-tail (max=513, no prefix) falls THROUGH to basic
            # on purpose -- low total work + no pfx skew means basic's smaller
            # wrapper/launch overhead wins.
            _use_wca = (
                _is_ragged_pfx  # F: spec-decode-style small-ext + pfx-skew
                or (max_len_extend >= 1024 and _waste_frac > 0.05)
                or (batch_size >= 8 and _total_pfx_est >= batch_size * 1024)
                or (batch_size >= 8 and max_len_extend >= 768 and _waste_frac >= 0.4)
                # Gate condition D on B>=4: random-perf bench found B=3
                # max_ext~964 no-pfx land here and WCA small is 1.6x slower
                # than basic (3 seqs * 32 heads * few tiles = small grid,
                # basic's simpler dispatch wins on the coordination cost).
                or (max_len_extend >= 768 and _waste_frac >= 0.2 and batch_size >= 4)
                or (batch_size >= 16 and _waste_frac >= 0.2)
            )
        else:
            # D=64 BF16 -- basic dispatch is more aggressive about picking
            # wide tiles, so WCA has a smaller window of wins. Keep the old
            # ext_ratio-based heuristic (requires min_len_extend for the ratio,
            # fall back to conservative no-ratio path if missing).
            ext_ratio = (
                max_len_extend / max(1, min_len_extend)
                if min_len_extend else float("inf")
            )
            _use_wca = (
                # Big-B with long prefix (multi-turn chat, long-context decode).
                # Basic launches 1 CTA per (seq, head) and each streams its full
                # per-seq prefix -- longest-prefix CTA dominates. WCA rebalances
                # across CUs; random-perf bench showed 2-3x wins for B>=8 with
                # avg_pfx >= 1k per seq and max_ext >= 32.
                # Exclude spec-decode (max<=8): basic with tiny grid
                # (B*H*1 tile) is already fast there and WCA's scheduler
                # overhead nets a regression.
                (batch_size >= 8 and _total_pfx_est >= batch_size * 1024
                 and max_len_extend >= 32)
                or (ext_ratio > 4.0 and max_len_extend >= 256)
                # High ext_ratio small-B *with* no prefix lands the basic path
                # in a small-grid regime that distributes fine without WCA's
                # scheduler overhead; bench shows WCA small ~50% slower there.
                # Keep the rule firing only when there's real reason to WCA.
                or (ext_ratio > 20.0 and max_len_extend >= 64
                    and (batch_size >= 5 or _total_pfx_est >= 512))
            )
        if _use_wca:
            _wca_fn = _launch_persistent_fp8 if _kv_is_fp8 else _launch_persistent
            # Three persistent flavors (identical kernel body, different
            # scheduler / tile geometry):
            #   * bsearch (TILE_MAP_MODE=0, BM=128 NW=8 NS=4) -- binary-search
            #     tile->seq lookup. Beats linear-scan when every tile does
            #     substantial work because the O(log B) lookup per tile
            #     amortizes. Use for "full-batch prefill" shapes: B >= 16,
            #     max_ext >= 2048, avg_ext >= 512 (no long tail of 17-token
            #     seqs skewing the average).
            #   * WCA default (TILE_MAP_MODE=1, BM=128 NW=8 NS=4) -- linear-
            #     scan scheduler; 8 warps + 4 stages pipeline KV prefetch
            #     well for prefix-dominated workloads (chat-mix / long-context
            #     decode).
            #   * WCA small (TILE_MAP_MODE=1, BM=64 NW=4 NS=2) -- halves
            #     register pressure + keeps tile count high, better CU spread
            #     when work is ext-dominated or extremely skewed (B=4
            #     big-prefill ext=[4096,2048,1024,512] with pfx=0).
            # Thresholds calibrated against bench_real_het.py.
            _wca_kw = dict()
            _avg_ext = _total_ext / max(1, batch_size)
            _use_bsearch = (
                Lq == 128
                and batch_size >= 16
                and max_len_extend >= 2048
                and _avg_ext >= 512
            )
            if _use_bsearch:
                _wca_kw.update(_force_tile_map_mode=0)
            elif Lq == 128 and _total_pfx_est < 4 * _total_ext:
                _wca_kw.update(
                    _force_block_m=64,
                    _force_num_warps=4,
                    _force_num_stages=2,
                )
            _bump_dispatch("wca_persistent")
            if _trace_enabled() and _dispatch_counters["wca_persistent"] <= 20:
                logger.warning(
                    "gluon extend WCA: B=%d minE=%s maxE=%d pfx_est=%d Lq=%d "
                    "sw=%s sink=%s kw=%s",
                    batch_size, min_len_extend, max_len_extend,
                    _total_pfx_est, Lq, sliding_window_size,
                    sinks is not None, _wca_kw,
                )
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
                **_wca_kw,
            )
            return

    # Late config-cache lookup for heterogeneous batches that either failed
    # the uniform-like guard or fell out of the `_is_ragged` block without
    # routing to WCA. They land on the basic path with the same dispatch
    # cfg as a uniform batch of matching (B, max_ext, pfx_bucket); if that
    # cfg is in `_config_cache` (installed by an earlier uniform call or by
    # prewarm phase 2), hit the fast runner here instead of paying the
    # full-dispatch path cost (~9ms first-call) on every such shape.
    if (not _uniform_like) and _cfg_cache_eligible:
        _cc_entry = _try_config_cache()
        if _cc_entry is not None:
            _bump_dispatch("late_cache_hit")
            _run_config_cache_entry(_cc_entry)
            return

    if _is_fast_eligible:
        _BLOCK_DMODEL, _ACTUAL_BLOCK_DMODEL = _resolve_qk_split_dims(Lq)

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
            if _need_persistent and _ragged_routable:
                if min_len_extend is None:
                    # See _launch_persistent: avoid 6ms CPU sync; conservative
                    # fallback is uniform-ratio (max), which only mildly affects
                    # tile-size selection downstream.
                    min_len_extend = max_len_extend
                if _kv_is_fp8:
                    _p_bm = 64 if Lq == 128 else 32
                    _p_bn = 128
                    _p_nw = 4
                    _p_ns = 1
                else:
                    _p_bm = _p_bn = _p_nw = _p_ns = None
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
                    _force_block_m=_p_bm,
                    _force_num_warps=_p_nw,
                    _force_num_stages=_p_ns,
                    _force_block_n=_p_bn,
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

        # D256: check for prefix-aware WCA routing before basic dispatch.
        # FP8 D256 persistent exceeds LDS budget (204KB > 160KB) and
        # hits LLVM codegen bugs in some split-K configs — skip for FP8.
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
            if _need_persistent_256 and _ragged_routable:
                if min_len_extend is None:
                    min_len_extend = max_len_extend
                _wca_fn = _launch_persistent_fp8 if _kv_is_fp8 else _launch_persistent
                _bump_dispatch("wca_persistent_d256")
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
            )
        )
        if _force_block_n is not None:
            _BN = _force_block_n
        if _kv_is_fp8 and _force_num_warps is not None:
            _NW = _force_num_warps
        if _force_num_stages is not None:
            _NS = _force_num_stages
            if _kv_is_fp8:
                _EXT_NS = _force_num_stages

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
            "BLOCK_DPE": 0, "ACTUAL_BLOCK_DPE": 0,
            "BLOCK_DV": _BLOCK_DMODEL, "ACTUAL_BLOCK_DV": Lq,
            "ASYNC_PAD_K": _PAD_K, "ASYNC_PAD_V": _PAD_V,
        } if _kv_is_fp8 else {}
        # V_PRELOAD / UNIFY_CAUSAL_PATH are hardcoded to False inside the
        # kernel body (dead-code eliminated at Gluon JIT time), so they are
        # not kernel-signature constexprs anymore.

        _frozen_kw = dict(
            IS_CAUSAL=is_causal,
            USE_CUSTOM_MASK=False,
            ENABLE_PREFIX_UNMASKED=is_causal,
            BLOCK_M=_BM, BLOCK_N=_BN,
            BLOCK_DMODEL=_BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL=_ACTUAL_BLOCK_DMODEL,
            NUM_STAGES=_NS,
            **_kernel_extra,
            Sinks=sinks, HAS_SINK=sinks is not None,
            LOGIT_CAP=logit_cap,
            XAI_TEMPERATURE_LEN=xai_temperature_len,
            SLIDING_WINDOW_SIZE=sliding_window_size,
            v_scale=v_scale,
            num_warps=_NW, num_stages=1, waves_per_eu=2,
            matrix_instr_nonkdim=32,
            warmup=False,
        )
        _frozen_grid = (batch_size, head_num, (max_len_extend + _BM - 1) // _BM)
        _frozen_scalars = (_sm, _kv_gn)
        _frozen_strides = (
            _q_s0, _q_s1, _k_s0, _k_s1, _v_s0, _v_s1,
            _o_s0, _o_s1, _kb_s0, _kb_s1, _vb_s0, _vb_s1,
        )
        # First call: go through JITFunction.run (compiles if needed, runs the
        # expensive ~55us specialization/binder path). The return value is the
        # CompiledKernel, which we capture and reuse on subsequent calls via
        # _make_fast_runner to skip the binder entirely (BF16 path only for now).
        _bump_dispatch("basic_slow_first")
        if _trace_enabled():
            _cfg_for_log = (_BM, _BN, _NW, _NS, _PAD_K, _PAD_V, _EXT_BN, _EXT_NS)
            logger.warning(
                "gluon extend JIT: B=%d maxE=%d pfxB=%d Lq=%d H=%d kvH=%d "
                "causal=%s sw=%s cap=%s xai=%s sink=%s cfg=%s",
                batch_size, max_len_extend,
                _pfx_b_full, Lq, head_num, k_extend.shape[1],
                is_causal, sliding_window_size, logit_cap,
                xai_temperature_len, sinks is not None, _cfg_for_log,
            )
        _compiled = _kernel_fn.run(
            q_extend, k_extend, v_extend, o_extend,
            k_buffer, v_buffer,
            qo_indptr, kv_indptr, kv_indices,
            _dummy_cm, _dummy_mi[: q_extend.shape[0] + 1], _wkvo,
            *_frozen_scalars, *_frozen_strides,
            grid=_frozen_grid, **_frozen_kw,
        )

        # Populate config cache for future calls with same dispatch config.
        # For het batches, `_get_basic_dispatch_config` returns the same BM/BN/NW/NS
        # as the full dispatch path would, so the cached CompiledKernel is valid.
        # We no longer gate on `sinks is None` / `window_kv_offsets is None`:
        # `_fast_run` now takes those as per-call args, and the cache key
        # distinguishes HAS_SINK-True vs HAS_SINK-False (kernel bodies differ).
        #
        # DTYPE SAFETY: Triton specializes CompiledKernel on each tensor arg's
        # dtype (pointer element type is baked into the SASS). If we cache a
        # kernel compiled for one dtype and later call with a different dtype,
        # the kernel reads wrong stride bytes -> garbage pointer math ->
        # HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION (async, attributed to
        # the NEXT kernel via a SystemError in Triton's launch hooks).
        # Fix: include (qo_indptr.dtype, kv_indptr.dtype, kv_indices.dtype) in
        # the cache key so every distinct indptr/indices dtype combination gets
        # its own CompiledKernel. Callers pick their natural dtype (SGLang
        # triton_backend hands us int64/int32/int64, AITER hands us
        # int32/int32/int64, etc.) and each combination has a dedicated fast
        # path with no per-call cast.
        if (
            _force_block_n is None
            and _force_num_warps is None
            and custom_mask is None
            and not _cfg_cache_disabled()
        ):
            _cfg = (_BM, _BN, _NW, _NS, _PAD_K, _PAD_V, _EXT_BN, _EXT_NS)
            _has_sink_w = sinks is not None
            _cc_key = (_cfg, Lq, _kv_is_fp8,
                       is_causal, logit_cap, sliding_window_size, xai_temperature_len,
                       head_num, k_extend.shape[1], _has_sink_w,
                       qo_indptr.dtype, kv_indptr.dtype, kv_indices.dtype)
            if _cc_key not in _config_cache:
                if _kv_is_fp8:
                    # FP8 has extra _kernel_extra constexprs not handled by
                    # _make_fast_runner yet — keep the legacy JITFunction.run
                    # path with baked kwargs for FP8.
                    _config_cache[_cc_key] = (
                        "legacy",
                        _kernel_fn.run,
                        _sm, _kv_gn, _frozen_strides, _frozen_kw, _BM,
                    )
                else:
                    _config_cache[_cc_key] = (
                        "fast",
                        _make_fast_runner(_compiled, _frozen_kw, _kv_gn),
                        _sm, _frozen_strides, _BM,
                    )
        return

    # -- Full dispatch path (heterogeneous batches, custom mask, test overrides) --
    _bump_dispatch("full_dispatch")

    BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL = _resolve_qk_split_dims(Lq)

    # min_len_extend is only needed by the D>=256 dispatch and the
    # persistent/splitk paths. Computing it eagerly costs a GPU sync
    # (~15us per call). Compute lazily below where actually used.

    _ensure_dummies(q_extend.device, q_extend.shape[0] + 1, batch_size)

    num_CUs = _get_num_CUs(q_extend.device)

    _BM_est = _force_block_m or (64 if Lq < 256 else 128)
    n_m_tiles_est = (max_len_extend + _BM_est - 1) // _BM_est
    total_tiles_est = batch_size * head_num * n_m_tiles_est

    use_splitk = False
    use_persistent = False

    if _force_use_splitk:
        use_splitk = True
    elif _force_use_persistent:
        use_persistent = True
    elif _force_use_splitk is False and _force_use_persistent is False:
        pass  # Both explicitly disabled → fall through to basic kernel.
    elif _force_use_splitk is False:
        pass  # splitk explicitly off; leave persistent to default rules.
    elif _force_use_persistent is False:
        pass  # persistent explicitly off; leave splitk to default rules.
    elif Lq >= 256:
        use_persistent = False
    elif custom_mask is None and Lq <= 128:
        use_splitk = True

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
            BLOCK_M=_force_block_m or 64,
            BLOCK_N=_BN,
            num_warps=_force_num_warps or 4,
            NUM_STAGES=_force_num_stages or 2,
            _force_waves_per_eu=_force_waves_per_eu,
            total_prefix_len=total_prefix_len,
        )
        return

    if use_persistent:
        if min_len_extend is None:
            min_len_extend = max_len_extend
        _wca_fn = _launch_persistent_fp8 if _kv_is_fp8 else _launch_persistent
        _wca_fn(
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
            k_scale=k_scale,
            v_scale=v_scale,
            sm_scale=sm_scale,
            logit_cap=logit_cap,
            skip_prefix_custom_mask=skip_prefix_custom_mask,
            sliding_window_size=sliding_window_size,
            sinks=sinks,
            window_kv_offsets=window_kv_offsets,
            xai_temperature_len=xai_temperature_len,
            _force_block_m=_force_block_m,
            _force_num_warps=_force_num_warps,
            _force_num_stages=_force_num_stages,
            _force_waves_per_eu=_force_waves_per_eu,
            _force_block_n=_force_block_n,
            min_len_extend=min_len_extend,
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
    if _force_block_n is not None:
        BLOCK_N = _force_block_n
    if _kv_is_fp8 and BLOCK_DMODEL < 256:
        BLOCK_N = 128
    EXT_BLOCK_N = BLOCK_N
    NUM_STAGES = 1

    if _force_block_m is not None and _force_num_warps is not None:
        BLOCK_M = _force_block_m
        num_warps = _force_num_warps
    elif BLOCK_DMODEL >= 256:
        if total_prefix_len is None or total_extend_len is None:
            # Cheap 2-scalar sync, but still ~100us on hot path. Caller should
            # pass these hints; when they don't, fall back to kv_indices.numel
            # (CPU-side shape) and q shape instead of GPU reductions.
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
        # D64 full-path dispatch. BM=64 beats BM=128/256 for high-batch small-extend.
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

    if _force_num_stages is not None:
        NUM_STAGES = _force_num_stages
    elif BLOCK_DMODEL >= 256:
        if _force_block_m is not None and _force_num_warps is not None:
            NUM_STAGES = 4 if BLOCK_M >= 128 else 2
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

    if _kv_is_fp8:
        if BLOCK_DMODEL < 128:
            # D=64 FP8 basic: the 4-warp DMA path assumes BLOCK_DMODEL>=128
            # for its layout bases; 8-warp DMA path is likewise broken for
            # D<128. Only NW=4 NS=1 compiles cleanly. For pipelined execution
            # on D=64 FP8, dispatch should route to the persistent kernel
            # which has a working D<128 branch. NW=4 + NS=1 is safe because
            # USE_SERIAL=True routes to the non-pipelined _dma_simple/_short
            # variants (no warp_pipeline_stage, no NS>=2 requirement).
            NUM_STAGES = _force_num_stages or 1
            num_warps = _force_num_warps or 4
            if _force_block_m is not None:
                BLOCK_M = _force_block_m
            elif batch_size >= 16 and max_len_extend <= 128:
                BLOCK_M = 64
            else:
                BLOCK_M = 128  # Retune Apr 2026: BM=128 wins 2-3x vs BM=256
        else:
            # D=128 FP8 basic fallback (het-batch / custom-mask / test
            # overrides). The 8-warp pipelined path requires NS>=2 for
            # determinism (static_assert in attn_fwd_inner_{extend,prefix}
            # _pipelined). Keep defaults aligned with the fast-path.
            NUM_STAGES = _force_num_stages or 1
            if _force_num_warps is not None:
                num_warps = _force_num_warps
            else:
                num_warps = 8
            if _force_block_m is not None:
                BLOCK_M = _force_block_m
            elif batch_size == 1 and max_len_extend <= 256:
                BLOCK_M = 64
            else:
                BLOCK_M = 128
    EXT_NUM_STAGES = NUM_STAGES

    # Correctness guards.
    if BLOCK_DMODEL == 64 and num_warps == 8:
        if BLOCK_M in (64, 128) and NUM_STAGES == 1:
            NUM_STAGES = 2
        if BLOCK_M == 256 and NUM_STAGES in (1, 3, 4):
            NUM_STAGES = 2
    if BLOCK_DMODEL >= 256 and num_warps == 8 and NUM_STAGES == 1:
        if _force_num_stages is None:
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
        "BLOCK_DPE": 0, "ACTUAL_BLOCK_DPE": 0,
        "BLOCK_DV": BLOCK_DMODEL, "ACTUAL_BLOCK_DV": Lq,
        "ASYNC_PAD_K": 16, "ASYNC_PAD_V": 16,
    } if _kv_is_fp8 else {}
    # V_PRELOAD / UNIFY_CAUSAL_PATH are hardcoded to False inside the BF16
    # kernel body. _ck_v_preload on the wrapper is now a no-op (kept for
    # backward-compat with oracle/test scripts).

    _kernel_fn.run(
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
        BLOCK_DMODEL=BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL=ACTUAL_BLOCK_DMODEL,
        NUM_STAGES=NUM_STAGES,
        **_kernel_extra_full,
        Sinks=sinks, HAS_SINK=sinks is not None,
        LOGIT_CAP=logit_cap,
        XAI_TEMPERATURE_LEN=xai_temperature_len,
        SLIDING_WINDOW_SIZE=sliding_window_size,
        v_scale=v_scale,
        num_warps=num_warps, num_stages=1,
        waves_per_eu=_force_waves_per_eu if _force_waves_per_eu is not None else 2,
        matrix_instr_nonkdim=32,
        grid=(batch_size, head_num, triton.cdiv(max_len_extend, BLOCK_M)),
        warmup=False,
    )


# ===-----------------------------------------------------------------------===#
# Test / Benchmark Helpers
# ===-----------------------------------------------------------------------===#


def _run_gluon(q, k, v, kb, vb, qo, kv, ki, o, elens, causal, **kw):
    gluon_extend_attention_fwd(
        q,
        k,
        v,
        o,
        kb,
        vb,
        qo,
        kv,
        ki,
        custom_mask=None,
        is_causal=causal,
        mask_indptr=None,
        max_len_extend=max(elens),
        sm_scale=1.0 / math.sqrt(q.shape[-1]),
        min_len_extend=min(elens),
        **kw,
    )


def _run_gluon_persistent(q, k, v, kb, vb, qo, kv, ki, o, elens, causal, **kw):
    _launch_persistent(
        q,
        k,
        v,
        o,
        kb,
        vb,
        qo,
        kv,
        ki,
        custom_mask=None,
        is_causal=causal,
        mask_indptr=None,
        max_len_extend=max(elens),
        sm_scale=1.0 / math.sqrt(q.shape[-1]),
        min_len_extend=min(elens),
        **kw,
    )


# ===-----------------------------------------------------------------------===#
# FP8 Persistent / Split-K Launchers
# ===-----------------------------------------------------------------------===#


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
    _force_block_m=None,
    _force_num_warps=None,
    _force_num_stages=None,
    _force_waves_per_eu=None,
    _force_block_n=None,
    _ck_v_preload=False,
    min_len_extend=None,
    SPLIT_K=1,
    total_prefix_len=None,
):
    """FP8 persistent launcher: calls the FP8 persistent kernel."""
    Lq = q_extend.shape[-1]
    Lv = v_extend.shape[-1]
    BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL = _resolve_qk_split_dims(Lq)

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
    BLOCK_N = 128 if Lq <= 128 else (32 if BLOCK_DMODEL >= 256 else 64)
    if _force_block_n is not None:
        BLOCK_N = _force_block_n
    batch_size = qo_indptr.shape[0] - 1

    if min_len_extend is None:
        min_len_extend = max_len_extend
    head_num = q_extend.shape[1]

    if _force_block_m is not None and _force_num_warps is not None:
        BLOCK_M = _force_block_m
        num_warps = _force_num_warps
    elif max(BLOCK_DMODEL, BLOCK_DV) >= 256:
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

    if _force_num_stages is not None:
        NUM_STAGES = _force_num_stages
    else:
        NUM_STAGES = int(os.environ.get('_GLUON_FP8_NS', '2'))

    EXT_BLOCK_N = int(os.environ.get('_GLUON_FP8_EXT_BN', '64'))
    EXT_NUM_STAGES = int(os.environ.get('_GLUON_FP8_EXT_NS', '3'))

    ASYNC_PAD_K = 8 if BLOCK_DMODEL >= 256 else 16
    ASYNC_PAD_V = 32 if BLOCK_DV >= 256 else 16

    sm_scale = sm_scale or 1.0 / math.sqrt(Lq)
    sm_scale = sm_scale * k_scale
    kv_group_num = q_extend.shape[1] // k_extend.shape[1]

    device = q_extend.device
    n_m_tiles = (max_len_extend + BLOCK_M - 1) // BLOCK_M
    total_output_tiles = batch_size * head_num * n_m_tiles
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
        ACTUAL_BLOCK_DMODEL=ACTUAL_BLOCK_DMODEL,
        BLOCK_DPE=0,
        ACTUAL_BLOCK_DPE=0,
        BLOCK_DV=BLOCK_DV,
        ACTUAL_BLOCK_DV=Lq,
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
        IS_PERSISTENT=True,
        SPLIT_K=SPLIT_K,
        V_PRELOAD=False,
        num_warps=num_warps,
        num_stages=1,
        waves_per_eu=_force_waves_per_eu if _force_waves_per_eu is not None else 2,
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
    _force_waves_per_eu=None,
    total_prefix_len=None,
):
    """FP8 split-K launcher: determines SPLIT_K and delegates to
    _launch_persistent_fp8 with the right split factor."""
    head_num = q_extend.shape[1]
    device = q_extend.device
    batch_size = qo_indptr.shape[0] - 1

    BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL = _resolve_qk_split_dims(Lq)
    BLOCK_DV = BLOCK_DMODEL

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
        )
        return

    SPLIT_K = _select_k_splits(total_output_tiles, num_CUs)

    if max(BLOCK_DMODEL, BLOCK_DV) < 256:
        BLOCK_M = 128
        num_warps = 8
        NUM_STAGES = int(os.environ.get('_GLUON_FP8_NS', '2'))

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
        _force_block_m=BLOCK_M,
        _force_num_warps=num_warps,
        _force_num_stages=NUM_STAGES,
        _force_waves_per_eu=_force_waves_per_eu,
        SPLIT_K=SPLIT_K,
    )
