# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon extend-attention dispatch for gfx950 (MI350X / CDNA4).

Public entry point: :func:`gluon_extend_attention_fwd`. Supports symmetric
heads only (Lq == Lv) at D in {64, 128, 256} with BF16 or FP8 KV cache.
Mixed-dim / DeepSeek MLA heads live on a separate ``mla_prefill`` branch
and are out of scope here.

Dispatch picks between four kernel bodies (BF16 basic, BF16 WCA persistent,
FP8 basic, FP8 WCA persistent) via a config-keyed cache; on cache hit the
launcher invokes HIPLauncher directly and bypasses Triton's JITFunction
specialization.
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

_dummy_cm = None
_dummy_mi = None
_dummy_mi_size = 0
_dummy_wkvo = None
_dummy_wkvo_size = 0
# Singleton passed to the kernel when HAS_SINK=False (HIPLauncher arg list
# is fixed; the kernel body never loads from it and DCE eliminates the path).
_dummy_sinks = None

_LOGGED_FP8_KV_MODE = False
logger = logging.getLogger(__name__)

# Config-keyed dispatch cache.
#
# Key on the OUTPUT of `_get_basic_dispatch_config` rather than the raw
# (batch_size, max_len_extend) input. Dispatch is a step function with
# ~6 configs per (Lq, is_fp8) pair, so calls that land on the same config
# share one cache entry regardless of the exact driving inputs. Hit rate
# is near-100% in production; per-call we only recompute grid = (B, H,
# ceildiv(max_len_extend, BM)).
#
# Entry layout: ("fast", fast_run_closure, sm, strides, BM). Both BF16 and
# FP8 paths use `_make_fast_runner` / `_make_fast_runner_fp8` to invoke the
# HIPLauncher directly, bypassing Triton's JITFunction specialization.
_config_cache = {}

# Dispatch-path counters. Set SGLANG_GLUON_TRACE=<N> to get a WARN log every
# N calls summarising where calls landed; unset they are effectively free.
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


_TRACE_INTERVAL = int(os.getenv("SGLANG_GLUON_TRACE", "0"))
_TRACE_LAST_TOTAL = 0


def _trace_enabled():
    return _TRACE_INTERVAL


def _maybe_trace_dispatch():
    if _TRACE_INTERVAL <= 0:
        return
    global _TRACE_LAST_TOTAL
    total = _dispatch_counters["total"]
    if total - _TRACE_LAST_TOTAL < _TRACE_INTERVAL:
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


# Env escape hatches: read once at import time so per-call cost is a
# simple global lookup. Semantics:
#   SGLANG_GLUON_DISABLE_CFG_CACHE=1  bypass the fast-runner cache entirely
#   SGLANG_GLUON_FP8_KV_FORCE_BF16=1  route FP8 KV through the BF16 kernels
#                                     (pays a per-call dtype cast)
#   SGLANG_GLUON_UNIFY_CAUSAL_PATH=1  collapse the causal kernel's
#                                     full/masked split into one loop
_CFG_CACHE_DISABLED = int(os.getenv("SGLANG_GLUON_DISABLE_CFG_CACHE", "0")) != 0
_FP8_KV_FORCE_BF16 = int(os.getenv("SGLANG_GLUON_FP8_KV_FORCE_BF16", "0")) != 0
_UNIFY_CAUSAL_PATH = int(os.getenv("SGLANG_GLUON_UNIFY_CAUSAL_PATH", "0")) != 0


def _use_fp8_kv_bf16_bridge():
    return _FP8_KV_FORCE_BF16


def _use_unify_causal_path():
    return _UNIFY_CAUSAL_PATH


_cached_num_xcds = {}


def _make_fast_runner(compiled_kernel, frozen_kw: dict, kv_gn: int):
    """Build a HIPLauncher-direct closure for the BF16 basic kernel.

    Once dispatch has picked a config, all constexprs are fixed; only the
    tensors, sm_scale, 12 stride ints, and grid change per call. Baking the
    constexprs into a closure avoids Triton's JITFunction specialization
    pass on every launch.

    Assumes ``knobs.runtime.launch_enter_hook`` / ``launch_exit_hook`` are
    ``None`` at construction time. Callers that later install a profiler
    should clear ``_config_cache`` so the closures are rebuilt.
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
    # Sinks pointer is a runtime arg (varies per layer on GPT-OSS); HAS_SINK
    # is a compile-time constant baked into the specialized kernel and part
    # of the cache key, so swapping the pointer across calls is safe.
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


def _make_fast_runner_fp8(compiled_kernel, frozen_kw: dict, kv_gn: int):
    """FP8 twin of :func:`_make_fast_runner`.

    The FP8 kernel signature carries extra constexprs
    (``SKIP_PREFIX_CUSTOM_MASK``, ``ENABLE_MASK_SPLIT``, ``BLOCK_DPE``/``DV``,
    ``EXT_BLOCK_N``/``EXT_NUM_STAGES``, ``ASYNC_PAD_K``/``V``) so the arg list
    is longer; order below matches the ``@gluon.jit`` definition in
    ``_fp8_kv_extend_basic_gfx950.py``.
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
    SKIP_PREFIX_CUSTOM_MASK = frozen_kw['SKIP_PREFIX_CUSTOM_MASK']
    ENABLE_PREFIX_UNMASKED = frozen_kw['ENABLE_PREFIX_UNMASKED']
    ENABLE_MASK_SPLIT = frozen_kw['ENABLE_MASK_SPLIT']
    BLOCK_M = frozen_kw['BLOCK_M']
    BLOCK_N = frozen_kw['BLOCK_N']
    BLOCK_DMODEL = frozen_kw['BLOCK_DMODEL']
    ACTUAL_BLOCK_DMODEL = frozen_kw['ACTUAL_BLOCK_DMODEL']
    BLOCK_DPE = frozen_kw['BLOCK_DPE']
    ACTUAL_BLOCK_DPE = frozen_kw['ACTUAL_BLOCK_DPE']
    BLOCK_DV = frozen_kw['BLOCK_DV']
    ACTUAL_BLOCK_DV = frozen_kw['ACTUAL_BLOCK_DV']
    NUM_STAGES = frozen_kw['NUM_STAGES']
    EXT_BLOCK_N = frozen_kw['EXT_BLOCK_N']
    EXT_NUM_STAGES = frozen_kw['EXT_NUM_STAGES']
    ASYNC_PAD_K = frozen_kw['ASYNC_PAD_K']
    ASYNC_PAD_V = frozen_kw['ASYNC_PAD_V']
    HAS_SINK = frozen_kw['HAS_SINK']
    LOGIT_CAP = frozen_kw['LOGIT_CAP']
    XAI_TEMPERATURE_LEN = frozen_kw['XAI_TEMPERATURE_LEN']
    SLIDING_WINDOW_SIZE = frozen_kw['SLIDING_WINDOW_SIZE']
    v_scale = frozen_kw['v_scale']

    def _fast_run_fp8(q, k, v, o, kb, vb, qo_i, kv_i, kv_x,
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
            IS_CAUSAL, USE_CUSTOM_MASK, SKIP_PREFIX_CUSTOM_MASK,
            ENABLE_PREFIX_UNMASKED, ENABLE_MASK_SPLIT,
            BLOCK_M, BLOCK_N, BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL,
            BLOCK_DPE, ACTUAL_BLOCK_DPE, BLOCK_DV, ACTUAL_BLOCK_DV,
            NUM_STAGES, EXT_BLOCK_N, EXT_NUM_STAGES,
            ASYNC_PAD_K, ASYNC_PAD_V,
            sinks, HAS_SINK,
            LOGIT_CAP, XAI_TEMPERATURE_LEN, SLIDING_WINDOW_SIZE,
            v_scale,
        )

    return _fast_run_fp8


def _get_num_xcds(device):
    idx = device.index if hasattr(device, 'index') and device.index is not None else 0
    if idx not in _cached_num_xcds:
        num_CUs = torch.cuda.get_device_properties(device).multi_processor_count
        # 38 CUs/XCD on MI300X (304/8); MI350X has 32 CUs/XCD (256/8) but
        # both divide cleanly enough here to not need special-casing.
        _cached_num_xcds[idx] = max(1, num_CUs // 38)
    return _cached_num_xcds[idx]


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


# Cached dispatch config (AITER-style per-shape lookup).

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
    head_num=None,
):
    """Pick (BLOCK_M, BLOCK_N, num_warps, NUM_STAGES, PAD_K, PAD_V,
    EXT_BLOCK_N, EXT_NUM_STAGES) for the basic (non-persistent) kernel.

    Cached per shape signature. ``pfx_bucket`` buckets avg prefix length
    (see :func:`_pfx_bucket`) so the cache stays small while keeping the
    prefix-vs-no-prefix distinction. ``sliding_window_size`` selects SWA
    configs where a smaller BM tends to win.

    ``head_num`` is the number of Q heads; B=1 uses it to choose BM since
    the win from BM=256 depends on how the grid ``(1, H, ceil(S/BM))`` fills
    waves across 256 CUs.
    """
    if Lq == 256:
        _BN = 32
        _total_ext = batch_size * max_len_extend
        # Coarse avg-prefix proxy derived from pfx_bucket (matches
        # _pfx_bucket thresholds).
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
        # D=64 BF16 basic dispatch tree. Per-shape winners from the
        # AITER/CK grid (see bench_fp8_vs_ck.py).
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
            # B=1..3 single/small-seq serving. The B=1 cross-over from
            # BM=128 NW=8 NS=4 to BM=256 NW=8 depends on H_q: H_q>=64 wants
            # S>=1600 and H_q=32 wants S>=1500 (lower H already under-
            # subscribes so BM=256's bigger per-tile work dominates earlier).
            # B=2,3 keep the conservative S>=2048 boundary.
            _h = head_num if head_num is not None else 0
            if batch_size == 1 and _h >= 64 and max_len_extend >= 1600:
                _BM, _NW, _NS = 256, 8, 2
            elif batch_size == 1 and _h == 32 and max_len_extend >= 1500:
                _BM, _NW, _NS = 256, 8, 2
            elif max_len_extend >= 2048:
                _BM, _NW, _NS = 256, 8, 2
            else:
                _BM, _NW, _NS = 128, 8, 4

        # B>=16 ext==256 full-attn misdispatches to BM=64 (undersubscribes
        # 256 CUs by 4x); route to BM=256.
        if (
            sliding_window_size <= 0
            and batch_size >= 16 and max_len_extend == 256
        ):
            _BM, _BN, _NW, _NS = 256, 64, 8, 2
            _EXT_BN, _EXT_NS = 64, 2

        # BM=256 NW=8 gets an extra stage on batched prefill shapes with
        # occupancy headroom. Skip for B=1 — single-seq ShareGPT already sits
        # at the occupancy floor and the extra stage only eats LDS/regs.
        if (
            _BM == 256 and _NW == 8 and _NS == 2
            and max_len_extend >= 1024
            and batch_size != 1
            and (max_len_extend <= 8192 or _total_ext <= 32768)
        ):
            _NS = 4
    else:
        # D=128 BF16 basic dispatch. NS=1 is unsafe on the prefix-
        # pipelined path (races on the DMA ring) and benches within +-3%
        # of NS=2, so NS>=2 throughout.
        #
        # Note: BM=256 NW=8 NS=2 is avoided at D=128 — it has a latent
        # reduction-order issue (max_err ~5e-3 to 1.3e-2 on ~0.1% of outputs,
        # concentrated on a few head-dim lanes). BM=128 NS=2 at H>=64 is
        # correct and captures most of the win; H=32 stays on BM=64 NW=4
        # since BM=128 cannot fill enough waves to hide the bigger tile work.
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
            _BM, _NW, _NS = 128, 8, 2
        elif batch_size == 1:
            _h = head_num if head_num is not None else 0
            if _h >= 64 and max_len_extend >= 2000:
                _BM, _NW, _NS = 128, 8, 2
            elif _h >= 64 and max_len_extend >= 1024:
                _BM, _NW, _NS = 128, 8, 4
            else:
                _BM, _NW, _NS = 64, 4, 2
        elif batch_size <= 4:
            _BM, _NW, _NS = 64, 4, 2
        elif _total_ext >= 32768:
            _BM, _NW, _NS = 128, 8, 2
        else:
            _BM, _NW, _NS = 64, 4, 2

    if is_fp8:
        # FP8 basic dispatch. Invariants:
        #   * NS>=2 on the 8-warp pipelined path (NS=1 is
        #     bit-nondeterministic; NS>=3 OOMs LDS at BN=128).
        #   * D<128 only has the NW=4 NS=1 serial path; pipelined layouts
        #     are not wired up. BM>128 at NW=4 blows register pressure.
        _PAD_K, _PAD_V = 16, 16
        _total_ext = batch_size * max_len_extend
        if Lq == 128:
            _BM, _BN, _NW, _NS = 128, 128, 8, 2
            _EXT_BN, _EXT_NS = 128, 2
            if batch_size == 1 and max_len_extend <= 256:
                _BM = 64
            elif (
                batch_size >= 32
                and max_len_extend <= 8
                and pfx_bucket <= 2
            ):
                # Decode / spec-verify / draft-extend on short-prefix
                # continuous batches: BM=64 NW=4 avoids pad-heavy tiles.
                _BM, _NW = 64, 4
        elif Lq == 64:
            _BM, _BN, _NW, _NS = 128, 128, 4, 1
            _EXT_BN, _EXT_NS = 128, 1
            # BM=64 wins wherever tiles are pad-heavy.
            if max_len_extend <= 8 or (
                batch_size >= 16 and max_len_extend <= 128
            ):
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

    # BF16 D=64 SWA override. The per-tile key band is ~(BM + sw) wide
    # regardless of tile count, so BM=256 (base-tree win for plain causal)
    # wastes work. Smaller BM splits the static window across more CTAs
    # and hides memory latency. Gates tuned on D=64 H=32 kvH=4 sw=127.
    if (
        not is_fp8
        and Lq == 64
        and sliding_window_size > 0
        and sliding_window_size < max_len_extend
    ):
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
            _BM, _NS = 128, 4
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
    # Cache shape tuples once; downstream accesses use tuple indexing.
    _q_shape = q_extend.shape
    Lq = _q_shape[-1]
    Lv = v_extend.shape[-1]
    _kb_dtype = k_buffer.dtype
    _kv_is_fp8 = _kb_dtype is torch.float8_e4m3fn or _kb_dtype is torch.float8_e4m3fnuz
    _kv_was_fp8 = _kv_is_fp8
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
        if _use_fp8_kv_bf16_bridge():
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

    # Config-keyed dispatch cache (see module-level comment). The
    # basic-path cache is gated on `_uniform_like` because the basic
    # grid is `(B, H, ceil(max_ext / BM))` — heterogeneous batches
    # waste most of those tiles and route through WCA persistent.
    _basic_forced_only = (
        (_force_use_persistent is False or _force_use_persistent is None)
        and (_force_use_splitk is False or _force_use_splitk is None)
    )
    # Spec-decode-style small-ext + big-pfx-skew shapes look uniform but
    # the longest-prefix CTA dominates; WCA persistent regains ~1.77x,
    # so exclude them from the basic cache. D=128 only (below WCA branch).
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
    # Structural gate: no force_* overrides, no custom mask. Sinks and
    # window_kv_offsets are runtime args keyed via HAS_SINK /
    # SLIDING_WINDOW_SIZE, so they don't need to be part of this check.
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
        and not _CFG_CACHE_DISABLED
    )
    _has_sink = sinks is not None

    def _try_config_cache():
        """Return the cache entry for the current shape, or None on miss."""
        _pfx_b_local = _pfx_bucket(total_prefix_len, batch_size)
        _cfg_local = _get_basic_dispatch_config(
            Lq, batch_size, max_len_extend, _pfx_b_local, _kv_is_fp8,
            sliding_window_size=sliding_window_size,
            head_num=head_num,
        )
        # Key includes head_num/k_head_num (kv_group_num is baked), HAS_SINK
        # (distinct kernel bodies), and the indptr/indices dtypes (Triton
        # bakes pointer element type into SASS).
        return _config_cache.get(
            (_cfg_local, Lq, _kv_is_fp8,
             is_causal, logit_cap, sliding_window_size, xai_temperature_len,
             head_num, k_extend.shape[1], _has_sink,
             qo_indptr.dtype, kv_indptr.dtype, kv_indices.dtype)
        )

    def _run_config_cache_entry(_cc_entry):
        """Execute a cache hit.

        Strides and sm_scale are recomputed per call: both are HIPLauncher
        runtime args, and per-layer tensors may have different strides or
        KV scales than the install-time sample.
        """
        _mi_n = _total_extend_rows + 1
        _wkvo_n = batch_size if batch_size > 0 else 1
        if _dummy_mi_size < _mi_n or _dummy_wkvo_size < _wkvo_n or _dummy_sinks is None:
            _ensure_dummies(q_extend.device, _mi_n, _wkvo_n)
        _sinks_arg = sinks if sinks is not None else _dummy_sinks
        _wkvo_arg = (
            window_kv_offsets
            if window_kv_offsets is not None
            else _dummy_wkvo[:_wkvo_n]
        )
        _qs = q_extend.stride(); _ks = k_extend.stride(); _vs = v_extend.stride()
        _os = o_extend.stride(); _kbs = k_buffer.stride(); _vbs = v_buffer.stride()
        _live_strides = (
            _qs[0], _qs[1], _ks[0], _ks[1], _vs[0], _vs[1],
            _os[0], _os[1], _kbs[0], _kbs[1], _vbs[0], _vbs[1],
        )
        _sm_live = (sm_scale if sm_scale is not None else Lq**-0.5) * k_scale
        _tag = _cc_entry[0]
        if _tag == "fast":
            _, _cc_run, _cc_sm, _cc_strides, _cc_BM = _cc_entry
            _cc_run(
                q_extend, k_extend, v_extend, o_extend,
                k_buffer, v_buffer,
                qo_indptr, kv_indptr, kv_indices,
                _dummy_cm, _dummy_mi[:_mi_n],
                _wkvo_arg, _sinks_arg,
                _sm_live, _live_strides,
                (batch_size, head_num,
                 (max_len_extend + _cc_BM - 1) // _cc_BM),
            )
        else:
            # Legacy JITFunction.run path, kept as a forward-compat fallback.
            _, _cc_run, _cc_sm, _cc_gn, _cc_strides, _cc_kw, _cc_BM = _cc_entry
            if "Sinks" in _cc_kw:
                _cc_kw = {**_cc_kw, "Sinks": _sinks_arg}
            _cc_run(
                q_extend, k_extend, v_extend, o_extend,
                k_buffer, v_buffer,
                qo_indptr, kv_indptr, kv_indices,
                _dummy_cm, _dummy_mi[:_mi_n],
                _wkvo_arg,
                _sm_live, _cc_gn, *_live_strides,
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
        # Cache miss falls through to the full dispatch, which populates it.

    # Fast-path eligibility: no custom mask, no ``_force_*`` overrides.
    # ``_force_use_*=False`` is accepted (explicit "basic, please").
    _is_fast_eligible = (
        _force_block_m is None
        and _force_block_n is None
        and not _ck_v_preload
        and _basic_forced_only
        and custom_mask is None
    )
    _is_uniform = (batch_size <= 1 or min_len_extend == max_len_extend
                   or _uniform_by_shape)
    _ragged_routable = (
        _is_fast_eligible
        and (_force_use_persistent is not False)
        and (_force_use_splitk is not False)
    )
    _is_ragged_ext = _ragged_routable and not _is_uniform and batch_size >= 2
    # Small-ext + big-prefix-skew (e.g. spec-decode) — looks uniform but
    # the longest-prefix CTA dominates; WCA persistent regains ~1.7x.
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

    # FP8 KV + ragged on WCA persistent has ~10% per-element error
    # (BM=256 masked-tail x FP8 MFMA accumulation order); keep FP8 het
    # batches on the basic path until fixed.
    if _kv_is_fp8 and not _is_uniform:
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
            _wca_fn = _launch_persistent_fp8 if _kv_is_fp8 else _launch_persistent
            # BF16 persistent has three variants (thresholds from
            # bench_real_het.py): bsearch tile map for full-batch prefill,
            # WCA default for prefix-dominated chat-mix, and WCA small
            # (BM=64 NW=4 NS=2) for ext-dominated / heavily-skewed batches.
            # The force_* knobs only exist on the BF16 launcher.
            _wca_kw = dict()
            _avg_ext = _total_ext / max(1, batch_size)
            _use_bsearch = (
                (not _kv_is_fp8)
                and Lq == 128
                and batch_size >= 16
                and max_len_extend >= 2048
                and _avg_ext >= 512
            )
            if _use_bsearch:
                _wca_kw.update(_force_tile_map_mode=0)
            elif (not _kv_is_fp8) and Lq == 128 and _total_pfx_est < 4 * _total_ext:
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

    # Late config-cache lookup for het batches that bypassed the early
    # uniform path and then fell out of the ``_is_ragged`` block without
    # routing to WCA. Skip full dispatch on cache hit.
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
                    # Avoid the 6ms CPU sync that computing min_len_extend
                    # would cost; falling back to max is conservative.
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
                head_num=head_num,
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
        # First call: pay JITFunction.run once (compiles if needed), capture
        # the CompiledKernel, and install a direct HIPLauncher closure for
        # subsequent hits via `_make_fast_runner[_fp8]`.
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

        # Install the direct-HIPLauncher closure for future calls with
        # matching (config, shape, dtype) — including the indptr/indices
        # dtypes since Triton specializes CompiledKernel on pointer element
        # type.
        if (
            _force_block_n is None
            and _force_num_warps is None
            and custom_mask is None
            and not _CFG_CACHE_DISABLED
        ):
            _cfg = (_BM, _BN, _NW, _NS, _PAD_K, _PAD_V, _EXT_BN, _EXT_NS)
            _has_sink_w = sinks is not None
            _cc_key = (_cfg, Lq, _kv_is_fp8,
                       is_causal, logit_cap, sliding_window_size, xai_temperature_len,
                       head_num, k_extend.shape[1], _has_sink_w,
                       qo_indptr.dtype, kv_indptr.dtype, kv_indices.dtype)
            if _cc_key not in _config_cache:
                _runner_factory = _make_fast_runner_fp8 if _kv_is_fp8 else _make_fast_runner
                _config_cache[_cc_key] = (
                    "fast",
                    _runner_factory(_compiled, _frozen_kw, _kv_gn),
                    _sm, _frozen_strides, _BM,
                )
        return

    # Full dispatch path: heterogeneous batches, custom_mask, test overrides.
    _bump_dispatch("full_dispatch")

    BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL = _resolve_qk_split_dims(Lq)

    # min_len_extend is only needed below on the D>=256 / persistent / splitk
    # branches. Computing it eagerly would cost a ~15us GPU sync; compute
    # lazily at the point of use.

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
    elif (_force_use_splitk is False) or (_force_use_persistent is False):
        # At least one of splitk / persistent explicitly disabled — let
        # the remaining default rules below decide.
        pass
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
            # D<128 FP8: only the NW=4 NS=1 serial path is wired up.
            NUM_STAGES = _force_num_stages or 1
            num_warps = _force_num_warps or 4
            if _force_block_m is not None:
                BLOCK_M = _force_block_m
            elif batch_size >= 16 and max_len_extend <= 128:
                BLOCK_M = 64
            else:
                BLOCK_M = 128
        else:
            # D>=128 FP8 basic fallback (het / custom mask / forced).
            # Mirrors the fast-path defaults; NS>=2 required by the
            # 8-warp pipelined path.
            NUM_STAGES = _force_num_stages or 1
            num_warps = _force_num_warps if _force_num_warps is not None else 8
            if _force_block_m is not None:
                BLOCK_M = _force_block_m
            elif batch_size == 1 and max_len_extend <= 256:
                BLOCK_M = 64
            else:
                BLOCK_M = 128
    EXT_NUM_STAGES = NUM_STAGES

    # Correctness guards (D=64 NW=8 BM=256 is nondeterministic at NS!=2;
    # D>=256 NW=8 NS=1 races on the DMA ring).
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
    """Launch the FP8 persistent-CTA kernel, managing split-K workspace
    and the dummy mask tensors when ``USE_CUSTOM_MASK`` is False."""
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
    """Pick ``SPLIT_K`` and delegate to ``_launch_persistent_fp8``; falls
    back to SPLIT_K=1 when the tiles-per-CU / pfx-per-BN heuristics
    don't justify a multi-split reduction."""
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
