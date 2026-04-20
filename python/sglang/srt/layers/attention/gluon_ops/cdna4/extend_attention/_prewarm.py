# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""JIT-free prewarm for the extend-attention dispatch.

Walks every (BM, BN, NW, NS) config the dispatcher can select for a
given model and compiles the corresponding kernel variant in a parallel
thread pool, so live serving never JITs. After ``prewarm_extend_attention``
(or ``prewarm_for_model``) returns, both the on-disk Triton cache and the
in-process ``_config_cache`` fast-runner table are populated.

The full-model prewarm footprint is typically 10-15 kernel compiles; the
runtime-cache population phase then routes one tiny dummy call through
each unique config so the direct-HIPLauncher closures are installed.

Callers: ``TritonAttnBackend.__init__`` invokes ``prewarm_for_model``
with the model's HF config. FP8 kernels are populated lazily on the
first live call (their shape space is small and the saturation
warmup rounds in every serving bench already absorb the one-time
JIT cost). The ``include_fp8`` parameter is reserved for a future
Phase-1 FP8 warmup pass and currently raises ``NotImplementedError``
if set to ``True``.
"""

from __future__ import annotations

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Sequence

import torch
import triton

logger = logging.getLogger(__name__)

from ._bf16_extend_gfx950 import gluon_extend_attn_fwd as _kfn_basic
from ._bf16_extend_persistent_gfx950 import (
    gluon_extend_attn_fwd_persistent as _kfn_persistent,
)
from ._launch_helpers import _resolve_qk_split_dims
from .extend_attention_gfx950 import (
    _get_basic_dispatch_config,
)
from .extend_attention_gfx950 import gluon_extend_attention_fwd as _dispatch_fwd

# Shape grid used to enumerate dispatch outputs. Chosen to cover the
# realistic sglang serving range (B up to 128, ext up to 16384 tokens,
# prefix buckets 0..4). Exhaustive against the selector since the
# dispatch collapses everything to a small (BM,NW,NS) set.
_BATCH_GRID = (1, 2, 4, 8, 16, 32, 64, 128)
_EXTEND_GRID = (1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
_PFX_BUCKET_GRID = (0, 1, 2, 3, 4)


def enumerate_basic_configs(
    head_dim: int,
    is_fp8: bool = False,
    sliding_window_size: int = -1,
    head_num: Optional[int] = None,
) -> list[tuple]:
    """Unique (BM, BN, NW, NS, PAD_K, PAD_V, EXT_BN, EXT_NS) tuples the
    basic-path dispatch can emit for this head dim.

    Purely static: walks ``_get_basic_dispatch_config`` over the shape
    grid, no GPU calls. ``sliding_window_size`` and ``head_num`` must
    match the live model — SWA configs and the B=1 H-aware BM=256
    variants are distinct constexpr specializations, so omitting either
    lets the first live request JIT-compile the missing variant.
    """
    cfgs: set[tuple] = set()
    for b in _BATCH_GRID:
        for e in _EXTEND_GRID:
            for p in _PFX_BUCKET_GRID:
                cfgs.add(
                    _get_basic_dispatch_config(
                        head_dim,
                        b,
                        e,
                        p,
                        is_fp8,
                        sliding_window_size=sliding_window_size,
                        head_num=head_num,
                    )
                )
    return sorted(cfgs)


def enumerate_persistent_configs(head_dim: int) -> list[tuple]:
    """Unique (BM, BN, NW, NS, SPLIT_K) tuples the persistent launcher
    can emit.

    Mirrors the branches inside ``_launch_persistent`` — keep in sync
    with that function.
    """
    BLOCK_DMODEL, _ = _resolve_qk_split_dims(head_dim)
    BLOCK_N = 32 if BLOCK_DMODEL >= 256 else 64
    cfgs: set[tuple] = set()
    for batch_size in _BATCH_GRID:
        for max_len_extend in _EXTEND_GRID:
            # Branches on _het_ratio = max/min; sweep uniform + skewed
            # proxies to cover the _use_small_tile transitions.
            for min_len_extend in (
                max_len_extend,
                max(1, max_len_extend // 2),
                max(1, max_len_extend // 4),
                1,
            ):
                het_ratio = max_len_extend / max(min_len_extend, 1)
                use_small_tile = (
                    max_len_extend <= 128
                    or (max_len_extend <= 256 and het_ratio >= 2.0)
                    or (
                        min_len_extend < 64
                        and max_len_extend <= 512
                        and batch_size <= 4
                    )
                )

                if BLOCK_DMODEL >= 256 or use_small_tile:
                    BM, NW = 64, 4
                else:
                    # Launcher always clamps to BM=128 (BM=256 hits an
                    # iota_range compiler crash on persistent).
                    BM, NW = 128, 8

                if BLOCK_DMODEL >= 256:
                    NS = 1
                elif BM == 64 and NW == 4:
                    NS = 2
                elif BM == 64:
                    NS = 1
                else:
                    NS = 4

                if BM == 128 and NW == 8 and NS == 2:
                    NS = 3

                # SPLIT_K>1 is only emitted on BM=128 NW=8 NS=4; all
                # other tile shapes stay at SPLIT_K=1.
                split_set = (
                    (1, 2, 4, 8) if (BM == 128 and NW == 8 and NS == 4) else (1,)
                )
                for split_k in split_set:
                    cfgs.add((BM, BLOCK_N, NW, NS, split_k))
    return sorted(cfgs)


def _make_dummy_tensors(
    device: torch.device,
    dtype: torch.dtype,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    has_sink: bool,
):
    """Minimal dummies with realistic strides so the binder specializes
    identically to a runtime call.

    Dtype matrix MUST match SGLang's ``triton_backend.py``:
      qo_indptr / kv_indices / mask_indptr: int64
      kv_indptr / window_kv_offsets:        int32

    Triton bakes each tensor's pointer element type into SASS, so a
    dtype mismatch silently strides past allocated memory instead of
    raising — it just faults asynchronously on the first OOB access.
    """
    q = torch.zeros((1, num_q_heads, head_dim), dtype=dtype, device=device)
    k = torch.zeros((1, num_kv_heads, head_dim), dtype=dtype, device=device)
    v = torch.zeros((1, num_kv_heads, head_dim), dtype=dtype, device=device)
    o = torch.zeros((1, num_q_heads, head_dim), dtype=dtype, device=device)
    # k/v buffers: need at least 1 entry; strides match the real KV cache.
    kb = torch.zeros((2, num_kv_heads, head_dim), dtype=dtype, device=device)
    vb = torch.zeros((2, num_kv_heads, head_dim), dtype=dtype, device=device)
    qo_indptr = torch.tensor([0, 1], dtype=torch.int64, device=device)
    kv_indptr = torch.tensor([0, 0], dtype=torch.int32, device=device)
    kv_indices = torch.zeros(1, dtype=torch.int64, device=device)
    custom_mask = torch.empty(0, dtype=torch.uint8, device=device)
    mask_indptr = torch.zeros(2, dtype=torch.int64, device=device)
    window_kv_offsets = torch.zeros(1, dtype=torch.int32, device=device)
    partial_out = torch.empty(1, dtype=torch.float32, device=device)
    partial_lse = torch.empty(1, dtype=torch.float32, device=device)
    # tile_done is an int32 atomic counter (see _ensure_splitk_workspace);
    # the dummy must match the runtime dtype so the persistent/split-K
    # kernel doesn't specialize on a float32 pointer and JIT-miss at
    # first live call.
    tile_done = torch.empty(1, dtype=torch.int32, device=device)
    cum_tiles_per_batch = torch.zeros(2, dtype=torch.int32, device=device)
    # HAS_SINK dtype must match the runtime tensor (bf16 for GPT-OSS).
    sinks = torch.zeros(num_q_heads, dtype=dtype, device=device) if has_sink else None
    return {
        "q": q,
        "k": k,
        "v": v,
        "o": o,
        "kb": kb,
        "vb": vb,
        "qo_indptr": qo_indptr,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "custom_mask": custom_mask,
        "mask_indptr": mask_indptr,
        "window_kv_offsets": window_kv_offsets,
        "partial_out": partial_out,
        "partial_lse": partial_lse,
        "tile_done": tile_done,
        "cum_tiles_per_batch": cum_tiles_per_batch,
        "sinks": sinks,
    }


def _warm_basic_variant(
    tensors: dict,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    config: tuple,
    is_causal: bool,
    has_sink: bool,
    sliding_window_size: int,
    logit_cap: float,
    xai_temperature_len: int,
):
    """Compile one basic-kernel variant via ``warmup=True`` and return
    the ``CompiledKernel`` so callers can install it in the dispatch
    cache for the direct-HIPLauncher bypass path."""
    BM, BN, NW, NS, PAD_K, PAD_V, EXT_BN, EXT_NS = config
    BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL = _resolve_qk_split_dims(head_dim)
    kv_group_num = num_q_heads // num_kv_heads
    sm_scale = 1.0 / math.sqrt(head_dim)

    q, k, v, o = tensors["q"], tensors["k"], tensors["v"], tensors["o"]
    kb, vb = tensors["kb"], tensors["vb"]

    return _kfn_basic.run(
        q,
        k,
        v,
        o,
        kb,
        vb,
        tensors["qo_indptr"],
        tensors["kv_indptr"],
        tensors["kv_indices"],
        tensors["custom_mask"],
        tensors["mask_indptr"],
        tensors["window_kv_offsets"],
        sm_scale,
        kv_group_num,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        o.stride(0),
        o.stride(1),
        kb.stride(0),
        kb.stride(1),
        vb.stride(0),
        vb.stride(1),
        IS_CAUSAL=is_causal,
        USE_CUSTOM_MASK=False,
        ENABLE_PREFIX_UNMASKED=is_causal,
        BLOCK_M=BM,
        BLOCK_N=BN,
        BLOCK_DMODEL=BLOCK_DMODEL,
        ACTUAL_BLOCK_DMODEL=ACTUAL_BLOCK_DMODEL,
        NUM_STAGES=NS,
        Sinks=tensors["sinks"],
        HAS_SINK=has_sink,
        LOGIT_CAP=logit_cap,
        XAI_TEMPERATURE_LEN=xai_temperature_len,
        SLIDING_WINDOW_SIZE=sliding_window_size,
        v_scale=1.0,
        num_warps=NW,
        num_stages=1,
        waves_per_eu=2,
        matrix_instr_nonkdim=32,
        grid=(1, num_q_heads, 1),
        warmup=True,
    )


def _warm_persistent_variant(
    tensors: dict,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    config: tuple,
    is_causal: bool,
    has_sink: bool,
    sliding_window_size: int,
    logit_cap: float,
    xai_temperature_len: int,
    tile_map_mode: int = 0,
):
    """Compile one persistent-kernel variant (inc. split-K).

    ``tile_map_mode``: 0 = cum_tiles binary search, 1 = WCA inline scan.
    """
    BM, BN, NW, NS, SPLIT_K = config
    BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL = _resolve_qk_split_dims(head_dim)
    kv_group_num = num_q_heads // num_kv_heads
    sm_scale = 1.0 / math.sqrt(head_dim)

    q, k, v, o = tensors["q"], tensors["k"], tensors["v"], tensors["o"]
    kb, vb = tensors["kb"], tensors["vb"]

    return _kfn_persistent.run(
        q,
        k,
        v,
        o,
        kb,
        vb,
        tensors["qo_indptr"],
        tensors["kv_indptr"],
        tensors["kv_indices"],
        tensors["custom_mask"],
        tensors["mask_indptr"],
        tensors["window_kv_offsets"],
        sm_scale,
        kv_group_num,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        o.stride(0),
        o.stride(1),
        kb.stride(0),
        kb.stride(1),
        vb.stride(0),
        vb.stride(1),
        IS_CAUSAL=is_causal,
        USE_CUSTOM_MASK=False,
        ENABLE_PREFIX_UNMASKED=is_causal,
        BLOCK_M=BM,
        BLOCK_N=BN,
        BLOCK_DMODEL=BLOCK_DMODEL,
        ACTUAL_BLOCK_DMODEL=ACTUAL_BLOCK_DMODEL,
        NUM_STAGES=NS,
        Sinks=tensors["sinks"],
        HAS_SINK=has_sink,
        LOGIT_CAP=logit_cap,
        XAI_TEMPERATURE_LEN=xai_temperature_len,
        SLIDING_WINDOW_SIZE=sliding_window_size,
        v_scale=1.0,
        num_heads=num_q_heads,
        n_m_tiles=1,
        total_valid_tiles=1,
        total_programs=1,
        partial_out=tensors["partial_out"],
        partial_lse=tensors["partial_lse"],
        tile_done=tensors["tile_done"],
        cum_tiles_per_batch=tensors["cum_tiles_per_batch"],
        actual_batch_size=1,
        IS_PERSISTENT=True,
        SPLIT_K=SPLIT_K,
        MAX_BATCH_LOG2=8,
        TILE_MAP_MODE=tile_map_mode,
        num_warps=NW,
        num_stages=1,
        waves_per_eu=2,
        matrix_instr_nonkdim=32,
        grid=(1,),
        warmup=True,
    )


def _build_work_list(
    head_dim: int,
    is_causal_modes: Sequence[bool],
    include_basic: bool,
    include_persistent: bool,
    persistent_tile_map_modes: Sequence[int] = (0, 1),
    sliding_window_size: int = -1,
    head_num: Optional[int] = None,
) -> list[tuple]:
    """Build the list of ``(path, config, is_causal, tile_map_mode)``
    tuples to warm.

    ``path`` is one of ``{"basic", "persistent"}``. Persistent
    variants are keyed on ``TILE_MAP_MODE`` as well — each mode is a
    distinct constexpr specialization. FP8 variants are not emitted
    here; they are compiled lazily by Phase 2's live dispatch when
    the model's KV dtype is FP8.
    """
    work: list[tuple] = []
    basic_cfgs = enumerate_basic_configs(
        head_dim,
        is_fp8=False,
        sliding_window_size=sliding_window_size,
        head_num=head_num,
    )
    persistent_cfgs = enumerate_persistent_configs(head_dim)
    for is_causal in is_causal_modes:
        if include_basic:
            for cfg in basic_cfgs:
                work.append(("basic", cfg, is_causal, 0))
        if include_persistent:
            for cfg in persistent_cfgs:
                for tmm in persistent_tile_map_modes:
                    work.append(("persistent", cfg, is_causal, tmm))
    return work


def prewarm_extend_attention(
    *,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bfloat16,
    # Model-level constants (become kernel constexprs)
    is_causal_modes: Sequence[bool] = (True,),
    has_sink: bool = False,
    sliding_window_size: int = -1,
    logit_cap: float = 0.0,
    xai_temperature_len: int = -1,
    # Scope knobs
    include_basic: bool = True,
    include_persistent: bool = True,
    include_fp8: bool = False,
    # Concurrency
    parallel: int = 4,
    # Diagnostics
    verbose: bool = False,
) -> dict:
    """Compile every kernel variant the dispatch can produce for ONE
    (attention-pattern, head-dim) pair.

    For models that mix attention patterns across layers (GPT-OSS,
    Gemma 3, Qwen 3, Llama 4 all do this), call ``prewarm_for_model``
    instead — it walks every unique pattern the HF config exposes and
    de-dupes the work list.

    ``parallel > 0`` compiles through ``triton.AsyncCompileMode`` +
    a thread pool; Triton releases the GIL on the heavy LLVM passes,
    so threading gives near-linear speedup. ``parallel = 0`` is
    sequential. Returns a dict with timings and variant counts.
    """
    if include_fp8:
        raise NotImplementedError(
            "include_fp8=True is reserved for a future Phase-1 FP8 warmup "
            "pass. FP8 kernels are compiled lazily on the first live "
            "call; the saturation warmup rounds in every standard serving "
            "bench absorb this one-time cost. Pass include_fp8=False."
        )

    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())

    tensors = _make_dummy_tensors(
        device,
        dtype,
        num_q_heads,
        num_kv_heads,
        head_dim,
        has_sink,
    )

    work = _build_work_list(
        head_dim,
        is_causal_modes,
        include_basic,
        include_persistent,
        sliding_window_size=sliding_window_size,
        head_num=num_q_heads,
    )
    if verbose:
        logger.info(
            "prewarm: %d variants (basic=%d persistent=%d) for D=%d H=%d kvH=%d",
            len(work),
            sum(1 for w in work if w[0] == "basic"),
            sum(1 for w in work if w[0] == "persistent"),
            head_dim,
            num_q_heads,
            num_kv_heads,
        )

    def _run_one(item):
        path, cfg, is_causal, tile_map_mode = item
        t0 = time.time()
        if path == "basic":
            _warm_basic_variant(
                tensors,
                num_q_heads,
                num_kv_heads,
                head_dim,
                cfg,
                is_causal=is_causal,
                has_sink=has_sink,
                sliding_window_size=sliding_window_size,
                logit_cap=logit_cap,
                xai_temperature_len=xai_temperature_len,
            )
        elif path == "persistent":
            _warm_persistent_variant(
                tensors,
                num_q_heads,
                num_kv_heads,
                head_dim,
                cfg,
                is_causal=is_causal,
                has_sink=has_sink,
                sliding_window_size=sliding_window_size,
                logit_cap=logit_cap,
                xai_temperature_len=xai_temperature_len,
                tile_map_mode=tile_map_mode,
            )
        else:
            raise ValueError(f"unknown path {path}")
        dt = time.time() - t0
        if verbose:
            logger.info(
                "  [%s] cfg=%s causal=%s tmm=%d: %.2fs",
                path,
                cfg,
                is_causal,
                tile_map_mode,
                dt,
            )
        return dt

    wall_start = time.time()
    seq_total = 0.0

    if parallel and parallel > 0 and len(work) > 1:
        executor = ThreadPoolExecutor(max_workers=parallel)
        try:
            with triton.AsyncCompileMode(executor):
                for item in work:
                    seq_total += _run_one(item)
        finally:
            executor.shutdown(wait=True)
    else:
        for item in work:
            seq_total += _run_one(item)

    # Phase 2: real-launch pass through the live dispatch.
    # Phase 1 (warmup=True) compiles AMDGCN but leaves two runtime
    # costs unpaid: hipModuleLoadData (~14ms per unique kernel) and
    # `_config_cache` installation (~20ms via JITFunction.run). Phase 2
    # routes a tiny call through each dispatch path so both costs are
    # paid before first serving token.
    phase2_start = time.time()
    try:
        _populate_runtime_caches(
            device=device,
            dtype=dtype,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            is_causal_modes=is_causal_modes,
            has_sink=has_sink,
            sliding_window_size=sliding_window_size,
            logit_cap=logit_cap,
            xai_temperature_len=xai_temperature_len,
            include_basic=include_basic,
            include_persistent=include_persistent,
            verbose=verbose,
        )
    except Exception as e:
        if verbose:
            logger.warning("prewarm phase 2 (real-launch) failed: %r", e)
    phase2_time = time.time() - phase2_start

    wall_time = time.time() - wall_start

    result = {
        "num_variants": len(work),
        "wall_time": wall_time,
        "sequential_time_est": seq_total,
        "phase2_time": phase2_time,
        "head_dim": head_dim,
        "parallel": parallel,
    }
    if verbose:
        logger.info(
            "prewarm: done. %d variants in %.2fs wall "
            "(phase1 compile %.2fs, phase2 real-launch %.2fs, seq estimate %.2fs)",
            len(work),
            wall_time,
            wall_time - phase2_time,
            phase2_time,
            seq_total,
        )
    return result


# Runtime-cache population (HIP module load + _config_cache).


def _make_realistic_tensors(
    device: torch.device,
    dtype: torch.dtype,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    batch_size: int,
    max_ext: int,
    max_pfx: int,
    kv_pool_size: int,
    has_sink: bool,
    qo_indptr_dtype: torch.dtype = torch.int64,
):
    """Tensors matching an attention backend's runtime extend call.

    Dtype contract varies by backend (triton_backend uses qo_indptr=int64,
    aiter_backend uses int32). ``_config_cache`` keys on the dtype triple,
    so each combo needs its own prewarm pass; ``qo_indptr_dtype`` selects.
    """
    total_ext = max(1, batch_size * max_ext)
    total_pfx = max(0, batch_size * max_pfx)
    q = torch.zeros((total_ext, num_q_heads, head_dim), dtype=dtype, device=device)
    k = torch.zeros((total_ext, num_kv_heads, head_dim), dtype=dtype, device=device)
    v = torch.zeros((total_ext, num_kv_heads, head_dim), dtype=dtype, device=device)
    o = torch.zeros((total_ext, num_q_heads, head_dim), dtype=dtype, device=device)
    kb = torch.zeros(
        (max(2, kv_pool_size), num_kv_heads, head_dim), dtype=dtype, device=device
    )
    vb = torch.zeros(
        (max(2, kv_pool_size), num_kv_heads, head_dim), dtype=dtype, device=device
    )
    qo_indptr = (
        torch.arange(0, batch_size + 1, dtype=qo_indptr_dtype, device=device) * max_ext
    )
    kv_indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * max_pfx
    )
    kv_indices = (
        torch.arange(total_pfx, dtype=torch.int64, device=device) % max(1, kv_pool_size)
        if total_pfx > 0
        else torch.zeros(0, dtype=torch.int64, device=device)
    )
    mask_indptr = torch.zeros(batch_size + 1, dtype=torch.int64, device=device)
    # Sinks dtype must match runtime (SGLang's GPT-OSS path uses bf16
    # under Triton backend). Using fp32 here would compile a variant
    # runtime never hits, defeating the prewarm.
    sinks = torch.zeros(num_q_heads, dtype=dtype, device=device) if has_sink else None
    return q, k, v, o, kb, vb, qo_indptr, kv_indptr, kv_indices, mask_indptr, sinks


def _enumerate_basic_shape_reps(
    head_dim: int,
    sliding_window_size: int = -1,
    head_num: Optional[int] = None,
) -> list[tuple]:
    """One representative ``(batch_size, max_ext, total_prefix_len)``
    tuple per unique basic dispatch output.

    Natural dispatch (no ``_force_*`` overrides) is required so the
    cache-install branch inside ``gluon_extend_attention_fwd`` fires.
    ``head_num`` must match the live model — B=1 dispatch branches on
    H_q and mismatched warming would miss the live cache key.
    """
    _gcfg = _get_basic_dispatch_config
    pfx_avg_for_bucket = {0: 0, 1: 256, 2: 1024, 3: 4096, 4: 16384}
    seen: dict[tuple, tuple] = {}
    for b in _BATCH_GRID:
        for e in _EXTEND_GRID:
            for p in _PFX_BUCKET_GRID:
                cfg = _gcfg(
                    head_dim,
                    b,
                    e,
                    p,
                    False,
                    sliding_window_size=sliding_window_size,
                    head_num=head_num,
                )
                if cfg in seen:
                    continue
                # Keeping the (b, e) pair uniform ensures `_uniform_like`
                # is True at dispatch time so the fast-path cache install
                # branch fires.
                pfx_total = b * pfx_avg_for_bucket[p]
                seen[cfg] = (b, e, pfx_total)
    return list(seen.values())


def _warm_persistent_dispatch_cases(
    *,
    device: torch.device,
    dtype: torch.dtype,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    is_causal: bool,
    has_sink: bool,
    sliding_window_size: int,
    logit_cap: float,
    xai_temperature_len: int,
    verbose: bool,
    qo_dt: torch.dtype,
    kv_pool_size: int,
):
    """Run the 4 persistent-dispatch warmup cases for one
    ``(is_causal, qo_dt)`` combo. Split out so the outer loop can iterate
    over ``qo_indptr`` dtypes without an extra indent level."""
    sm_scale_local = 1.0 / math.sqrt(head_dim)

    # Case 1: big-ext het -> WCA small / WCA default.
    bs = 4
    elens = [4096, 2048, 1024, 512]
    pfx_per = 0
    q, k, v, o, kb, vb, _, kv_ip, kv_ix, mi_ip, sinks = _make_realistic_tensors(
        device,
        dtype,
        num_q_heads,
        num_kv_heads,
        head_dim,
        bs,
        max(elens),
        pfx_per,
        kv_pool_size,
        has_sink,
    )
    total_ext = sum(elens)
    qo_ip = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(elens), 0).tolist()),
        dtype=qo_dt,
        device=device,
    )
    q = q[:total_ext].contiguous()
    k = k[:total_ext].contiguous()
    v = v[:total_ext].contiguous()
    o = o[:total_ext].contiguous()
    try:
        _dispatch_fwd(
            q,
            k,
            v,
            o,
            kb,
            vb,
            qo_ip,
            kv_ip,
            kv_ix,
            None,
            is_causal,
            mi_ip,
            max(elens),
            k_scale=1.0,
            v_scale=1.0,
            sm_scale=sm_scale_local,
            logit_cap=logit_cap,
            sliding_window_size=sliding_window_size,
            sinks=sinks,
            xai_temperature_len=xai_temperature_len,
            total_prefix_len=0,
            total_extend_len=total_ext,
            min_len_extend=min(elens),
        )
    except Exception as e:
        if verbose:
            logger.warning("phase2 persistent big-het causal=%s: %r", is_causal, e)

    # Case 2: spec-decode style (small ext, big pfx skew) -> D=128 WCA.
    bs = 8
    elens = [1] * 8
    plens = [4096] * 8
    total_pfx = sum(plens)
    q, k, v, o, kb, vb, _, _, _, mi_ip, sinks = _make_realistic_tensors(
        device,
        dtype,
        num_q_heads,
        num_kv_heads,
        head_dim,
        bs,
        1,
        0,
        kv_pool_size,
        has_sink,
    )
    total_ext = sum(elens)
    qo_ip = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(elens), 0).tolist()),
        dtype=qo_dt,
        device=device,
    )
    kv_ip = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(plens), 0).tolist()),
        dtype=torch.int32,
        device=device,
    )
    kv_ix = torch.arange(total_pfx, dtype=torch.int64, device=device) % kv_pool_size
    q = q[:total_ext].contiguous()
    k = k[:total_ext].contiguous()
    v = v[:total_ext].contiguous()
    o = o[:total_ext].contiguous()
    try:
        _dispatch_fwd(
            q,
            k,
            v,
            o,
            kb,
            vb,
            qo_ip,
            kv_ip,
            kv_ix,
            None,
            is_causal,
            mi_ip,
            1,
            k_scale=1.0,
            v_scale=1.0,
            sm_scale=sm_scale_local,
            logit_cap=logit_cap,
            sliding_window_size=sliding_window_size,
            sinks=sinks,
            xai_temperature_len=xai_temperature_len,
            total_prefix_len=total_pfx,
            total_extend_len=total_ext,
            min_len_extend=1,
        )
    except Exception as e:
        if verbose:
            logger.warning("phase2 persistent spec-decode causal=%s: %r", is_causal, e)

    # Case 3: big-batch heterogeneous prefill -> bsearch.
    elens = [
        4096,
        4096,
        2048,
        2048,
        2048,
        2048,
        1024,
        1024,
        1024,
        1024,
        512,
        512,
        512,
        512,
        512,
        512,
    ]
    plens = [
        8192,
        4096,
        2048,
        0,
        2048,
        1024,
        8192,
        4096,
        2048,
        0,
        4096,
        2048,
        1024,
        0,
        2048,
        1024,
    ]
    bs = len(elens)
    total_ext = sum(elens)
    total_pfx = sum(plens)
    q, k, v, o, kb, vb, _, _, _, mi_ip, sinks = _make_realistic_tensors(
        device,
        dtype,
        num_q_heads,
        num_kv_heads,
        head_dim,
        bs,
        max(elens),
        0,
        kv_pool_size,
        has_sink,
    )
    qo_ip = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(elens), 0).tolist()),
        dtype=qo_dt,
        device=device,
    )
    kv_ip = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(plens), 0).tolist()),
        dtype=torch.int32,
        device=device,
    )
    kv_ix = (
        torch.arange(
            max(total_pfx, 1),
            dtype=torch.int64,
            device=device,
        )
        % kv_pool_size
    )
    q = q[:total_ext].contiguous()
    k = k[:total_ext].contiguous()
    v = v[:total_ext].contiguous()
    o = o[:total_ext].contiguous()
    try:
        _dispatch_fwd(
            q,
            k,
            v,
            o,
            kb,
            vb,
            qo_ip,
            kv_ip,
            kv_ix,
            None,
            is_causal,
            mi_ip,
            max(elens),
            k_scale=1.0,
            v_scale=1.0,
            sm_scale=sm_scale_local,
            logit_cap=logit_cap,
            sliding_window_size=sliding_window_size,
            sinks=sinks,
            xai_temperature_len=xai_temperature_len,
            total_prefix_len=total_pfx,
            total_extend_len=total_ext,
            min_len_extend=min(elens),
        )
    except Exception as e:
        if verbose:
            logger.warning("phase2 persistent bsearch causal=%s: %r", is_causal, e)

    # Case 4: chat-mix style het+pfx -> WCA default (D>=128 only).
    if head_dim >= 128:
        bs = 8
        elens = [512, 256, 128, 64, 512, 256, 128, 64]
        plens = [4096, 2048, 1024, 0, 4096, 2048, 1024, 0]
        total_ext = sum(elens)
        total_pfx = sum(plens)
        q, k, v, o, kb, vb, _, _, _, mi_ip, sinks = _make_realistic_tensors(
            device,
            dtype,
            num_q_heads,
            num_kv_heads,
            head_dim,
            bs,
            max(elens),
            0,
            kv_pool_size,
            has_sink,
        )
        qo_ip = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(elens), 0).tolist()),
            dtype=qo_dt,
            device=device,
        )
        kv_ip = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(plens), 0).tolist()),
            dtype=torch.int32,
            device=device,
        )
        kv_ix = torch.arange(total_pfx, dtype=torch.int64, device=device) % kv_pool_size
        q = q[:total_ext].contiguous()
        k = k[:total_ext].contiguous()
        v = v[:total_ext].contiguous()
        o = o[:total_ext].contiguous()
        try:
            _dispatch_fwd(
                q,
                k,
                v,
                o,
                kb,
                vb,
                qo_ip,
                kv_ip,
                kv_ix,
                None,
                is_causal,
                mi_ip,
                max(elens),
                k_scale=1.0,
                v_scale=1.0,
                sm_scale=sm_scale_local,
                logit_cap=logit_cap,
                sliding_window_size=sliding_window_size,
                sinks=sinks,
                xai_temperature_len=xai_temperature_len,
                total_prefix_len=total_pfx,
                total_extend_len=total_ext,
                min_len_extend=min(elens),
            )
        except Exception as e:
            if verbose:
                logger.warning(
                    "phase2 persistent wca-default causal=%s: %r",
                    is_causal,
                    e,
                )


def _populate_runtime_caches(
    device: torch.device,
    dtype: torch.dtype,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    is_causal_modes: Sequence[bool],
    has_sink: bool,
    sliding_window_size: int,
    logit_cap: float,
    xai_temperature_len: int,
    include_basic: bool,
    include_persistent: bool,
    verbose: bool,
    qo_indptr_dtypes: Sequence[torch.dtype] = (torch.int64, torch.int32),
):
    """Route a small real call through ``gluon_extend_attention_fwd``
    for each unique basic config and each persistent variant. Loads the
    HIP module into the runtime and populates ``_config_cache`` so the
    first live call hits the fast path.

    Uses natural dispatch (no ``_force_*`` args) so the cache-install
    branch inside the dispatcher fires. Warms both int64 and int32
    ``qo_indptr`` conventions (triton_backend vs aiter_backend) since
    the cache keys on dtype.
    """
    if include_basic:
        kv_pool_size = 64 * 1024
        shape_reps = _enumerate_basic_shape_reps(
            head_dim,
            sliding_window_size=sliding_window_size,
            head_num=num_q_heads,
        )
        for qo_dt in qo_indptr_dtypes:
            for is_causal in is_causal_modes:
                for bs, ext, pfx_total in shape_reps:
                    q, k, v, o, kb, vb, qo_ip, kv_ip, kv_ix, mi_ip, sinks = (
                        _make_realistic_tensors(
                            device,
                            dtype,
                            num_q_heads,
                            num_kv_heads,
                            head_dim,
                            bs,
                            ext,
                            pfx_total // max(1, bs),
                            kv_pool_size,
                            has_sink,
                            qo_indptr_dtype=qo_dt,
                        )
                    )
                    try:
                        _dispatch_fwd(
                            q,
                            k,
                            v,
                            o,
                            kb,
                            vb,
                            qo_ip,
                            kv_ip,
                            kv_ix,
                            None,
                            is_causal,
                            mi_ip,
                            ext,
                            k_scale=1.0,
                            v_scale=1.0,
                            sm_scale=1.0 / math.sqrt(head_dim),
                            logit_cap=logit_cap,
                            sliding_window_size=sliding_window_size,
                            sinks=sinks,
                            xai_temperature_len=xai_temperature_len,
                            total_prefix_len=pfx_total,
                            total_extend_len=bs * ext,
                            min_len_extend=ext,
                        )
                    except Exception as e:
                        if verbose:
                            logger.warning(
                                "phase2 basic bs=%d ext=%d pfx=%d causal=%s qo_dt=%s failed: %r",
                                bs,
                                ext,
                                pfx_total,
                                is_causal,
                                qo_dt,
                                e,
                            )

    # Persistent path: ragged batches to exercise WCA and bsearch.
    if include_persistent:
        kv_pool_size = 128 * 1024
        for qo_dt in qo_indptr_dtypes:
            for is_causal in is_causal_modes:
                _warm_persistent_dispatch_cases(
                    device=device,
                    dtype=dtype,
                    num_q_heads=num_q_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    is_causal=is_causal,
                    has_sink=has_sink,
                    sliding_window_size=sliding_window_size,
                    logit_cap=logit_cap,
                    xai_temperature_len=xai_temperature_len,
                    verbose=verbose,
                    qo_dt=qo_dt,
                    kv_pool_size=kv_pool_size,
                )

    torch.cuda.synchronize(device)


# Model-aware prewarm.
#
# Real models mix attention patterns across layers (e.g. GPT-OSS
# alternates full / SWA, Gemma 3 uses 5:1 local:global, Llama 4 iRoPE,
# Qwen 3 ``max_window_layers`` split). Each distinct pattern is a
# unique kernel variant because ``SLIDING_WINDOW_SIZE``, ``HAS_SINK``,
# ``LOGIT_CAP``, and ``XAI_TEMPERATURE_LEN`` are all constexprs.
# ``prewarm_for_model`` enumerates every pattern, de-dupes, and warms
# each unique one.


def enumerate_layer_patterns(
    layers: Sequence[dict],
) -> list[tuple]:
    """De-dupe per-layer attention configs into unique constexpr tuples.

    Each ``layers[i]`` is a dict with keys ``sliding_window_size``,
    ``has_sink``, ``logit_cap``, ``xai_temperature_len``, ``is_causal``.
    Returns a sorted list of the distinct combinations so callers can
    walk unique patterns instead of all N layers.
    """
    seen: set[tuple] = set()
    for L in layers:
        seen.add(
            (
                int(L.get("sliding_window_size", -1)),
                bool(L.get("has_sink", False)),
                float(L.get("logit_cap", 0.0)),
                int(L.get("xai_temperature_len", -1)),
                bool(L.get("is_causal", True)),
            )
        )
    return sorted(seen)


# Built-in model presets: each returns the per-layer attention configs
# for the named architecture. Covers the common sglang target models
# for gfx950.


def spec_gpt_oss(num_layers: int = 36, sliding_window: int = 128) -> list[dict]:
    """GPT-OSS 20B (24 layers) / 120B (36 layers): alternating full /
    SWA(=128), attention sinks on every layer. D=64, GQA 64->8."""
    return [
        {
            "sliding_window_size": sliding_window if (i % 2 == 1) else -1,
            "has_sink": True,
            "logit_cap": 0.0,
            "xai_temperature_len": -1,
            "is_causal": True,
        }
        for i in range(num_layers)
    ]


def spec_gemma3(num_layers: int = 26, sliding_window: int = 1024) -> list[dict]:
    """Gemma 3 27B: 5:1 local(SWA=1024):global(full) pattern, no sinks."""
    return [
        {
            "sliding_window_size": -1 if ((i + 1) % 6 == 0) else sliding_window,
            "has_sink": False,
            "logit_cap": 0.0,
            "xai_temperature_len": -1,
            "is_causal": True,
        }
        for i in range(num_layers)
    ]


def spec_qwen3(
    num_layers: int = 32,
    max_window_layers: int = 28,
    sliding_window: int = 4096,
    use_sliding_window: bool = True,
) -> list[dict]:
    """Qwen 3: layers < ``max_window_layers`` are full, the rest are SWA
    when ``use_sliding_window``. Many Qwen 3 configs ship with SWA off —
    pass ``use_sliding_window=False`` to warm a single pattern."""
    patterns = []
    for i in range(num_layers):
        sw = sliding_window if (use_sliding_window and i >= max_window_layers) else -1
        patterns.append(
            {
                "sliding_window_size": sw,
                "has_sink": False,
                "logit_cap": 0.0,
                "xai_temperature_len": -1,
                "is_causal": True,
            }
        )
    return patterns


def spec_llama4(num_layers: int = 48, sliding_window: int = 8191) -> list[dict]:
    """Llama 4 (iRoPE): 3-of-4 chunked-window layers, every 4th full NoPE.

    Matches ``llama4.py``'s ``use_rope = (layer_id + 1) % 4 != 0``.
    ``sliding_window`` is SGLang's exclusive value (HF's 8192-token
    ``attention_chunk_size`` maps to 8191 exclusive).
    """
    return [
        {
            "sliding_window_size": sliding_window if ((i + 1) % 4 != 0) else -1,
            "has_sink": False,
            "logit_cap": 0.0,
            "xai_temperature_len": -1,
            "is_causal": True,
        }
        for i in range(num_layers)
    ]


def spec_gemma2(num_layers: int = 42, sliding_window: int = 4095) -> list[dict]:
    """Gemma 2 (9B/27B): alternating SWA/full (SWA on even layer ids),
    attn_logit_softcapping=50.0. Matches ``gemma2.py``."""
    return [
        {
            "sliding_window_size": sliding_window if (i % 2 == 0) else -1,
            "has_sink": False,
            "logit_cap": 50.0,
            "xai_temperature_len": -1,
            "is_causal": True,
        }
        for i in range(num_layers)
    ]


def spec_cohere2(
    num_layers: int = 40,
    sliding_window: int = 4096,
    layer_types: Optional[Sequence[str]] = None,
) -> list[dict]:
    """Cohere Command R+ v2: 3 sliding + 1 full, repeated. Cohere v2
    uses the raw HF ``sliding_window`` value (no off-by-one)."""
    if layer_types is None:
        layer_types = [
            "full_attention" if ((i + 1) % 4 == 0) else "sliding_attention"
            for i in range(num_layers)
        ]
    out = []
    for lt in layer_types:
        out.append(
            {
                "sliding_window_size": (
                    sliding_window if lt == "sliding_attention" else -1
                ),
                "has_sink": False,
                "logit_cap": 0.0,
                "xai_temperature_len": -1,
                "is_causal": True,
            }
        )
    return out


def spec_grok(
    num_layers: int = 64,
    logit_cap: float = 30.0,
    xai_temperature_len: int = -1,
) -> list[dict]:
    """Grok-1: uniform full attention with LOGIT_CAP=30 (default).
    Pass ``xai_temperature_len > 0`` to warm the temperature-routing
    variant."""
    return [
        {
            "sliding_window_size": -1,
            "has_sink": False,
            "logit_cap": logit_cap,
            "xai_temperature_len": xai_temperature_len,
            "is_causal": True,
        }
        for _ in range(num_layers)
    ]


def spec_llama3(num_layers: int = 32) -> list[dict]:
    """Uniform full-attention, no sinks / caps / sliding.

    Covers Llama 3.x, Qwen 2 / 2.5, Mistral 7B, Mixtral, DBRX, OLMo 2
    (no-``layer_types`` configs), Arcee, and the dense-decoder family
    generally.
    """
    return [
        {
            "sliding_window_size": -1,
            "has_sink": False,
            "logit_cap": 0.0,
            "xai_temperature_len": -1,
            "is_causal": True,
        }
        for _ in range(num_layers)
    ]


# Named preset dispatch. Tuples are ``(head_dim, H_q, H_kv, layer_spec)``.
# Values from each model's public HF config. Alphabetised within family
# blocks.
MODEL_PRESETS = {
    # Cohere Command R+ v2 (interleaved SWA, D=128, GQA 96->8)
    "cohere-command-r-plus": lambda: (128, 96, 8, spec_cohere2(num_layers=64)),
    # DBRX (Mosaic, dense full attention, D=128, 40 heads MHA -> 8 KV GQA)
    "dbrx-instruct": lambda: (128, 48, 8, spec_llama3(num_layers=40)),
    # Gemma 2 (alternating SWA, attn_logit_softcap=50)
    "gemma2-9b": lambda: (256, 16, 8, spec_gemma2(num_layers=42)),
    "gemma2-27b": lambda: (128, 32, 16, spec_gemma2(num_layers=46)),
    # Gemma 3 (5:1 local:global, attn_logit_softcap=0; model uses final_logit_softcap)
    "gemma3-27b": lambda: (256, 32, 16, spec_gemma3(num_layers=62)),
    # GPT-OSS (alternating SWA, attention sinks)
    "gpt-oss-20b": lambda: (64, 64, 8, spec_gpt_oss(num_layers=24)),
    "gpt-oss-120b": lambda: (64, 64, 8, spec_gpt_oss(num_layers=36)),
    # Grok-1 (uniform full, attn_logit_softcap=30, optional temperature_len)
    "grok-1": lambda: (128, 48, 8, spec_grok(num_layers=64)),
    # Llama 3 family (uniform full attention)
    "llama3-8b": lambda: (128, 32, 8, spec_llama3(num_layers=32)),
    "llama3-70b": lambda: (128, 64, 8, spec_llama3(num_layers=80)),
    "llama3.1-405b": lambda: (128, 128, 8, spec_llama3(num_layers=126)),
    # Llama 4 iRoPE (every 4th layer NoPE global, the rest chunked-window)
    "llama4-maverick": lambda: (128, 128, 8, spec_llama4(num_layers=48)),
    "llama4-scout": lambda: (128, 48, 8, spec_llama4(num_layers=48)),
    # Mistral (sliding_window is defined in HF config but ignored by sglang)
    "mistral-7b-v0.3": lambda: (128, 32, 8, spec_llama3(num_layers=32)),
    "mistral-nemo-12b": lambda: (128, 32, 8, spec_llama3(num_layers=40)),
    "mistral-small-24b": lambda: (128, 32, 8, spec_llama3(num_layers=40)),
    # Mixtral (MoE, dense full attention per layer)
    "mixtral-8x7b": lambda: (128, 32, 8, spec_llama3(num_layers=32)),
    "mixtral-8x22b": lambda: (128, 48, 8, spec_llama3(num_layers=56)),
    # OLMo 2 (uniform full; 7B/13B both have `layer_types` but uniformly full)
    "olmo2-7b": lambda: (128, 32, 32, spec_llama3(num_layers=32)),
    "olmo2-13b": lambda: (128, 40, 40, spec_llama3(num_layers=40)),
    # Qwen 2.5 / Qwen 3 (Qwen3 ignores use_sliding_window in sglang; treat as uniform)
    "qwen2.5-72b": lambda: (128, 64, 8, spec_llama3(num_layers=80)),
    "qwen3-8b": lambda: (128, 32, 8, spec_qwen3(num_layers=36)),
    "qwen3-32b": lambda: (128, 64, 8, spec_qwen3(num_layers=64)),
}


def prewarm_for_model(
    layers: Sequence[dict],
    *,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bfloat16,
    include_basic: bool = True,
    include_persistent: bool = True,
    include_fp8: bool = False,
    parallel: int = 4,
    verbose: bool = False,
) -> dict:
    """Prewarm every unique kernel variant across a model's layer patterns.

    ``layers`` is a sequence of per-layer dicts (see
    ``enumerate_layer_patterns``) — build one manually or use a
    ``spec_*`` helper or a ``MODEL_PRESETS`` entry. Returns a dict with
    per-pattern stats and aggregate wall time.
    """
    patterns = enumerate_layer_patterns(layers)
    if verbose:
        logger.info(
            "prewarm_for_model: D=%d H=%d kvH=%d -- %d layers collapse to %d unique patterns",
            head_dim,
            num_q_heads,
            num_kv_heads,
            len(layers),
            len(patterns),
        )
        for swa, sink, cap, xai_len, causal in patterns:
            logger.info(
                "  pattern: swa=%d sink=%s cap=%.3g xai_len=%d causal=%s",
                swa,
                sink,
                cap,
                xai_len,
                causal,
            )

    pat_stats = []
    t0 = time.time()
    for swa, sink, cap, xai_len, causal in patterns:
        sub = prewarm_extend_attention(
            head_dim=head_dim,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            device=device,
            dtype=dtype,
            is_causal_modes=(causal,),
            has_sink=sink,
            sliding_window_size=swa,
            logit_cap=cap,
            xai_temperature_len=xai_len,
            include_basic=include_basic,
            include_persistent=include_persistent,
            include_fp8=include_fp8,
            parallel=parallel,
            verbose=verbose,
        )
        pat_stats.append(
            {
                "pattern": (swa, sink, cap, xai_len, causal),
                **sub,
            }
        )

    wall = time.time() - t0
    total_variants = sum(p["num_variants"] for p in pat_stats)
    if verbose:
        logger.info(
            "prewarm_for_model: %d patterns, %d total variants, %.2fs wall",
            len(patterns),
            total_variants,
            wall,
        )
    return {
        "num_patterns": len(patterns),
        "num_layers": len(layers),
        "total_variants": total_variants,
        "wall_time": wall,
        "per_pattern": pat_stats,
    }


def prewarm_preset(
    preset_name: str,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bfloat16,
    parallel: int = 4,
    verbose: bool = False,
) -> dict:
    """Prewarm from a named preset (see MODEL_PRESETS)."""
    if preset_name not in MODEL_PRESETS:
        raise ValueError(
            f"unknown preset {preset_name!r}; " f"available: {sorted(MODEL_PRESETS)}"
        )
    head_dim, num_q_heads, num_kv_heads, layers = MODEL_PRESETS[preset_name]()
    return prewarm_for_model(
        layers,
        head_dim=head_dim,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        device=device,
        dtype=dtype,
        parallel=parallel,
        verbose=verbose,
    )
