"""Fused DSA indexer top-k (LiteTopk) for SM100 (Blackwell).

Streams KV in tiles and fuses fp8 MQA scoring (tcgen05 UMMA) + an online
bucketed gate + a compact exact top-k, so the ``[num_q, seq_len]`` logit
matrix is never materialized. Kernels are vendored from LiteTopK
(https://github.com/Heisenberg-Yin/LiteTopK), whose vLLM integration is
PR #48726 (``csrc/dsa_litetopk/``); this module orchestrates the three
primitives (``seed_prep`` / ``scan`` / ``select``) and allocates scratch.

Prefill (ragged extend) only. The output indices are gathered-KV absolute
positions, padded with ``-1`` -- the dense path's RAGGED top-k transform
contract, not the PAGED one (KV-pool locations).

Caveats:
  * Exact top-k SET by construction (conservative gate), but tie-breaking at
    the k-th value follows atomic arrival order and is nondeterministic; the
    within-row output order is unsorted. Excluded from deterministic mode.
  * GLM DSA shape only (H=32, D=128); fp8 index-K cache only.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

NUM_HEADS = 32
HEAD_DIM = 128
_BLOCK_Q = 4  # q rows per scan CTA (dsa_indexer.cuh BLOCK_Q)
_NUM_BUCKETS = 256
_SAMPLE_LEN = 8192
_REFRESH_EVERY = 64
# Initial gate: the rank within the calibration sample whose score becomes the
# first threshold. The safe rank (topk) is a guaranteed bound but admits
# topk/sample_len of every row (25% at the defaults), so the scan opens with an
# estimate of the row's own topk quantile, scaled by the margin, and rows whose
# candidate count comes back short are rescanned from the safe threshold.
_GATE_MARGIN = 1.5
_GATE_K_MIN = 64


@cache_once
def dsa_litetopk_is_supported() -> bool:
    """True iff the current device can run the LiteTopk kernels (SM100)."""
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] == 10


@cache_once
def _jit_dsa_litetopk_module() -> Module:
    return load_jit(
        "dsa_litetopk",
        cuda_files=["dsa_litetopk/entry.cuh"],
        cuda_wrappers=[
            ("seed_prep", "dsa_litetopk_seed_prep"),
            ("scan", "dsa_litetopk_scan"),
            ("select", "dsa_litetopk_select"),
        ],
        extra_cuda_cflags=[
            "-O3",
            "-DNDEBUG",
            "-DCUTE_USE_PACKED_TUPLE=1",
            "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        ],
        # Also supplies the deep_gemm headers the kernel includes.
        extra_dependencies=["cutlass"],
    )


def _tma_aligned_scales(kv_scales: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    """``kv_scales[lo:hi]`` with a 16B-aligned base, copying only when needed.

    deep_gemm builds the calibration KV-scale TMA descriptor straight off the
    tensor's ``data_ptr`` (``make_tma_2d_desc`` -> ``cuTensorMapEncodeTiled``),
    which requires a 16B-aligned global address, and reads the scale dim rounded
    up to 16B. fp8 KV rows are 128B so any per-request slice stays aligned, but
    the fp32 scales are 4B-granular: a request whose gathered KV starts at a
    position that is not a multiple of 4 hands the driver a misaligned
    descriptor and it fails with CUDA_ERROR_INVALID_VALUE. Materialize such a
    slice into a fresh buffer (allocator-aligned, padded to a multiple of 4 so
    the rounded-up tail read stays in bounds); the same copy covers a slice
    whose rounded-up tail would run past the end of ``kv_scales`` itself. Slices
    that are aligned and have a readable tail pass through as views, so the
    common shapes -- and single-request prefill -- never copy.
    """
    width = hi - lo
    tail = (hi + 3) // 4 * 4
    if lo % 4 == 0 and tail <= kv_scales.shape[0]:
        return kv_scales[lo:hi]
    buf = kv_scales.new_empty((width + 3) // 4 * 4)
    buf[:width] = kv_scales[lo:hi]
    return buf[:width]


def _pad_sample_logits_for_vec4(slog: torch.Tensor) -> torch.Tensor:
    """seed_prep reads each row with 16B float4 loads, so the row stride must
    be a multiple of 4 floats or every odd row is misaligned (CUDA fault).
    deep_gemm's logits already have an aligned row stride (the width itself
    is arbitrary: min(sample_len, kv_len)), so they pass through without the
    512MB copy a .contiguous() would cost; a strided or misaligned buffer is
    padded with -inf, which every seed_prep pass skips via its isfinite guard
    (the same fill clean_logits uses for the out-of-causal-range tail)."""
    if slog.stride(1) == 1 and slog.stride(0) % 4 == 0:
        return slog
    rem = slog.shape[1] % 4
    return torch.nn.functional.pad(
        slog, (0, (4 - rem) % 4), value=float("-inf")
    ).contiguous()


def _gate_k(kv_len: int, sample_len: int, topk: int) -> int:
    if kv_len <= sample_len:
        return topk
    est = math.ceil(_GATE_MARGIN * topk * sample_len / kv_len)
    return min(topk, max(_GATE_K_MIN, est))


def _rescan_short_rows(
    *,
    module: Module,
    q_fp8: torch.Tensor,
    kv_fp8: torch.Tensor,
    kv_scales: torch.Tensor,
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    origin: torch.Tensor,
    inv_delta: torch.Tensor,
    th_bucket: torch.Tensor,
    th_safe: torch.Tensor,
    cand_val: torch.Tensor,
    cand_idx: torch.Tensor,
    cand_cnt: torch.Tensor,
    bcount: torch.Tensor,
    num_buckets: int,
    topk: int,
    cap: int,
    refresh_every: int,
) -> None:
    """Verify the estimated gate and rescan the q-blocks it failed on.

    A row is exact when at least ``topk`` scores passed its gate (the k-th best
    score then lies above the threshold, so every top-k element was emitted)
    and none were dropped at ``cand_cap``. Every row of a q-block with a
    failing row restarts from its safe threshold with an empty candidate list;
    the scan skips the other q-blocks. No host sync: the mask is built on the
    device and an all-clear pass costs one near-empty launch.
    """
    num_q = cand_cnt.shape[0]
    num_q_blocks = (num_q + _BLOCK_Q - 1) // _BLOCK_Q
    bad = (cand_cnt < topk) | (cand_cnt > cap)
    bad_pad = torch.zeros(num_q_blocks * _BLOCK_Q, dtype=torch.bool, device=bad.device)
    bad_pad[:num_q] = bad
    qblock_mask = bad_pad.view(num_q_blocks, _BLOCK_Q).any(dim=1)
    row_mask = qblock_mask.repeat_interleave(_BLOCK_Q)[:num_q]
    torch.where(row_mask, th_safe, th_bucket, out=th_bucket)
    cand_cnt.masked_fill_(row_mask, 0)
    # The refresh histogram must restart with the candidate list: counts left
    # from the first pass would be doubled by the re-emitted positions and
    # tighten the threshold past the true k-th score.
    bcount.masked_fill_(row_mask.unsqueeze(1), 0)
    module.scan(
        q_fp8,
        kv_fp8,
        kv_scales,
        weights,
        ks,
        ke,
        origin,
        inv_delta,
        th_bucket,
        cand_val,
        cand_idx,
        cand_cnt,
        bcount,
        num_buckets,
        topk,
        refresh_every,
        -1,
        0,
        0,
        qblock_mask.to(torch.int32),
    )


def dsa_litetopk_indexer(
    q_fp8: torch.Tensor,
    kv_fp8: torch.Tensor,
    kv_scales: torch.Tensor,
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    topk: int,
    out_indices: Optional[torch.Tensor] = None,
    *,
    req_bounds: Sequence[Tuple[int, int, int, int]],
    num_buckets: int = _NUM_BUCKETS,
    sample_len: int = _SAMPLE_LEN,
    cand_cap: Optional[int] = None,
    refresh_every: int = _REFRESH_EVERY,
) -> torch.Tensor:
    """Fused indexer top-k. Writes ``topk`` gathered-KV indices per row.

    Args:
        q_fp8: ``[num_q, 32, 128]`` fp8_e4m3 indexer queries (contiguous).
        kv_fp8: ``[seq_len_kv, 128]`` fp8_e4m3 gathered index-K (contiguous).
        kv_scales: ``[seq_len_kv]`` fp32 per-position dequant scales.
        weights: ``[num_q, 32]`` fp32 per-head gates (q-scale folded in).
        ks / ke: ``[num_q]`` int32 per-row causal [start, end) into ``kv_fp8``.
        topk: number of KV indices to select per row.
        out_indices: optional ``[num_q, topk]`` int32 output buffer.
        req_bounds: per-request ``(row_start, row_end, kv_start, kv_end)``
            groups (rows of one request share ``ks == kv_start``). Gate
            calibration samples each request's own KV prefix so thresholds are
            causally valid across requests (unlike upstream's single shared
            ``kv[:sample_len]`` prefix, which can over-tighten the gate with
            scores from positions a row can never select).
    Returns:
        ``[num_q, topk]`` int32 gathered-KV positions, ``-1`` padded.
    """
    import deep_gemm

    num_q = q_fp8.shape[0]
    dev = q_fp8.device
    cap = cand_cap if cand_cap is not None else max(4 * topk, 16384)
    assert q_fp8.is_contiguous() and kv_fp8.is_contiguous()
    assert weights.dtype == torch.float32

    module = _jit_dsa_litetopk_module()

    origin = torch.empty(num_q, dtype=torch.float32, device=dev)
    inv_delta = torch.empty(num_q, dtype=torch.float32, device=dev)
    th_bucket = torch.empty(num_q, dtype=torch.int32, device=dev)
    th_safe = torch.empty(num_q, dtype=torch.int32, device=dev)
    bcount = torch.zeros(num_q, num_buckets, dtype=torch.int32, device=dev)
    cand_val = torch.empty(num_q, cap, dtype=torch.float32, device=dev)
    cand_idx = torch.empty(num_q, cap, dtype=torch.int32, device=dev)
    cand_cnt = torch.empty(num_q, dtype=torch.int32, device=dev)
    out_val = torch.empty(num_q, topk, dtype=torch.float32, device=dev)
    if out_indices is None:
        out_indices = torch.empty(num_q, topk, dtype=torch.int32, device=dev)

    # Gate calibration: per request, score the request's own bounded KV prefix
    # (<< seq_len) with the existing dense kernel and derive per-row bucket
    # params + initial thresholds. emit_limit=0: calibration only, no seed
    # candidates, bcount zeroed -- recall comes from the full scan below.
    # clean_logits=True + per-row causal ke keep the sample rows free of both
    # uninitialized columns and causally-invalid scores.
    any_tight = False
    for row_start, row_end, kv_start, kv_end in req_bounds:
        if row_end <= row_start:
            continue
        sl = min(sample_len, kv_end - kv_start)
        gate_k = _gate_k(kv_end - kv_start, sample_len, topk)
        any_tight |= gate_k < topk
        rows = slice(row_start, row_end)
        ks0 = torch.zeros(row_end - row_start, dtype=torch.int32, device=dev)
        ke_s = (ke[rows].to(torch.int32) - kv_start).clamp_(min=0, max=sl)
        sample_logits = _pad_sample_logits_for_vec4(
            deep_gemm.fp8_mqa_logits(
                q_fp8[rows],
                (
                    kv_fp8[kv_start : kv_start + sl],
                    _tma_aligned_scales(kv_scales, kv_start, kv_start + sl),
                ),
                weights[rows],
                ks0,
                ke_s,
                clean_logits=True,
            )
        )
        module.seed_prep(
            sample_logits,
            num_buckets,
            topk,
            cap,
            0,  # emit_limit=0: calibrate only
            0.0,  # headroom
            0,  # probe_stride_tok
            1,  # hist_stride
            origin[rows],
            inv_delta[rows],
            th_bucket[rows],
            bcount[rows],
            cand_val[rows],
            cand_idx[rows],
            cand_cnt[rows],
            gate_k,
            th_safe[rows],
        )

    ks = ks.to(torch.int32)
    ke = ke.to(torch.int32)
    module.scan(
        q_fp8,
        kv_fp8,
        kv_scales,
        weights,
        ks,
        ke,
        origin,
        inv_delta,
        th_bucket,
        cand_val,
        cand_idx,
        cand_cnt,
        bcount,
        num_buckets,
        topk,
        refresh_every,
        -1,  # num_kv_splits_override: auto
        0,  # probe_group: probe compaction off
        0,  # probe_add_max
        None,
    )

    if any_tight:
        _rescan_short_rows(
            module=module,
            q_fp8=q_fp8,
            kv_fp8=kv_fp8,
            kv_scales=kv_scales,
            weights=weights,
            ks=ks,
            ke=ke,
            origin=origin,
            inv_delta=inv_delta,
            th_bucket=th_bucket,
            th_safe=th_safe,
            cand_val=cand_val,
            cand_idx=cand_idx,
            cand_cnt=cand_cnt,
            bcount=bcount,
            num_buckets=num_buckets,
            topk=topk,
            cap=cap,
            refresh_every=refresh_every,
        )

    # Candidate values are already in bucket space (the scan folds the per-row
    # affine into the register weights), so select rebases with identity.
    zero = torch.zeros(num_q, dtype=torch.float32, device=dev)
    one = torch.ones(num_q, dtype=torch.float32, device=dev)
    module.select(
        cand_val,
        cand_idx,
        cand_cnt,
        zero,
        one,
        th_bucket,
        num_buckets,
        topk,
        out_val,
        out_indices,
    )
    return out_indices
