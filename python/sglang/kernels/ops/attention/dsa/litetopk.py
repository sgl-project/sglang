"""Fused DSA indexer top-k (LiteTopk) for SM100 (Blackwell).

Streams KV in tiles and fuses fp8 MQA scoring (tcgen05 UMMA) + an online
bucketed gate + a compact exact top-k, so the ``[num_q, seq_len]`` logit
matrix is never materialized. Kernels are vendored from vLLM PR #48726
(``csrc/dsa_litetopk/``); this module orchestrates the three primitives
(``seed_prep`` / ``scan`` / ``select``) and allocates scratch.

Prefill (ragged extend) only. The output indices are gathered-KV absolute
positions -- the same coordinates the dense ``fp8_mqa_logits`` +
``fast_topk_transform_ragged_fused`` path produces -- padded with ``-1``.

Caveats:
  * Exact top-k SET by construction (conservative gate), but tie-breaking at
    the k-th value follows atomic arrival order and is nondeterministic; the
    within-row output order is unsorted. Excluded from deterministic mode.
  * GLM DSA shape only (H=32, D=128); fp8 index-K cache only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import torch

from sglang.kernels.jit.utils import (
    KERNEL_PATH,
    cache_once,
    load_jit,
    override_jit_cuda_arch,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

NUM_HEADS = 32
HEAD_DIM = 128
_NUM_BUCKETS = 256
_SAMPLE_LEN = 8192
_REFRESH_EVERY = 64


@cache_once
def dsa_litetopk_is_supported() -> bool:
    """True iff the current device can run the LiteTopk kernels (SM100)."""
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] == 10


@cache_once
def _jit_dsa_litetopk_module() -> Module:
    with override_jit_cuda_arch(10, 0, "a"):
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
            extra_include_paths=[
                str(KERNEL_PATH / "csrc" / "dsa_litetopk" / "vendor_deep_gemm"),
            ],
            extra_dependencies=["cutlass"],
        )


def _pad_scales_for_tma(kv_scales: torch.Tensor) -> torch.Tensor:
    """The scan TMA descriptor rounds the scales dim up to 16B; pad if needed."""
    rem = kv_scales.shape[0] % 4
    if rem == 0:
        return kv_scales
    return torch.nn.functional.pad(kv_scales, (0, 4 - rem))


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
    for row_start, row_end, kv_start, kv_end in req_bounds:
        if row_end <= row_start:
            continue
        sl = min(sample_len, kv_end - kv_start)
        rows = slice(row_start, row_end)
        ks0 = torch.zeros(row_end - row_start, dtype=torch.int32, device=dev)
        ke_s = (ke[rows].to(torch.int32) - kv_start).clamp_(min=0, max=sl)
        sample_logits = deep_gemm.fp8_mqa_logits(
            q_fp8[rows],
            (kv_fp8[kv_start : kv_start + sl], kv_scales[kv_start : kv_start + sl]),
            weights[rows],
            ks0,
            ke_s,
            clean_logits=True,
        ).contiguous()
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
        )

    module.scan(
        q_fp8,
        kv_fp8,
        _pad_scales_for_tma(kv_scales),
        weights,
        ks.to(torch.int32),
        ke.to(torch.int32),
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
