"""fp4 (MX e2m1) DSA prefill indexer with sampled-threshold candidate filtering (SM100).

One launch scores every visible key of each query row of the DeepSeek-V4 index
layers (64 heads x 128 for V4-Pro, 32 x 128 for V4.1-Flash; fp4 index K with
MX scales, the operands of DeepGEMM's ``fp8_fp4_mqa_logits``) on the tensor
cores and keeps, per row, the keys whose score reaches a threshold set from a
strided sample of the row's own scores. The ``[rows, keys]`` logits are never
written; the output is a per-row candidate slab for a separate top-k kernel.
Kernel source: ``kernels/jit/csrc/dsa/litetopk_prefill/fp4_indexer.cuh``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

FP4_INDEXER_HEAD_DIM = 128
# Slab width: the real-data candidate maximum is ~6K per row (DESIGN.md).
FP4_INDEXER_CAP = 12288
# KV ring depth, 16 KB fp4 + 1 KB scales per stage (DeepGEMM uses 10): the time is flat from 6 up (B300), and the
# 4-row config spends the smem on the staging ring instead.
DEFAULT_KV_STAGES = {64: 8, 32: 6}
# Staging-ring depth in 256-key tiles (BLOCK_Q x 512 B each); the sample tiles of a block stay in the ring until its
# threshold is selected, so the sample cap is the ring minus 8.
DEFAULT_DRAIN_SLOTS = {64: 40, 32: 28}


class Fp4IndexerCandidates(NamedTuple):
    scores: torch.Tensor  # [rows, cap] bf16 (rounded to nearest-even), unordered
    indices: torch.Tensor  # [rows, cap] int32, key - ks[row]; not written for T = -inf rows of dense blocks (= slot)
    counts: torch.Tensor  # [rows] int32, scores >= T (may exceed cap)
    thresholds: torch.Tensor  # [rows] fp32 T
    overflow: torch.Tensor  # [rows] int32, 1 when count > cap
    samples: torch.Tensor  # [rows] int32, samples the threshold was taken from


@cache_once
def is_fp4_indexer_supported() -> bool:
    if torch.version.cuda is None or not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] == 10


# Blocks are scheduled dynamically from a [1] int32 counter that each launch leaves zeroed. One per process: launches
# sharing it must not overlap (indexer calls on two streams at once).
# TODO(DarkSharpness): borrow it from a per-stream zero allocator;
# a kernel that leaves its scratch zero can share it.
_SCHED_WORKSPACE: Optional[torch.Tensor] = None


def _sched_workspace(device: torch.device) -> torch.Tensor:
    global _SCHED_WORKSPACE
    if _SCHED_WORKSPACE is None:
        _SCHED_WORKSPACE = torch.zeros((1,), dtype=torch.int32, device=device)
    return _SCHED_WORKSPACE


@cache_once
def _jit_fp4_indexer_module(num_heads: int, kv_stages: int, drain_slots: int) -> Module:
    if not is_fp4_indexer_supported():
        raise RuntimeError("fp4_indexer_candidates requires CUDA SM100")
    if num_heads not in (32, 64):
        raise RuntimeError(f"num_heads must be 32 or 64, got {num_heads}")
    args = make_cpp_args(num_heads, kv_stages, drain_slots)
    return load_jit(
        "dsa_prefill_fp4_indexer",
        *args,
        cuda_files=["dsa/litetopk_prefill/fp4_indexer.cuh"],
        cuda_wrappers=[("run", f"Fp4IndexerKernel<{args}>::run")],
    )


def fp4_indexer_candidates(
    q_fp4: torch.Tensor,
    q_sf: torch.Tensor,
    weights: torch.Tensor,
    kv_fp4: torch.Tensor,
    kv_sf: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    topk: int,
    *,
    num_ctas: int = 0,
    kv_stages: Optional[int] = None,
    drain_slots: Optional[int] = None,
    out: Optional[Fp4IndexerCandidates] = None,
) -> Fp4IndexerCandidates:
    """Row ``r`` scores keys ``j`` in ``[ks[r], ke[r])`` exactly as DeepGEMM's
    ``fp8_fp4_mqa_logits`` does with these operands and bf16 weights (its bf16
    reduce), rounds the scores to nearest-even bf16 and stores every ``(score,
    j - ks[r])`` with ``score >= thresholds[r]`` into the slab, unordered.
    ``thresholds[r]`` is ``-inf`` (every key) when the row sees at most
    ``topk`` keys or its block's rows all see at most 6144, otherwise exactly
    a bf16 sample score at or below its ``topk``-th score, so the slab holds
    the exact bf16 top-k; ``counts[r]`` keys reach it, the first
    ``min(count, cap)`` are stored, ``overflow[r]`` flags the rest. A
    ``-inf`` row holds every key in key order (slot ``s`` is key ``s``), and
    rows of dense blocks leave ``indices`` unwritten.

    ``q_fp4`` ``[rows, heads, 64]`` int8, ``q_sf`` ``[rows, heads]`` int32,
    ``kv_fp4`` ``[keys, 64]`` int8 with ``keys`` a multiple of 4, ``kv_sf``
    ``[keys]`` int32, ``weights`` ``[rows, heads]`` bf16, ``ks`` / ``ke``
    ``[rows]`` int32. ``num_ctas`` 0 launches one persistent CTA per SM.
    """
    rows, num_heads = q_sf.shape
    if kv_stages is None:
        kv_stages = DEFAULT_KV_STAGES[num_heads]
    if drain_slots is None:
        drain_slots = DEFAULT_DRAIN_SLOTS[num_heads]
    if out is None:
        device = q_fp4.device
        out = Fp4IndexerCandidates(
            scores=torch.empty(
                (rows, FP4_INDEXER_CAP), dtype=torch.bfloat16, device=device
            ),
            indices=torch.empty(
                (rows, FP4_INDEXER_CAP), dtype=torch.int32, device=device
            ),
            counts=torch.empty((rows,), dtype=torch.int32, device=device),
            thresholds=torch.empty((rows,), dtype=torch.float32, device=device),
            overflow=torch.empty((rows,), dtype=torch.int32, device=device),
            samples=torch.empty((rows,), dtype=torch.int32, device=device),
        )
    _jit_fp4_indexer_module(num_heads, kv_stages, drain_slots).run(
        q_fp4,
        q_sf,
        kv_fp4,
        kv_sf,
        weights,
        ks,
        ke,
        out.scores,
        out.indices,
        out.counts,
        out.overflow,
        out.thresholds,
        out.samples,
        _sched_workspace(q_fp4.device),
        topk,
        num_ctas,
    )
    return out
