"""FP8 DSA prefill indexer with sampled-threshold candidate filtering (SM100).

One launch scores every visible key of each query row (32 or 64 index heads
x 128, e.g. GLM-5.2 / 5.3 with 32; fp8 index K with a per-key fp32 scale) on
the tensor cores and keeps, per row, the keys whose score reaches a threshold set
from a strided sample of the row's scores. The ``[rows, keys]`` logits are
never written; the output is a per-row candidate slab for a separate top-k
kernel. Kernel source: ``kernels/jit/csrc/dsa/litetopk_prefill/fp8_indexer.cuh``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

FP8_INDEXER_NUM_HEADS = (32, 64)
FP8_INDEXER_HEAD_DIM = 128
# Slab width: the real-data candidate maximum is ~10K per row (DESIGN.md).
FP8_INDEXER_CAP = 12288
# Pipeline depths: 3 x 32 KB KV stages (4 and 5 measured equal), 2 x 16 KB Q stages (the
# third went to the 16 KB candidate staging).
DEFAULT_KV_STAGES = 3
DEFAULT_Q_STAGES = 2
# Query rows per CTA block by head count (kBlockQ = 256 / heads in fp8_indexer.cuh).
FP8_INDEXER_BLOCK_ROWS = {32: 8, 64: 4}


@cache_once
def _num_sms(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


class Fp8IndexerCandidates(NamedTuple):
    scores: torch.Tensor  # [rows, cap] fp32, unordered
    indices: torch.Tensor  # [rows, cap] int32, key - ks[row]
    counts: torch.Tensor  # [rows] int32, scores >= T (may exceed cap)
    thresholds: torch.Tensor  # [rows] fp32
    overflow: torch.Tensor  # [rows] int32, 1 when count > cap


# Fixed cost of a scan unit (pipeline fill, block-end sync) in 256-key tiles; fitted on B200 (128K-1M keys,
# 64-1024 rows).
UNIT_OVERHEAD_TILES = 10


def pick_tail_chunks(num_blocks: int, num_ctas: int, block_tiles: int) -> int:
    """Pieces per block of the last, partial wave: the one of 1, 2, ..., 32 that ends the launch first. With grid =
    min(num_ctas, num_blocks C), the full waves take num_blocks // grid block times and the tail of r blocks in C
    pieces ceil(r C / grid) / C, plus UNIT_OVERHEAD_TILES / block_tiles per unit a CTA runs."""
    unit_cost = UNIT_OVERHEAD_TILES / max(block_tiles, 1)

    def launch_time(c: int) -> float:
        grid = min(num_ctas, num_blocks * c)
        tail = num_blocks % grid
        units = -(-tail * c // grid)
        return num_blocks // grid * (1 + unit_cost) + (
            units / c + units * unit_cost if tail else 0.0
        )

    return min((1, 2, 4, 8, 16, 32), key=launch_time)


# The scratch buffers below are one per process: launches sharing them must not overlap (indexer calls on two
# streams at once).
# TODO(DarkSharpness): borrow them from a per-stream zero allocator;
# a kernel that leaves its scratch zero can share it.
_WORKSPACE: Optional[torch.Tensor] = None
_SAMPLE_CODES: Optional[torch.Tensor] = None


def _workspace(device: torch.device, size: int) -> torch.Tensor:
    """int32 scratch of the scan pieces (zero between launches; every launch leaves it zero)."""
    global _WORKSPACE
    if _WORKSPACE is None or _WORKSPACE.numel() < size:
        _WORKSPACE = torch.zeros(max(size, 4096), dtype=torch.int32, device=device)
    return _WORKSPACE


def _sample_codes(device: torch.device) -> torch.Tensor:
    """Scratch of the sample codes between the two threshold passes: per CTA 32 sample tiles x 2 key halves x 256
    math threads x 4 rows x 16 bits (2 rows used at 64 heads; no zeroing needed)."""
    global _SAMPLE_CODES
    if _SAMPLE_CODES is None:
        _SAMPLE_CODES = torch.empty(
            _num_sms(device) * 32 * 2 * 256 * 2, dtype=torch.int32, device=device
        )
    return _SAMPLE_CODES


@cache_once
def is_fp8_indexer_supported() -> bool:
    if torch.version.cuda is None or not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] == 10


@cache_once
def _jit_fp8_indexer_module(num_heads: int, kv_stages: int, q_stages: int) -> Module:
    if not is_fp8_indexer_supported():
        raise RuntimeError("fp8_indexer_candidates requires CUDA SM100")
    if num_heads not in FP8_INDEXER_NUM_HEADS:
        raise RuntimeError(
            f"num_heads must be one of {FP8_INDEXER_NUM_HEADS}, got {num_heads}"
        )
    args = make_cpp_args(num_heads, kv_stages, q_stages)
    return load_jit(
        "dsa_prefill_fp8_indexer",
        *args,
        cuda_files=["dsa/litetopk_prefill/fp8_indexer.cuh"],
        cuda_wrappers=[("run", f"Fp8IndexerKernel<{args}>::run")],
    )


def fp8_indexer_candidates(
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    kv_fp8: torch.Tensor,
    kv_scale: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    topk: int,
    *,
    cap: int = FP8_INDEXER_CAP,
    dense_bound: Optional[int] = None,
    num_ctas: int = 0,
    kv_stages: int = DEFAULT_KV_STAGES,
    q_stages: int = DEFAULT_Q_STAGES,
    tail_chunks: Optional[int] = None,
    out: Optional[Fp8IndexerCandidates] = None,
) -> Fp8IndexerCandidates:
    """``q_fp8`` is ``[rows, heads, 128]`` with 32 or 64 heads, ``weights``
    ``[rows, heads]``. Row ``r`` scores keys ``j`` in ``[ks[r], ke[r])`` as
    ``kv_scale[j] * sum_h weights[r, h] * relu(q_fp8[r, h] . kv_fp8[j])``
    (DeepGEMM ``fp8_mqa_logits`` arithmetic) and keeps every ``(score, j -
    ks[r])`` with ``score >= T[r]``, ``T[r]`` the j'-th largest of a sample
    of single keys on an odd stride (65, 129 or 257 by span) of the row's
    scores (``j' = ceil((1 + 4 sqrt(2 / j)) j)``,
    ``j = topk * samples / keys``), so the candidates contain the row's exact
    top-``topk``. Rows with at most ``dense_bound`` keys (default ``topk``)
    keep every key (``T = -inf``). ``counts`` keeps counting past ``cap``; such
    rows set ``overflow``. The ``s w`` half of ``relu(s) w`` comes from the
    tensor core (q' in four e4m3 terms), so the scores differ from DeepGEMM's
    by the e4m3 split error of q' (measured <= 1e-6 relative); the candidate
    set still contains the exact top-k. The blocks of the last, partial wave
    are scanned in ``tail_chunks`` pieces each (default: the count that ends
    first, see :func:`pick_tail_chunks`); a cached per-device workspace
    coordinates the pieces, so calls on one device must not overlap. Meant for
    calls whose rows see more than ~12K keys; shorter calls are faster
    on the dense path (DeepGEMM logits + top-k).
    """
    rows, num_heads = q_fp8.shape[:2]
    module = _jit_fp8_indexer_module(num_heads, kv_stages, q_stages)
    if dense_bound is None:
        dense_bound = topk
    num_blocks = -(-rows // FP8_INDEXER_BLOCK_ROWS[num_heads])
    if tail_chunks is None:
        num_sms = _num_sms(q_fp8.device)
        grid = min(num_ctas if num_ctas > 0 else num_sms, num_sms)
        tail_chunks = pick_tail_chunks(num_blocks, grid, kv_fp8.shape[0] // 256)
    if out is None:
        device = q_fp8.device
        out = Fp8IndexerCandidates(
            scores=torch.empty((rows, cap), dtype=torch.float32, device=device),
            indices=torch.empty((rows, cap), dtype=torch.int32, device=device),
            counts=torch.empty((rows,), dtype=torch.int32, device=device),
            thresholds=torch.empty((rows,), dtype=torch.float32, device=device),
            overflow=torch.empty((rows,), dtype=torch.int32, device=device),
        )
    module.run(
        q_fp8,
        weights,
        kv_fp8,
        kv_scale,
        ks,
        ke,
        _workspace(q_fp8.device, 2 * num_blocks),
        _sample_codes(q_fp8.device),
        out.scores,
        out.indices,
        out.counts,
        out.thresholds,
        out.overflow,
        topk,
        dense_bound,
        num_ctas,
        tail_chunks,
    )
    return out
