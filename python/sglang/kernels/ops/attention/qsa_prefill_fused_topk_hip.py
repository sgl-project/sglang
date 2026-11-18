"""QSA prefill block selection without the fp32 logits, on AMD (csrc/attention/qsa_prefill_fused_topk_hip.cuh).

One launch scores 32 query rows over their key range and keeps each row's candidates above a running
threshold; a second takes their exact top-k and writes the expanded token indices.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


def _candidate_slots(topk: int) -> int:
    """Candidate slots per row (the kernel's kCap): 1024 at topk 512."""
    return 2 * ((topk + 255) // 256) * 256


def qsa_prefill_fused_topk_row_bytes(topk: int) -> int:
    """Workspace bytes per row: an int32 key and an int16 column per slot, a count and two bounds."""
    return 6 * _candidate_slots(topk) + 12


@cache_once
def _jit_module(topk: int, compress_ratio: int, token_topk: int) -> Module:
    args = make_cpp_args(topk, compress_ratio, token_topk)
    return load_jit(
        "qsa_prefill_fused_topk_hip",
        *args,
        cuda_files=["attention/qsa_prefill_fused_topk_hip.cuh"],
        cuda_wrappers=[("run", f"QSAPrefillFusedTopKKernel<{args}>::run")],
    )


def qsa_prefill_fused_topk(
    q: torch.Tensor,
    k: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    out: torch.Tensor,
    *,
    topk: int,
    compress_ratio: int,
    token_topk: int,
) -> torch.Tensor:
    """Write row b's expanded token selection into ``out[b]``.

    The tokens are those of the unscaled relative TileLang scores, ``topk_transform_ragged_v2``
    with deterministic ties (the smallest block indices) and ``expand_qsa_block_indices``, with
    the blocks in ascending order. q [B, 4, 128] bf16 contiguous; k [N, 128] bf16 contiguous (the
    compressed keys' one head); starts / ends [B] int32 (row b scores keys starts[b]..ends[b]-1,
    at most 65,536); positions [B] int64; sequence_lengths [B] int32; out [B, token_topk +
    compress_ratio - 1] int32 with unit column stride. The QSA checkpoints' topk 512,
    compress_ratio 4 and token_topk 2,048 are the sizes tested. Takes B *
    ``qsa_prefill_fused_topk_row_bytes(topk)`` bytes of workspace.
    """
    rows = q.shape[0]
    slots = _candidate_slots(topk)
    keys = torch.empty((rows, slots), dtype=torch.int32, device=q.device)
    cols = torch.empty((rows, slots), dtype=torch.int16, device=q.device)
    count = torch.empty((rows,), dtype=torch.int32, device=q.device)
    bounds = torch.empty((rows, 2), dtype=torch.int32, device=q.device)
    _jit_module(topk, compress_ratio, token_topk).run(
        q, k, starts, ends, positions, sequence_lengths, keys, cols, count, bounds, out
    )
    return out
