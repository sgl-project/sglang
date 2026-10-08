"""Fused candidate-aware top-k for the dense FP4 prefill indexer."""

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

from .utils import make_name


@cache_once
def _candidate_topk_module():
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        make_name("prefill_candidate_topk"),
        *args,
        cuda_files=["deepseek_v4/prefill_candidate_topk.cuh"],
        cuda_wrappers=[("run", f"prefill_candidate::Kernel<{args}>::run")],
    )


def topk_prefill_candidates(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    block_mask: torch.Tensor,
    block_size: int,
    out_offsets: torch.Tensor,
    out_indices: torch.Tensor,
) -> None:
    """Select within each query's candidate blocks, without modifying scores.

    Columns are request-local, seq_lens are nonnegative and bounded by the score
    width. Output indices include out_offsets; invalid slots are -1. Row strides
    must be multiples of four floats for the vectorized top-k reads.
    """
    _candidate_topk_module().run(
        scores, seq_lens, block_mask, out_offsets, out_indices, block_size
    )
