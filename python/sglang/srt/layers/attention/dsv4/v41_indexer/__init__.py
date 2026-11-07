"""The top-k selection of the DeepSeek V4.1 ratio-1/2 index layers: the full top-k
(``full_topk``), and the candidate schemes split by what a source publishes, a
DeepGEMM sparse table (``sparse_table``) or dense block ids (``dense_blocks``)."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Tuple

import torch

from sglang.kernels.arch import current_arch
from sglang.kernels.ops.attention.dsv4.index_logits import use_fp8_prefill_logits

from .full_topk import FullTopKIndexer
from .types import (
    CandidateMetadata,
    CapturedPrefillInputs,
    DecodeCandidates,
    DecodeInputs,
    PrefillCandidates,
    PrefillInputs,
    RowShard,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

__all__ = [
    "CandidateMetadata",
    "CapturedPrefillInputs",
    "DecodeCandidates",
    "DecodeInputs",
    "FullTopKIndexer",
    "PrefillCandidates",
    "PrefillInputs",
    "RowShard",
    "has_dense_fp4_indexer",
    "is_sm100",
    "make_candidate_indexer",
    "make_full_topk_indexer",
]


@functools.cache
def is_sm100() -> bool:
    # The DeepGEMM fp8_fp4 mqa-logits paths are SM100-family only; SM120 and
    # non-CUDA devices take the torch / Triton indexer.
    return current_arch().is_sm(10)


@functools.cache
def has_dense_fp4_indexer() -> bool:
    if not torch.cuda.is_available() or torch.version.cuda is None:
        return False
    try:
        import deep_gemm
    except ImportError:
        return False
    return hasattr(deep_gemm, "fp8_fp4_mqa_logits")


@functools.cache
def _prefill_path(*, index_n_heads: int) -> Tuple[bool, bool]:
    """``(use_deep_gemm_prefill, prefill_fp8_logits)``: whether the prefill
    top-k runs on DeepGEMM, and whether it scores fp8 operands read back from
    the fp4 pool rather than the fp4 operands themselves."""
    fp8_logits = use_fp8_prefill_logits(index_n_heads=index_n_heads)
    return (is_sm100() and has_dense_fp4_indexer()) or fp8_logits, fp8_logits


def make_full_topk_indexer(
    *,
    token_to_kv_pool: DeepSeekV4TokenToKVPool,
    req_to_token: torch.Tensor,
    index_n_heads: int,
) -> FullTopKIndexer:
    use_deep_gemm_prefill, prefill_fp8_logits = _prefill_path(
        index_n_heads=index_n_heads
    )
    return FullTopKIndexer(
        token_to_kv_pool=token_to_kv_pool,
        req_to_token=req_to_token,
        use_deep_gemm_prefill=use_deep_gemm_prefill,
        prefill_fp8_logits=prefill_fp8_logits,
        use_deep_gemm_decode=is_sm100(),
    )


def make_candidate_indexer(
    *,
    token_to_kv_pool: DeepSeekV4TokenToKVPool,
    req_to_token: torch.Tensor,
    page_size: int,
    candidate_topk_blocks: int,
    candidate_block_size: int,
    index_n_heads: int,
) -> Tuple[PrefillCandidates, DecodeCandidates]:
    from sglang.srt.runtime_context import get_parallel

    from .dense_blocks import DenseBlocksBackend

    use_deep_gemm_prefill, prefill_fp8_logits = _prefill_path(
        index_n_heads=index_n_heads
    )
    dense_blocks = DenseBlocksBackend(
        token_to_kv_pool=token_to_kv_pool,
        req_to_token=req_to_token,
        candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size,
        use_deep_gemm_prefill=use_deep_gemm_prefill,
        prefill_fp8_logits=prefill_fp8_logits,
    )
    # Without candidate blocks no layer is a candidate source or consumer.
    if not is_sm100() or candidate_topk_blocks <= 0:
        return dense_blocks, dense_blocks

    from sglang.srt.layers.deep_gemm_wrapper.configurer import (
        DEEPGEMM_PAGED_SPARSE_MQA_LOGITS,
    )

    if not DEEPGEMM_PAGED_SPARSE_MQA_LOGITS:
        raise RuntimeError(
            "the candidate indexer needs DeepGEMM's paged sparse MQA logits "
            "(sgl-deep-gemm >= 0.2.0 with SGLANG_ENABLE_JIT_DEEPGEMM on)"
        )
    from .sparse_table import SparseTableBackend

    sparse_table = SparseTableBackend(
        token_to_kv_pool=token_to_kv_pool,
        page_size=page_size,
        candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size,
        prefill_fp8_logits=prefill_fp8_logits,
    )
    # A CP rank's rows are an interleaved subset of the batch, and without the dense
    # fp4 indexer prefill scores on torch: both keep the sparse table on decode only.
    if not use_deep_gemm_prefill or get_parallel().attn_cp_size > 1:
        return dense_blocks, sparse_table
    return sparse_table, sparse_table
