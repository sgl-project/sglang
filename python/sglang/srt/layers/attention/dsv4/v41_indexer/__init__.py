"""The top-k selection of the DeepSeek V4.1 ratio-1/2 index layers: the full top-k
(``full_topk``), and the candidate schemes split by what a source publishes, a
DeepGEMM sparse table (``sparse_table``) or dense block ids (``dense_blocks``)."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Tuple

import torch

from .full_topk import FullTopKIndexer
from .types import (
    CandidateMetadata,
    CapturedPrefillInputs,
    DecodeCandidates,
    DecodeInputs,
    PrefillCandidates,
    PrefillInputs,
    Selection,
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
    "Selection",
    "has_dense_fp4_indexer",
    "is_sm100_or_newer",
    "make_candidate_indexer",
    "make_full_topk_indexer",
]


@functools.cache
def is_sm100_or_newer() -> bool:
    # DeepGEMM's fp8_fp4 mqa-logits kernels need SM100+; Hopper takes the torch indexer.
    return torch.cuda.get_device_capability()[0] >= 10


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
def _use_deep_gemm_prefill() -> bool:
    # A CP rank scores rank-local rows, which only DeepGEMM serves, so CP ignores the env.
    from sglang.srt.environ import envs
    from sglang.srt.runtime_context import get_parallel

    return (
        is_sm100_or_newer()
        and has_dense_fp4_indexer()
        and (
            get_parallel().attn_cp_size > 1
            or not envs.SGLANG_DSV41_TORCH_PREFILL_INDEXER.get()
        )
    )


def make_full_topk_indexer(
    *,
    token_to_kv_pool: DeepSeekV4TokenToKVPool,
    req_to_token: torch.Tensor,
) -> FullTopKIndexer:
    return FullTopKIndexer(
        token_to_kv_pool=token_to_kv_pool,
        req_to_token=req_to_token,
        use_deep_gemm_prefill=_use_deep_gemm_prefill(),
        use_deep_gemm_decode=is_sm100_or_newer(),
    )


def make_candidate_indexer(
    *,
    token_to_kv_pool: DeepSeekV4TokenToKVPool,
    req_to_token: torch.Tensor,
    page_size: int,
    candidate_topk_blocks: int,
    candidate_block_size: int,
) -> Tuple[PrefillCandidates, DecodeCandidates]:
    from sglang.srt.runtime_context import get_parallel

    from .dense_blocks import DenseBlocksBackend

    use_deep_gemm_prefill = _use_deep_gemm_prefill()
    dense_blocks = DenseBlocksBackend(
        token_to_kv_pool=token_to_kv_pool,
        req_to_token=req_to_token,
        candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size,
        use_deep_gemm_prefill=use_deep_gemm_prefill,
    )
    # Without candidate blocks no layer is a candidate source or consumer.
    if not is_sm100_or_newer() or candidate_topk_blocks <= 0:
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
        req_to_token=req_to_token,
        page_size=page_size,
        candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size,
    )
    # A CP rank's rows are an interleaved subset of the batch; the torch prefill
    # env keeps the sparse table on decode only.
    if not use_deep_gemm_prefill or get_parallel().attn_cp_size > 1:
        return dense_blocks, sparse_table
    return sparse_table, sparse_table
