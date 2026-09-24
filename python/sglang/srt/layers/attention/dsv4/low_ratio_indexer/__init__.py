"""The top-k selection of the DeepSeek V4.1 ratio-1/2 index layers.

The dense layers go through ``DenseIndexer``; a candidate source and its
consumers go through the ``CandidateIndexer`` that ``make_candidate_indexer``
builds. The two share only the scoring utilities. Nothing else in this package
is meant to be named from outside it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.runtime_context import get_parallel

from .candidate_indexer import CandidateIndexer
from .caps import (
    has_dense_fp4_indexer,
    is_sm100_or_newer,
    use_deep_gemm_prefill,
)
from .dense_indexer import DenseIndexer
from .inputs import (
    CandidateMetadata,
    CapturedPrefillInputs,
    DecodeInputs,
    PrefillInputs,
    Selection,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

__all__ = [
    "CandidateIndexer",
    "CandidateMetadata",
    "CapturedPrefillInputs",
    "DecodeInputs",
    "DenseIndexer",
    "PrefillInputs",
    "Selection",
    "has_dense_fp4_indexer",
    "is_sm100_or_newer",
    "make_candidate_indexer",
]


def make_candidate_indexer(
    *,
    token_to_kv_pool: DeepSeekV4TokenToKVPool,
    req_to_token: torch.Tensor,
    page_size: int,
    candidate_topk_blocks: int,
    candidate_block_size: int,
) -> CandidateIndexer:
    from .torch_backend import TorchCandidateBackend

    torch_backend = TorchCandidateBackend(
        token_to_kv_pool=token_to_kv_pool, req_to_token=req_to_token
    )
    # Without candidate blocks no layer is a candidate source or consumer.
    if not is_sm100_or_newer() or candidate_topk_blocks <= 0:
        return CandidateIndexer(prefill=torch_backend, decode=torch_backend)

    from sglang.srt.layers.deep_gemm_wrapper.configurer import (
        DEEPGEMM_PAGED_SPARSE_MQA_LOGITS,
    )

    if not DEEPGEMM_PAGED_SPARSE_MQA_LOGITS:
        raise RuntimeError(
            "the candidate indexer needs DeepGEMM's paged sparse MQA logits "
            "(sgl-deep-gemm >= 0.2.0 with SGLANG_ENABLE_JIT_DEEPGEMM on)"
        )
    from .deep_gemm_backend import (
        DeepGEMMCandidateBackend,
        DeepGEMMCPCandidateBackend,
    )

    deep_gemm_backend = DeepGEMMCandidateBackend(
        token_to_kv_pool=token_to_kv_pool,
        req_to_token=req_to_token,
        page_size=page_size,
        candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size,
    )

    if not use_deep_gemm_prefill():
        prefill_backend = torch_backend
    elif get_parallel().attn_cp_size > 1:
        prefill_backend = DeepGEMMCPCandidateBackend(
            token_to_kv_pool=token_to_kv_pool,
            req_to_token=req_to_token,
            candidate_topk_blocks=candidate_topk_blocks,
            candidate_block_size=candidate_block_size,
        )
    else:
        prefill_backend = deep_gemm_backend
    return CandidateIndexer(prefill=prefill_backend, decode=deep_gemm_backend)
