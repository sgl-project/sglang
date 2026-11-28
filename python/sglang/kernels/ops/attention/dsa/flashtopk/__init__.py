# SPDX-License-Identifier: Apache-2.0
"""FlashTopK prefill indexer top-k for DSA (`Indexer`) and KPool DSA (`IndexerKPool`).

Derives from FlashTopK (https://github.com/Tencent-Hunyuan/FlashTopK, Apache-2.0) at
commit ``090902abad86e5bb3324519124a680122fff91a1``, trimmed to what SGLang's FP8 DSA
prefill indexers use (FP8 operands, adaptive key pruning, unordered exact top-k with
deterministic tie-breaking, caller-owned transient scratch); files derived from
FlashInfer, TensorRT-LLM, and LiteTopK state so in their headers.
"""

from .prefill import (
    FLASHTOPK_ADAPTIVE_MIN_KEYS,
    FLASHTOPK_HEAD_DIM,
    FLASHTOPK_HEADS,
    FLASHTOPK_MAX_TOPK,
    flashtopk_fp8_indexer_topk,
    flashtopk_kpool_indexer_topk,
    flashtopk_kpool_should_use,
    flashtopk_min_keys,
    flashtopk_prunes,
    flashtopk_should_use,
    is_flashtopk_supported,
    warmup_flashtopk_prefill,
)

__all__ = [
    "FLASHTOPK_ADAPTIVE_MIN_KEYS",
    "FLASHTOPK_HEAD_DIM",
    "FLASHTOPK_HEADS",
    "FLASHTOPK_MAX_TOPK",
    "flashtopk_fp8_indexer_topk",
    "flashtopk_kpool_indexer_topk",
    "flashtopk_kpool_should_use",
    "flashtopk_min_keys",
    "flashtopk_prunes",
    "flashtopk_should_use",
    "is_flashtopk_supported",
    "warmup_flashtopk_prefill",
]
