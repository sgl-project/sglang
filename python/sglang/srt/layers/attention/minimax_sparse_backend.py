# SPDX-License-Identifier: Apache-2.0
"""Backward-compatible re-exports for ``sglang.srt.layers.attention.minimax_sparse``."""

from __future__ import annotations

from sglang.srt.layers.attention.minimax_sparse import (
    MiniMaxBF16SparseAttnBackend,
    MiniMaxHybridAttnBackend,
    MiniMaxMSAV1AttnBackend,
    MiniMaxNPUSparseAttnBackend,
    MiniMaxROCmSparseAttnBackend,
    MiniMaxSparseAttnBackend,
    MiniMaxSparseBaseBackend,
    MiniMaxSparseForwardMetadata,
    MiniMaxSparseNvfp4AttnBackend,
    build_minimax_attn_backend,
    create_minimax_sparse_backend,
    resolve_minimax_sparse_backend_cls,
)
from sglang.srt.layers.attention.minimax_sparse.metadata import _quant_q_fp8
from sglang.srt.layers.attention.minimax_sparse.npu_backend import (
    _idx_cache_to_bnsd,
    _kv_cache_to_bnsd,
    _native_attn_enabled,
    _native_indexer_enabled,
)

__all__ = [
    "MiniMaxBF16SparseAttnBackend",
    "MiniMaxHybridAttnBackend",
    "MiniMaxMSAV1AttnBackend",
    "MiniMaxNPUSparseAttnBackend",
    "MiniMaxROCmSparseAttnBackend",
    "MiniMaxSparseAttnBackend",
    "MiniMaxSparseBaseBackend",
    "MiniMaxSparseForwardMetadata",
    "MiniMaxSparseNvfp4AttnBackend",
    "_idx_cache_to_bnsd",
    "_kv_cache_to_bnsd",
    "_native_attn_enabled",
    "_native_indexer_enabled",
    "_quant_q_fp8",
    "build_minimax_attn_backend",
    "create_minimax_sparse_backend",
    "resolve_minimax_sparse_backend_cls",
]
