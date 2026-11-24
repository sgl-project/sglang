# SPDX-License-Identifier: Apache-2.0
"""Backward-compatible re-exports for ``minimax_sparse.msa_v1_backend``."""

from __future__ import annotations

from sglang.srt.layers.attention.minimax_sparse.msa_v1_backend import (
    MiniMaxMSAV1AttnBackend,
    MiniMaxSparseNvfp4AttnBackend,
)
from sglang.srt.layers.attention.minimax_sparse_ops.msa_plan import (
    MSAV1DecodeAttentionPlan,
    MSAV1DecodeIndexerPlan,
    MSAV1PrefillPlan,
    build_msa_v1_page_table,
    msa_v1_qmul4_decode_capability,
)

__all__ = [
    "MSAV1DecodeAttentionPlan",
    "MSAV1DecodeIndexerPlan",
    "MSAV1PrefillPlan",
    "MiniMaxMSAV1AttnBackend",
    "MiniMaxSparseNvfp4AttnBackend",
    "build_msa_v1_page_table",
    "msa_v1_qmul4_decode_capability",
]
