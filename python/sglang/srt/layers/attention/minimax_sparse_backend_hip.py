# SPDX-License-Identifier: Apache-2.0
"""Backward-compatible re-exports for ``minimax_sparse.rocm_backend``."""

from __future__ import annotations

from sglang.srt.layers.attention.minimax_sparse.rocm_backend import (
    MiniMaxROCmSparseAttnBackend,
    dense_metadata_unused,
    init_sparse_backend,
    nvfp4_layer_caches,
    verify_num_draft,
)

__all__ = [
    "MiniMaxROCmSparseAttnBackend",
    "dense_metadata_unused",
    "init_sparse_backend",
    "nvfp4_layer_caches",
    "verify_num_draft",
]
