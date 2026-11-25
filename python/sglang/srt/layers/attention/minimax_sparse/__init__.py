# SPDX-License-Identifier: Apache-2.0
"""MiniMax sparse attention subpackage."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from sglang.srt.configs.model_config import uses_minimax_m3_nvfp4_native_stack
from sglang.srt.layers.attention.minimax_sparse.base import (
    MiniMaxSparseAttnBackend,
    MiniMaxSparseBaseBackend,
    create_minimax_sparse_backend,
    resolve_minimax_sparse_backend_cls,
)
from sglang.srt.layers.attention.minimax_sparse.hybrid_backend import (
    MiniMaxHybridAttnBackend,
)
from sglang.srt.layers.attention.minimax_sparse.metadata import (
    MiniMaxSparseForwardMetadata,
)

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.layers.attention.minimax_sparse.bf16_backend import (
        MiniMaxBF16SparseAttnBackend,
    )
    from sglang.srt.layers.attention.minimax_sparse.msa_v1_backend import (
        MiniMaxMSAV1AttnBackend,
        MiniMaxSparseNvfp4AttnBackend,
    )
    from sglang.srt.layers.attention.minimax_sparse.npu_backend import (
        MiniMaxNPUSparseAttnBackend,
    )
    from sglang.srt.layers.attention.minimax_sparse.rocm_backend import (
        MiniMaxROCmSparseAttnBackend,
    )
    from sglang.srt.model_executor.model_runner import ModelRunner

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "MiniMaxBF16SparseAttnBackend": (".bf16_backend", "MiniMaxBF16SparseAttnBackend"),
    "MiniMaxMSAV1AttnBackend": (".msa_v1_backend", "MiniMaxMSAV1AttnBackend"),
    "MiniMaxSparseNvfp4AttnBackend": (
        ".msa_v1_backend",
        "MiniMaxSparseNvfp4AttnBackend",
    ),
    "MiniMaxNPUSparseAttnBackend": (".npu_backend", "MiniMaxNPUSparseAttnBackend"),
    "MiniMaxROCmSparseAttnBackend": (".rocm_backend", "MiniMaxROCmSparseAttnBackend"),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def build_minimax_attn_backend(
    runner: ModelRunner, full_attn_backend: AttentionBackend
) -> AttentionBackend:
    """Build the top-level attention backend for a MiniMax sparse model."""
    if uses_minimax_m3_nvfp4_native_stack(runner.model_config.quantization):
        from sglang.srt.layers.attention.minimax_sparse.msa_v1_backend import (
            MiniMaxMSAV1AttnBackend,
        )

        return MiniMaxMSAV1AttnBackend(runner)
    sparse_backend = create_minimax_sparse_backend(runner)
    return MiniMaxHybridAttnBackend(
        full_attn_backend, sparse_backend, sparse_backend.sparse_layer_ids
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
    "build_minimax_attn_backend",
    "create_minimax_sparse_backend",
    "resolve_minimax_sparse_backend_cls",
]
