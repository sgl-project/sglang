"""DeepSeek-V3.2 only."""

from .elementwise import (
    fused_k_indexer_norm_rope_first_hadamard,
    fused_k_indexer_norm_rope_store,
)

__all__ = [
    "fused_k_indexer_norm_rope_first_hadamard",
    "fused_k_indexer_norm_rope_store",
]
