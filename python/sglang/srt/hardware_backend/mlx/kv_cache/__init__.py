"""Unified KV cache module for MLX backend.

Shared KV pool with trie-based prefix matching (radix cache) and
native paged attention integration.
"""

from sglang.srt.hardware_backend.mlx.kv_cache.attention_wrapper import (
    MLXAttentionWrapper,
    PagedAttentionContext,
    clear_context,
    get_context,
    set_context,
)
from sglang.srt.hardware_backend.mlx.kv_cache.kv_pool import MlxKVPool
from sglang.srt.hardware_backend.mlx.kv_cache.model_patching import (
    find_attention_layers,
    get_num_layers,
    patch_model_attention,
)
from sglang.srt.hardware_backend.mlx.kv_cache.radix_trie import MlxRadixTrie

__all__ = [
    "MLXAttentionWrapper",
    "MlxKVPool",
    "MlxRadixTrie",
    "PagedAttentionContext",
    "clear_context",
    "find_attention_layers",
    "get_context",
    "get_num_layers",
    "patch_model_attention",
    "set_context",
]
