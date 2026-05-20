"""KV cache components for the MLX backend."""

from sglang.srt.hardware_backend.mlx.kv_cache.attention_contract import (
    get_head_dim,
    get_num_heads,
    get_num_kv_heads,
    is_attention_module,
)
from sglang.srt.hardware_backend.mlx.kv_cache.attention_wrapper import (
    BatchedDecodeContext,
    MLXAttentionWrapper,
    clear_context,
    get_context,
    set_context,
)
from sglang.srt.hardware_backend.mlx.kv_cache.contiguous_cache import (
    ContiguousKVCache,
    OffsetCache,
    PoolBackedCache,
)
from sglang.srt.hardware_backend.mlx.kv_cache.kv_pool import MlxKVPool
from sglang.srt.hardware_backend.mlx.kv_cache.model_patching import (
    find_attention_layers,
    get_num_layers,
    patch_model_attention,
)

__all__ = [
    "BatchedDecodeContext",
    "clear_context",
    "ContiguousKVCache",
    "find_attention_layers",
    "get_head_dim",
    "get_context",
    "get_num_layers",
    "get_num_heads",
    "get_num_kv_heads",
    "is_attention_module",
    "MLXAttentionWrapper",
    "MlxKVPool",
    "OffsetCache",
    "patch_model_attention",
    "PoolBackedCache",
    "set_context",
]
