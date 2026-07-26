"""Cache components for the MLX backend."""

from sglang.srt.hardware_backend.mlx.kv_cache.attention_contract import (
    get_head_dim,
    get_num_heads,
    get_num_kv_heads,
    is_attention_module,
    uses_sliding_window_attention,
)
from sglang.srt.hardware_backend.mlx.kv_cache.attention_kv_cache import (
    AttentionOffsetCache,
    ContiguousAttentionKVCache,
    PoolBackedAttentionKVCache,
)
from sglang.srt.hardware_backend.mlx.kv_cache.attention_kv_pool import (
    MlxAttentionKVPool,
    MlxBlockAttentionKVPool,
)
from sglang.srt.hardware_backend.mlx.kv_cache.attention_wrapper import (
    BatchedDecodeContext,
    MLXAttentionWrapper,
    clear_context,
    get_context,
    set_context,
)
from sglang.srt.hardware_backend.mlx.kv_cache.auxiliary_state import (
    MlxAuxiliaryStateComponent,
    MlxAuxiliaryStatePool,
    MlxAuxiliaryStateReqToTokenPool,
)
from sglang.srt.hardware_backend.mlx.kv_cache.block_metadata import (
    MLXBlockPagedAttentionMetadata,
    build_decode_block_metadata,
)
from sglang.srt.hardware_backend.mlx.kv_cache.layout import MlxModelCacheLayout
from sglang.srt.hardware_backend.mlx.kv_cache.model_patching import (
    find_attention_layers,
    get_num_layers,
    patch_model_attention,
)

__all__ = [
    "BatchedDecodeContext",
    "clear_context",
    "AttentionOffsetCache",
    "ContiguousAttentionKVCache",
    "find_attention_layers",
    "get_head_dim",
    "get_context",
    "get_num_layers",
    "get_num_heads",
    "get_num_kv_heads",
    "is_attention_module",
    "MLXAttentionWrapper",
    "MLXBlockPagedAttentionMetadata",
    "MlxAttentionKVPool",
    "MlxAuxiliaryStateComponent",
    "MlxAuxiliaryStatePool",
    "MlxAuxiliaryStateReqToTokenPool",
    "MlxBlockAttentionKVPool",
    "MlxModelCacheLayout",
    "patch_model_attention",
    "PoolBackedAttentionKVCache",
    "build_decode_block_metadata",
    "set_context",
    "uses_sliding_window_attention",
]
