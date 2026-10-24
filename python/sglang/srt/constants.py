# GPU Memory Types
GPU_MEMORY_TYPE_KV_CACHE = "kv_cache"
GPU_MEMORY_TYPE_WEIGHTS = "weights"
GPU_MEMORY_TYPE_CUDA_GRAPH = "cuda_graph"
# Platform-gated tag: only honored where a platform registered a sleep-comm
# backend (see sglang.srt.utils.sleep_comm). It releases device-side
# communication resources and, on those platforms, implies CUDA graph drop +
# recapture (captured graphs embed comm buffer addresses).
GPU_MEMORY_TYPE_COMM = "comm"

# Tags every platform can release. Platform-specific tags (see
# sglang.srt.utils.sleep_comm.tags.PLATFORM_GATED_TAGS) are appended by the
# platform itself.
GPU_MEMORY_ALL_TYPES = [
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS,
    GPU_MEMORY_TYPE_CUDA_GRAPH,
]

HEALTH_CHECK_RID_PREFIX = "HEALTH_CHECK"

# Placeholder token inserted between items in Multi-Item Scoring sequences:
# query<delim>item1<delim>item2<delim>... Positions are pre-computed from item
# lengths (multi_item_delimiter_indices); the token only exists for FlashInfer
# attention mask compat and logprob column indexing. Will be removed once the
# attention backend supports position-only MIS.
MIS_DELIMITER_TOKEN_ID = 9999

GIB_BYTES = 1073741824  # 1024**3
