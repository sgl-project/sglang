"""Compatibility imports for the multimodal CUDA IPC transport.

New code should import from :mod:`sglang.srt.multimodal.transport.cuda_ipc`.
"""

from sglang.srt.multimodal.transport.cuda_ipc import (
    DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY,
    MM_FEATURE_CACHE_SIZE,
    MM_ITEM_MEMORY_POOL_RECYCLE_INTERVAL,
    CudaIpcTensorTransportProxy,
    MmItemMemoryPool,
    _pool_handle_cache_clear,
    get_mm_feature_pool_size_per_worker,
)

__all__ = [
    "DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY",
    "MM_FEATURE_CACHE_SIZE",
    "MM_ITEM_MEMORY_POOL_RECYCLE_INTERVAL",
    "CudaIpcTensorTransportProxy",
    "MmItemMemoryPool",
    "_pool_handle_cache_clear",
    "get_mm_feature_pool_size_per_worker",
]
