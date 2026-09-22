"""Compatibility imports for the multimodal CUDA IPC transport.

New code should import from :mod:`sglang.srt.multimodal.transport.cuda_ipc`.
"""

from sglang.srt.multimodal.transport.cuda_ipc import (
    DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY,
    MM_ITEM_MEMORY_POOL_RECYCLE_INTERVAL,
    CudaIpcTensorTransportProxy,
    MmItemMemoryPool,
    _pool_handle_cache_clear,
    get_mm_feature_pool_size_per_worker,
    mm_feature_cache_size,
)

__all__ = [
    "DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY",
    "MM_FEATURE_CACHE_SIZE",
    "MM_ITEM_MEMORY_POOL_RECYCLE_INTERVAL",
    "CudaIpcTensorTransportProxy",
    "MmItemMemoryPool",
    "_pool_handle_cache_clear",
    "get_mm_feature_pool_size_per_worker",
    "mm_feature_cache_size",
]


def __getattr__(name: str):
    # The former module constant, kept for old import paths; resolved on access
    # so it reflects the arguments.
    if name == "MM_FEATURE_CACHE_SIZE":
        return mm_feature_cache_size()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
