"""Infrastructure for parallel GPU test execution."""

from .gpu_allocator import (
    GPUAllocator,
    GPUInfo,
    GPUSlot,
    get_gpu_memory_usage,
    get_open_port,
    get_physical_device_indices,
    nvml_context,
    wait_for_gpu_memory_to_clear,
)
from .model_pool import ModelInstance, ModelPool
from .model_specs import MODEL_SPECS

__all__ = [
    # GPU allocation
    "GPUAllocator",
    "GPUInfo",
    "GPUSlot",
    # GPU utilities
    "nvml_context",
    "get_open_port",
    "get_physical_device_indices",
    "get_gpu_memory_usage",
    "wait_for_gpu_memory_to_clear",
    # Model management
    "ModelInstance",
    "ModelPool",
    "MODEL_SPECS",
]
