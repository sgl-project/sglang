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
from .model_specs import (  # Default model paths; Model groups
    CHAT_MODELS,
    DEFAULT_EMBEDDING_MODEL_PATH,
    DEFAULT_ENABLE_THINKING_MODEL_PATH,
    DEFAULT_GPT_OSS_MODEL_PATH,
    DEFAULT_MISTRAL_FUNCTION_CALLING_MODEL_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_QWEN_FUNCTION_CALLING_MODEL_PATH,
    DEFAULT_REASONING_MODEL_PATH,
    DEFAULT_SMALL_MODEL_PATH,
    EMBEDDING_MODELS,
    FUNCTION_CALLING_MODELS,
    MODEL_SPECS,
    REASONING_MODELS,
)

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
    # Default model paths
    "DEFAULT_MODEL_PATH",
    "DEFAULT_SMALL_MODEL_PATH",
    "DEFAULT_REASONING_MODEL_PATH",
    "DEFAULT_ENABLE_THINKING_MODEL_PATH",
    "DEFAULT_QWEN_FUNCTION_CALLING_MODEL_PATH",
    "DEFAULT_MISTRAL_FUNCTION_CALLING_MODEL_PATH",
    "DEFAULT_GPT_OSS_MODEL_PATH",
    "DEFAULT_EMBEDDING_MODEL_PATH",
    # Model groups
    "CHAT_MODELS",
    "EMBEDDING_MODELS",
    "REASONING_MODELS",
    "FUNCTION_CALLING_MODELS",
]
