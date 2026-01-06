"""Infrastructure for parallel GPU test execution."""

from .constants import (  # Enums; Convenience sets; Fixture parameters; Defaults; Environment variables
    CLOUD_RUNTIMES,
    DEFAULT_HOST,
    DEFAULT_MODEL,
    DEFAULT_ROUTER_TIMEOUT,
    DEFAULT_STARTUP_TIMEOUT,
    ENV_BACKENDS,
    ENV_MODEL,
    ENV_MODELS,
    ENV_SHOW_ROUTER_LOGS,
    ENV_SHOW_WORKER_LOGS,
    ENV_SKIP_BACKEND_SETUP,
    ENV_SKIP_MODEL_POOL,
    ENV_STARTUP_TIMEOUT,
    HEALTH_CHECK_INTERVAL,
    LOCAL_MODES,
    LOCAL_RUNTIMES,
    PARAM_BACKEND_ROUTER,
    PARAM_MODEL,
    PARAM_SETUP_BACKEND,
    ConnectionMode,
    Runtime,
    WorkerType,
)
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
from .process_utils import (
    detect_ib_device,
    kill_process_tree,
    terminate_process,
    wait_for_health,
    wait_for_workers_ready,
)
from .run_eval import run_eval

__all__ = [
    # Enums
    "ConnectionMode",
    "WorkerType",
    "Runtime",
    # Convenience sets
    "LOCAL_MODES",
    "LOCAL_RUNTIMES",
    "CLOUD_RUNTIMES",
    # Fixture params
    "PARAM_SETUP_BACKEND",
    "PARAM_BACKEND_ROUTER",
    "PARAM_MODEL",
    # Defaults
    "DEFAULT_MODEL",
    "DEFAULT_HOST",
    "DEFAULT_STARTUP_TIMEOUT",
    "DEFAULT_ROUTER_TIMEOUT",
    "HEALTH_CHECK_INTERVAL",
    # Env vars
    "ENV_MODELS",
    "ENV_BACKENDS",
    "ENV_MODEL",
    "ENV_STARTUP_TIMEOUT",
    "ENV_SKIP_MODEL_POOL",
    "ENV_SKIP_BACKEND_SETUP",
    "ENV_SHOW_ROUTER_LOGS",
    "ENV_SHOW_WORKER_LOGS",
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
    # Process utilities
    "kill_process_tree",
    "terminate_process",
    "wait_for_health",
    "wait_for_workers_ready",
    "detect_ib_device",
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
    # Evaluation
    "run_eval",
]
