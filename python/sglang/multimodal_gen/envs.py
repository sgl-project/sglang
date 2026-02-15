# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/envs.py

import logging
import os
from typing import TYPE_CHECKING, Any, Callable

from sglang.multimodal_gen.runtime.utils.common import get_bool_env_var

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    SGLANG_DIFFUSION_RINGBUFFER_WARNING_INTERVAL: int = 60
    SGLANG_DIFFUSION_NCCL_SO_PATH: str | None = None
    LD_LIBRARY_PATH: str | None = None
    LOCAL_RANK: int = 0
    CUDA_VISIBLE_DEVICES: str | None = None
    SGLANG_DIFFUSION_CACHE_ROOT: str = os.path.expanduser("~/.cache/sgl_diffusion")
    SGLANG_DIFFUSION_CONFIG_ROOT: str = os.path.expanduser("~/.config/sgl_diffusion")
    SGLANG_DIFFUSION_CONFIGURE_LOGGING: int = 1
    SGLANG_DIFFUSION_LOGGING_LEVEL: str = "INFO"
    SGLANG_DIFFUSION_LOGGING_PREFIX: str = ""
    SGLANG_DIFFUSION_LOGGING_CONFIG_PATH: str | None = None
    SGLANG_DIFFUSION_TRACE_FUNCTION: int = 0
    SGLANG_DIFFUSION_WORKER_MULTIPROC_METHOD: str = "fork"
    SGLANG_DIFFUSION_TARGET_DEVICE: str = "cuda"
    MAX_JOBS: str | None = None
    NVCC_THREADS: str | None = None
    CMAKE_BUILD_TYPE: str | None = None
    VERBOSE: bool = False
    SGLANG_DIFFUSION_SERVER_DEV_MODE: bool = False
    SGLANG_DIFFUSION_STAGE_LOGGING: bool = False
    # cache-dit env vars (primary transformer)
    SGLANG_CACHE_DIT_ENABLED: bool = False
    SGLANG_CACHE_DIT_FN: int = 1
    SGLANG_CACHE_DIT_BN: int = 0
    SGLANG_CACHE_DIT_WARMUP: int = 4
    SGLANG_CACHE_DIT_RDT: float = 0.24
    SGLANG_CACHE_DIT_MC: int = 3
    SGLANG_CACHE_DIT_TAYLORSEER: bool = False
    SGLANG_CACHE_DIT_TS_ORDER: int = 1
    SGLANG_CACHE_DIT_SCM_PRESET: str = "none"
    SGLANG_CACHE_DIT_SCM_COMPUTE_BINS: str | None = None
    SGLANG_CACHE_DIT_SCM_CACHE_BINS: str | None = None
    SGLANG_CACHE_DIT_SCM_POLICY: str = "dynamic"
    # cache-dit env vars (secondary transformer, e.g., Wan2.2 low-noise expert)
    SGLANG_CACHE_DIT_SECONDARY_FN: int = 1
    SGLANG_CACHE_DIT_SECONDARY_BN: int = 0
    SGLANG_CACHE_DIT_SECONDARY_WARMUP: int = 4
    SGLANG_CACHE_DIT_SECONDARY_RDT: float = 0.24
    SGLANG_CACHE_DIT_SECONDARY_MC: int = 3
    SGLANG_CACHE_DIT_SECONDARY_TAYLORSEER: bool = False
    SGLANG_CACHE_DIT_SECONDARY_TS_ORDER: int = 1
    # model loading
    SGLANG_USE_RUNAI_MODEL_STREAMER: bool = True


def get_default_cache_root() -> str:
    return os.getenv(
        "XDG_CACHE_HOME",
        os.path.join(os.path.expanduser("~"), ".cache"),
    )


def get_default_config_root() -> str:
    return os.getenv(
        "XDG_CONFIG_HOME",
        os.path.join(os.path.expanduser("~"), ".config"),
    )


def maybe_convert_int(value: str | None) -> int | None:
    return int(value) if value is not None else None


# helpers for environment variable definitions
def _lazy_str(key: str, default: str | None = None) -> Callable[[], str | None]:
    return lambda: os.getenv(key, default)


def _lazy_int(key: str, default: str | int | None = None) -> Callable[[], int | None]:
    def _getter():
        val = os.getenv(key)
        if val is None:
            return int(default) if default is not None else None
        return int(val)

    return _getter


def _lazy_float(key: str, default: str | float) -> Callable[[], float]:
    return lambda: float(os.getenv(key, str(default)))


def _lazy_bool(key: str, default: str = "false") -> Callable[[], bool]:
    return lambda: get_bool_env_var(key, default)


def _lazy_bool_any(keys: list[str], default: str = "false") -> Callable[[], bool]:
    def _getter():
        for key in keys:
            if get_bool_env_var(key, "false"):
                return True
        return (
            get_bool_env_var("", default)
            if not keys
            else get_bool_env_var(keys[0], default)
        )

    return _getter


def _lazy_path(
    key: str, default_func: Callable[[], str] | None = None
) -> Callable[[], str | None]:
    def _getter():
        val = os.getenv(key)
        if val is None:
            if default_func is None:
                return None
            val = default_func()
        return os.path.expanduser(val)

    return _getter


# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

# begin-env-vars-definition

environment_variables: dict[str, Callable[[], Any]] = {
    # ================== Installation Time Env Vars ==================
    # Target device of sglang-diffusion, supporting [cuda (by default),
    # rocm, neuron, cpu, openvino]
    "SGLANG_DIFFUSION_TARGET_DEVICE": _lazy_str(
        "SGLANG_DIFFUSION_TARGET_DEVICE", "cuda"
    ),
    # Maximum number of compilation jobs to run in parallel.
    # By default this is the number of CPUs
    "MAX_JOBS": _lazy_str("MAX_JOBS"),
    # Number of threads to use for nvcc
    # By default this is 1.
    # If set, `MAX_JOBS` will be reduced to avoid oversubscribing the CPU.
    "NVCC_THREADS": _lazy_str("NVCC_THREADS"),
    # If set, sgl_diffusion will use precompiled binaries (*.so)
    "SGLANG_DIFFUSION_USE_PRECOMPILED": _lazy_bool_any(
        [
            "SGLANG_DIFFUSION_USE_PRECOMPILED",
            "SGLANG_DIFFUSION_PRECOMPILED_WHEEL_LOCATION",
        ]
    ),
    # CMake build type
    # If not set, defaults to "Debug" or "RelWithDebInfo"
    # Available options: "Debug", "Release", "RelWithDebInfo"
    "CMAKE_BUILD_TYPE": _lazy_str("CMAKE_BUILD_TYPE"),
    # If set, sgl_diffusion will print verbose logs during installation
    "VERBOSE": _lazy_bool("VERBOSE"),
    # Root directory for SGL-diffusion configuration files
    # Defaults to `~/.config/sgl_diffusion` unless `XDG_CONFIG_HOME` is set
    # Note that this not only affects how sgl_diffusion finds its configuration files
    # during runtime, but also affects how sgl_diffusion installs its configuration
    # files during **installation**.
    "SGLANG_DIFFUSION_CONFIG_ROOT": _lazy_path(
        "SGLANG_DIFFUSION_CONFIG_ROOT",
        lambda: os.path.join(get_default_config_root(), "sgl_diffusion"),
    ),
    # ================== Runtime Env Vars ==================
    # Root directory for SGL-diffusion cache files
    # Defaults to `~/.cache/sgl_diffusion` unless `XDG_CACHE_HOME` is set
    "SGLANG_DIFFUSION_CACHE_ROOT": _lazy_path(
        "SGLANG_DIFFUSION_CACHE_ROOT",
        lambda: os.path.join(get_default_cache_root(), "sgl_diffusion"),
    ),
    # Interval in seconds to log a warning message when the ring buffer is full
    "SGLANG_DIFFUSION_RINGBUFFER_WARNING_INTERVAL": _lazy_int(
        "SGLANG_DIFFUSION_RINGBUFFER_WARNING_INTERVAL", 60
    ),
    # Path to the NCCL library file. It is needed because nccl>=2.19 brought
    # by PyTorch contains a bug: https://github.com/NVIDIA/nccl/issues/1234
    "SGLANG_DIFFUSION_NCCL_SO_PATH": _lazy_str("SGLANG_DIFFUSION_NCCL_SO_PATH"),
    # when `SGLANG_DIFFUSION_NCCL_SO_PATH` is not set, sgl_diffusion will try to find the nccl
    # library file in the locations specified by `LD_LIBRARY_PATH`
    "LD_LIBRARY_PATH": _lazy_str("LD_LIBRARY_PATH"),
    # Internal flag to enable Dynamo fullgraph capture
    "SGLANG_DIFFUSION_TEST_DYNAMO_FULLGRAPH_CAPTURE": _lazy_bool(
        "SGLANG_DIFFUSION_TEST_DYNAMO_FULLGRAPH_CAPTURE", "1"
    ),
    # local rank of the process in the distributed setting, used to determine
    # the GPU device id
    "LOCAL_RANK": _lazy_int("LOCAL_RANK", 0),
    # used to control the visible devices in the distributed setting
    "CUDA_VISIBLE_DEVICES": _lazy_str("CUDA_VISIBLE_DEVICES"),
    # timeout for each iteration in the engine
    "SGLANG_DIFFUSION_ENGINE_ITERATION_TIMEOUT_S": _lazy_int(
        "SGLANG_DIFFUSION_ENGINE_ITERATION_TIMEOUT_S", 60
    ),
    # Logging configuration
    # If set to 0, sgl_diffusion will not configure logging
    # If set to 1, sgl_diffusion will configure logging using the default configuration
    #    or the configuration file specified by SGLANG_DIFFUSION_LOGGING_CONFIG_PATH
    "SGLANG_DIFFUSION_CONFIGURE_LOGGING": _lazy_int(
        "SGLANG_DIFFUSION_CONFIGURE_LOGGING", 1
    ),
    "SGLANG_DIFFUSION_LOGGING_CONFIG_PATH": _lazy_str(
        "SGLANG_DIFFUSION_LOGGING_CONFIG_PATH"
    ),
    # this is used for configuring the default logging level
    "SGLANG_DIFFUSION_LOGGING_LEVEL": _lazy_str(
        "SGLANG_DIFFUSION_LOGGING_LEVEL", "INFO"
    ),
    # if set, SGLANG_DIFFUSION_LOGGING_PREFIX will be prepended to all log messages
    "SGLANG_DIFFUSION_LOGGING_PREFIX": _lazy_str("SGLANG_DIFFUSION_LOGGING_PREFIX", ""),
    # Trace function calls
    # If set to 1, sgl_diffusion will trace function calls
    # Useful for debugging
    "SGLANG_DIFFUSION_TRACE_FUNCTION": _lazy_int("SGLANG_DIFFUSION_TRACE_FUNCTION", 0),
    # Path to the attention configuration file. Only used for sliding tile
    # attention for now.
    "SGLANG_DIFFUSION_ATTENTION_CONFIG": _lazy_path(
        "SGLANG_DIFFUSION_ATTENTION_CONFIG"
    ),
    # Optional override to force a specific attention backend (e.g. "aiter")
    "SGLANG_DIFFUSION_ATTENTION_BACKEND": _lazy_str(
        "SGLANG_DIFFUSION_ATTENTION_BACKEND"
    ),
    # Use dedicated multiprocess context for workers.
    # Both spawn and fork work
    "SGLANG_DIFFUSION_WORKER_MULTIPROC_METHOD": _lazy_str(
        "SGLANG_DIFFUSION_WORKER_MULTIPROC_METHOD", "fork"
    ),
    # Enables torch profiler if set. Path to the directory where torch profiler
    # traces are saved. Note that it must be an absolute path.
    "SGLANG_DIFFUSION_TORCH_PROFILER_DIR": _lazy_path(
        "SGLANG_DIFFUSION_TORCH_PROFILER_DIR"
    ),
    # If set, sgl_diffusion will run in development mode, which will enable
    # some additional endpoints for developing and debugging,
    # e.g. `/reset_prefix_cache`
    "SGLANG_DIFFUSION_SERVER_DEV_MODE": _lazy_bool("SGLANG_DIFFUSION_SERVER_DEV_MODE"),
    # If set, sgl_diffusion will enable stage logging, which will print the time
    # taken for each stage
    "SGLANG_DIFFUSION_STAGE_LOGGING": _lazy_bool("SGLANG_DIFFUSION_STAGE_LOGGING"),
    # ================== cache-dit Env Vars ==================
    # Enable cache-dit acceleration for DiT inference
    "SGLANG_CACHE_DIT_ENABLED": _lazy_bool("SGLANG_CACHE_DIT_ENABLED"),
    # Number of first blocks to always compute (DBCache F parameter)
    "SGLANG_CACHE_DIT_FN": _lazy_int("SGLANG_CACHE_DIT_FN", 1),
    # Number of last blocks to always compute (DBCache B parameter)
    "SGLANG_CACHE_DIT_BN": _lazy_int("SGLANG_CACHE_DIT_BN", 0),
    # Warmup steps before caching (DBCache W parameter)
    "SGLANG_CACHE_DIT_WARMUP": _lazy_int("SGLANG_CACHE_DIT_WARMUP", 4),
    # Residual difference threshold (DBCache R parameter)
    "SGLANG_CACHE_DIT_RDT": _lazy_float("SGLANG_CACHE_DIT_RDT", 0.24),
    # Maximum continuous cached steps (DBCache MC parameter)
    "SGLANG_CACHE_DIT_MC": _lazy_int("SGLANG_CACHE_DIT_MC", 3),
    # Enable TaylorSeer calibrator
    "SGLANG_CACHE_DIT_TAYLORSEER": _lazy_bool("SGLANG_CACHE_DIT_TAYLORSEER", "false"),
    # TaylorSeer order (1 or 2)
    "SGLANG_CACHE_DIT_TS_ORDER": _lazy_int("SGLANG_CACHE_DIT_TS_ORDER", 1),
    # SCM preset: none, slow, medium, fast, ultra
    "SGLANG_CACHE_DIT_SCM_PRESET": _lazy_str("SGLANG_CACHE_DIT_SCM_PRESET", "none"),
    # SCM custom compute bins (e.g., "8,3,3,2,2")
    "SGLANG_CACHE_DIT_SCM_COMPUTE_BINS": _lazy_str("SGLANG_CACHE_DIT_SCM_COMPUTE_BINS"),
    # SCM custom cache bins (e.g., "1,2,2,2,3")
    "SGLANG_CACHE_DIT_SCM_CACHE_BINS": _lazy_str("SGLANG_CACHE_DIT_SCM_CACHE_BINS"),
    # SCM policy: dynamic or static
    "SGLANG_CACHE_DIT_SCM_POLICY": _lazy_str("SGLANG_CACHE_DIT_SCM_POLICY", "dynamic"),
    # model loading
    "SGLANG_USE_RUNAI_MODEL_STREAMER": _lazy_bool(
        "SGLANG_USE_RUNAI_MODEL_STREAMER", "true"
    ),
}

# Add cache-dit Secondary Transformer Env Vars via programmatic generation to reduce duplication
_CACHE_DIT_SECONDARY_CONFIGS = [
    ("FN", int, "1"),
    ("BN", int, "0"),
    ("WARMUP", int, "4"),
    ("RDT", float, "0.24"),
    ("MC", int, "3"),
    ("TS_ORDER", int, "1"),
]


def _create_secondary_getter(suffix, type_func, default_val):
    primary_key = f"SGLANG_CACHE_DIT_{suffix}"
    secondary_key = f"SGLANG_CACHE_DIT_SECONDARY_{suffix}"

    def _getter():
        val = os.getenv(secondary_key)
        if val is not None:
            return type_func(val)
        return type_func(os.getenv(primary_key, str(default_val)))

    return secondary_key, _getter


for suffix, type_func, default_val in _CACHE_DIT_SECONDARY_CONFIGS:
    key, getter = _create_secondary_getter(suffix, type_func, default_val)
    environment_variables[key] = getter


# Special handling for boolean secondary var (TaylorSeer)
def _secondary_taylorseer_getter():
    return get_bool_env_var(
        "SGLANG_CACHE_DIT_SECONDARY_TAYLORSEER",
        default=os.getenv("SGLANG_CACHE_DIT_TAYLORSEER", "false"),
    )


environment_variables["SGLANG_CACHE_DIT_SECONDARY_TAYLORSEER"] = (
    _secondary_taylorseer_getter
)


# end-env-vars-definition
def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())
