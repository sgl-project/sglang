# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/envs.py

import logging
import os
from typing import TYPE_CHECKING, Any, Callable

from sglang.multimodal_gen.runtime.utils.common import get_bool_env_var

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    SGLANG_DIFFUSION_NCCL_SO_PATH: str | None = None
    LD_LIBRARY_PATH: str | None = None
    LOCAL_RANK: int = 0
    CUDA_VISIBLE_DEVICES: str | None = None
    SGLANG_DIFFUSION_CACHE_ROOT: str = os.path.expanduser("~/.cache/sgl_diffusion")
    SGLANG_DIFFUSION_CONFIG_ROOT: str = os.path.expanduser("~/.config/sgl_diffusion")
    SGLANG_DIFFUSION_LOGGING_LEVEL: str = "INFO"
    SGLANG_DIFFUSION_LOGGING_PREFIX: str = ""
    SGLANG_DIFFUSION_TRACE_FUNCTION: int = 0
    SGLANG_DIFFUSION_DISABLE_EARLY_VAE_DECODER_CAST: bool = False
    SGLANG_DIFFUSION_DISABLE_VAE_DECODER_STORE: bool = False
    SGLANG_DIFFUSION_DISABLE_LORA_MERGE_CACHE: bool = False
    SGLANG_DIFFUSION_WORKER_MULTIPROC_METHOD: str = "fork"
    SGLANG_DIFFUSION_TARGET_DEVICE: str = "cuda"
    SGLANG_DIFFUSION_PLATFORM_OVERRIDE: str = ""
    SGLANG_EXTERNAL_MODEL_PACKAGE: str = ""
    MAX_JOBS: str | None = None
    NVCC_THREADS: str | None = None
    CMAKE_BUILD_TYPE: str | None = None
    VERBOSE: bool = False
    SGLANG_DIFFUSION_SERVER_DEV_MODE: bool = False
    SGLANG_DIFFUSION_DISABLE_MAPPED_COURIER: bool = False
    SGLANG_DIFFUSION_TEST_FORCE_HOST_AVAILABLE_GIB: float | None = None
    SGLANG_DIFFUSION_TEST_CAP_DEVICE_MEMORY_GIB: float | None = None
    SGLANG_DIFFUSION_STAGE_LOGGING: bool = False
    SGLANG_DIFFUSION_CFG_GATE_STEP: float = 1.0
    # cache-dit env vars (primary transformer)
    # on by default; engages only on 2 ranks with peer-to-peer access and falls
    # back to NCCL when unavailable. Set 0 to force NCCL. Keep this in step with
    # the resolver below -- that is the value the runtime reads.
    SGLANG_DIFFUSION_IPC_A2A: bool = True
    # a deadlock backstop, not a per-step budget: a rank can legitimately stall
    # for seconds (layerwise offload, wan2.2 expert-tower swaps), and expiry now
    # retires the transport on every rank and fails the request
    SGLANG_DIFFUSION_IPC_A2A_TIMEOUT_MS: float = 10000.0
    # distinct (n_local, n_peer, dtype) staging pairs kept; each is two
    # slots and is never freed, so multi-resolution serving needs a cap
    SGLANG_DIFFUSION_IPC_A2A_MAX_BUFFERS: int = 16
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
    SGLANG_LINGBOT_ENABLE_INTERACTIVE_KV_WINDOW: bool = False
    SGLANG_LINGBOT_LAZY_VAE_ENCODE_BLACK_FRAMES: int | None = None
    SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND: str | None = None
    SGLANG_DIFFUSION_ENABLE_W8A8_FP8_GEMM: bool = False
    SGLANG_DIFFUSION_FP8_WEIGHT_DEQUANT_CACHE: bool = True
    SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D: str = "auto"
    SGLANG_USE_ROCM_VAE: bool = False
    SGLANG_USE_ROCM_CUDNN_BENCHMARK: bool = False
    SGLANG_USE_ROCM_VAE_CONV2D: bool = False
    SGLANG_USE_ROCM_VAE_CONV2D_BF16: bool = False


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


def _lazy_optional_float(key: str) -> Callable[[], float | None]:
    def _getter():
        val = os.getenv(key)
        return float(val) if val is not None else None

    return _getter


def _lazy_bool(key: str, default: str = "false") -> Callable[[], bool]:
    return lambda: get_bool_env_var(key, default)


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
    # Path to the NCCL library file. It is needed because nccl>=2.19 brought
    # by PyTorch contains a bug: https://github.com/NVIDIA/nccl/issues/1234
    "SGLANG_DIFFUSION_NCCL_SO_PATH": _lazy_str("SGLANG_DIFFUSION_NCCL_SO_PATH"),
    # when `SGLANG_DIFFUSION_NCCL_SO_PATH` is not set, sgl_diffusion will try to find the nccl
    # library file in the locations specified by `LD_LIBRARY_PATH`
    "LD_LIBRARY_PATH": _lazy_str("LD_LIBRARY_PATH"),
    # local rank of the process in the distributed setting, used to determine
    # the GPU device id
    "LOCAL_RANK": _lazy_int("LOCAL_RANK", 0),
    # used to control the visible devices in the distributed setting
    "CUDA_VISIBLE_DEVICES": _lazy_str("CUDA_VISIBLE_DEVICES"),
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
    # Internal per-worker platform override used by disaggregated role launch.
    # Empty means normal platform auto-detection.
    "SGLANG_DIFFUSION_PLATFORM_OVERRIDE": _lazy_str(
        "SGLANG_DIFFUSION_PLATFORM_OVERRIDE", ""
    ),
    # Import an installed package that registers out-of-tree diffusion models
    # and pipelines. This is shared with the SRT model plugin mechanism.
    "SGLANG_EXTERNAL_MODEL_PACKAGE": _lazy_str("SGLANG_EXTERNAL_MODEL_PACKAGE", ""),
    # Enables torch profiler if set. Path to the directory where torch profiler
    # traces are saved. Note that it must be an absolute path.
    "SGLANG_DIFFUSION_TORCH_PROFILER_DIR": _lazy_path(
        "SGLANG_DIFFUSION_TORCH_PROFILER_DIR"
    ),
    # If set, sgl_diffusion will run in development mode, which will enable
    # some additional endpoints for developing and debugging,
    # e.g. `/reset_prefix_cache`
    "SGLANG_DIFFUSION_SERVER_DEV_MODE": _lazy_bool("SGLANG_DIFFUSION_SERVER_DEV_MODE"),
    # Kill-switch for the courier thread that ships checkpoint-mapped layers to
    # the device off the compute thread. The courier already falls back to the
    # synchronous copy on any failure; this forces that path up front.
    "SGLANG_DIFFUSION_DISABLE_MAPPED_COURIER": _lazy_bool(
        "SGLANG_DIFFUSION_DISABLE_MAPPED_COURIER"
    ),
    # Test hook: make the host memory budget behave as if the machine had this
    # many GiB of RAM (available = this figure minus the process's own
    # anonymous memory). CI uses it to exercise the constrained placement
    # paths -- mapped weights, partial pinning, the courier -- on runners whose
    # real hosts are never short of memory.
    "SGLANG_DIFFUSION_TEST_FORCE_HOST_AVAILABLE_GIB": _lazy_optional_float(
        "SGLANG_DIFFUSION_TEST_FORCE_HOST_AVAILABLE_GIB"
    ),
    # Test-only: cap the CUDA caching allocator at this many GiB, so a large
    # CI card behaves like the consumer card a case is written for. Without
    # the cap the allocator is free to reserve past the pretended budget and
    # a peak-VRAM baseline stops meaning "fits the card".
    "SGLANG_DIFFUSION_TEST_CAP_DEVICE_MEMORY_GIB": _lazy_optional_float(
        "SGLANG_DIFFUSION_TEST_CAP_DEVICE_MEMORY_GIB"
    ),
    # If set, sgl_diffusion will enable stage logging, which will print the time
    # taken for each stage
    "SGLANG_DIFFUSION_STAGE_LOGGING": _lazy_bool("SGLANG_DIFFUSION_STAGE_LOGGING"),
    # Fraction of denoising steps that run both CFG branches before reusing the
    # last conditional-minus-unconditional residual. Keep 1.0 to disable.
    "SGLANG_DIFFUSION_CFG_GATE_STEP": _lazy_float(
        "SGLANG_DIFFUSION_CFG_GATE_STEP", 1.0
    ),
    "SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D": _lazy_str(
        "SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D", "auto"
    ),
    # Kill-switch: keep VAE decoder weights in their checkpoint dtype at load
    # instead of the decode compute dtype the decode stage would round them to
    # on first use anyway.
    "SGLANG_DIFFUSION_DISABLE_EARLY_VAE_DECODER_CAST": _lazy_bool(
        "SGLANG_DIFFUSION_DISABLE_EARLY_VAE_DECODER_CAST"
    ),
    # Kill-switch: keep the decode-dtype VAE decoder weights in anonymous host
    # memory instead of a file-backed cache mapping the page cache can drop.
    "SGLANG_DIFFUSION_DISABLE_VAE_DECODER_STORE": _lazy_bool(
        "SGLANG_DIFFUSION_DISABLE_VAE_DECODER_STORE"
    ),
    # Kill-switch: keep LoRA-merged weights in anonymous host memory instead
    # of the file-backed LoRA merge cache.
    "SGLANG_DIFFUSION_DISABLE_LORA_MERGE_CACHE": _lazy_bool(
        "SGLANG_DIFFUSION_DISABLE_LORA_MERGE_CACHE"
    ),
    # ================== cache-dit Env Vars ==================
    # Enable cache-dit acceleration for DiT inference
    # CUDA-IPC transport for 2-rank Ulysses all-to-all (NVLink same-node)
    "SGLANG_DIFFUSION_IPC_A2A": _lazy_bool("SGLANG_DIFFUSION_IPC_A2A", "true"),
    "SGLANG_DIFFUSION_IPC_A2A_TIMEOUT_MS": _lazy_float(
        "SGLANG_DIFFUSION_IPC_A2A_TIMEOUT_MS", 10000.0
    ),
    "SGLANG_DIFFUSION_IPC_A2A_MAX_BUFFERS": _lazy_int(
        "SGLANG_DIFFUSION_IPC_A2A_MAX_BUFFERS", 16
    ),
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
    "SGLANG_LINGBOT_ENABLE_INTERACTIVE_KV_WINDOW": _lazy_bool(
        "SGLANG_LINGBOT_ENABLE_INTERACTIVE_KV_WINDOW"
    ),
    "SGLANG_LINGBOT_LAZY_VAE_ENCODE_BLACK_FRAMES": _lazy_int(
        "SGLANG_LINGBOT_LAZY_VAE_ENCODE_BLACK_FRAMES"
    ),
    # FlashInfer FP4 GEMM backend override for diffusion NVFP4.
    # When unset, diffusion ModelOpt NVFP4 defaults to flashinfer_trtllm.
    # Supported values:
    # - auto
    # - flashinfer_cudnn
    # - flashinfer_cutlass
    # - flashinfer_trtllm
    # Legacy aliases `cudnn` and `trtllm` are also accepted.
    "SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND": _lazy_str(
        "SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND"
    ),
    # Experimental opt-in for W8A8 FP8 GEMM in diffusion weight-only FP8 linears.
    # When disabled, FP8 weights are dequantized to compute dtype before matmul.
    "SGLANG_DIFFUSION_ENABLE_W8A8_FP8_GEMM": _lazy_bool(
        "SGLANG_DIFFUSION_ENABLE_W8A8_FP8_GEMM"
    ),
    # Dequantize storage-only FP8 linear weights to the compute dtype once,
    # at first use (bit-identical outputs; trades weight VRAM for skipping
    # the per-forward dequant pass). Weights are kept FP8-resident when free
    # memory is low or when this flag is disabled.
    "SGLANG_DIFFUSION_FP8_WEIGHT_DEQUANT_CACHE": _lazy_bool(
        "SGLANG_DIFFUSION_FP8_WEIGHT_DEQUANT_CACHE", "true"
    ),
    # ROCm: use AITer GroupNorm in VAE for improved performance
    "SGLANG_USE_ROCM_VAE": _lazy_bool("SGLANG_USE_ROCM_VAE"),
    # ROCm: enable cudnn.benchmark (MIOpen auto-tuning) for VAE conv layers
    "SGLANG_USE_ROCM_CUDNN_BENCHMARK": _lazy_bool("SGLANG_USE_ROCM_CUDNN_BENCHMARK"),
    # ROCm: replace CausalConv3d with temporal-unfolded batched Conv2D in VAE
    "SGLANG_USE_ROCM_VAE_CONV2D": _lazy_bool("SGLANG_USE_ROCM_VAE_CONV2D"),
    # ROCm: use BF16 compute for the Conv2D replacement (implies CONV2D=true)
    "SGLANG_USE_ROCM_VAE_CONV2D_BF16": _lazy_bool("SGLANG_USE_ROCM_VAE_CONV2D_BF16"),
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
