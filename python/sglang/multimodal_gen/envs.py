# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
import importlib.util

# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/envs.py
import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import diffusers
import torch
from packaging import version

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


def _is_hip():
    has_rocm = torch.version.hip is not None
    return has_rocm


def _is_cuda():
    has_cuda = torch.version.cuda is not None
    return has_cuda


def _is_musa():
    try:
        if hasattr(torch, "musa") and torch.musa.is_available():
            return True
    except ModuleNotFoundError:
        return False


def _is_mps():
    return torch.backends.mps.is_available()


class PackagesEnvChecker:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PackagesEnvChecker, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        self.packages_info = {
            "has_aiter": self.check_aiter(),
            "diffusers_version": self.check_diffusers_version(),
        }

    def check_aiter(self):
        """
        Checks whether ROCm AITER library is installed
        """
        try:

            logger.info("Using AITER as the attention library")
            return True
        except:
            if _is_hip():
                logger.warning(
                    f'Using AMD GPUs, but library "aiter" is not installed, '
                    "defaulting to other attention mechanisms"
                )
            return False

    def check_flash_attn(self):
        if not torch.cuda.is_available():
            return False
        if _is_musa():
            logger.info(
                "Flash Attention library is not supported on MUSA for the moment."
            )
            return False
        try:
            return True
        except ImportError:
            logger.warning(
                f'Flash Attention library "flash_attn" not found, '
                f"using pytorch attention implementation"
            )
            return False

    def check_long_ctx_attn(self):
        if not torch.cuda.is_available():
            return False
        try:
            return importlib.util.find_spec("yunchang") is not None
        except ImportError:
            logger.warning(
                f'Ring Flash Attention library "yunchang" not found, '
                f"using pytorch attention implementation"
            )
            return False

    def check_diffusers_version(self):
        if version.parse(
            version.parse(diffusers.__version__).base_version
        ) < version.parse("0.30.0"):
            raise RuntimeError(
                f"Diffusers version: {version.parse(version.parse(diffusers.__version__).base_version)} is not supported,"
                f"please upgrade to version > 0.30.0"
            )
        return version.parse(version.parse(diffusers.__version__).base_version)

    def get_packages_info(self):
        return self.packages_info


PACKAGES_CHECKER = PackagesEnvChecker()


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
    if value is None:
        return None
    return int(value)


# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

# begin-env-vars-definition

environment_variables: dict[str, Callable[[], Any]] = {
    # ================== Installation Time Env Vars ==================
    # Target device of sglang-diffusion, supporting [cuda (by default),
    # rocm, neuron, cpu, openvino]
    "SGLANG_DIFFUSION_TARGET_DEVICE": lambda: os.getenv(
        "SGLANG_DIFFUSION_TARGET_DEVICE", "cuda"
    ),
    # Maximum number of compilation jobs to run in parallel.
    # By default this is the number of CPUs
    "MAX_JOBS": lambda: os.getenv("MAX_JOBS", None),
    # Number of threads to use for nvcc
    # By default this is 1.
    # If set, `MAX_JOBS` will be reduced to avoid oversubscribing the CPU.
    "NVCC_THREADS": lambda: os.getenv("NVCC_THREADS", None),
    # If set, sgl_diffusion will use precompiled binaries (*.so)
    "SGLANG_DIFFUSION_USE_PRECOMPILED": lambda: bool(
        os.environ.get("SGLANG_DIFFUSION_USE_PRECOMPILED")
    )
    or bool(os.environ.get("SGLANG_DIFFUSION_PRECOMPILED_WHEEL_LOCATION")),
    # CMake build type
    # If not set, defaults to "Debug" or "RelWithDebInfo"
    # Available options: "Debug", "Release", "RelWithDebInfo"
    "CMAKE_BUILD_TYPE": lambda: os.getenv("CMAKE_BUILD_TYPE"),
    # If set, sgl_diffusion will print verbose logs during installation
    "VERBOSE": lambda: bool(int(os.getenv("VERBOSE", "0"))),
    # Root directory for FASTVIDEO configuration files
    # Defaults to `~/.config/sgl_diffusion` unless `XDG_CONFIG_HOME` is set
    # Note that this not only affects how sgl_diffusion finds its configuration files
    # during runtime, but also affects how sgl_diffusion installs its configuration
    # files during **installation**.
    "SGLANG_DIFFUSION_CONFIG_ROOT": lambda: os.path.expanduser(
        os.getenv(
            "SGLANG_DIFFUSION_CONFIG_ROOT",
            os.path.join(get_default_config_root(), "sgl_diffusion"),
        )
    ),
    # ================== Runtime Env Vars ==================
    # Root directory for FASTVIDEO cache files
    # Defaults to `~/.cache/sgl_diffusion` unless `XDG_CACHE_HOME` is set
    "SGLANG_DIFFUSION_CACHE_ROOT": lambda: os.path.expanduser(
        os.getenv(
            "SGLANG_DIFFUSION_CACHE_ROOT",
            os.path.join(get_default_cache_root(), "sgl_diffusion"),
        )
    ),
    # Interval in seconds to log a warning message when the ring buffer is full
    "SGLANG_DIFFUSION_RINGBUFFER_WARNING_INTERVAL": lambda: int(
        os.environ.get("SGLANG_DIFFUSION_RINGBUFFER_WARNING_INTERVAL", "60")
    ),
    # Path to the NCCL library file. It is needed because nccl>=2.19 brought
    # by PyTorch contains a bug: https://github.com/NVIDIA/nccl/issues/1234
    "SGLANG_DIFFUSION_NCCL_SO_PATH": lambda: os.environ.get(
        "SGLANG_DIFFUSION_NCCL_SO_PATH", None
    ),
    # when `SGLANG_DIFFUSION_NCCL_SO_PATH` is not set, sgl_diffusion will try to find the nccl
    # library file in the locations specified by `LD_LIBRARY_PATH`
    "LD_LIBRARY_PATH": lambda: os.environ.get("LD_LIBRARY_PATH", None),
    # Internal flag to enable Dynamo fullgraph capture
    "SGLANG_DIFFUSION_TEST_DYNAMO_FULLGRAPH_CAPTURE": lambda: bool(
        os.environ.get("SGLANG_DIFFUSION_TEST_DYNAMO_FULLGRAPH_CAPTURE", "1") != "0"
    ),
    # local rank of the process in the distributed setting, used to determine
    # the GPU device id
    "LOCAL_RANK": lambda: int(os.environ.get("LOCAL_RANK", "0")),
    # used to control the visible devices in the distributed setting
    "CUDA_VISIBLE_DEVICES": lambda: os.environ.get("CUDA_VISIBLE_DEVICES", None),
    # timeout for each iteration in the engine
    "SGLANG_DIFFUSION_ENGINE_ITERATION_TIMEOUT_S": lambda: int(
        os.environ.get("SGLANG_DIFFUSION_ENGINE_ITERATION_TIMEOUT_S", "60")
    ),
    # Logging configuration
    # If set to 0, sgl_diffusion will not configure logging
    # If set to 1, sgl_diffusion will configure logging using the default configuration
    #    or the configuration file specified by SGLANG_DIFFUSION_LOGGING_CONFIG_PATH
    "SGLANG_DIFFUSION_CONFIGURE_LOGGING": lambda: int(
        os.getenv("SGLANG_DIFFUSION_CONFIGURE_LOGGING", "1")
    ),
    "SGLANG_DIFFUSION_LOGGING_CONFIG_PATH": lambda: os.getenv(
        "SGLANG_DIFFUSION_LOGGING_CONFIG_PATH"
    ),
    # this is used for configuring the default logging level
    "SGLANG_DIFFUSION_LOGGING_LEVEL": lambda: os.getenv(
        "SGLANG_DIFFUSION_LOGGING_LEVEL", "INFO"
    ),
    # if set, SGLANG_DIFFUSION_LOGGING_PREFIX will be prepended to all log messages
    "SGLANG_DIFFUSION_LOGGING_PREFIX": lambda: os.getenv(
        "SGLANG_DIFFUSION_LOGGING_PREFIX", ""
    ),
    # Trace function calls
    # If set to 1, sgl_diffusion will trace function calls
    # Useful for debugging
    "SGLANG_DIFFUSION_TRACE_FUNCTION": lambda: int(
        os.getenv("SGLANG_DIFFUSION_TRACE_FUNCTION", "0")
    ),
    # Path to the attention configuration file. Only used for sliding tile
    # attention for now.
    "SGLANG_DIFFUSION_ATTENTION_CONFIG": lambda: (
        None
        if os.getenv("SGLANG_DIFFUSION_ATTENTION_CONFIG", None) is None
        else os.path.expanduser(os.getenv("SGLANG_DIFFUSION_ATTENTION_CONFIG", "."))
    ),
    # Use dedicated multiprocess context for workers.
    # Both spawn and fork work
    "SGLANG_DIFFUSION_WORKER_MULTIPROC_METHOD": lambda: os.getenv(
        "SGLANG_DIFFUSION_WORKER_MULTIPROC_METHOD", "fork"
    ),
    # Enables torch profiler if set. Path to the directory where torch profiler
    # traces are saved. Note that it must be an absolute path.
    "SGLANG_DIFFUSION_TORCH_PROFILER_DIR": lambda: (
        None
        if os.getenv("SGLANG_DIFFUSION_TORCH_PROFILER_DIR", None) is None
        else os.path.expanduser(os.getenv("SGLANG_DIFFUSION_TORCH_PROFILER_DIR", "."))
    ),
    # If set, sgl_diffusion will run in development mode, which will enable
    # some additional endpoints for developing and debugging,
    # e.g. `/reset_prefix_cache`
    "SGLANG_DIFFUSION_SERVER_DEV_MODE": lambda: get_bool_env_var(
        "SGLANG_DIFFUSION_SERVER_DEV_MODE"
    ),
    # If set, sgl_diffusion will enable stage logging, which will print the time
    # taken for each stage
    "SGLANG_DIFFUSION_STAGE_LOGGING": lambda: get_bool_env_var(
        "SGLANG_DIFFUSION_STAGE_LOGGING"
    ),
    # ================== cache-dit Env Vars ==================
    # Enable cache-dit acceleration for DiT inference
    "SGLANG_CACHE_DIT_ENABLED": lambda: get_bool_env_var("SGLANG_CACHE_DIT_ENABLED"),
    # Number of first blocks to always compute (DBCache F parameter)
    "SGLANG_CACHE_DIT_FN": lambda: int(os.getenv("SGLANG_CACHE_DIT_FN", "1")),
    # Number of last blocks to always compute (DBCache B parameter)
    "SGLANG_CACHE_DIT_BN": lambda: int(os.getenv("SGLANG_CACHE_DIT_BN", "0")),
    # Warmup steps before caching (DBCache W parameter)
    "SGLANG_CACHE_DIT_WARMUP": lambda: int(os.getenv("SGLANG_CACHE_DIT_WARMUP", "4")),
    # Residual difference threshold (DBCache R parameter)
    "SGLANG_CACHE_DIT_RDT": lambda: float(os.getenv("SGLANG_CACHE_DIT_RDT", "0.24")),
    # Maximum continuous cached steps (DBCache MC parameter)
    "SGLANG_CACHE_DIT_MC": lambda: int(os.getenv("SGLANG_CACHE_DIT_MC", "3")),
    # Enable TaylorSeer calibrator
    "SGLANG_CACHE_DIT_TAYLORSEER": lambda: get_bool_env_var(
        "SGLANG_CACHE_DIT_TAYLORSEER", default="false"
    ),
    # TaylorSeer order (1 or 2)
    "SGLANG_CACHE_DIT_TS_ORDER": lambda: int(
        os.getenv("SGLANG_CACHE_DIT_TS_ORDER", "1")
    ),
    # SCM preset: none, slow, medium, fast, ultra
    "SGLANG_CACHE_DIT_SCM_PRESET": lambda: os.getenv(
        "SGLANG_CACHE_DIT_SCM_PRESET", "none"
    ),
    # SCM custom compute bins (e.g., "8,3,3,2,2")
    "SGLANG_CACHE_DIT_SCM_COMPUTE_BINS": lambda: os.getenv(
        "SGLANG_CACHE_DIT_SCM_COMPUTE_BINS", None
    ),
    # SCM custom cache bins (e.g., "1,2,2,2,3")
    "SGLANG_CACHE_DIT_SCM_CACHE_BINS": lambda: os.getenv(
        "SGLANG_CACHE_DIT_SCM_CACHE_BINS", None
    ),
    # SCM policy: dynamic or static
    "SGLANG_CACHE_DIT_SCM_POLICY": lambda: os.getenv(
        "SGLANG_CACHE_DIT_SCM_POLICY", "dynamic"
    ),
    # ================== cache-dit Secondary Transformer Env Vars ==================
    # For dual-transformer models like Wan2.2 (high-noise + low-noise experts)
    # These parameters configure the secondary transformer (transformer_2)
    # If not set, they inherit from the primary transformer settings
    # Number of first blocks to always compute for secondary transformer
    "SGLANG_CACHE_DIT_SECONDARY_FN": lambda: int(
        os.getenv(
            "SGLANG_CACHE_DIT_SECONDARY_FN", os.getenv("SGLANG_CACHE_DIT_FN", "1")
        )
    ),
    # Number of last blocks to always compute for secondary transformer
    "SGLANG_CACHE_DIT_SECONDARY_BN": lambda: int(
        os.getenv(
            "SGLANG_CACHE_DIT_SECONDARY_BN", os.getenv("SGLANG_CACHE_DIT_BN", "0")
        )
    ),
    # Warmup steps before caching for secondary transformer
    "SGLANG_CACHE_DIT_SECONDARY_WARMUP": lambda: int(
        os.getenv(
            "SGLANG_CACHE_DIT_SECONDARY_WARMUP",
            os.getenv("SGLANG_CACHE_DIT_WARMUP", "4"),
        )
    ),
    # Residual difference threshold for secondary transformer
    "SGLANG_CACHE_DIT_SECONDARY_RDT": lambda: float(
        os.getenv(
            "SGLANG_CACHE_DIT_SECONDARY_RDT", os.getenv("SGLANG_CACHE_DIT_RDT", "0.24")
        )
    ),
    # Maximum continuous cached steps for secondary transformer
    "SGLANG_CACHE_DIT_SECONDARY_MC": lambda: int(
        os.getenv(
            "SGLANG_CACHE_DIT_SECONDARY_MC", os.getenv("SGLANG_CACHE_DIT_MC", "3")
        )
    ),
    # Enable TaylorSeer for secondary transformer
    "SGLANG_CACHE_DIT_SECONDARY_TAYLORSEER": lambda: get_bool_env_var(
        "SGLANG_CACHE_DIT_SECONDARY_TAYLORSEER",
        default=os.getenv("SGLANG_CACHE_DIT_TAYLORSEER", "false"),
    ),
    # TaylorSeer order for secondary transformer
    "SGLANG_CACHE_DIT_SECONDARY_TS_ORDER": lambda: int(
        os.getenv(
            "SGLANG_CACHE_DIT_SECONDARY_TS_ORDER",
            os.getenv("SGLANG_CACHE_DIT_TS_ORDER", "1"),
        )
    ),
}


# end-env-vars-definition


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())


def get_torch_distributed_backend() -> str:
    if torch.cuda.is_available():
        return "nccl"
    elif _is_musa():
        return "mccl"
    elif _is_mps():
        return "gloo"
    else:
        raise NotImplementedError(
            "No Accelerators(AMD/NV/MTT GPU, AMD MI instinct accelerators) available"
        )


def get_device(local_rank: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", local_rank)
    elif _is_musa():
        return torch.device("musa", local_rank)
    elif _is_mps():
        return torch.device("mps")
    else:
        return torch.device("cpu")
