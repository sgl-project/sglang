import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Callable, Dict, List, Optional, Tuple

from tqdm.contrib.concurrent import thread_map

from sglang.srt.layers.quantization.deep_gemm_wrapper.configurer import (
    DEEPGEMM_BLACKWELL,
    ENABLE_JIT_DEEPGEMM,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_bool_env_var, get_int_env_var

logger = logging.getLogger(__name__)

if ENABLE_JIT_DEEPGEMM and not DEEPGEMM_BLACKWELL:
    from deep_gemm import get_num_sms
    from deep_gemm.jit import build
    from deep_gemm.jit_kernels.gemm import get_best_configs
    from deep_gemm.jit_kernels.runtime import FP8GemmRuntime, GemmType


_BUILTIN_M_LIST = list(range(1, 1024 * 16 + 1))
_ENABLE_JIT_DEEPGEMM_PRECOMPILE = get_bool_env_var(
    "SGL_JIT_DEEPGEMM_PRECOMPILE", "true"
)
_DO_COMPILE_ALL = True
_IS_FIRST_RANK_ON_NODE = get_bool_env_var("SGL_IS_FIRST_RANK_ON_NODE", "true")
_COMPILE_WORKERS = get_int_env_var("SGL_JIT_DEEPGEMM_COMPILE_WORKERS", 4)
_IN_PRECOMPILE_STAGE = get_bool_env_var("SGL_IN_DEEPGEMM_PRECOMPILE_STAGE", "false")

# Force redirect deep_gemm cache_dir
os.environ["DG_JIT_CACHE_DIR"] = os.getenv(
    "SGL_DG_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "deep_gemm")
)

# Refer to https://github.com/deepseek-ai/DeepGEMM/commit/d75b218b7b8f4a5dd5406ac87905039ead3ae42f
# NVRTC may have performance loss with some cases.
# And NVCC JIT speed is also 9x faster in the ref commit
os.environ["DG_JIT_USE_NVRTC"] = os.getenv("SGL_DG_USE_NVRTC", "0")


def update_deep_gemm_config(gpu_id: int, server_args: ServerArgs):
    global _BUILTIN_M_LIST
    global _DO_COMPILE_ALL
    global _IS_FIRST_RANK_ON_NODE

    # Generate m_max
    m_max = 1024 * 16
    if server_args.chunked_prefill_size < 1:
        m_max = 1024 * 64
    elif server_args.chunked_prefill_size > 8192:
        m_max = server_args.chunked_prefill_size * 2
    m_max = min(1024 * 128, m_max)
    _BUILTIN_M_LIST = list(range(1, m_max + 1))

    _IS_FIRST_RANK_ON_NODE = ServerArgs.base_gpu_id == gpu_id

    # Check if is the first rank on node.
    # Default each rank will try compile all Ms to
    # load all symbols at the launch stages.
    # Avoid loading symbols at the serving stages.
    _DO_COMPILE_ALL = _IS_FIRST_RANK_ON_NODE or not _IN_PRECOMPILE_STAGE


class DeepGemmKernelType(IntEnum):
    GROUPED_GEMM_NT_F8F8BF16_MASKED = auto()
    GROUPED_GEMM_NT_F8F8BF16_CONTIG = auto()
    GEMM_NT_F8F8BF16 = auto()


@dataclass
class DeepGemmKernelHelper:
    name: str
    compile_func: Callable[
        [
            int,
            int,
            int,
            Tuple[int, int, int, int, Tuple[int, bool], Tuple[int, int, int]],
        ],
        None,
    ]
    configure_func: Callable[
        [int, int, int, int, int],
        Tuple[int, int, int, int, Tuple[int, bool], Tuple[int, int, int]],
    ]


_INITIALIZATION_DICT: Dict[Tuple[DeepGemmKernelType, int, int, int], bool] = dict()


# TODO improve naming
def _compile_warning_1():
    if not _IN_PRECOMPILE_STAGE and _IS_FIRST_RANK_ON_NODE:
        logger.warning(
            "Entering DeepGEMM JIT Pre-Compile session. "
            "It may takes a long time (typically 10-20 mins) "
            "if you have not run `sglang.compile_deep_gemm`. "
            "It is recommended to run `sglang.compile_deep_gemm` with same args as `sglang.launch_server`"
            " for pre-compilation to reduce the overhead if you have not run it before. "
            "For example: "
            "`python3 -m sglang.compile_deep_gemm --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code`"
        )


def _maybe_compile_deep_gemm_one_type_all(
    kernel_type: DeepGemmKernelType,
    n: int,
    k: int,
    num_groups: int,
    m_list: Optional[List[int]] = None,
) -> None:
    global _INITIALIZATION_DICT
    global _BUILTIN_M_LIST

    query_key = (kernel_type, n, k, num_groups)
    if (
        _ENABLE_JIT_DEEPGEMM_PRECOMPILE
        and _DO_COMPILE_ALL
        and _INITIALIZATION_DICT.get(query_key) is None
    ):
        _INITIALIZATION_DICT[query_key] = True

        _compile_warning_1()
        logger.info(
            f"Try DeepGEMM JIT Compiling for "
            f"<{kernel_type.name}> N={n}, K={k}, num_groups={num_groups} with all Ms."
            f"{' It only takes a little time (typically 1 sec) if you have run `python3 -m sglang.compile_deep_gemm`. ' if not _IN_PRECOMPILE_STAGE else ''}"
        )

        # NOTE(alcanderian): get_num_sms should be change when 2-batch-overlap is introduced
        for m in m_list if m_list is not None else _BUILTIN_M_LIST:
            # TODO can use multi thread
            _compile_one_deepgemm()


@contextmanager
def deep_gemm_execution_hook(
    m: int, n: int, k: int, num_groups: int, kernel_type: DeepGemmKernelType
):
    _maybe_compile_deep_gemm_one_type_all(kernel_type, n, k, num_groups)
    yield
