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
_USE_NVRTC_DEFAULT = "0"
if ENABLE_JIT_DEEPGEMM:
    try:
        from deep_gemm.jit.compiler import get_nvcc_compiler

        get_nvcc_compiler()
    except:
        logger.warning(
            "NVCC Compiler not found, use NVRTC for DeepGEMM JIT "
            "and may have performance loss with some cases."
        )
        _USE_NVRTC_DEFAULT = "1"
os.environ["DG_JIT_USE_NVRTC"] = os.getenv("SGL_DG_USE_NVRTC", _USE_NVRTC_DEFAULT)


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


# TODO improve naming
def _compile_warning_2():
    logger.warning(
        "Entering DeepGEMM JIT Single Kernel Compile session. "
        "And it will makes inference throughput becomes flaky. "
        "Please run `sglang.compile_deep_gemm` with same args as `sglang.launch_server`"
        " for pre-compilation to solve this issue. "
        "For example: "
        "`python3 -m sglang.compile_deep_gemm --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code`"
    )


def _compile_grouped_gemm_nt_f8f8bf16_masked_one(
    n: int,
    k: int,
    num_groups: int,
    config: Tuple[int, int, int, int, Tuple[int, bool], Tuple[int, int, int]],
) -> None:
    num_sms, block_m, block_n, num_stages, tma_multicast_config, smem_config = config
    block_k = 128
    num_tma_threads = 128
    num_math_threads_per_group = 128

    kwargs = {
        "GEMM_TYPE": GemmType.GroupedMasked,
        "NUM_TMA_THREADS": num_tma_threads,
        "NUM_MATH_THREADS_PER_GROUP": num_math_threads_per_group,
        "N": n,
        "K": k,
        "NUM_GROUPS": 1,
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "SWIZZLE_D_MODE": smem_config[1],
        "BLOCK_N_PADDING": smem_config[2],
        "NUM_STAGES": num_stages,
        "NUM_TMA_MULTICAST": tma_multicast_config[0],
        "IS_TMA_MULTICAST_ON_A": tma_multicast_config[1],
        "NUM_SMS": num_sms,
        "SMEM_SIZE": smem_config[0],
    }

    code = FP8GemmRuntime.generate(kwargs)
    _ = build("m_grouped_gemm_fp8_fp8_bf16_nt", code, FP8GemmRuntime, kwargs)


def _compile_grouped_gemm_nt_f8f8bf16_contig_one(
    n: int,
    k: int,
    num_groups: int,
    config: Tuple[int, int, int, int, Tuple[int, bool], Tuple[int, int, int]],
) -> None:
    num_sms, block_m, block_n, num_stages, tma_multicast_config, smem_config = config
    block_k = 128
    num_tma_threads = 128
    num_math_threads_per_group = 128
    kwargs = {
        "GEMM_TYPE": GemmType.GroupedContiguous,
        "NUM_TMA_THREADS": num_tma_threads,
        "NUM_MATH_THREADS_PER_GROUP": num_math_threads_per_group,
        "N": n,
        "K": k,
        "NUM_GROUPS": 1,
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "SWIZZLE_D_MODE": smem_config[1],
        "BLOCK_N_PADDING": smem_config[2],
        "NUM_STAGES": num_stages,
        "NUM_TMA_MULTICAST": tma_multicast_config[0],
        "IS_TMA_MULTICAST_ON_A": tma_multicast_config[1],
        "NUM_SMS": num_sms,
        "SMEM_SIZE": smem_config[0],
    }

    code = FP8GemmRuntime.generate(kwargs)
    _ = build("m_grouped_gemm_fp8_fp8_bf16_nt", code, FP8GemmRuntime, kwargs)


def _compile_gemm_nt_f8f8bf16_one(
    n: int,
    k: int,
    _: int,  # _ is a dummy parameter to align with other interfaces
    config: Tuple[int, int, int, int, Tuple[int, bool], Tuple[int, int, int]],
) -> None:
    num_sms, block_m, block_n, num_stages, tma_multicast_config, smem_config = config
    block_k = 128
    num_tma_threads = 128
    num_math_threads_per_group = 128
    kwargs = {
        "GEMM_TYPE": GemmType.Normal,
        "NUM_TMA_THREADS": num_tma_threads,
        "NUM_MATH_THREADS_PER_GROUP": num_math_threads_per_group,
        "N": n,
        "K": k,
        "NUM_GROUPS": 1,
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "SWIZZLE_D_MODE": smem_config[1],
        "BLOCK_N_PADDING": smem_config[2],
        "NUM_STAGES": num_stages,
        "NUM_TMA_MULTICAST": tma_multicast_config[0],
        "IS_TMA_MULTICAST_ON_A": tma_multicast_config[1],
        "NUM_SMS": num_sms,
        "SMEM_SIZE": smem_config[0],
    }

    code = FP8GemmRuntime.generate(kwargs)
    _ = build("gemm_fp8_fp8_bf16_nt", code, FP8GemmRuntime, kwargs)


# TODO further refactor warmup-related
_KERNEL_HELPER_DICT: Dict[DeepGemmKernelType, DeepGemmKernelHelper] = {
    DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED: DeepGemmKernelHelper(
        name="m_grouped_gemm_fp8_fp8_bf16_nt_masked",
        compile_func=_compile_grouped_gemm_nt_f8f8bf16_masked_one,
        configure_func=lambda m, n, k, num_groups, num_sms: get_best_configs(
            m, n, k, num_groups, num_sms, is_grouped_masked=True
        ),
    ),
    DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG: DeepGemmKernelHelper(
        name="m_grouped_gemm_fp8_fp8_bf16_nt_contiguous",
        compile_func=_compile_grouped_gemm_nt_f8f8bf16_contig_one,
        configure_func=lambda m, n, k, _, num_sms: get_best_configs(
            m, n, k, 1, num_sms, is_grouped_contiguous=True
        ),
    ),
    DeepGemmKernelType.GEMM_NT_F8F8BF16: DeepGemmKernelHelper(
        name="gemm_fp8_fp8_bf16_nt",
        compile_func=_compile_gemm_nt_f8f8bf16_one,
        configure_func=lambda m, n, k, _, num_sms: get_best_configs(
            m, n, k, 1, num_sms
        ),
    ),
}


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

        kernel_helper = _KERNEL_HELPER_DICT[kernel_type]
        _compile_warning_1()
        logger.info(
            f"Try DeepGEMM JIT Compiling for "
            f"<{kernel_helper.name}> N={n}, K={k}, num_groups={num_groups} with all Ms."
            f"{' It only takes a little time (typically 1 sec) if you have run `python3 -m sglang.compile_deep_gemm`. ' if not _IN_PRECOMPILE_STAGE else ''}"
        )

        # NOTE(alcanderian): get_num_sms should be change when 2-batch-overlap is introduced
        num_sms = get_num_sms()
        collected_configs = set()
        for m in m_list if m_list is not None else _BUILTIN_M_LIST:
            # Put config into set to get unique configs and reduce cases to be compiled
            collected_configs.add(
                kernel_helper.configure_func(m, n, k, num_groups, num_sms)
            )
        compile_func = lambda config: kernel_helper.compile_func(
            n, k, num_groups, config
        )
        thread_map(compile_func, collected_configs, max_workers=_COMPILE_WORKERS)


@contextmanager
def _log_jit_build(M: int, N: int, K: int, kernel_type: DeepGemmKernelType):
    if _IN_PRECOMPILE_STAGE:
        yield
        return

    from deep_gemm.jit.runtime import RuntimeCache

    origin_func = RuntimeCache.get

    def __patched_func(self, *args, **kwargs):
        ret = origin_func(self, *args, **kwargs)
        if ret is None:
            kernel_helper = _KERNEL_HELPER_DICT[kernel_type]
            if not DEEPGEMM_BLACKWELL:
                _compile_warning_2()
            logger.warning(
                f"DeepGEMM JIT Compiling for <{kernel_helper.name}> M={M}, N={N}, K={K}. Please wait."
            )
        return ret

    RuntimeCache.get = __patched_func
    yield
    RuntimeCache.get = origin_func


@contextmanager
def deep_gemm_execution_hook(
    m: int, n: int, k: int, num_groups: int, kernel_type: DeepGemmKernelType
):
    # not supported yet
    if not DEEPGEMM_BLACKWELL:
        _maybe_compile_deep_gemm_one_type_all(kernel_type, n, k, num_groups)

    with _log_jit_build(m, n, k, kernel_type):
        yield
