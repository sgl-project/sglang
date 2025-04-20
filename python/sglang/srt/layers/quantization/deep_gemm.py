import logging
import os
from contextlib import contextmanager
from enum import IntEnum, auto
from typing import Tuple

import torch
from tqdm.contrib.concurrent import thread_map

from sglang.srt.utils import get_bool_env_var, get_device_sm, get_int_env_var, is_cuda

ENABLE_JIT_DEEPGEMM = False
ENABLE_JIT_DEEPGEMM_BMM = False
ENABLE_JIT_DEEPGEMM_PRECOMPILE = get_bool_env_var("SGL_JIT_DEEPGEMM_PRECOMPILE", "true")
ENABLE_JIT_DEEPGEMM_LOG_BUILD = get_bool_env_var(
    "SGL_JIT_DEEPGEMM_LOG_BUILD", "false" if ENABLE_JIT_DEEPGEMM_PRECOMPILE else "true"
)
COMPILE_WORKERS = get_int_env_var("SGL_JIT_DEEPGEMM_COMPILE_WORKERS", 4)

if is_cuda():
    import deep_gemm
    from deep_gemm import get_num_sms
    from deep_gemm.jit_kernels.gemm import get_best_configs
    from deep_gemm.jit_kernels.gemm import includes as deep_gemm_includes
    from deep_gemm.jit_kernels.gemm import template as deep_gemm_gemm_template
    from deep_gemm.jit_kernels.m_grouped_gemm import (
        template as deep_gemm_grouped_gemm_template,
    )
    from deep_gemm.jit_kernels.tuner import jit_tuner

    sm_version = get_device_sm()
    if sm_version == 90:
        if get_bool_env_var("SGL_ENABLE_JIT_DEEPGEMM", default="false"):
            ENABLE_JIT_DEEPGEMM = True
        if get_bool_env_var("SGL_ENABLE_JIT_DEEPGEMM_BMM", default="false"):
            ENABLE_JIT_DEEPGEMM_BMM = True

logger = logging.getLogger(__name__)

INITIALIZATION_DICT = {}
PRE_COMPILE_M_LIST = list(range(1, 1024 * 64 + 1))

# Force redirect deep_gemm cache_dir
os.environ["DG_CACHE_DIR"] = os.getenv(
    "SGL_DG_CACHE_DIR", os.path.expanduser("~") + "/.cache/deep_gemm"
)


class DeepGemmKernelType(IntEnum):
    GROUPED_GEMM_NT_F8F8BF16_MASKED = auto()
    GROUPED_GEMM_NT_F8F8BF16_CONTIG = auto()
    GEMM_NT_F8F8BF16 = auto()


FUNCTION_NAME_DICT = {
    DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED: "m_grouped_gemm_fp8_fp8_bf16_nt_masked",
    DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG: "m_grouped_gemm_fp8_fp8_bf16_nt_contiguous",
    DeepGemmKernelType.GEMM_NT_F8F8BF16: "gemm_fp8_fp8_bf16_nt",
}


def compile_grouped_gemm_nt_f8f8bf16_masked_one(n, k, num_groups, config) -> None:
    # Auto-tuning with compilation
    global deep_gemm_includes, deep_gemm_grouped_gemm_template
    _, block_m, block_n, num_stages, tma_multicast_config, smem_config = config
    _ = jit_tuner.compile_and_tune(
        name="m_grouped_gemm_fp8_fp8_bf16_nt",
        keys={
            "N": n,
            "K": k,
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "SWIZZLE_D_MODE": smem_config[1],
            "BLOCK_N_PADDING": smem_config[2],
            "NUM_GROUPS": num_groups,
            "NUM_STAGES": num_stages,
            "NUM_TMA_MULTICAST": tma_multicast_config[0],
            "IS_TMA_MULTICAST_ON_A": tma_multicast_config[1],
            "GEMM_TYPE": "GroupedMasked",
        },
        space=(),
        includes=deep_gemm_includes,
        arg_defs=(
            ("lhs", torch.float8_e4m3fn),
            ("lhs_scales", torch.float),
            ("rhs", torch.float8_e4m3fn),
            ("rhs_scales", torch.float),
            ("out", torch.bfloat16),
            ("grouped_layout", torch.int32),
            ("m", int),
            ("stream", torch.cuda.Stream),
            ("num_sms", int),
            ("smem_size", int),
        ),
        template=deep_gemm_grouped_gemm_template,
        args=[],
    )


def compile_grouped_gemm_nt_f8f8bf16_contig_one(n, k, num_groups, config) -> None:
    global deep_gemm_includes, deep_gemm_grouped_gemm_template
    _, block_m, block_n, num_stages, tma_multicast_config, smem_config = config
    _ = jit_tuner.compile_and_tune(
        name="m_grouped_gemm_fp8_fp8_bf16_nt",
        keys={
            "N": n,
            "K": k,
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "SWIZZLE_D_MODE": smem_config[1],
            "BLOCK_N_PADDING": smem_config[2],
            "NUM_GROUPS": num_groups,
            "NUM_STAGES": num_stages,
            "NUM_TMA_MULTICAST": tma_multicast_config[0],
            "IS_TMA_MULTICAST_ON_A": tma_multicast_config[1],
            "GEMM_TYPE": "GroupedContiguous",
        },
        space=(),
        includes=deep_gemm_includes,
        arg_defs=(
            ("lhs", torch.float8_e4m3fn),
            ("lhs_scales", torch.float),
            ("rhs", torch.float8_e4m3fn),
            ("rhs_scales", torch.float),
            ("out", torch.bfloat16),
            ("grouped_layout", torch.int32),
            ("m", int),
            ("num_groups", int),
            ("stream", torch.cuda.Stream),
            ("num_sms", int),
            ("smem_size", int),
        ),
        template=deep_gemm_grouped_gemm_template,
        args=[],
    )


def compile_gemm_nt_f8f8bf16_one(n, k, config) -> None:
    global deep_gemm_includes, deep_gemm_gemm_template
    _, block_m, block_n, num_stages, tma_multicast_config, smem_config = config
    _ = jit_tuner.compile_and_tune(
        name="gemm_fp8_fp8_bf16_nt",
        keys={
            "N": n,
            "K": k,
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "SWIZZLE_D_MODE": smem_config[1],
            "BLOCK_N_PADDING": smem_config[2],
            "NUM_STAGES": num_stages,
            "NUM_TMA_MULTICAST": tma_multicast_config[0],
            "IS_TMA_MULTICAST_ON_A": tma_multicast_config[1],
        },
        space=(),
        includes=deep_gemm_includes,
        arg_defs=(
            ("lhs", torch.float8_e4m3fn),
            ("lhs_scales", torch.float),
            ("rhs", torch.float8_e4m3fn),
            ("rhs_scales", torch.float),
            ("out", torch.bfloat16),
            ("m", int),
            ("stream", torch.cuda.Stream),
            ("num_sms", int),
            ("smem_size", int),
        ),
        template=deep_gemm_gemm_template,
        args=[],
    )


def compile_grouped_gemm_nt_f8f8bf16_masked_all(n, k, num_groups) -> None:
    global PRE_COMPILE_M_LIST
    logger.info(
        f"DeepGEMM JIT Compiling for <m_grouped_gemm_fp8_fp8_bf16_nt_masked> N={n}, K={k}, num_groups={num_groups} with all Ms. Please wait."
    )
    num_sms = get_num_sms()
    collected_configs = set()
    for expected_m in PRE_COMPILE_M_LIST:
        collected_configs.add(
            get_best_configs(
                expected_m, n, k, num_groups, num_sms, is_grouped_masked=True
            )
        )
    compile_func = lambda config: compile_grouped_gemm_nt_f8f8bf16_masked_one(
        n, k, num_groups, config
    )
    thread_map(compile_func, collected_configs, max_workers=COMPILE_WORKERS)


def compile_grouped_gemm_nt_f8f8bf16_contig_all(n, k, num_groups) -> None:
    global PRE_COMPILE_M_LIST
    logger.info(
        f"DeepGEMM JIT Compiling for <m_grouped_gemm_fp8_fp8_bf16_nt_contiguous> for N={n}, K={k}, num_groups={num_groups} with all Ms. Please wait."
    )
    num_sms = get_num_sms()
    collected_configs = set()
    for m in PRE_COMPILE_M_LIST:
        collected_configs.add(
            get_best_configs(m, n, k, 1, num_sms, is_grouped_contiguous=True)
        )
    compile_func = lambda config: compile_grouped_gemm_nt_f8f8bf16_contig_one(
        n, k, num_groups, config
    )
    thread_map(compile_func, collected_configs, max_workers=COMPILE_WORKERS)


def compile_gemm_nt_f8f8bf16_all(n, k) -> None:
    global PRE_COMPILE_M_LIST
    logger.info(
        f"DeepGEMM JIT Compiling for <gemm_fp8_fp8_bf16_nt> N={n}, K={k} with all Ms. Please wait."
    )
    num_sms = get_num_sms()
    collected_configs = set()
    for m in PRE_COMPILE_M_LIST:
        collected_configs.add(get_best_configs(m, n, k, 1, num_sms))
    compile_func = lambda config: compile_gemm_nt_f8f8bf16_one(n, k, config)
    thread_map(compile_func, collected_configs, max_workers=COMPILE_WORKERS)


def grouped_gemm_nt_f8f8bf16_masked(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
):

    global INITIALIZATION_DICT
    num_groups, _, k = lhs[0].shape
    _, n, _ = rhs[0].shape

    query_key = (DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED, n, k, num_groups)
    if ENABLE_JIT_DEEPGEMM_PRECOMPILE and INITIALIZATION_DICT.get(query_key) is None:
        compile_grouped_gemm_nt_f8f8bf16_masked_all(n, k, num_groups)
        INITIALIZATION_DICT[query_key] = True

    with _log_jit_build(expected_m, n, k, query_key[0]):
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(
            lhs, rhs, out, masked_m, expected_m
        )


def grouped_gemm_nt_f8f8bf16_contig(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    m_indices: torch.Tensor,
):

    global INITIALIZATION_DICT
    m, k = lhs[0].shape
    num_groups, n, _ = rhs[0].shape

    query_key = (DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG, n, k, num_groups)
    if ENABLE_JIT_DEEPGEMM_PRECOMPILE and INITIALIZATION_DICT.get(query_key) is None:
        compile_grouped_gemm_nt_f8f8bf16_contig_all(n, k, num_groups)
        INITIALIZATION_DICT[query_key] = True

    with _log_jit_build(m, n, k, query_key[0]):
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(lhs, rhs, out, m_indices)


def gemm_nt_f8f8bf16(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
):

    global INITIALIZATION_DICT
    m, k = lhs[0].shape
    n, _ = rhs[0].shape

    query_key = (DeepGemmKernelType.GEMM_NT_F8F8BF16, n, k, 1)
    if ENABLE_JIT_DEEPGEMM_PRECOMPILE and INITIALIZATION_DICT.get(query_key) is None:
        compile_gemm_nt_f8f8bf16_all(n, k)
        INITIALIZATION_DICT[query_key] = True

    with _log_jit_build(m, n, k, query_key[0]):
        deep_gemm.gemm_fp8_fp8_bf16_nt(lhs, rhs, out)


@contextmanager
def _log_jit_build(M: int, N: int, K: int, func_type: DeepGemmKernelType):
    if not ENABLE_JIT_DEEPGEMM_LOG_BUILD:
        yield
        return

    from deep_gemm.jit.runtime import RuntimeCache

    origin_func = RuntimeCache.__getitem__

    def __patched_func(self, *args, **kwargs):
        ret = origin_func(self, *args, **kwargs)
        if ret is None:
            func_name = FUNCTION_NAME_DICT[func_type]
            logger.warning(
                f"DeepGEMM JIT Compiling for <{func_name}> M={M}, N={N}, K={K}. Please wait."
            )
        return ret

    RuntimeCache.__getitem__ = __patched_func
    yield
    RuntimeCache.__getitem__ = origin_func
