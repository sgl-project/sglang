import logging
import os
import time
from contextlib import contextmanager
from enum import IntEnum, auto
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    disable_symmetric_memory_context,
    restore_symmetric_memory_context,
)
from sglang.srt.environ import envs
from sglang.srt.layers.asym_gemm_wrapper.configurer import ENABLE_JIT_ASYMGEMM
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import ceil_div, get_available_gpu_memory

logger = logging.getLogger(__name__)

if ENABLE_JIT_ASYMGEMM:
    import asym_gemm

_BUILTIN_M_LIST = list(range(1, 1024 * 16 + 1))
_ENABLE_JIT_ASYMGEMM_PRECOMPILE = envs.SGLANG_JIT_ASYMGEMM_PRECOMPILE.get()
_DO_COMPILE_ALL = True
_IS_FIRST_RANK_ON_NODE = envs.SGLANG_IS_FIRST_RANK_ON_NODE.get()
_IN_PRECOMPILE_STAGE = envs.SGLANG_IN_ASYMGEMM_PRECOMPILE_STAGE.get()
_FAST_WARMUP = envs.SGLANG_JIT_ASYMGEMM_FAST_WARMUP.get()

# Force redirect asym_gemm cache_dir
os.environ["AG_JIT_CACHE_DIR"] = os.getenv(
    "SGLANG_AG_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "asym_gemm"),
)


def update_asym_gemm_config(gpu_id: int, server_args: ServerArgs):
    global _BUILTIN_M_LIST
    global _DO_COMPILE_ALL
    global _IS_FIRST_RANK_ON_NODE

    _BUILTIN_M_LIST = []

    if _FAST_WARMUP:
        _BUILTIN_M_LIST += list(range(1, 1025))

        next_m, sample_step = 1024, 2
        max_prefill_bs = (
            min(server_args.chunked_prefill_size, 32 * 1024)
            if server_args.chunked_prefill_size >= 1
            else 16 * 1024
        )
        while next_m < max_prefill_bs:
            _BUILTIN_M_LIST += list(range(next_m, 2 * next_m, sample_step))
            next_m = next_m * 2
            sample_step = sample_step * 2
        _BUILTIN_M_LIST.append(max_prefill_bs)
        _BUILTIN_M_LIST = sorted(list(set(_BUILTIN_M_LIST)))
    else:
        m_max = 1024 * 16
        if server_args.chunked_prefill_size < 1:
            m_max = 1024 * 64
        elif server_args.chunked_prefill_size > 8192:
            m_max = server_args.chunked_prefill_size * 2
        m_max = min(1024 * 128, m_max)
        _BUILTIN_M_LIST += list(range(1, m_max + 1))

    _IS_FIRST_RANK_ON_NODE = server_args.base_gpu_id == gpu_id
    _DO_COMPILE_ALL = _IS_FIRST_RANK_ON_NODE


class AsymGemmKernelType(IntEnum):
    GROUPED_GEMM_NT_F8F8BF16_MASKED = auto()
    GROUPED_GEMM_NT_F8F8BF16_CONTIG = auto()
    GROUPED_GEMM_NT_BF16_MASKED = auto()
    GROUPED_GEMM_NT_BF16_CONTIG = auto()


_INITIALIZATION_DICT: Dict[Tuple[AsymGemmKernelType, int, int, int], bool] = dict()


def _maybe_compile_asym_gemm_one_type_all(
    kernel_type: AsymGemmKernelType,
    n: int,
    k: int,
    num_groups: int,
) -> None:
    global _INITIALIZATION_DICT
    global _BUILTIN_M_LIST

    query_key = (kernel_type, n, k, num_groups)
    if (
        _ENABLE_JIT_ASYMGEMM_PRECOMPILE
        and _DO_COMPILE_ALL
        and _INITIALIZATION_DICT.get(query_key) is None
    ):
        _INITIALIZATION_DICT[query_key] = True

        if not _IN_PRECOMPILE_STAGE and _IS_FIRST_RANK_ON_NODE:
            logger.warning(
                "Entering AsymGEMM JIT Pre-Compile session. "
                "It may take a long time (typically 10-20 mins) "
                "if you have not pre-compiled."
            )

        logger.info(
            f"Try AsymGEMM JIT Compiling for "
            f"<{kernel_type.name}> N={n}, K={k}, num_groups={num_groups} with all Ms."
        )

        _compile_asym_gemm_one_type_all(
            kernel_type=kernel_type,
            n=n,
            k=k,
            num_groups=num_groups,
            m_list=_BUILTIN_M_LIST,
        )


def _compile_asym_gemm_one_type_all(
    kernel_type: AsymGemmKernelType,
    n: int,
    k: int,
    num_groups: int,
    m_list: List[int],
) -> None:
    saved_context = disable_symmetric_memory_context()
    try:
        if kernel_type in (
            AsymGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG,
            AsymGemmKernelType.GROUPED_GEMM_NT_BF16_CONTIG,
        ):
            m_alignment = asym_gemm.get_mk_alignment_for_contiguous_layout()
            m_list = sorted(list(set(m for m in m_list if m % m_alignment == 0)))

        memory_budget = get_available_gpu_memory(device="cuda", gpu_id=0)

        max_m = max(m_list)
        required_memory = _BaseWarmupExecutor.get_memory_requirement(
            kernel_type, max_m=max_m, n=n, k=k, num_groups=num_groups
        )
        logger.info(
            f"Required memory for warmup: {required_memory}GB, Available memory: {memory_budget}GB"
        )
        if memory_budget < required_memory:
            while (
                _BaseWarmupExecutor.get_memory_requirement(
                    kernel_type, max_m=max_m, n=n, k=k, num_groups=num_groups
                )
                > memory_budget
                and max_m > 4096
            ):
                max_m = max_m // 2
            logger.warning(
                f"Available memory {memory_budget}GB is less than required memory {required_memory}GB for warmup, reducing max_m to {max_m} to avoid out of memory"
            )
            m_list = [m for m in m_list if m <= max_m]

        executor = _BaseWarmupExecutor.create(
            kernel_type, max_m=max_m, n=n, k=k, num_groups=num_groups
        )

        old_compile_mode = asym_gemm.get_compile_mode()
        asym_gemm.set_compile_mode(1)
        for m in tqdm(m_list, desc=f"AsymGEMM warmup"):
            executor.execute(m=m)
        asym_gemm.set_compile_mode(old_compile_mode)

        _sync_t0 = time.perf_counter()
        torch.cuda.current_stream().synchronize()
        logger.info("[CUDA_SYNC] _compile_asym_gemm_one_type_all/stream.synchronize: %.3f ms", (time.perf_counter() - _sync_t0) * 1000)
        del executor
        torch.cuda.empty_cache()
    finally:
        restore_symmetric_memory_context(saved_context)


class _BaseWarmupExecutor:
    @staticmethod
    def create(kernel_type: AsymGemmKernelType, **kwargs):
        return {
            AsymGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG: _GroupedContWarmupExecutor,
            AsymGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED: _GroupedMaskedWarmupExecutor,
            AsymGemmKernelType.GROUPED_GEMM_NT_BF16_CONTIG: _GroupedContBf16WarmupExecutor,
            AsymGemmKernelType.GROUPED_GEMM_NT_BF16_MASKED: _GroupedMaskedBf16WarmupExecutor,
        }[kernel_type](**kwargs)

    @staticmethod
    def get_memory_requirement(
        kernel_type: AsymGemmKernelType, max_m: int, n: int, k: int, num_groups: int
    ) -> int:
        _GB = 1 << 30
        if kernel_type == AsymGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG:
            return (max_m * k + num_groups * n * k + max_m * 4 + max_m * n * 2) / _GB
        elif kernel_type == AsymGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED:
            return (
                num_groups * max_m * k
                + num_groups * n * k
                + num_groups * 4
                + num_groups * max_m * n * 2
            ) / _GB
        elif kernel_type == AsymGemmKernelType.GROUPED_GEMM_NT_BF16_CONTIG:
            return (max_m * k * 2 + num_groups * n * k * 2 + max_m * 4 + max_m * n * 2) / _GB
        elif kernel_type == AsymGemmKernelType.GROUPED_GEMM_NT_BF16_MASKED:
            return (
                num_groups * max_m * k * 2
                + num_groups * n * k * 2
                + num_groups * 4
                + num_groups * max_m * n * 2
            ) / _GB
        else:
            raise ValueError(f"Invalid kernel type: {kernel_type}")

    def execute(self, m):
        raise NotImplementedError


def _empty_token_fp8(size):
    *dims, k = size
    return (
        torch.empty(size, device="cuda", dtype=torch.float8_e4m3fn),
        torch.empty(
            (*dims, ceil_div(k, _BLOCK_SIZE)), device="cuda", dtype=torch.float32
        ),
    )


def _empty_block_fp8(size):
    *dims, n, k = size
    return (
        torch.empty(size, device="cuda", dtype=torch.float8_e4m3fn),
        torch.empty(
            (*dims, ceil_div(n, _BLOCK_SIZE), ceil_div(k, _BLOCK_SIZE)),
            device="cuda",
            dtype=torch.float32,
        ),
    )


_BLOCK_SIZE = 128


class _GroupedContWarmupExecutor(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q, self.lhs_s = _empty_token_fp8((max_m, k))
        self.rhs_q, self.rhs_s = _empty_block_fp8((num_groups, n, k))
        self.m_indices = torch.zeros((max_m,), device="cuda", dtype=torch.int32)
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)

    def execute(self, m):
        asym_gemm.m_grouped_fp8_asym_gemm_nt_contiguous(
            (self.lhs_q[:m], self.lhs_s[:m]),
            (self.rhs_q, self.rhs_s),
            self.out[:m],
            m_indices=self.m_indices[:m],
        )


class _GroupedMaskedWarmupExecutor(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q, self.lhs_s = _empty_token_fp8((num_groups, max_m, k))
        self.rhs_q, self.rhs_s = _empty_block_fp8((num_groups, n, k))
        self.masked_m = torch.zeros((num_groups,), device="cuda", dtype=torch.int32)
        self.out = torch.empty(
            (num_groups, max_m, n), device="cuda", dtype=torch.bfloat16
        )

    # def execute(self, m):
    #     asym_gemm.m_grouped_fp8_asym_gemm_nt_masked(
    #         (self.lhs_q, self.lhs_s),
    #         (self.rhs_q, self.rhs_s),
    #         self.out,
    #         masked_m=self.masked_m,
    #         expected_m=m,
    #     )

    def execute(self, m):
        return

class _GroupedContBf16WarmupExecutor(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs = torch.empty((max_m, k), device="cuda", dtype=torch.bfloat16)
        self.rhs = torch.empty((num_groups, n, k), device="cuda", dtype=torch.bfloat16)
        self.m_indices = torch.zeros((max_m,), device="cuda", dtype=torch.int32)
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)
        # Pre-allocate buffers for offsets/experts conversion
        self.offsets = torch.empty((max_m + 1,), device="cuda", dtype=torch.int32)
        self.experts = torch.empty((max_m + 1,), device="cuda", dtype=torch.int32)

    def _convert_m_indices(self, m: int):
        """Convert m_indices to offsets/experts format matching C++ build_offsets_experts_from_indices."""
        # Use simple conversion: each token with different expert ID starts a new segment
        # For warmup, we use a simple pattern where all tokens belong to expert 0
        self.m_indices[:m].fill_(0)
        self.offsets[:2].copy_(torch.tensor([0, m], device="cuda", dtype=torch.int32))
        self.experts[:2].copy_(torch.tensor([0, -1], device="cuda", dtype=torch.int32))
        return 2  # list_size

    def execute(self, m):
        list_size = self._convert_m_indices(m)
        asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous(
            self.lhs[:m],
            self.rhs,
            self.out[:m],
            self.offsets[:list_size],
            self.experts[:list_size],
            list_size,
        )


class _GroupedMaskedBf16WarmupExecutor(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs = torch.empty(
            (num_groups, max_m, k), device="cuda", dtype=torch.bfloat16
        )
        self.rhs = torch.empty(
            (num_groups, n, k), device="cuda", dtype=torch.bfloat16
        )
        self.masked_m = torch.zeros((num_groups,), device="cuda", dtype=torch.int32)
        self.out = torch.empty(
            (num_groups, max_m, n), device="cuda", dtype=torch.bfloat16
        )

    def execute(self, m):
        return
    # def execute(self, m):
    #     from .entrypoint import build_offsets_experts_from_masked_m
        
    #     num_groups = self.lhs.shape[0]
    #     offsets, experts, list_size = build_offsets_experts_from_masked_m(
    #         self.masked_m,
    #         num_groups,
    #     )

    #     rhs = self.rhs.detach().to("cpu", non_blocking=False).pin_memory()

    #     asym_gemm.m_grouped_bf16_asym_gemm_nt_masked(
    #         self.lhs,
    #         rhs,
    #         self.out,
    #         offsets,
    #         experts,
    #         list_size,
    #         m,
    #         compiled_dims="nk",
    #     )


@contextmanager
def asym_gemm_execution_hook(
    m: int, n: int, k: int, num_groups: int, kernel_type: AsymGemmKernelType
):
    if m > 0:
        _maybe_compile_asym_gemm_one_type_all(kernel_type, n, k, num_groups)
    yield
