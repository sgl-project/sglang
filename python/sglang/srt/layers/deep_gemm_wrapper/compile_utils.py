import logging
import math
import os
import time
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    disable_symmetric_memory_context,
    restore_symmetric_memory_context,
)
from sglang.srt.environ import envs
from sglang.srt.layers.deep_gemm_wrapper.configurer import ENABLE_JIT_DEEPGEMM
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_parallel, get_schedule
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    ceil_align,
    ceil_div,
    get_available_gpu_memory,
    is_musa,
    is_ppu,
)

logger = logging.getLogger(__name__)

_is_musa = is_musa()

if ENABLE_JIT_DEEPGEMM:
    import deep_gemm

_is_ppu = is_ppu()

_BUILTIN_M_LIST = list(range(1, 1024 * 16 + 1))
_ENABLE_JIT_DEEPGEMM_PRECOMPILE = envs.SGLANG_JIT_DEEPGEMM_PRECOMPILE.get()
_DO_COMPILE_ALL = True
_IS_FIRST_RANK_ON_NODE = envs.SGLANG_IS_FIRST_RANK_ON_NODE.get()
_IN_PRECOMPILE_STAGE = envs.SGLANG_IN_DEEPGEMM_PRECOMPILE_STAGE.get()
_FAST_WARMUP = envs.SGLANG_JIT_DEEPGEMM_FAST_WARMUP.get()

# Force redirect deep_gemm cache_dir. Defaults under SGLANG_CACHE_DIR so it
# sits with the other compiled-kernel caches; SGLANG_DG_CACHE_DIR still wins.
os.environ["DG_JIT_CACHE_DIR"] = envs.SGLANG_DG_CACHE_DIR.get()

# Refer to https://github.com/deepseek-ai/DeepGEMM/commit/d75b218b7b8f4a5dd5406ac87905039ead3ae42f
# NVRTC may have performance loss with some cases.
# And NVCC JIT speed is also 9x faster in the ref commit
os.environ["DG_JIT_USE_NVRTC"] = os.getenv("SGL_DG_USE_NVRTC", "0")


def update_deep_gemm_config(gpu_id: int, server_args: ServerArgs):
    global _BUILTIN_M_LIST
    global _DO_COMPILE_ALL
    global _IS_FIRST_RANK_ON_NODE

    _BUILTIN_M_LIST = []

    if _FAST_WARMUP:
        # In fast warmup mode, only compile a small set of typical Ms

        # First cover all the small bs to ensure decode performance
        _BUILTIN_M_LIST += list(range(1, 1025))

        # Then cover larger batch sizes with gradually increasing steps
        # For example, when chunekd prefill size is 16384
        # The sampled Ms would be:
        #   1024, 1026, ... 2046 (step 2)
        #   2048, 2052, ... 4092 (step 4)
        #   4096, 5004, ... 8184 (step 8)
        #   8192, 9008, ... 16384 (step 16)
        # Totally 1024 + 1024 / 2 + 2048 / 4 + 4096 / 8 + 8192 / 16 = 3072 kernels
        next_m, sample_step = 1024, 2
        chunked_prefill_size = get_schedule().chunked_prefill_size
        max_prefill_bs = (
            min(chunked_prefill_size, 32 * 1024)
            if chunked_prefill_size >= 1
            else 16 * 1024
        )
        while next_m < max_prefill_bs:
            _BUILTIN_M_LIST += list(range(next_m, 2 * next_m, sample_step))
            next_m = next_m * 2
            sample_step = sample_step * 2
        _BUILTIN_M_LIST.append(max_prefill_bs)
        _BUILTIN_M_LIST = sorted(list(set(_BUILTIN_M_LIST)))
    else:
        # When fast warmup isn't enabled, generate m_max and compile all the covered Ms.
        m_max = 1024 * 16
        chunked_prefill_size = get_schedule().chunked_prefill_size
        if chunked_prefill_size < 1:
            m_max = 1024 * 64
        elif chunked_prefill_size > 8192:
            m_max = chunked_prefill_size * 2
        m_max = min(1024 * 128, m_max)
        _BUILTIN_M_LIST += list(range(1, m_max + 1))

    _IS_FIRST_RANK_ON_NODE = server_args.base_gpu_id == gpu_id

    # Check if is the first rank on node.
    # Default each rank will try compile all Ms to
    # load all symbols at the launch stages.
    # Avoid loading symbols at the serving stages.
    _DO_COMPILE_ALL = _IS_FIRST_RANK_ON_NODE


class DeepGemmKernelType(IntEnum):
    GROUPED_GEMM_NT_F8F8BF16_MASKED = auto()
    GROUPED_GEMM_NT_F8F8BF16_CONTIG = auto()
    GROUPED_GEMM_NT_BF16_MASKED = auto()
    GROUPED_GEMM_NT_BF16_CONTIG = auto()
    GEMM_NT_F8F8BF16 = auto()
    GEMM_NT_BF16BF16F32 = auto()

    # <NOTE>
    # new added int8/bf16/fp4/int4 kernel type and nopad impl.
    # </NOTE>
    GROUPED_GEMM_NT_F8F8BF16_NOPAD = auto()
    GROUPED_GEMM_NT_F8F8BF16_MASKED_CHANNEL = auto()
    GROUPED_GEMM_NT_F8F8BF16_CONTIG_CHANNEL = auto()
    GEMM_NT_F8F8BF16_CHANNEL = auto()
    GROUPED_GEMM_NT_F8F8BF16_NOPAD_CHANNEL = auto()

    GROUPED_GEMM_NT_I8I8BF16_MASKED = auto()
    GROUPED_GEMM_NT_I8I8BF16_CONTIG = auto()
    GROUPED_GEMM_NT_I8I8BF16_NOPAD = auto()
    GEMM_NT_I8I8BF16 = auto()

    GROUPED_GEMM_NT_BF16_NOPAD = auto()

    GROUPED_GEMM_NT_F4F4BF16_MASKED = auto()
    GROUPED_GEMM_NT_F4F4BF16_NOPAD = auto()
    GEMM_NT_F4F4BF16 = auto()
    GROUPED_GEMM_NT_F4F4BF16_MASKED_BIAS = auto()
    GROUPED_GEMM_NT_F4F4BF16_NOPAD_BIAS = auto()
    GEMM_NT_F4F4BF16_BIAS = auto()

    GROUPED_GEMM_NT_BF16I4BF16_MASKED = auto()
    GROUPED_GEMM_NT_BF16I4BF16_NOPAD = auto()

    # fused variants (gather-read GEMM, no physical scatter)
    GROUPED_GEMM_NT_F8F8BF16_FUSED = auto()
    GROUPED_GEMM_NT_F8F8BF16_FUSED_CHANNEL = auto()
    GROUPED_GEMM_NT_I8I8BF16_FUSED = auto()
    GROUPED_GEMM_NT_BF16_FUSED = auto()
    GROUPED_GEMM_NT_F4F4BF16_FUSED = auto()

    # Append new kernel types to preserve all existing IntEnum values.
    GROUPED_GEMM_NT_BF16I4BF16_MMA_MASKED = auto()
    GROUPED_GEMM_NT_BF16I4BF16_MMA_NOPAD = auto()
    GROUPED_GEMM_NT_BF16I4BF16_MMA_FUSED = auto()
    GROUPED_GEMM_NT_BF16I4BF16_FUSED = auto()
    GROUPED_GEMM_NT_BF16I4BF16_VALU_FUSED = auto()
    GEMM_NT_BF16BF16BF16 = auto()


_INITIALIZATION_DICT: Dict[Tuple[DeepGemmKernelType, int, int, int], bool] = dict()


# TODO improve code
def _maybe_compile_deep_gemm_one_type_all(
    kernel_type: DeepGemmKernelType,
    n: int,
    k: int,
    num_groups: int,
) -> None:
    global _INITIALIZATION_DICT
    global _BUILTIN_M_LIST

    query_key = (kernel_type, n, k, num_groups)
    if (
        _ENABLE_JIT_DEEPGEMM_PRECOMPILE
        and _DO_COMPILE_ALL
        and not torch.cuda.is_current_stream_capturing()
        and _INITIALIZATION_DICT.get(query_key) is None
    ):
        _INITIALIZATION_DICT[query_key] = True

        # TODO maybe improve logs
        if not _IN_PRECOMPILE_STAGE and _IS_FIRST_RANK_ON_NODE:
            logger.warning(
                "Entering DeepGEMM JIT Pre-Compile session. "
                "It may take a long time (typically 10-20 mins) "
                "if you have not run `sglang.compile_deep_gemm`. "
                "It is recommended to run `sglang.compile_deep_gemm` with same args as `sglang.launch_server`"
                " for pre-compilation to reduce the overhead if you have not run it before. "
                "For example: "
                "`python3 -m sglang.compile_deep_gemm --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code`"
            )

        logger.info(
            f"Try DeepGEMM JIT Compiling for "
            f"<{kernel_type.name}> N={n}, K={k}, num_groups={num_groups} with all Ms."
            f"{' It only takes a little time (typically 1 sec) if you have run `python3 -m sglang.compile_deep_gemm`. ' if not _IN_PRECOMPILE_STAGE else ''}"
        )

        _compile_deep_gemm_one_type_all(
            kernel_type=kernel_type,
            n=n,
            k=k,
            num_groups=num_groups,
            m_list=_BUILTIN_M_LIST,
        )


# NOTE(alcanderian): get_num_sms should be change when 2-batch-overlap is introduced
def _compile_deep_gemm_one_type_all(
    kernel_type: DeepGemmKernelType,
    n: int,
    k: int,
    num_groups: int,
    m_list: List[int],
) -> None:
    # Symmetric memory allocation performs a collective operation across all the GPUs.
    # Temporary disable symmetric memory during compilation since it only runs on the first rank.
    saved_context = disable_symmetric_memory_context()
    try:
        if kernel_type in [
            DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG,
            DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG_CHANNEL,
            DeepGemmKernelType.GROUPED_GEMM_NT_I8I8BF16_CONTIG,
        ]:
            # TODO: PPU deepgemm does not have get_mk_alignment_for_contiguous_layout, use get_m_alignment_for_contiguous_layout instead
            if not _is_ppu:
                m_alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
            else:
                m_alignment = deep_gemm.get_m_alignment_for_contiguous_layout()
            m_list = sorted(list(set(m for m in m_list if m % m_alignment == 0)))
        elif kernel_type == DeepGemmKernelType.GROUPED_GEMM_NT_BF16_CONTIG:
            m_alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
            m_list = sorted(list(set(m for m in m_list if m % m_alignment == 0)))

        # Here the precompilation is only run on the first rank, so gpu_id should be 0
        memory_budget = get_available_gpu_memory(
            device="cuda", gpu_id=torch.cuda.current_device()
        )

        # If the memory budget is less memory requirement, we need to reduce max_m to avoid out of memory, which might further cause hanging during warmup
        max_m = max(m_list)
        required_memory = _BaseWarmupExecutor.get_memory_requirement(
            kernel_type, max_m=max_m, n=n, k=k, num_groups=num_groups
        )
        logger.info(
            f"Required memory for warmup: {required_memory}GB, Available memory: {memory_budget}GB"
        )
        if memory_budget < required_memory:
            # TODO: Maybe compute the max_m based on the memory budget
            while (
                _BaseWarmupExecutor.get_memory_requirement(
                    kernel_type, max_m=max_m, n=n, k=k, num_groups=num_groups
                )
                > memory_budget
                and max_m > 2048
            ):
                max_m = max_m // 2
            logger.warning(
                f"Available memory {memory_budget}GB is less than required memory {required_memory}GB for warmup, reducing max_m to {max_m} to avoid out of memory"
            )
            m_list = [m for m in m_list if m <= max_m]

        # Need some methods to estimate needed memory for warmup
        executor = _BaseWarmupExecutor.create(
            kernel_type, max_m=max_m, n=n, k=k, num_groups=num_groups
        )

        has_compile_mode_api = hasattr(deep_gemm, "get_compile_mode") and hasattr(
            deep_gemm, "set_compile_mode"
        )
        if has_compile_mode_api:
            old_compile_mode = deep_gemm.get_compile_mode()
            deep_gemm.set_compile_mode(1)

        # TODO can use multi thread
        for m in tqdm(m_list, desc="DeepGEMM warmup"):
            executor.execute(m=m)
        if has_compile_mode_api:
            deep_gemm.set_compile_mode(old_compile_mode)

        # clean up input buffers
        torch.cuda.current_stream().synchronize()
        del executor
        torch.cuda.empty_cache()
    finally:
        # Restore symmetric memory context
        restore_symmetric_memory_context(saved_context)


def check_and_skip_execute(func):
    """Decorator that checks if the m exceeds self.max_m"""

    def wrapper(self, m):
        if hasattr(self, "max_m") and m >= getattr(self, "max_m"):
            logger.warning(
                f"WARN: NOT calling deepgemm function since {m=} >= max_m={getattr(self, 'max_m')}."
            )
            return
        return func(self, m)

    return wrapper


class _BaseWarmupExecutorMeta(type):
    """Metaclass that automatically decorates 'execute' methods"""

    def __new__(cls, name, bases, attrs):
        if "execute" in attrs:
            attrs["execute"] = check_and_skip_execute(attrs["execute"])
        return super().__new__(cls, name, bases, attrs)


class _BaseWarmupExecutor(metaclass=_BaseWarmupExecutorMeta):
    @staticmethod
    def create(kernel_type: DeepGemmKernelType, **kwargs):
        return {
            DeepGemmKernelType.GEMM_NT_F8F8BF16: _NormalWarmupExecutor,
            DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG: _GroupedContWarmupExecutor,
            DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED: _GroupedMaskedWarmupExecutor,
            DeepGemmKernelType.GEMM_NT_BF16BF16F32: _BF16F32WarmupExecutor,
            DeepGemmKernelType.GEMM_NT_BF16BF16BF16: _BF16WarmupExecutor,
            DeepGemmKernelType.GROUPED_GEMM_NT_BF16_CONTIG: _BF16GroupedContWarmupExecutor,
            DeepGemmKernelType.GROUPED_GEMM_NT_BF16_MASKED: _BF16GroupedMaskedWarmupExecutor,
            # <NOTE>
            # new executor for int8/bf16/fp4/int4 kernel type as well as nopad interface on ppu
            # </NOTE>
            DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_NOPAD: _GroupedNopadWarmupExecutor,
            DeepGemmKernelType.GEMM_NT_F8F8BF16_CHANNEL: _NormalWarmupExecutor_channel,
            DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG_CHANNEL: _GroupedContWarmupExecutor_channel,
            DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED_CHANNEL: _GroupedMaskedWarmupExecutor_channel,
            DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_NOPAD_CHANNEL: _GroupedNopadWarmupExecutor_channel,
            DeepGemmKernelType.GROUPED_GEMM_NT_I8I8BF16_MASKED: _GroupedMaskedWarmupExecutor_int8,
            DeepGemmKernelType.GROUPED_GEMM_NT_I8I8BF16_CONTIG: _GroupedContWarmupExecutor_int8,
            DeepGemmKernelType.GROUPED_GEMM_NT_I8I8BF16_NOPAD: _GroupedNopadWarmupExecutor_int8,
            DeepGemmKernelType.GEMM_NT_I8I8BF16: _NormalWarmupExecutor_int8,
            DeepGemmKernelType.GROUPED_GEMM_NT_BF16_NOPAD: _GroupedNopadWarmupExecutor_bf16,
            DeepGemmKernelType.GROUPED_GEMM_NT_F4F4BF16_MASKED: _GroupedMaskedWarmupExecutor_fp4,
            DeepGemmKernelType.GROUPED_GEMM_NT_F4F4BF16_NOPAD: _GroupedNopadWarmupExecutor_fp4,
            DeepGemmKernelType.GEMM_NT_F4F4BF16: _NormalWarmupExecutor_fp4,
            DeepGemmKernelType.GROUPED_GEMM_NT_F4F4BF16_MASKED_BIAS: _GroupedMaskedWarmupExecutor_fp4_bias,
            DeepGemmKernelType.GROUPED_GEMM_NT_F4F4BF16_NOPAD_BIAS: _GroupedNopadWarmupExecutor_fp4_bias,
            DeepGemmKernelType.GEMM_NT_F4F4BF16_BIAS: _NormalWarmupExecutor_fp4_bias,
            DeepGemmKernelType.GROUPED_GEMM_NT_BF16I4BF16_MASKED: _GroupedMaskedWarmupExecutor_int4,
            DeepGemmKernelType.GROUPED_GEMM_NT_BF16I4BF16_NOPAD: _GroupedNopadWarmupExecutor_int4,
            DeepGemmKernelType.GROUPED_GEMM_NT_BF16I4BF16_MMA_MASKED: (
                _GroupedMaskedWarmupExecutor_int4_mma
            ),
            DeepGemmKernelType.GROUPED_GEMM_NT_BF16I4BF16_MMA_NOPAD: (
                _GroupedNopadWarmupExecutor_int4_mma
            ),
            DeepGemmKernelType.GROUPED_GEMM_NT_BF16I4BF16_MMA_FUSED: (
                _GroupedFusedWarmupExecutor_int4_mma
            ),
            DeepGemmKernelType.GROUPED_GEMM_NT_BF16I4BF16_FUSED: (
                _GroupedFusedWarmupExecutor_int4
            ),
            DeepGemmKernelType.GROUPED_GEMM_NT_BF16I4BF16_VALU_FUSED: (
                _GroupedFusedWarmupExecutor_mxfp4_valu
            ),
            # fused variants
            DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_FUSED: _GroupedFusedWarmupExecutor,
            DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_FUSED_CHANNEL: _GroupedFusedWarmupExecutor_fp8_channel,
            DeepGemmKernelType.GROUPED_GEMM_NT_I8I8BF16_FUSED: _GroupedFusedWarmupExecutor_int8,
            DeepGemmKernelType.GROUPED_GEMM_NT_BF16_FUSED: _GroupedFusedWarmupExecutor_bf16,
            DeepGemmKernelType.GROUPED_GEMM_NT_F4F4BF16_FUSED: _GroupedFusedWarmupExecutor_fp4,
        }[kernel_type](**kwargs)

    @staticmethod
    def get_memory_requirement(
        kernel_type: DeepGemmKernelType, max_m: int, n: int, k: int, num_groups: int
    ) -> int:
        # Return the required memory space in GB for warmup executor
        _GB = 1 << 30
        if kernel_type in [
            DeepGemmKernelType.GEMM_NT_F8F8BF16,
            DeepGemmKernelType.GEMM_NT_F8F8BF16_CHANNEL,
            DeepGemmKernelType.GEMM_NT_I8I8BF16,
            DeepGemmKernelType.GEMM_NT_F4F4BF16,
        ]:
            return (max_m * k + n * k + max_m * n * 2) / _GB
        elif kernel_type in [
            DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG,
            DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG_CHANNEL,
            DeepGemmKernelType.GROUPED_GEMM_NT_I8I8BF16_CONTIG,
            DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_NOPAD,
            DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_NOPAD_CHANNEL,
            DeepGemmKernelType.GROUPED_GEMM_NT_I8I8BF16_NOPAD,
            DeepGemmKernelType.GROUPED_GEMM_NT_F4F4BF16_NOPAD,
        ]:
            return (max_m * k + num_groups * n * k + max_m * 4 + max_m * n * 2) / _GB
        elif kernel_type in [
            DeepGemmKernelType.GROUPED_GEMM_NT_BF16_CONTIG,
            DeepGemmKernelType.GROUPED_GEMM_NT_BF16_NOPAD,
        ]:
            return (
                max_m * k * 2 + num_groups * n * k * 2 + max_m * 4 + max_m * n * 2
            ) / _GB
        elif kernel_type in [
            DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED,
            DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED_CHANNEL,
            DeepGemmKernelType.GROUPED_GEMM_NT_I8I8BF16_MASKED,
            DeepGemmKernelType.GROUPED_GEMM_NT_F4F4BF16_MASKED,
        ]:
            return (
                num_groups * max_m * k
                + num_groups * n * k
                + num_groups * 4
                + num_groups * max_m * n * 2
            ) / _GB
        elif kernel_type == DeepGemmKernelType.GEMM_NT_BF16BF16F32:
            # bf16 lhs + bf16 rhs + fp32 out
            return (max_m * k * 2 + n * k * 2 + max_m * n * 4) / _GB
        elif kernel_type == DeepGemmKernelType.GEMM_NT_BF16BF16BF16:
            return (max_m * k * 2 + n * k * 2 + max_m * n * 2) / _GB
        elif kernel_type == DeepGemmKernelType.GROUPED_GEMM_NT_BF16_MASKED:
            return (
                num_groups * max_m * k * 2
                + num_groups * n * k * 2
                + num_groups * 4
                + num_groups * max_m * n * 2
            ) / _GB
        elif kernel_type in [DeepGemmKernelType.GROUPED_GEMM_NT_F4F4BF16_NOPAD_BIAS]:
            return (
                max_m * k
                + num_groups * n * k
                + num_groups * n * 4
                + max_m * 4
                + max_m * n * 2
            ) / _GB
        elif kernel_type in [DeepGemmKernelType.GROUPED_GEMM_NT_F4F4BF16_MASKED_BIAS]:
            return (
                num_groups * max_m * k
                + num_groups * n * k
                + num_groups * n * 4
                + num_groups * 4
                + num_groups * max_m * n * 2
            ) / _GB
        elif kernel_type in [DeepGemmKernelType.GEMM_NT_F4F4BF16_BIAS]:
            return (max_m * k + n * k + n * 4 + max_m * n * 2) / _GB
        elif kernel_type in [DeepGemmKernelType.GROUPED_GEMM_NT_BF16I4BF16_NOPAD]:
            return (
                max_m * k * 2
                + num_groups * (k // 16) * (n * 2) * 4
                + num_groups * (k // 32) * n * 2
                + max_m * 4
                + max_m * n * 2
            ) / _GB
        elif kernel_type in [DeepGemmKernelType.GROUPED_GEMM_NT_BF16I4BF16_MASKED]:
            return (
                num_groups * max_m * k * 2
                + num_groups * (k // 16) * (n * 2) * 4
                + num_groups * (k // 32) * n * 2
                + num_groups * 4
                + num_groups * max_m * n * 2
            ) / _GB
        elif kernel_type in [DeepGemmKernelType.GROUPED_GEMM_NT_BF16I4BF16_MMA_NOPAD]:
            return (
                max_m * k * 2
                + num_groups * n * (k // 2)
                + num_groups * n * (k // 32)
                + max_m * 4
                + max_m * n * 2
            ) / _GB
        elif kernel_type in [DeepGemmKernelType.GROUPED_GEMM_NT_BF16I4BF16_MMA_MASKED]:
            return (
                num_groups * max_m * k * 2
                + num_groups * n * (k // 2)
                + num_groups * n * (k // 32)
                + num_groups * 4
                + num_groups * max_m * n * 2
            ) / _GB
        elif kernel_type in [
            DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_FUSED,
            DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_FUSED_CHANNEL,
            DeepGemmKernelType.GROUPED_GEMM_NT_I8I8BF16_FUSED,
            DeepGemmKernelType.GROUPED_GEMM_NT_BF16_FUSED,
            DeepGemmKernelType.GROUPED_GEMM_NT_F4F4BF16_FUSED,
            DeepGemmKernelType.GROUPED_GEMM_NT_BF16I4BF16_MMA_FUSED,
            DeepGemmKernelType.GROUPED_GEMM_NT_BF16I4BF16_FUSED,
            DeepGemmKernelType.GROUPED_GEMM_NT_BF16I4BF16_VALU_FUSED,
        ]:
            # moe_align output memory (all int32)
            _BLOCK_M_MIN = 16  # min block_m config, used for conservative upper bound of max_num_m_blocks
            _BLOCK_M_MAX = (
                256  # max block_m config, used for conservative upper bound of s_total
            )
            _numel = max_m  # warmup topk=1
            _n_align = _numel + 1 - num_groups
            _max_blocks_ub = num_groups - 1 + ceil_div(_n_align, _BLOCK_M_MIN)
            _s_total_ub = (
                num_groups - 1 + ceil_div(_n_align, _BLOCK_M_MAX)
            ) * _BLOCK_M_MAX
            _bs = (
                1 << max(0, num_groups - 1).bit_length() if num_groups > 0 else 1
            )  # next_pow2 with boundary check
            _num_blocks_pad = ceil_div(_s_total_ub, _bs)
            _moe_align_mem = (
                40  # config (10 ints)
                + num_groups * 4  # m_rows
                + _max_blocks_ub * 16  # expert_ids_and_cumsum (max_num_m_blocks, 4)
                + _s_total_ub * 4  # sorted_token_ids (max_num_m_blocks, block_m)
                + 4  # aligned_num_m_blocks
                + _numel * 4  # inv_perm
                + _numel * 4  # m_indices
                + (2 * _num_blocks_pad + 2) * num_groups * 4  # intermediate_buffer
            )
            if kernel_type in [
                DeepGemmKernelType.GROUPED_GEMM_NT_BF16I4BF16_FUSED,
                DeepGemmKernelType.GROUPED_GEMM_NT_BF16I4BF16_VALU_FUSED,
            ]:
                scale_bytes = (
                    1
                    if kernel_type
                    == DeepGemmKernelType.GROUPED_GEMM_NT_BF16I4BF16_VALU_FUSED
                    else 2
                )
                return (
                    max_m * k * 2
                    + num_groups * (k // 16) * (n * 2) * 4
                    + num_groups * (k // 32) * n * scale_bytes
                    + max_m * n * 2
                    + _moe_align_mem
                ) / _GB
            elif kernel_type in [
                DeepGemmKernelType.GROUPED_GEMM_NT_BF16I4BF16_MMA_FUSED
            ]:
                return (
                    max_m * k * 2
                    + num_groups * n * (k // 2)
                    + num_groups * n * (k // 32)
                    + max_m * n * 2
                    + _moe_align_mem
                ) / _GB
            elif kernel_type in [DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_FUSED]:
                # lhs(fp8) + lhs_scale(f32 blockwise) + rhs(fp8) + rhs_scale(f32 blockwise) + out(bf16)
                return (
                    max_m * k
                    + max_m * ceil_div(k, _BLOCK_SIZE) * 4
                    + num_groups * n * k
                    + num_groups
                    * ceil_div(n, _BLOCK_SIZE)
                    * ceil_div(k, _BLOCK_SIZE)
                    * 4
                    + max_m * n * 2
                    + _moe_align_mem
                ) / _GB
            elif kernel_type in [
                DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_FUSED_CHANNEL,
                DeepGemmKernelType.GROUPED_GEMM_NT_I8I8BF16_FUSED,
            ]:
                # lhs(i8/fp8) + lhs_scale(f32 per-channel) + rhs(i8/fp8) + rhs_scale(f32 per-channel) + out(bf16)
                return (
                    max_m * k
                    + max_m * 4
                    + num_groups * n * k
                    + num_groups * n * 4
                    + max_m * n * 2
                    + _moe_align_mem
                ) / _GB
            elif kernel_type in [DeepGemmKernelType.GROUPED_GEMM_NT_F4F4BF16_FUSED]:
                _GROUP_SIZE = 32  # k is came from quanted tensor
                # lhs(uint8) + lhs_scale(uint16 groupwise) + rhs(uint8) + rhs_scale(uint16 groupwise) + out(bf16)
                return (
                    max_m * k
                    + max_m * ceil_div(k, _GROUP_SIZE) * 2  # 2 means uint16_t
                    + num_groups * n * k
                    + num_groups * n * ceil_div(k, _GROUP_SIZE) * 2  # 2 means uint16_t
                    + max_m * n * 2  # 2 means bfloat16
                    + _moe_align_mem
                ) / _GB
            else:  # BF16_FUSED
                # lhs(bf16) + rhs(bf16) + out(bf16)
                return (
                    max_m * k * 2
                    + num_groups * n * k * 2
                    + max_m * n * 2
                    + _moe_align_mem
                ) / _GB
        else:
            raise ValueError(f"Invalid kernel type: {kernel_type}")

    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        self.max_m = max_m
        while True:
            try:
                self.setup_tensors(self.max_m, n, k, num_groups)
                break
            except:
                self.max_m = self.max_m - 1024
                if self.max_m <= 0:
                    break

    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        raise NotImplementedError

    def execute(self, m):
        raise NotImplementedError


def _empty_token_fp8(size):
    *dims, k = size
    return (
        torch.empty(size, device="cuda", dtype=torch.float8_e4m3fn),
        torch.ones(
            (*dims, ceil_div(k, _BLOCK_SIZE)), device="cuda", dtype=torch.float32
        ),
    )


def _empty_block_fp8(size):
    *dims, n, k = size
    return (
        torch.empty(size, device="cuda", dtype=torch.float8_e4m3fn),
        torch.ones(
            (*dims, ceil_div(n, _BLOCK_SIZE), ceil_div(k, _BLOCK_SIZE)),
            device="cuda",
            dtype=torch.float32,
        ),
    )


_BLOCK_SIZE = 128


class _NormalWarmupExecutor(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        super().__init__(max_m, n, k, num_groups)

    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q, self.lhs_s = _empty_token_fp8((max_m, k))
        self.rhs_q, self.rhs_s = _empty_block_fp8((n, k))
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)

    def execute(self, m):
        deep_gemm.fp8_gemm_nt(
            (self.lhs_q[:m], self.lhs_s[:m]),
            (self.rhs_q, self.rhs_s),
            self.out[:m],
        )


class _GroupedContWarmupExecutor(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        super().__init__(max_m, n, k, num_groups)

    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q, self.lhs_s = _empty_token_fp8((max_m, k))
        self.rhs_q, self.rhs_s = _empty_block_fp8((num_groups, n, k))
        self.m_indices = torch.zeros((max_m,), device="cuda", dtype=torch.int32)
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)

    def execute(self, m):
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (self.lhs_q[:m], self.lhs_s[:m]),
            (self.rhs_q, self.rhs_s),
            self.out[:m],
            self.m_indices[:m],
        )


class _BF16GroupedContWarmupExecutor(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        super().__init__(max_m, n, k, num_groups)

    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        self.a = torch.empty((max_m, k), device="cuda", dtype=torch.bfloat16)
        self.b = torch.empty((num_groups, n, k), device="cuda", dtype=torch.bfloat16)
        self.m_indices = torch.zeros((max_m,), device="cuda", dtype=torch.int32)
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)

    def execute(self, m):
        deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_contiguous(
            self.a[:m],
            self.b,
            self.out[:m],
            self.m_indices[:m],
        )


class _GroupedMaskedWarmupExecutor(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        super().__init__(max_m, n, k, num_groups)

    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q, self.lhs_s = _empty_token_fp8((num_groups, max_m, k))
        self.rhs_q, self.rhs_s = _empty_block_fp8((num_groups, n, k))
        self.masked_m = torch.zeros((num_groups,), device="cuda", dtype=torch.int32)
        self.out = torch.empty(
            (num_groups, max_m, n), device="cuda", dtype=torch.bfloat16
        )

    def execute(self, m):
        deep_gemm.fp8_m_grouped_gemm_nt_masked(
            (self.lhs_q, self.lhs_s),
            (self.rhs_q, self.rhs_s),
            self.out,
            masked_m=self.masked_m,
            # DeepGEMM uses `expected_m` instead of input shape for `get_best_config`
            expected_m=m,
        )


class _BF16F32WarmupExecutor(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs = torch.empty((max_m, k), device="cuda", dtype=torch.bfloat16)
        self.rhs = torch.empty((n, k), device="cuda", dtype=torch.bfloat16)
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.float32)

    def execute(self, m):
        deep_gemm.bf16_gemm_nt(self.lhs[:m], self.rhs, self.out[:m])


class _BF16WarmupExecutor(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs = torch.empty((max_m, k), device="cuda", dtype=torch.bfloat16)
        self.rhs = torch.empty((n, k), device="cuda", dtype=torch.bfloat16)
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)

    def execute(self, m):
        deep_gemm.bf16_gemm_nt(self.lhs[:m], self.rhs, self.out[:m])


class _BF16GroupedMaskedWarmupExecutor(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        super().__init__(max_m, n, k, num_groups)

    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        self.a = torch.empty(
            (num_groups, max_m, k), device="cuda", dtype=torch.bfloat16
        )
        self.b = torch.empty((num_groups, n, k), device="cuda", dtype=torch.bfloat16)
        self.masked_m = torch.zeros((num_groups,), device="cuda", dtype=torch.int32)
        self.out = torch.empty(
            (num_groups, max_m, n), device="cuda", dtype=torch.bfloat16
        )

    def execute(self, m):
        deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_masked(
            self.a,
            self.b,
            self.out,
            masked_m=self.masked_m,
            # DeepGEMM uses `expected_m` instead of input shape for `get_best_config`
            expected_m=m,
        )


def deep_gemm_execution_hook(
    m: int, n: int, k: int, num_groups: int, kernel_type: DeepGemmKernelType
):
    if _is_musa:
        return nullcontext()

    return _deep_gemm_execution_hook(m, n, k, num_groups, kernel_type)


@contextmanager
def _deep_gemm_execution_hook(
    m: int, n: int, k: int, num_groups: int, kernel_type: DeepGemmKernelType
):
    if m > 0:
        _maybe_compile_deep_gemm_one_type_all(kernel_type, n, k, num_groups)
    yield


def pp_parallel_deep_gemm_warmup(runner) -> None:
    """Run per-PP-rank dummy DECODE+EXTEND forwards so each rank's
    DeepGEMM JIT compiles in parallel instead of serially via the warmup
    /generate flowing through the pipeline. Opt-in via
    SGLANG_PP_PARALLEL_DEEPGEMM_WARMUP.

    Driven from BaseRunner.warmup(), which passes the runner; the dummy
    forwards go through runner._dummy_run (the autotune/dummy-run machinery now
    lives on BaseRunner). ModelRunner state is read via runner.model_runner.
    """
    model_runner = runner.model_runner
    # n_splits ~= n_sms / ceil(bs/block_m) with block_m=64; sweep 5 bs to
    # cover the brackets real /generate hits (smallest decode shape,
    # mid-low, two mid, and n_splits=1 for ~5K+ token prefill). Ceil-align
    # bs to the CP padding alignment (cp_size, or 2*cp_size for DSA
    # in-seq-split). _dummy_run does not pad q/hidden like the real flow, so
    # an unaligned bs makes DSA's padded num_splits longer than the q tokens
    # and trips FlashMLA's "num_splits must have shape (b+1)" check.
    from sglang.srt.layers.cp.padding import get_cp_padding_align_size
    from sglang.srt.utils.common import require_mlp_sync

    n_sms = torch.cuda.get_device_properties(model_runner.device).multi_processor_count
    block_m = 64
    cp = max(get_cp_padding_align_size(), 1)

    attn_tp_size = get_parallel().attn_tp_size
    mlp_sync = require_mlp_sync(model_runner.server_args)

    def _align(bs: int) -> int:
        # Align to lcm(cp, attn_tp_size) so the CP multiple isn't undone by a
        # later attn_tp align (e.g. cp=2, attn_tp=3: 128 -> 128 -> 129).
        align = cp
        if mlp_sync and attn_tp_size > 1:
            align = math.lcm(cp, attn_tp_size)
        return ceil_align(bs, align)

    batch_sizes = sorted(
        {
            _align(bs)
            for bs in (
                1,
                2 * block_m,
                max(n_sms // 8, 2) * block_m,
                max(n_sms // 4, 4) * block_m,
                n_sms * block_m,
            )
        }
    )

    # In PD, prefill-only nodes never decode (indexer would OOM at large
    # bs) and decode-only nodes never extend.
    disagg_mode = model_runner.server_args.disaggregation_mode
    run_decode = model_runner.is_generation and disagg_mode != "prefill"
    run_extend = disagg_mode != "decode"

    logger.info(
        "PP-parallel DeepGEMM warmup start "
        "(pp_rank=%d, tp_rank=%d, batch_sizes=%s, disagg=%s).",
        model_runner.ps.pp_rank,
        model_runner.ps.tp_rank,
        batch_sizes,
        disagg_mode,
    )

    # One buffer set sized to the largest shape, reused across the sweep
    # (the decode runner's max_bs is too small for n_sms*block_m).
    dummy_buffers = runner._alloc_dummy_decode_buffers(max(batch_sizes))

    t0 = time.perf_counter()
    with torch.inference_mode():
        for bs in batch_sizes:
            if run_decode:
                runner._dummy_run(
                    batch_size=bs,
                    forward_mode_override=ForwardMode.DECODE,
                    buffers=dummy_buffers,
                )
            if run_extend:
                runner._dummy_run(
                    batch_size=bs,
                    forward_mode_override=ForwardMode.EXTEND,
                    buffers=dummy_buffers,
                )

    logger.info(
        "PP-parallel DeepGEMM warmup done in %.2fs (pp_rank=%d).",
        time.perf_counter() - t0,
        model_runner.ps.pp_rank,
    )


def _empty_token_int8(size):
    *dims, k = size
    return (
        torch.empty(size, device="cuda", dtype=torch.int8),
        torch.empty((*dims, 1), device="cuda", dtype=torch.float32),
    )


def _empty_block_int8(size):
    *dims, n, k = size
    return (
        torch.empty(size, device="cuda", dtype=torch.int8),
        torch.empty(
            (*dims, n, 1),
            device="cuda",
            dtype=torch.float32,
        ),
    )


def _empty_token_fp8_channel(size):
    *dims, k = size
    return (
        torch.empty(size, device="cuda", dtype=torch.float8_e4m3fn),
        torch.empty((*dims, 1), device="cuda", dtype=torch.float32),
    )


def _empty_block_fp8_channel(size):
    *dims, n, k = size
    return (
        torch.empty(size, device="cuda", dtype=torch.float8_e4m3fn),
        torch.empty(
            (*dims, n, 1),
            device="cuda",
            dtype=torch.float32,
        ),
    )


def _empty_token_bf16(size):
    *dims, k = size
    return torch.empty(size, device="cuda", dtype=torch.bfloat16)


def _empty_block_bf16(size):
    *dims, n, k = size
    return torch.empty(size, device="cuda", dtype=torch.bfloat16)


def _empty_token_uint8(size):
    *dims, k = size
    return (
        torch.empty(size, device="cuda", dtype=torch.uint8),
        torch.empty((*dims, k // 16), device="cuda", dtype=torch.uint8),
    )


def _empty_block_uint8(size):
    *dims, n, k = size
    return (
        torch.empty(size, device="cuda", dtype=torch.uint8),
        torch.empty(
            (*dims, n, k // 16),
            device="cuda",
            dtype=torch.uint8,
        ),
    )


def _empty_marlin_int4(size, scale_dtype=torch.bfloat16):
    # Currently deepgemm w4a16 only supports group_size=32
    *dims, n, k = size
    group_size = 32
    return (
        torch.empty(
            (*dims, k // 16, n * 2),
            device="cuda",
            dtype=torch.int32,
        ),
        torch.empty(
            (*dims, k // group_size, n),
            device="cuda",
            dtype=scale_dtype,
        ),
    )


def _empty_mxfp4_w4a16_mma(size):
    *dims, n, k = size
    if n % 64 != 0 or k % 64 != 0:
        raise ValueError(f"W4A16 MMA requires 64-aligned N/K, got N={n}, K={k}.")
    return (
        torch.empty((*dims, n, k // 2), device="cuda", dtype=torch.uint8),
        torch.empty((*dims, n // 64, k * 2), device="cuda", dtype=torch.uint8),
    )


class _GroupedNopadWarmupExecutor(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        super().__init__(max_m, n, k, num_groups)

    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q, self.lhs_s = _empty_token_fp8((max_m, k))
        self.rhs_q, self.rhs_s = _empty_block_fp8((num_groups, n, k))
        self.m_indices = torch.zeros((max_m,), device="cuda", dtype=torch.int32)
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)

    def execute(self, m):
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_nopad(
            (self.lhs_q[:m], self.lhs_s[:m]),
            (self.rhs_q, self.rhs_s),
            self.out[:m],
            m_indices=self.m_indices[:m],
        )


class _NormalWarmupExecutor_channel(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        super().__init__(max_m, n, k, num_groups)

    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q, self.lhs_s = _empty_token_fp8_channel((max_m, k))
        self.rhs_q, self.rhs_s = _empty_block_fp8_channel((n, k))
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)

    def execute(self, m):
        deep_gemm.fp8_gemm_nt(
            (self.lhs_q[:m], self.lhs_s[:m]),
            (self.rhs_q, self.rhs_s),
            self.out[:m],
        )


class _GroupedContWarmupExecutor_channel(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        super().__init__(max_m, n, k, num_groups)

    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q, self.lhs_s = _empty_token_fp8_channel((max_m, k))
        self.rhs_q, self.rhs_s = _empty_block_fp8_channel((num_groups, n, k))
        self.m_indices = torch.zeros((max_m,), device="cuda", dtype=torch.int32)
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)

    def execute(self, m):
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (self.lhs_q[:m], self.lhs_s[:m]),
            (self.rhs_q, self.rhs_s),
            self.out[:m],
            m_indices=self.m_indices[:m],
        )


class _GroupedMaskedWarmupExecutor_channel(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        super().__init__(max_m, n, k, num_groups)

    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q, self.lhs_s = _empty_token_fp8_channel((num_groups, max_m, k))
        self.rhs_q, self.rhs_s = _empty_block_fp8_channel((num_groups, n, k))
        self.masked_m = torch.zeros((num_groups,), device="cuda", dtype=torch.int32)
        self.out = torch.empty(
            (num_groups, max_m, n), device="cuda", dtype=torch.bfloat16
        )

    def execute(self, m):
        deep_gemm.fp8_m_grouped_gemm_nt_masked(
            (self.lhs_q, self.lhs_s),
            (self.rhs_q, self.rhs_s),
            self.out,
            masked_m=self.masked_m,
            # DeepGEMM uses `expected_m` instead of input shape for `get_best_config`
            expected_m=m,
        )


class _GroupedNopadWarmupExecutor_channel(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        super().__init__(max_m, n, k, num_groups)

    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q, self.lhs_s = _empty_token_fp8_channel((max_m, k))
        self.rhs_q, self.rhs_s = _empty_block_fp8_channel((num_groups, n, k))
        self.m_indices = torch.zeros((max_m,), device="cuda", dtype=torch.int32)
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)

    def execute(self, m):
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_nopad(
            (self.lhs_q[:m], self.lhs_s[:m]),
            (self.rhs_q, self.rhs_s),
            self.out[:m],
            m_indices=self.m_indices[:m],
        )


class _NormalWarmupExecutor_int8(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        super().__init__(max_m, n, k, num_groups)

    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q, self.lhs_s = _empty_token_int8((max_m, k))
        self.rhs_q, self.rhs_s = _empty_block_int8((n, k))
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)

    def execute(self, m):
        deep_gemm.gemm_int8_int8_bf16_nt(
            (self.lhs_q[:m], self.lhs_s[:m]),
            (self.rhs_q, self.rhs_s),
            self.out[:m],
        )


class _GroupedContWarmupExecutor_int8(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        super().__init__(max_m, n, k, num_groups)

    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q, self.lhs_s = _empty_token_int8((max_m, k))
        self.rhs_q, self.rhs_s = _empty_block_int8((num_groups, n, k))
        self.m_indices = torch.zeros((max_m,), device="cuda", dtype=torch.int32)
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)

    def execute(self, m):
        deep_gemm.m_grouped_gemm_int8_int8_bf16_nt_contiguous(
            (self.lhs_q[:m], self.lhs_s[:m]),
            (self.rhs_q, self.rhs_s),
            self.out[:m],
            m_indices=self.m_indices[:m],
        )


class _GroupedMaskedWarmupExecutor_int8(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        super().__init__(max_m, n, k, num_groups)

    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q, self.lhs_s = _empty_token_int8((num_groups, max_m, k))
        self.rhs_q, self.rhs_s = _empty_block_int8((num_groups, n, k))
        self.masked_m = torch.zeros((num_groups,), device="cuda", dtype=torch.int32)
        self.out = torch.empty(
            (num_groups, max_m, n), device="cuda", dtype=torch.bfloat16
        )

    def execute(self, m):
        deep_gemm.m_grouped_gemm_int8_int8_bf16_nt_masked(
            (self.lhs_q, self.lhs_s),
            (self.rhs_q, self.rhs_s),
            self.out,
            masked_m=self.masked_m,
            # DeepGEMM uses `expected_m` instead of input shape for `get_best_config`
            expected_m=m,
        )


class _GroupedNopadWarmupExecutor_int8(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        super().__init__(max_m, n, k, num_groups)

    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q, self.lhs_s = _empty_token_int8((max_m, k))
        self.rhs_q, self.rhs_s = _empty_block_int8((num_groups, n, k))
        self.m_indices = torch.zeros((max_m,), device="cuda", dtype=torch.int32)
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)

    def execute(self, m):
        deep_gemm.m_grouped_gemm_int8_int8_bf16_nt_nopad(
            (self.lhs_q[:m], self.lhs_s[:m]),
            (self.rhs_q, self.rhs_s),
            self.out[:m],
            m_indices=self.m_indices[:m],
        )


class _GroupedNopadWarmupExecutor_bf16(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        super().__init__(max_m, n, k, num_groups)

    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q = _empty_token_bf16((max_m, k))
        self.rhs_q = _empty_block_bf16((num_groups, n, k))
        self.m_indices = torch.zeros((max_m,), device="cuda", dtype=torch.int32)
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)

    def execute(self, m):
        deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_nopad(
            self.lhs_q[:m],
            self.rhs_q,
            self.out[:m],
            m_indices=self.m_indices[:m],
        )


class _GroupedMaskedWarmupExecutor_fp4(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        super().__init__(max_m, n, k, num_groups)

    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q, self.lhs_s = _empty_token_uint8((num_groups, max_m, k))
        self.rhs_q, self.rhs_s = _empty_block_uint8((num_groups, n, k))
        self.lhs_s = deep_gemm.preprocess_mxfp4_scales(self.lhs_s)
        self.rhs_s = deep_gemm.preprocess_mxfp4_scales(self.rhs_s)
        self.bias = None
        self.masked_m = torch.zeros((num_groups,), device="cuda", dtype=torch.int32)
        self.out = torch.empty(
            (num_groups, max_m, n), device="cuda", dtype=torch.bfloat16
        )

    def execute(self, m):
        deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_masked(
            (self.lhs_q, self.lhs_s),
            (self.rhs_q, self.rhs_s),
            self.bias,
            self.out,
            masked_m=self.masked_m,
            # DeepGEMM uses `expected_m` instead of input shape for `get_best_config`
            expected_m=m,
        )


class _GroupedNopadWarmupExecutor_fp4(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        super().__init__(max_m, n, k, num_groups)

    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q, self.lhs_s = _empty_token_uint8((max_m, k))
        self.rhs_q, self.rhs_s = _empty_block_uint8((num_groups, n, k))
        self.lhs_s = deep_gemm.preprocess_mxfp4_scales(self.lhs_s)
        self.rhs_s = deep_gemm.preprocess_mxfp4_scales(self.rhs_s)
        self.bias = None
        self.m_indices = torch.zeros((max_m,), device="cuda", dtype=torch.int32)
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)

    def execute(self, m):
        deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_nopad(
            (self.lhs_q[:m], self.lhs_s[:m]),
            (self.rhs_q, self.rhs_s),
            self.bias,
            self.out[:m],
            m_indices=self.m_indices[:m],
        )


class _NormalWarmupExecutor_fp4(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        super().__init__(max_m, n, k, num_groups)

    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q, self.lhs_s = _empty_token_uint8((max_m, k))
        self.rhs_q, self.rhs_s = _empty_block_uint8((n, k))
        self.lhs_s = deep_gemm.preprocess_mxfp4_scales(self.lhs_s)
        self.rhs_s = deep_gemm.preprocess_mxfp4_scales(self.rhs_s)
        self.bias = None
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)

    def execute(self, m):
        deep_gemm.gemm_fp4_fp4_bf16_nt(
            (self.lhs_q[:m], self.lhs_s[:m]),
            (self.rhs_q, self.rhs_s),
            self.bias,
            self.out[:m],
        )


class _GroupedMaskedWarmupExecutor_fp4_bias(_GroupedMaskedWarmupExecutor_fp4):
    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        super().setup_tensors(max_m, n, k, num_groups)
        self.bias = torch.zeros((num_groups, n), device="cuda", dtype=torch.float32)


class _GroupedNopadWarmupExecutor_fp4_bias(_GroupedNopadWarmupExecutor_fp4):
    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        super().setup_tensors(max_m, n, k, num_groups)
        self.bias = torch.zeros((num_groups, n), device="cuda", dtype=torch.float32)


class _NormalWarmupExecutor_fp4_bias(_NormalWarmupExecutor_fp4):
    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        super().setup_tensors(max_m, n, k, num_groups)
        self.bias = torch.zeros((n,), device="cuda", dtype=torch.float32)


class _GroupedMaskedWarmupExecutor_int4(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        super().__init__(max_m, n, k, num_groups)

    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs = _empty_token_bf16((num_groups, max_m, k))
        self.rhs_q, self.rhs_s = _empty_marlin_int4((num_groups, n, k))
        self.masked_m = torch.zeros((num_groups,), device="cuda", dtype=torch.int32)
        self.out = torch.empty(
            (num_groups, max_m, n), device="cuda", dtype=torch.bfloat16
        )

    def execute(self, m):
        deep_gemm.m_grouped_gemm_w4a16_masked(
            self.lhs,
            (self.rhs_q, self.rhs_s),
            self.out,
            masked_m=self.masked_m,
            # DeepGEMM uses `expected_m` instead of input shape for `get_best_config`
            expected_m=m,
        )


class _GroupedNopadWarmupExecutor_int4(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        super().__init__(max_m, n, k, num_groups)

    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs = _empty_token_bf16((max_m, k))
        self.rhs_q, self.rhs_s = _empty_marlin_int4((num_groups, n, k))
        self.m_indices = torch.zeros((max_m,), device="cuda", dtype=torch.int32)
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)

    def execute(self, m):
        deep_gemm.m_grouped_gemm_w4a16_nopad(
            self.lhs[:m],
            (self.rhs_q, self.rhs_s),
            self.out[:m],
            m_indices=self.m_indices[:m],
        )


class _GroupedMaskedWarmupExecutor_int4_mma(_GroupedMaskedWarmupExecutor_int4):
    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs = _empty_token_bf16((num_groups, max_m, k))
        self.rhs_q, self.rhs_s = _empty_mxfp4_w4a16_mma((num_groups, n, k))
        self.masked_m = torch.zeros((num_groups,), device="cuda", dtype=torch.int32)
        self.out = torch.empty(
            (num_groups, max_m, n), device="cuda", dtype=torch.bfloat16
        )


class _GroupedNopadWarmupExecutor_int4_mma(_GroupedNopadWarmupExecutor_int4):
    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs = _empty_token_bf16((max_m, k))
        self.rhs_q, self.rhs_s = _empty_mxfp4_w4a16_mma((num_groups, n, k))
        self.m_indices = torch.zeros((max_m,), device="cuda", dtype=torch.int32)
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)


class _GroupedFusedWarmupExecutor_int4(_BaseWarmupExecutor):
    scale_dtype = torch.bfloat16

    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        super().__init__(max_m, n, k, num_groups)

    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs = _empty_token_bf16((max_m, k))
        self.rhs_q, self.rhs_s = _empty_marlin_int4(
            (num_groups, n, k), self.scale_dtype
        )
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)
        self.topk_ids = torch.randint(
            0, num_groups, (max_m, 1), device="cuda", dtype=torch.int32
        )

    def execute(self, m):
        (
            config,
            m_rows,
            expert_ids_and_cumsum,
            sorted_token_ids,
            aligned_num_m_blocks,
            _,
            _,
        ) = deep_gemm.moe_align_block_size(
            self.lhs[:m], self.rhs_q, self.topk_ids[:m], perchannel_quant=False
        )
        deep_gemm.m_grouped_gemm_w4a16_fused(
            self.lhs[:m],
            (self.rhs_q, self.rhs_s),
            self.out[:m],
            m_rows,
            expert_ids_and_cumsum,
            sorted_token_ids,
            aligned_num_m_blocks,
            config,
        )


class _GroupedFusedWarmupExecutor_int4_mma(_GroupedFusedWarmupExecutor_int4):
    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs = _empty_token_bf16((max_m, k))
        self.rhs_q, self.rhs_s = _empty_mxfp4_w4a16_mma((num_groups, n, k))
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)
        self.topk_ids = torch.randint(
            0, num_groups, (max_m, 1), device="cuda", dtype=torch.int32
        )


class _GroupedFusedWarmupExecutor_mxfp4_valu(_GroupedFusedWarmupExecutor_int4):
    scale_dtype = torch.uint8


class _GroupedFusedWarmupExecutor(_BaseWarmupExecutor):
    """Warmup executor for GROUPED_GEMM_NT_F8F8BF16_FUSED."""

    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        super().__init__(max_m, n, k, num_groups)

    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q, self.lhs_s = _empty_token_fp8((max_m, k))
        self.rhs_q, self.rhs_s = _empty_block_fp8((num_groups, n, k))
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)
        self.topk_ids = torch.randint(
            0, num_groups, (max_m, 1), device="cuda", dtype=torch.int32
        )

    def execute(self, m):
        (
            config,
            m_rows,
            expert_ids_and_cumsum,
            sorted_token_ids,
            aligned_num_m_blocks,
            _,
            _,
        ) = deep_gemm.moe_align_block_size(
            self.lhs_q[:m], self.rhs_q, self.topk_ids[:m], perchannel_quant=False
        )
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_fused(
            (self.lhs_q[:m], self.lhs_s[:m]),
            (self.rhs_q, self.rhs_s),
            self.out[:m],
            m_rows,
            expert_ids_and_cumsum,
            sorted_token_ids,
            aligned_num_m_blocks,
            config,
        )


class _GroupedFusedWarmupExecutor_int8(_BaseWarmupExecutor):
    """Warmup executor for GROUPED_GEMM_NT_I8I8BF16_FUSED."""

    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        super().__init__(max_m, n, k, num_groups)

    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q, self.lhs_s = _empty_token_int8((max_m, k))
        self.rhs_q, self.rhs_s = _empty_block_int8((num_groups, n, k))
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)
        self.topk_ids = torch.randint(
            0, num_groups, (max_m, 1), device="cuda", dtype=torch.int32
        )

    def execute(self, m):
        (
            config,
            m_rows,
            expert_ids_and_cumsum,
            sorted_token_ids,
            aligned_num_m_blocks,
            _,
            _,
        ) = deep_gemm.moe_align_block_size(
            self.lhs_q[:m], self.rhs_q, self.topk_ids[:m], perchannel_quant=True
        )
        deep_gemm.m_grouped_gemm_int8_int8_bf16_nt_fused(
            (self.lhs_q[:m], self.lhs_s[:m]),
            (self.rhs_q, self.rhs_s),
            self.out[:m],
            m_rows,
            expert_ids_and_cumsum,
            sorted_token_ids,
            aligned_num_m_blocks,
            config,
        )


class _GroupedFusedWarmupExecutor_bf16(_BaseWarmupExecutor):
    """Warmup executor for GROUPED_GEMM_NT_BF16_FUSED."""

    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        super().__init__(max_m, n, k, num_groups)

    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q = _empty_token_bf16((max_m, k))
        self.rhs_q = _empty_block_bf16((num_groups, n, k))
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)
        self.topk_ids = torch.randint(
            0, num_groups, (max_m, 1), device="cuda", dtype=torch.int32
        )

    def execute(self, m):
        (
            config,
            m_rows,
            expert_ids_and_cumsum,
            sorted_token_ids,
            aligned_num_m_blocks,
            _,
            _,
        ) = deep_gemm.moe_align_block_size(
            self.lhs_q[:m], self.rhs_q, self.topk_ids[:m], perchannel_quant=False
        )
        deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_fused(
            self.lhs_q[:m],
            self.rhs_q,
            self.out[:m],
            m_rows,
            expert_ids_and_cumsum,
            sorted_token_ids,
            aligned_num_m_blocks,
            config,
        )


class _GroupedFusedWarmupExecutor_fp8_channel(_BaseWarmupExecutor):
    """Warmup executor for GROUPED_GEMM_NT_F8F8BF16_FUSED_CHANNEL."""

    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        super().__init__(max_m, n, k, num_groups)

    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q, self.lhs_s = _empty_token_fp8_channel((max_m, k))
        self.rhs_q, self.rhs_s = _empty_block_fp8_channel((num_groups, n, k))
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)
        self.topk_ids = torch.randint(
            0, num_groups, (max_m, 1), device="cuda", dtype=torch.int32
        )

    def execute(self, m):
        (
            config,
            m_rows,
            expert_ids_and_cumsum,
            sorted_token_ids,
            aligned_num_m_blocks,
            _,
            _,
        ) = deep_gemm.moe_align_block_size(
            self.lhs_q[:m], self.rhs_q, self.topk_ids[:m], perchannel_quant=True
        )
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_fused(
            (self.lhs_q[:m], self.lhs_s[:m]),
            (self.rhs_q, self.rhs_s),
            self.out[:m],
            m_rows,
            expert_ids_and_cumsum,
            sorted_token_ids,
            aligned_num_m_blocks,
            config,
        )


class _GroupedFusedWarmupExecutor_fp4(_BaseWarmupExecutor):
    """Warmup executor for GROUPED_GEMM_NT_F4F4BF16_FUSED."""

    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        super().__init__(max_m, n, k, num_groups)

    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q, self.lhs_s = _empty_token_uint8((max_m, k))
        self.rhs_q, self.rhs_s = _empty_token_uint8((num_groups, n, k))
        ### lhs_s is uint16 k major(contiguous) for DG fused_moe
        self.lhs_s = self.lhs_s.view(torch.uint16)
        self.rhs_s = deep_gemm.preprocess_mxfp4_scales(self.rhs_s)
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)
        self.topk_ids = torch.randint(
            0, num_groups, (max_m, 1), device="cuda", dtype=torch.int32
        )

    def execute(self, m):
        (
            config,
            m_rows,
            expert_ids_and_cumsum,
            sorted_token_ids,
            aligned_num_m_blocks,
            _,
            _,
        ) = deep_gemm.moe_align_block_size(
            self.lhs_q[:m], self.rhs_q, self.topk_ids[:m], perchannel_quant=False
        )
        deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_fused(
            (self.lhs_q[:m], self.lhs_s[:m]),
            (self.rhs_q, self.rhs_s),
            self.out[:m],
            m_rows,
            expert_ids_and_cumsum,
            sorted_token_ids,
            aligned_num_m_blocks,
            config,
        )
