from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple, Optional, Tuple

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    get_device_properties,
    get_jit_cuda_arch,
    get_max_vector_bytes,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.kernels.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from tvm_ffi.module import Module


logger = logging.getLogger(__name__)


@cache_once
def _jit_qknorm_module(head_dim: int, dtype: torch.dtype) -> Module:
    args = make_cpp_args(head_dim, is_arch_support_pdl(), dtype)
    return load_jit(
        "qknorm",
        *args,
        cuda_files=["elementwise/qknorm.cuh"],
        cuda_wrappers=[("qknorm", f"QKNormKernel<{args}>::run")],
    )


_RMSNORM_MAX_HIDDEN_SIZE = 16384


def _next_pow_2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _div_ceil(a: int, b: int) -> int:
    return -(-a // b)


class Schedule(NamedTuple):
    vec_size: int  # elements per vectorized access
    num_threads: int  # threads cooperating on one row
    copy_mode: int = 0  # 0=reg, 1=cp.async, 2=TMA


def is_jit_rmsnorm_supported(hidden_size: int) -> bool:
    return 0 < hidden_size <= _RMSNORM_MAX_HIDDEN_SIZE and hidden_size % 2 == 0


@cache_once
def _schedule_rmsnorm(dim: int, dtype_bytes: int) -> Tuple[Schedule, Schedule, int]:
    dim_bytes = dim * dtype_bytes
    if dim_bytes % 32 != 0:
        # NOTE: slow path. Each thread 16 elem
        assert dim % 2 == 0
        schedule = Schedule(2, num_threads=dim // 16)
        return schedule, schedule, -1

    max_vec_bytes = get_max_vector_bytes()
    assert max_vec_bytes in (16, 32)
    max_vec_size = max_vec_bytes // dtype_bytes
    min_vec_size = 16 // dtype_bytes

    def _try_vectorize(schedule: Schedule) -> Schedule:
        if dim % schedule.num_threads != 0 or schedule.copy_mode != 0:
            return schedule
        vec_size = schedule.vec_size
        dim_per_thread = dim // schedule.num_threads
        while vec_size < max_vec_size and dim_per_thread % vec_size == 0:
            vec_size *= 2
        return Schedule(vec_size, schedule.num_threads, copy_mode=0)

    # norm fit into 1 warp
    if dim_bytes <= 2048:  # at most 64B / thread
        # try 32B per thread if possible; always fit into warp
        num_threads = min(_next_pow_2(dim_bytes // 32), 32)
        assert num_threads in (1, 2, 4, 8, 16, 32)
        schedule = Schedule(min_vec_size, num_threads=num_threads)
        schedule = _try_vectorize(schedule)
        return schedule, schedule, -1

    props = get_device_properties()
    cc_major = get_jit_cuda_arch().major

    num_sm = props.multi_processor_count
    threads_per_sm = props.max_threads_per_multi_processor

    # 1. choose low latency config
    ll_bytes = 32 if dim_bytes > 16384 else 16
    num_threads_ll = _div_ceil(dim_bytes // ll_bytes, 32) * 32
    threshold = (threads_per_sm // num_threads_ll) * num_sm
    # NOTE: DO NOT vectorize LL mode
    schedule_ll = Schedule(min_vec_size, num_threads=num_threads_ll)

    # 2. choose high throughput config
    if cc_major >= 8 and dim_bytes >= 4096:
        copy_mode = 2 if cc_major >= 9 else 1  # use TMA after sm90
        schedule_tput = Schedule(min_vec_size, num_threads=128, copy_mode=copy_mode)
    else:
        if dim_bytes <= 3072:
            num_threads = 32  # NOTE: 32thr x 16B x 6
        elif dim_bytes <= 4096:
            num_threads = 64  # NOTE: 64thr x 16B x 4
        elif dim_bytes <= 10240:
            num_threads = 128  # NOTE: 128thr x 16B x 5
        elif dim_bytes <= 16384:
            num_threads = 256  # NOTE: 256thr x 16B x 4
        else:
            num_threads = _div_ceil(dim_bytes // 64, 64) * 64  # NOTE: 64B = 16B x 4
        schedule_tput = Schedule(min_vec_size, num_threads=num_threads)
        schedule_tput = _try_vectorize(schedule_tput)
    return schedule_ll, schedule_tput, threshold


@cache_once
def _schedule_fused_add_rmsnorm(dim: int, dtype_bytes: int) -> Schedule:
    """The old C++ launcher's shape: one widest-possible vector per thread.

    Unlike :func:`_schedule_rmsnorm` there is no latency/throughput split --
    the old kernel had a single form and picked it regardless of batch size.
    """
    dim_bytes = dim * dtype_bytes
    if dim_bytes % 32 != 0:
        # NOTE: slow path. Each thread 16 elem
        assert dim % 2 == 0
        return Schedule(2, num_threads=dim // 16)

    vec_size = get_max_vector_bytes() // dtype_bytes
    num_threads = _div_ceil(dim_bytes // 32, 32) * 32
    if num_threads > 512:
        while num_threads >= 512:
            num_threads //= 2
    num_threads = _div_ceil(num_threads, 32) * 32
    return Schedule(vec_size, num_threads=num_threads)


@cache_once
def _jit_rmsnorm_module(
    hidden_size: int,
    dtype: torch.dtype,
    cast_x_before_out_mul: bool,
    schedule: Schedule,
) -> Module:
    args = make_cpp_args(
        dtype,
        hidden_size,
        is_arch_support_pdl(),
        cast_x_before_out_mul,
        *schedule,
    )
    return load_jit(
        "rmsnorm",
        *args,
        cuda_files=["elementwise/rmsnorm.cuh"],
        cuda_wrappers=[("rmsnorm", f"RMSNormKernel<{args}>::run")],
    )


@cache_once
def _jit_fused_add_rmsnorm_module(
    dim: int,
    dtype: torch.dtype,
    cast_x_before_out_mul: bool,
    schedule: Schedule,
) -> Module:
    args = make_cpp_args(
        dtype, dim, is_arch_support_pdl(), cast_x_before_out_mul, *schedule
    )
    return load_jit(
        "fused_add_rmsnorm",
        *args,
        cuda_files=["elementwise/rmsnorm.cuh"],
        cuda_wrappers=[("fused_add_rmsnorm", f"FusedAddRMSNormKernel<{args}>::run")],
    )


@cache_once
def _jit_qknorm_across_heads_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "qknorm_across_heads",
        *args,
        cuda_files=["elementwise/qknorm_across_heads.cuh"],
        cuda_wrappers=[
            ("qknorm_across_heads", f"QKNormAcrossHeadsKernel<{args}>::run")
        ],
    )


@torch.compiler.assume_constant_result
@cache_once
def can_use_fused_inplace_qknorm(head_dim: int, dtype: torch.dtype) -> bool:
    if head_dim not in [64, 128, 256, 512, 1024]:
        logger.warning(f"Unsupported head_dim={head_dim} for JIT QK-Norm kernel")
        return False
    try:
        _jit_qknorm_module(head_dim, dtype)
        return True
    except Exception as e:
        logger.warning(f"Failed to load JIT QK-Norm kernel: {e}")
        return False


@debug_kernel_api
def fused_inplace_qknorm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    head_dim: int = 0,
) -> None:
    head_dim = head_dim or q.size(-1)
    module = _jit_qknorm_module(head_dim, q.dtype)
    module.qknorm(q, k, q_weight, k_weight, eps)


@debug_kernel_api
def rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
    *,
    cast_x_before_out_mul: bool = False,
) -> torch.Tensor:
    """``(input / RMS(input)) * weight``, returned as a new tensor.

    Out-of-place: ``input`` is left untouched unless it is also passed as
    ``out``. Note the contrast with :func:`fused_add_rmsnorm`, which writes
    both of its inputs in place and returns ``None``.

    :param out: destination tensor; allocated when omitted.
    :param cast_x_before_out_mul: round the normalized value to the activation
        dtype *before* the weight multiply (HuggingFace ``LlamaRMSNorm``
        semantics) instead of keeping that multiply in fp32.
    """
    if out is None:
        out = torch.empty_like(input)
    num_tokens, hidden_size = input.size()
    *schedules, threshold = _schedule_rmsnorm(hidden_size, input.element_size())
    module = _jit_rmsnorm_module(
        hidden_size,
        input.dtype,
        cast_x_before_out_mul,
        schedules[0 if num_tokens <= threshold else 1],
    )
    module.rmsnorm(input, weight, out, eps)
    return out


@debug_kernel_api
def fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    cast_x_before_out_mul: bool = False,
) -> None:
    hidden_size = weight.size(-1)
    module = _jit_fused_add_rmsnorm_module(
        hidden_size,
        input.dtype,
        cast_x_before_out_mul,
        _schedule_fused_add_rmsnorm(hidden_size, input.element_size()),
    )
    module.fused_add_rmsnorm(input, residual, weight, eps)


@debug_kernel_api
def fused_inplace_qknorm_across_heads(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float = 1e-6,
) -> None:
    """
    Fused inplace QK normalization across all heads.

    Args:
        q: Query tensor of shape [batch_size, num_heads * head_dim]
        k: Key tensor of shape [batch_size, num_heads * head_dim]
        q_weight: Query weight tensor of shape [num_heads * head_dim]
        k_weight: Key weight tensor of shape [num_heads * head_dim]
        eps: Epsilon for numerical stability
    """
    module = _jit_qknorm_across_heads_module(q.dtype)
    module.qknorm_across_heads(q, k, q_weight, k_weight, eps)
