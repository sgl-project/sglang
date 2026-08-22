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
    """Whether the schedules below can cover `hidden_size`.

    The row has to divide into 32-byte units: every schedule hands each thread
    a whole number of 16-byte vectors, and `_try_vectorize` may widen that to
    32. 16 elements is the strictest form of that over the supported dtypes
    (fp16/bf16 need 16, fp32 needs 8), so it is the single gate.
    """
    return 0 < hidden_size <= _RMSNORM_MAX_HIDDEN_SIZE and hidden_size % 16 == 0


@cache_once
def _schedule_rmsnorm(dim: int, dtype_bytes: int) -> Tuple[Schedule, Schedule, int]:
    dim_bytes = dim * dtype_bytes
    assert (
        dim_bytes % 32 == 0
    ), f"{dim=} is not schedulable; see is_jit_rmsnorm_supported"

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
        # `cp.async` needs sm80. The bulk copy needs sm90 and is limited to
        # the arches where it is known to behave: sm120's support for it is
        # poor enough that a kernel using it reserves gigabytes of extra device
        # memory, which buys a couple of percent over cp.async at the deepest
        # tiles. Newer or unmeasured arches take the per-thread copy too.
        copy_mode = 2 if cc_major in (9, 10) else 1
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
    assert (
        dim_bytes % 32 == 0
    ), f"{dim=} is not schedulable; see is_jit_rmsnorm_supported"

    vec_size = get_max_vector_bytes() // dtype_bytes
    num_threads = _div_ceil(dim_bytes // 32, 32) * 32
    if num_threads > 512:
        while num_threads >= 512:
            num_threads //= 2
    num_threads = _div_ceil(num_threads, 32) * 32
    return Schedule(vec_size, num_threads=num_threads)


def _warmup_module(module, kind: str, dim: int, dtype: torch.dtype) -> None:
    """Launch `module` once so CUDA loads its cubin now rather than later.

    Module loading is lazy: the cubin is brought into the context by the *first*
    launch, and that allocation can fail once a server has sized its KV cache to
    fill the device. A schedule that only runs at large batch would first launch
    mid-serving and take the engine down with `CUDA error: out of memory`, so
    every variant is made resident here, while the caller is still warming up.
    """
    if torch.cuda.is_current_stream_capturing():
        # allocating here would land in the graph's pool and the launch would be
        # recorded into the graph; leave the load lazy rather than corrupt it
        return
    x = torch.zeros(1, dim, dtype=dtype, device="cuda")
    weight = torch.zeros(dim, dtype=dtype, device="cuda")
    if kind == "rmsnorm":
        module.rmsnorm(x, weight, torch.empty_like(x), 1e-6)
    else:
        module.fused_add_rmsnorm(x, torch.zeros_like(x), weight, 1e-6)


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
    module = load_jit(
        "rmsnorm",
        *args,
        cuda_files=["elementwise/rmsnorm.cuh"],
        cuda_wrappers=[("rmsnorm", f"RMSNormKernel<{args}>::run")],
    )
    _warmup_module(module, "rmsnorm", hidden_size, dtype)
    return module


@cache_once
def _jit_rmsnorm_modules(
    hidden_size: int,
    dtype: torch.dtype,
    cast_x_before_out_mul: bool,
    schedules: Tuple[Schedule, ...],
) -> Tuple[Module, ...]:
    """Build every schedule this hidden size can dispatch to, in one go.

    The two schedules are selected by token count, so the throughput one is
    first needed part way through CUDA graph capture -- which is where its JIT
    compile would land, inflating the captured graph. Compiling both on the
    first call keeps that off the capture path.
    """
    return tuple(
        _jit_rmsnorm_module(hidden_size, dtype, cast_x_before_out_mul, schedule)
        for schedule in schedules
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
    module = load_jit(
        "fused_add_rmsnorm",
        *args,
        cuda_files=["elementwise/rmsnorm.cuh"],
        cuda_wrappers=[("fused_add_rmsnorm", f"FusedAddRMSNormKernel<{args}>::run")],
    )
    _warmup_module(module, "fused_add_rmsnorm", dim, dtype)
    return module


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
    modules = _jit_rmsnorm_modules(
        hidden_size, input.dtype, cast_x_before_out_mul, tuple(schedules)
    )
    module = modules[0 if num_tokens <= threshold else 1]
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
