from __future__ import annotations

import enum
import functools
import logging
from typing import TYPE_CHECKING, NamedTuple, Optional, Tuple

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    get_jit_cuda_arch,
    is_arch_support_pdl,
    is_hip_runtime,
    is_musa_runtime,
    load_jit,
    make_cpp_args,
    parallel_compile,
)
from sglang.kernels.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from tvm_ffi.module import Module


logger = logging.getLogger(__name__)


@cache_once
def _jit_qknorm_module(
    head_dim: int, dtype: torch.dtype, half_warp: bool = True
) -> Module:
    """``half_warp=False`` keeps the one-row-per-warp kernel where the two-rows
    variant would apply; tests use it as the bit-exactness oracle."""
    args = make_cpp_args(head_dim, is_arch_support_pdl(), dtype, half_warp)
    return load_jit(
        "qknorm",
        *args,
        cuda_files=["elementwise/qknorm.cuh"],
        cuda_wrappers=[("qknorm", f"QKNormKernel<{args}>::run")],
    )


class RMSNormSchedule(NamedTuple):
    """One JIT instantiation of the rmsnorm template (see `elementwise/rmsnorm.cuh`)."""

    vec_bytes: int  # bytes per vector access: 32 on Blackwell, 16 before
    threads_per_row: int  # whole warps per row
    rows_per_cta: int  # rows handled by one CTA


class RMSNormMode(enum.Enum):
    PLAIN = enum.auto()  # out = norm(x) * w
    FUSED_ADD = enum.auto()  # residual = T(x + residual); x = norm(residual) * w
    QUANT = enum.auto()  # out (fp8 e4m3) = norm(x) * w / scale
    FUSED_ADD_QUANT = enum.auto()

    @property
    def cpp_name(self) -> str:
        """The C++ enumerator, resolved by `using enum RMSNormMode` in `rmsnorm.cuh`."""
        return self.name

    @property
    def fused_add(self) -> bool:
        return self in (RMSNormMode.FUSED_ADD, RMSNormMode.FUSED_ADD_QUANT)


_RMSNORM_MAX_HIDDEN_SIZE = 16384


def is_supported_rmsnorm_hidden_size(hidden_size: int) -> bool:
    # whole 32 B vectors; tuned for the sizes in the schedule table, generic elsewhere
    return hidden_size % 16 == 0 and 32 <= hidden_size <= _RMSNORM_MAX_HIDDEN_SIZE


def _div_ceil(a: int, b: int) -> int:
    return -(-a // b)


# Measured on B200
_RMSNORM_SCHEDULE_OVERRIDES = {
    1536: ((96, 1), (96, 2)),
    3072: ((192, 1), (64, 1)),
}
_RMSNORM_MAX_THREADS = 256
_RMSNORM_MAX_THREADS_FUSED = 512


@cache_once
def _schedule_rmsnorm(
    dim: int,
    dtype_bytes: int,
    fused_add: bool,
) -> Tuple[RMSNormSchedule, RMSNormSchedule, int]:
    """(decode schedule, prefill schedule, token threshold)."""

    def _rmsnorm_vec_bytes() -> int:
        if is_hip_runtime() or is_musa_runtime():
            return 16
        return 32 if get_jit_cuda_arch().major >= 10 else 16

    def _threads_per_row(num_vecs: int, vecs_per_thread: int, max_threads: int) -> int:
        if num_vecs < 32:
            # a row shorter than a warp takes a power-of-two slice of it
            # only one vector per thread
            return 1 << (num_vecs - 1).bit_length()
        threads = _div_ceil(_div_ceil(num_vecs, vecs_per_thread), 32) * 32
        return min(max_threads, threads)

    vec_bytes = _rmsnorm_vec_bytes()
    num_vecs = _div_ceil(dim * dtype_bytes, vec_bytes)
    threads = _threads_per_row(num_vecs, 2, _RMSNORM_MAX_THREADS)
    if num_vecs < 32:
        # several short rows share a warp; the CTA is always 4 warps, one vector per thread
        decode = prefill = (threads, 128 // threads)
    else:
        default_config = (threads, 1), (threads, max(1, min(8, 256 // threads)))
        decode, prefill = _RMSNORM_SCHEDULE_OVERRIDES.get(dim, default_config)
        if fused_add:
            # two streams per thread: one vector each at decode, at most two at prefill
            decode = (_threads_per_row(num_vecs, 1, _RMSNORM_MAX_THREADS_FUSED), 1)
            threads = _threads_per_row(num_vecs, 2, _RMSNORM_MAX_THREADS_FUSED)
            if threads > prefill[0]:
                prefill = (threads, max(1, min(8, 256 // threads)))
    # small rows switch to the packed form earlier: at D <= 1536 the one-row CTA loses from N = 256
    threshold = 128 if dim <= 1536 else 512
    return (
        RMSNormSchedule(vec_bytes, *decode),
        RMSNormSchedule(vec_bytes, *prefill),
        threshold,
    )


@cache_once
def _jit_rmsnorm_module(
    dim: int,
    dtype: torch.dtype,
    mode: RMSNormMode,
    cast_x_before_out_mul: bool,
    schedule: RMSNormSchedule,
) -> Module:
    args = make_cpp_args(
        dtype,
        dim,
        mode.cpp_name,
        cast_x_before_out_mul,
        is_arch_support_pdl(),
        schedule.vec_bytes,
        schedule.threads_per_row,
        schedule.rows_per_cta,
    )
    return load_jit(
        "rmsnorm",
        *args,
        cuda_files=["elementwise/rmsnorm.cuh"],
        cuda_wrappers=[("rmsnorm", f"RMSNormKernel<{args}>::run")],
    )


class RMSNormDispatch:
    """The two schedules of one (hidden size, dtype, mode, HF) signature, picked by token count."""

    def __init__(self, decode: Module, prefill: Module, threshold: int):
        self._decode = decode
        self._prefill = prefill
        self._threshold = threshold

    def __call__(self, num_tokens: int) -> Module:
        return self._decode if num_tokens <= self._threshold else self._prefill


@cache_once
def precompile_rmsnorm(
    dim: int,
    dtype: torch.dtype,
    mode: RMSNormMode,
    cast_x_before_out_mul: bool,
) -> RMSNormDispatch:
    """Build both schedules of a signature at once, so a token count seen for the first time
    (e.g. inside a CUDA graph capture) never compiles."""
    decode, prefill, threshold = _schedule_rmsnorm(dim, dtype.itemsize, mode.fused_add)
    modules = parallel_compile(
        functools.partial(
            _jit_rmsnorm_module, dim, dtype, mode, cast_x_before_out_mul, s
        )
        for s in dict.fromkeys((decode, prefill))
    )
    return RMSNormDispatch(modules[0], modules[-1], threshold)


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
    out: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    *,
    cast_x_before_out_mul: bool = False,
) -> None:
    out = out if out is not None else input
    hidden_size = input.size(-1)
    dispatch = precompile_rmsnorm(
        hidden_size, input.dtype, RMSNormMode.PLAIN, cast_x_before_out_mul
    )
    dispatch(input.size(0)).rmsnorm(input, weight, out, None, None, eps)


@debug_kernel_api
def fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    cast_x_before_out_mul: bool = False,
) -> None:
    """In place: residual = T(input + residual); input = rmsnorm(residual) * weight."""
    dispatch = precompile_rmsnorm(
        input.size(-1), input.dtype, RMSNormMode.FUSED_ADD, cast_x_before_out_mul
    )
    dispatch(input.size(0)).rmsnorm(input, weight, None, residual, None, eps)


@debug_kernel_api
def rmsnorm_quant(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    out: torch.Tensor,
    eps: float = 1e-6,
) -> None:
    """out (fp8 e4m3) = sat(rmsnorm(input) * weight / scale); scale is a device fp32 scalar."""
    dispatch = precompile_rmsnorm(input.size(-1), input.dtype, RMSNormMode.QUANT, False)
    dispatch(input.size(0)).rmsnorm(input, weight, out, None, scale, eps)


@debug_kernel_api
def fused_add_rmsnorm_quant(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    out: torch.Tensor,
    eps: float = 1e-6,
) -> None:
    """In place residual = T(input + residual); out (fp8) = sat(rmsnorm(residual) * weight / scale)."""
    dispatch = precompile_rmsnorm(
        input.size(-1), input.dtype, RMSNormMode.FUSED_ADD_QUANT, False
    )
    dispatch(input.size(0)).rmsnorm(input, weight, out, residual, scale, eps)
