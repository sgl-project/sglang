"""Fused BF16 RMSNorm + packed-UE8M0 FP8 group quantization.

The public API accepts any supported reduction size. Each static size gets a
JIT specialization so the CUDA kernel can retain one input vector per thread
across the RMS reduction. Its narrow quantization path is self-contained so
the existing quant-v2 kernel remains untouched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    get_jit_cuda_arch,
    is_arch_support_pdl,
    is_hip_runtime,
    load_jit,
    make_cpp_args,
)
from sglang.kernels.ops.quantization.fp8_kernel import (
    create_per_token_group_quant_fp8_output_scale,
    sglang_per_token_group_quant_fp8,
)
from sglang.srt.utils.common import get_cuda_version
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_GROUP_SIZE = 128
_MAX_HIDDEN_SIZE = 16384
# Keep this synchronized with SGL_ARCH_BLACKWELL_OR_GREATER in
# include/sgl_kernel/utils.cuh: this kernel instantiates a 32-byte
# AlignedVector, which that JIT ABI enables starting with CUDA 12.9.
_MIN_CUDA_VERSION = (12, 9)


def is_supported_rmsnorm_per_token_group_quant_fp8_hidden_size(
    hidden_size: int,
) -> bool:
    return (
        hidden_size >= _GROUP_SIZE
        and hidden_size % _GROUP_SIZE == 0
        and hidden_size <= _MAX_HIDDEN_SIZE
    )


def _is_supported_blackwell_runtime() -> bool:
    """Whether the JIT target enables the 32-byte Blackwell vector ABI."""
    return bool(
        torch.cuda.is_available()
        and not is_hip_runtime()
        and get_jit_cuda_arch().major == 10
        and get_cuda_version() >= _MIN_CUDA_VERSION
    )


def _require_supported_blackwell_runtime() -> None:
    if not _is_supported_blackwell_runtime():
        raise RuntimeError(
            "rmsnorm_per_token_group_quant_fp8 requires an NVIDIA SM10x "
            "Blackwell GPU and CUDA 12.9 or later"
        )


def _validate_inputs(input: torch.Tensor, weight: torch.Tensor) -> None:
    if input.dim() != 2:
        raise ValueError(f"input must be rank 2, got shape {tuple(input.shape)}")
    if input.dtype != torch.bfloat16:
        raise TypeError(f"input must be bfloat16, got {input.dtype}")
    if not input.is_cuda:
        raise ValueError("input must be a CUDA tensor")

    hidden_size = input.shape[1]
    if not is_supported_rmsnorm_per_token_group_quant_fp8_hidden_size(hidden_size):
        raise ValueError(
            f"hidden size must be a multiple of {_GROUP_SIZE} in "
            f"[{_GROUP_SIZE}, {_MAX_HIDDEN_SIZE}], got {hidden_size}"
        )
    if input.stride(1) != 1:
        raise ValueError("input must have a contiguous hidden dimension")
    if input.stride(0) % 16 != 0:
        raise ValueError("input rows must preserve 32-byte alignment")

    if tuple(weight.shape) != (hidden_size,):
        raise ValueError(
            f"weight must have shape ({hidden_size},), got {tuple(weight.shape)}"
        )
    if weight.dtype != torch.bfloat16:
        raise TypeError(f"weight must be bfloat16, got {weight.dtype}")
    if weight.device != input.device:
        raise ValueError("input and weight must be on the same CUDA device")
    if weight.stride(0) != 1:
        raise ValueError("weight must be contiguous")


def _is_fused_kernel_pointer_aligned(input: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether the live pointers satisfy the fused kernel's vector ABI."""
    return input.data_ptr() % 32 == 0 and weight.data_ptr() % 32 == 0


def _run_unfused_rmsnorm_quant_out(
    input: torch.Tensor,
    weight: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    output_norm: torch.Tensor,
    eps: float,
) -> None:
    """Run the ordinary RMSNorm + CUDA UE8M0 quant pair for unaligned views."""
    # sgl_kernel.rmsnorm itself requires 128-byte-aligned input and weight
    # pointers, so it cannot implement the fallback for the views routed here.
    normalized = torch.nn.functional.rms_norm(
        input,
        (input.shape[1],),
        weight,
        eps,
    )
    output_norm.copy_(normalized)
    quantized, scale = sglang_per_token_group_quant_fp8(
        normalized,
        _GROUP_SIZE,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )
    output_q.copy_(quantized)
    output_s.copy_(scale)


def _validate_outputs(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    output_norm: torch.Tensor,
) -> None:
    num_tokens, hidden_size = input.shape
    packed_scale_cols = (hidden_size // _GROUP_SIZE + 3) // 4

    if (
        tuple(output_q.shape) != tuple(input.shape)
        or output_q.dtype != torch.float8_e4m3fn
        or output_q.device != input.device
        or not output_q.is_contiguous()
        or output_q.data_ptr() % 16 != 0
    ):
        raise ValueError(
            "output_q must be a contiguous, 16-byte-aligned FP8 tensor "
            "matching input shape and device"
        )
    if (
        tuple(output_s.shape) != (num_tokens, packed_scale_cols)
        or output_s.dtype != torch.int32
        or output_s.device != input.device
        or output_s.stride(0) != 1
        or (
            num_tokens > 0
            and (output_s.stride(1) < num_tokens or output_s.stride(1) % 4 != 0)
        )
    ):
        raise ValueError(
            "output_s must use the packed column-major/TMA-aligned int32 layout"
        )
    if (
        tuple(output_norm.shape) != tuple(input.shape)
        or output_norm.dtype != torch.bfloat16
        or output_norm.device != input.device
        or not output_norm.is_contiguous()
        or output_norm.data_ptr() % 32 != 0
    ):
        raise ValueError(
            "output_norm must be a contiguous, 32-byte-aligned BF16 tensor "
            "matching input shape and device"
        )


@cache_once
def _jit_module(hidden_size: int, use_pdl: bool) -> Module:
    args = make_cpp_args(hidden_size, use_pdl)
    return load_jit(
        "rmsnorm_per_token_group_quant_fp8",
        *args,
        cuda_files=["gemm/rmsnorm_per_token_group_quant_fp8.cuh"],
        cuda_wrappers=[
            (
                "rmsnorm_per_token_group_quant_fp8",
                f"RMSNormPerTokenGroupQuantFP8Kernel<{args}>::run",
            )
        ],
        # Match quant-v2's arithmetic and keep the independent RMS reduction
        # lightweight now that bit-exact parity with another norm topology is
        # not part of this specialization's contract.
        extra_cuda_cflags=["--use_fast_math"],
    )


def can_use_rmsnorm_per_token_group_quant_fp8(
    input_dtype: torch.dtype, hidden_size: int
) -> bool:
    """Return whether the static runtime metadata supports this kernel."""
    if (
        input_dtype != torch.bfloat16
        or not is_supported_rmsnorm_per_token_group_quant_fp8_hidden_size(hidden_size)
    ):
        return False
    return _is_supported_blackwell_runtime()


@register_custom_op(
    op_name="rmsnorm_per_token_group_quant_fp8_out",
    mutates_args=["output_q", "output_s", "output_norm"],
)
def rmsnorm_per_token_group_quant_fp8_out(
    input: torch.Tensor,
    weight: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    output_norm: torch.Tensor,
    eps: float = 1e-5,
) -> None:
    """Custom-op boundary selecting fused or ordinary CUDA kernels per call."""
    _require_supported_blackwell_runtime()
    _validate_inputs(input, weight)
    _validate_outputs(input, output_q, output_s, output_norm)
    if input.shape[0] == 0:
        return
    if not _is_fused_kernel_pointer_aligned(input, weight):
        _run_unfused_rmsnorm_quant_out(
            input,
            weight,
            output_q,
            output_s,
            output_norm,
            eps,
        )
        return
    module = _jit_module(input.shape[1], is_arch_support_pdl())
    module.rmsnorm_per_token_group_quant_fp8(
        input, weight, output_q, output_s, output_norm, eps
    )


def rmsnorm_per_token_group_quant_fp8(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return RMSNorm and packed-UE8M0 outputs under the BF16 contract.

    Aligned inputs use the fused JIT kernel. Unaligned storage views use the
    ordinary RMSNorm plus CUDA group-quantization pair. This decision remains
    inside the opaque custom op so every eager or compiled invocation observes
    its live tensor pointers. Unsupported hardware and malformed tensor
    metadata are rejected before JIT compilation; the C++ launcher retains the
    fused path's ABI checks independently.
    """
    if not torch.compiler.is_compiling():
        _require_supported_blackwell_runtime()
        _validate_inputs(input, weight)
    output_q = torch.empty_like(input, dtype=torch.float8_e4m3fn)
    output_s = create_per_token_group_quant_fp8_output_scale(
        x_shape=input.shape,
        device=input.device,
        group_size=_GROUP_SIZE,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )
    output_norm = torch.empty_like(input, memory_format=torch.contiguous_format)
    rmsnorm_per_token_group_quant_fp8_out(
        input, weight, output_q, output_s, output_norm, eps
    )
    return output_q, output_s, output_norm


__all__ = [
    "can_use_rmsnorm_per_token_group_quant_fp8",
    "is_supported_rmsnorm_per_token_group_quant_fp8_hidden_size",
    "rmsnorm_per_token_group_quant_fp8",
    "rmsnorm_per_token_group_quant_fp8_out",
]
