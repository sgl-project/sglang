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
)
from sglang.srt.utils.common import get_cuda_version

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


def _validate_inputs(input: torch.Tensor) -> None:
    """Guard the two contract breaks the C++ launcher cannot report cleanly.

    A non-rank-2 input has no well-defined hidden size, and an unsupported
    hidden size would otherwise surface as an nvcc ``static_assert`` compile
    error. Every other tensor requirement (dtype, device, stride, pointer
    alignment, and the full weight/output contract) is verified by the C++
    launcher via ``TensorMatcher`` and ``RuntimeCheck``.
    """
    if input.dim() != 2:
        raise ValueError(f"input must be rank 2, got shape {tuple(input.shape)}")
    hidden_size = input.shape[1]
    if not is_supported_rmsnorm_per_token_group_quant_fp8_hidden_size(hidden_size):
        raise ValueError(
            f"hidden size must be a multiple of {_GROUP_SIZE} in "
            f"[{_GROUP_SIZE}, {_MAX_HIDDEN_SIZE}], got {hidden_size}"
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


def rmsnorm_per_token_group_quant_fp8_out(
    input: torch.Tensor,
    weight: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    output_norm: torch.Tensor,
    eps: float = 1e-5,
) -> None:
    """Run the fused JIT kernel, writing into the caller's output tensors.

    The model-side gate (``can_use_rmsnorm_per_token_group_quant_fp8``) decides
    eligibility before this runs, and the DeepSeek MLA caller never enters this
    path under ``torch.compile``. The C++ launcher verifies the shapes, dtypes,
    strides, and pointer alignment of every tensor.
    """
    _validate_inputs(input)
    if input.shape[0] == 0:
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
    """Return RMSNorm and packed-UE8M0 FP8 outputs under the BF16 contract."""
    _validate_inputs(input)
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
