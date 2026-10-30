from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    get_activation_cuda_cflags,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.kernels.jit.utils.common import empty_sentinel, is_sail_runtime

if TYPE_CHECKING:
    from tvm_ffi.module import Module


def _make_name(*args):
    return "kimi_k3_" + "_".join(str(a) for a in args)


@cache_once
def _jit_situ_and_mul_module(in_dtype: torch.dtype, out_dtype: torch.dtype) -> Module:
    """Compile and cache the JIT SiTU-and-mul module for an (in, out) dtype pair."""
    args = make_cpp_args(in_dtype, out_dtype, is_arch_support_pdl())
    return load_jit(
        _make_name("situ_and_mul"),
        *args,
        cuda_files=["kimi_k3/situ_and_mul.cuh"],
        cuda_wrappers=[("run", f"SituAndMulKernel<{args}>::run")],
        extra_cuda_cflags=get_activation_cuda_cflags(),
    )


def situ_and_mul(
    input: torch.Tensor,
    out: Optional[torch.Tensor],
    beta: float,
    linear_beta: Optional[float],
) -> torch.Tensor:
    """Fused SiTU (SoftCap-GLU) activation.

    gate_out = beta * tanh(gate / beta) * sigmoid(gate)
    up_out   = linear_beta * tanh(up / linear_beta)  [if linear_beta is not None]
    output   = gate_out * up_out

    Parameters
    ----------
    input       : bf16 or fp32 CUDA tensor [*, 2*D]
    out         : optional pre-allocated CUDA tensor [*, D]; its dtype selects
                  the output dtype (bf16 for an fp32 input)
    beta        : gate softcap scalar (e.g. 4.0)
    linear_beta : up softcap scalar (e.g. 25.0), or None to skip
    """
    hidden_size = input.shape[-1] // 2
    if out is None:
        out_dtype = torch.bfloat16 if input.dtype == torch.float32 else input.dtype
        out = input.new_empty(*input.shape[:-1], hidden_size, dtype=out_dtype)

    # 2D inputs may be row-strided (e.g. a slice of a fused-GEMM output) when
    # the row stride is a whole number of the kernel's 32-byte vectors of the
    # wider dtype; other inputs keep the dense-view path.
    widest = max(input.element_size(), out.element_size())
    if input.dim() == 2 and input.stride(1) == 1 and input.stride(0) * widest % 32 == 0:
        input_2d = input
    else:
        input_2d = input.contiguous().view(-1, hidden_size * 2)
    out_2d = out.view(-1, hidden_size)

    has_linear_beta = linear_beta is not None
    module = _jit_situ_and_mul_module(input_2d.dtype, out_2d.dtype)
    module.run(
        input_2d,
        out_2d,
        float(beta),
        float(linear_beta) if has_linear_beta else 0.0,
        has_linear_beta,
    )
    return out


@cache_once
def _jit_situ_masked_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        _make_name("situ_and_mul_masked"),
        *args,
        cuda_files=["kimi_k3/situ_and_mul.cuh"],
        cuda_wrappers=[("run", f"SituAndMulMaskedKernel<{args}>::run")],
        extra_cuda_cflags=get_activation_cuda_cflags(),
    )


@cache_once
def _jit_situ_mxfp4_module(masked: bool) -> Module:
    if not is_sail_runtime() and (
        torch.version.hip is not None
        or not torch.cuda.is_available()
        or torch.cuda.get_device_capability()[0] < 10
    ):
        raise RuntimeError(
            "SiTU MXFP4 requires SAIL or an SM100+ CUDA device with FP4 instructions."
        )
    args = make_cpp_args(masked, is_arch_support_pdl())
    return load_jit(
        _make_name("situ_and_mul_mxfp4"),
        *args,
        cuda_files=["kimi_k3/situ_and_mul.cuh"],
        cuda_wrappers=[("run", f"SituAndMulMxfp4Kernel<{args}>::run")],
        extra_cuda_cflags=[*get_activation_cuda_cflags(), "-DSGLANG_SITU_MXFP4"],
    )


def situ_and_mul_masked(
    input: torch.Tensor,
    out: Optional[torch.Tensor],
    masked_m: torch.Tensor,
    beta: float,
    linear_beta: Optional[float],
    topk: int = 8,
    expected_m: Optional[int] = None,
) -> torch.Tensor:
    """Only the valid rows selected by masked_m are written; expected_m is just a
    scheduling hint."""
    if out is None:
        out = input.new_empty(*input.shape[:-1], input.shape[-1] // 2)
    _jit_situ_masked_module().run(
        input,
        out,
        masked_m,
        topk,
        int(expected_m) if expected_m is not None else int(input.shape[1]),
        float(beta),
        float(linear_beta) if linear_beta is not None else 0.0,
        linear_beta is not None,
    )
    return out


def situ_and_mul_post_quant_mxfp4(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    beta: float,
    linear_beta: Optional[float],
) -> None:
    """SiTU rounds to BF16 before quantizing; scales are physically [D/64, T] uint16."""
    _jit_situ_mxfp4_module(False).run(
        input,
        output,
        output_scale,
        empty_sentinel(input.device, torch.int32),
        1,
        int(input.shape[0]),
        float(beta),
        float(linear_beta) if linear_beta is not None else 0.0,
        linear_beta is not None,
    )


def situ_and_mul_masked_post_quant_mxfp4(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    masked_m: torch.Tensor,
    beta: float,
    linear_beta: Optional[float],
    topk: int = 8,
    expected_m: Optional[int] = None,
) -> None:
    """Masked SiTU+MXFP4 that preserves padding; scales are [E, D/64, T] uint16."""
    _jit_situ_mxfp4_module(True).run(
        input,
        output,
        output_scale,
        masked_m,
        topk,
        int(expected_m) if expected_m is not None else int(input.shape[1]),
        float(beta),
        float(linear_beta) if linear_beta is not None else 0.0,
        linear_beta is not None,
    )
