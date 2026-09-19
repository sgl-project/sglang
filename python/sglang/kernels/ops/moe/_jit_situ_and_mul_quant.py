from __future__ import annotations

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)


def _make_name(*args):
    return "kimi_k3_" + "_".join(str(a) for a in args)


@cache_once
def _jit_situ_mul_quant_varlen_module(
    quant_group_size: int,
    scale_ue8m0: bool,
    swizzle: bool,
):
    args = make_cpp_args(
        quant_group_size,
        scale_ue8m0,
        swizzle,
        is_arch_support_pdl(),
    )
    return load_jit(
        _make_name("situ_mul_quant_varlen"),
        *args,
        cuda_files=["kimi_k3/situ_and_mul.cuh"],
        cuda_wrappers=[("run", f"SituAndMulMaskedPostQuantKernel<{args}>::run")],
        extra_cuda_cflags=["-use_fast_math"],
    )


def situ_and_mul_masked_post_quant(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    quant_group_size: int,
    masked_m: torch.Tensor,
    beta: float,
    linear_beta: float,
    scale_ue8m0: bool = False,
    topk: int = 8,
    transposed: bool = False,
    swizzle: bool = False,
) -> None:
    module = _jit_situ_mul_quant_varlen_module(quant_group_size, scale_ue8m0, swizzle)
    module.run(
        input,
        output,
        output_scale,
        masked_m,
        topk,
        transposed,
        float(beta),
        float(linear_beta),
    )


@cache_once
def _jit_situ_mul_quant_mxfp4_module():
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        _make_name("situ_mul_quant_mxfp4"),
        *args,
        cuda_files=["kimi_k3/situ_and_mul.cuh"],
        cuda_wrappers=[("run", f"SituAndMulPostQuantMxfp4Kernel<{args}>::run")],
        extra_cuda_cflags=["-use_fast_math"],
    )


@cache_once
def _jit_situ_mul_quant_mxfp4_varlen_module():
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        _make_name("situ_mul_quant_mxfp4_varlen"),
        *args,
        cuda_files=["kimi_k3/situ_and_mul.cuh"],
        cuda_wrappers=[("run", f"SituAndMulMaskedPostQuantMxfp4Kernel<{args}>::run")],
        extra_cuda_cflags=["-use_fast_math"],
    )


def situ_and_mul_post_quant_mxfp4(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    beta: float,
    linear_beta: float,
) -> None:
    module = _jit_situ_mul_quant_mxfp4_module()
    module.run(
        input,
        output,
        output_scale,
        float(beta),
        float(linear_beta),
    )


def situ_and_mul_masked_post_quant_mxfp4(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    masked_m: torch.Tensor,
    beta: float,
    linear_beta: float,
    topk: int = 8,
    expected_m: int | None = None,
) -> None:
    module = _jit_situ_mul_quant_mxfp4_varlen_module()
    module.run(
        input,
        output,
        output_scale,
        masked_m,
        topk,
        int(expected_m) if expected_m is not None else int(input.shape[1]),
        float(beta),
        float(linear_beta),
    )
