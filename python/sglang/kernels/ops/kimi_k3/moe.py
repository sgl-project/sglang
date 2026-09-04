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
