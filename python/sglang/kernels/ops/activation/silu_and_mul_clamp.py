"""SwiGLU with clamped branches, optionally fused with per-token-group quantization."""

from typing import Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.utils import is_xpu

_is_xpu = is_xpu()


def make_name(name: str) -> str:
    return f"dpsk_v4_{name}"


@cache_once
def _jit_silu_mul_quant_contig_module(
    quant_group_size: int,
    scale_ue8m0: bool,
    swizzle: bool,
    apply_swiglu_limit: bool,
):
    args = make_cpp_args(
        quant_group_size,
        scale_ue8m0,
        swizzle,
        is_arch_support_pdl(),
        apply_swiglu_limit,
    )
    return load_jit(
        make_name("silu_mul_quant_contig"),
        *args,
        cuda_files=["deepseek_v4/silu_and_mul_masked_post_quant.cuh"],
        cuda_wrappers=[("run", f"SiluAndMulContigPostQuantKernel<{args}>::run")],
        extra_cuda_cflags=["-use_fast_math"],
    )


@cache_once
def _jit_silu_and_mul_clamp_module(dtype: torch.dtype):
    args = make_cpp_args(dtype, is_arch_support_pdl())
    return load_jit(
        make_name("silu_and_mul_clamp"),
        *args,
        cuda_files=["deepseek_v4/silu_and_mul_masked_post_quant.cuh"],
        cuda_wrappers=[("run", f"SiluAndMulClampKernel<{args}>::run")],
        extra_cuda_cflags=["-use_fast_math"],
    )


def silu_and_mul_clamp(
    input: torch.Tensor,
    output: torch.Tensor,
    swiglu_limit: float,
) -> None:
    if _is_xpu:
        from sgl_kernel import silu_and_mul_clamp

        silu_and_mul_clamp(input, output, float(swiglu_limit))
    else:
        module = _jit_silu_and_mul_clamp_module(input.dtype)
        module.run(input, output, float(swiglu_limit))


def silu_and_mul_contig_post_quant(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    quant_group_size: int,
    scale_ue8m0: bool = False,
    transposed: bool = False,
    swiglu_limit: Optional[float] = None,
    swizzle: bool = False,
) -> None:
    apply_swiglu_limit = swiglu_limit is not None
    module = _jit_silu_mul_quant_contig_module(
        quant_group_size, scale_ue8m0, swizzle, apply_swiglu_limit
    )
    module.run(
        input,
        output,
        output_scale,
        transposed,
        float(swiglu_limit) if apply_swiglu_limit else 0.0,
    )
