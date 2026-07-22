from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


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
        cuda_files=["kimi_k3/situ_and_mul_masked_post_quant.cuh"],
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
def _jit_moe_finalize_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        _make_name("moe_finalize"),
        *args,
        cuda_files=["kimi_k3/moe_finalize.cuh"],
        cuda_wrappers=[("run", f"K3MoeFinalizeKernel<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


def moe_finalize(
    gemm2_out: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    expert_weights: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """out[t] = sum_k expert_weights[t, k] * gemm2_out[idx[t*16 + k]].

    gemm2_out: [P, H] bf16 in the trtllm-gen permuted layout; H % 8 == 0.
    expanded_idx_to_permuted_idx: [T*16] int32, -1 = dropped slot (skipped).
    expert_weights: [T, 16] bf16 (routed scaling already folded in).
    out: optional [T, H] bf16 destination (e.g. a symm-buffer slice).

    fp32 ascending-k accumulation: bit-identical to the trtllm-gen in-op
    finalize (do_finalize=True).
    """
    num_tokens = expert_weights.shape[0]
    if out is None:
        out = torch.empty(
            num_tokens,
            gemm2_out.shape[1],
            dtype=gemm2_out.dtype,
            device=gemm2_out.device,
        )
    _jit_moe_finalize_module().run(
        gemm2_out, expanded_idx_to_permuted_idx, expert_weights, out
    )
    return out
