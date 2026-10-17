"""Fused Kimi-K3 router top-k + aiter moe_sorting for decode-sized batches (gfx950).

One launch replaces aiter's biased top-k kernel and its two-phase opus
moe_sorting. The outputs follow aiter.fused_moe.moe_sorting's contract, so
they can stand in for its return value (see csrc/kimi_k3/route_sort.cuh).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

MAX_TOKENS: int = 64


class RouteSortOutput(NamedTuple):
    topk_weights: torch.Tensor  # [M, topk] fp32
    topk_ids: torch.Tensor  # [M, topk] int32
    sorted_ids: torch.Tensor
    sorted_weights: torch.Tensor
    sorted_expert_ids: torch.Tensor
    num_valid_ids: torch.Tensor
    moe_buf: torch.Tensor
    # Stage-1 MXFP8 activations as aiter.fused_dynamic_mxfp8_quant_moe_sort
    # returns them (None unless route_sort was given quant_input).
    a1: Optional[torch.Tensor] = None
    a1_scale: Optional[torch.Tensor] = None


@cache_once
def _jit_route_sort_module(
    num_experts: int,
    topk: int,
    bias_dtype: torch.dtype,
    fuse_quant: bool,
) -> Module:
    args = make_cpp_args(num_experts, topk, bias_dtype, fuse_quant)
    return load_jit(
        "kimi_k3_route_sort",
        *args,
        cuda_files=["kimi_k3/route_sort.cuh"],
        cuda_wrappers=[("run", f"K3RouteSortKernel<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


def covered(
    router_logits: torch.Tensor, correction_bias: torch.Tensor, topk: int
) -> bool:
    num_tokens, num_experts = router_logits.shape
    return (
        0 < num_tokens <= MAX_TOKENS
        and router_logits.dtype == torch.bfloat16
        and router_logits.stride(1) == 1
        and router_logits.stride(0) % 2 == 0
        and router_logits.data_ptr() % 4 == 0
        and correction_bias.dtype in (torch.bfloat16, torch.float32)
        and correction_bias.is_contiguous()
        and num_experts % 128 == 0
        and num_experts <= 1024
        and 0 < topk <= 64
    )


def quant_covered(quant_input: torch.Tensor) -> bool:
    cols = quant_input.shape[-1]
    return (
        quant_input.dim() == 2
        and quant_input.dtype == torch.bfloat16
        and quant_input.stride(1) == 1
        and quant_input.stride(0) % 8 == 0
        and quant_input.data_ptr() % 16 == 0
        and cols % 32 == 0
        and cols // 32 <= 128
    )


def route_sort(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    *,
    topk: int,
    block_m: int,
    model_dim: int,
    moe_buf: Optional[torch.Tensor],
    routed_scaling_factor: float,
    renormalize: bool,
    quant_input: Optional[torch.Tensor] = None,
) -> RouteSortOutput:
    """Top-k of sigmoid(router_logits) + bias, sorted into aiter's block-padded
    route layout; moe_buf ([M, model_dim], allocated when None) is zeroed.
    With quant_input ([M, D] bf16, the routed activations) the stage-1 MXFP8
    quant + scale sort is fused in as well."""
    num_tokens, num_experts = router_logits.shape
    device = router_logits.device
    # Same sizes as aiter.fused_moe._moe_sorting_impl.
    max_rows = num_tokens * topk + num_experts * block_m - topk
    max_blocks = (max_rows + block_m - 1) // block_m
    max_rows = max_blocks * block_m
    topk_weights = torch.empty((num_tokens, topk), dtype=torch.float32, device=device)
    topk_ids = torch.empty((num_tokens, topk), dtype=torch.int32, device=device)
    sorted_ids = torch.empty(max_rows, dtype=torch.int32, device=device)
    sorted_weights = torch.empty(max_rows, dtype=torch.float32, device=device)
    sorted_expert_ids = torch.empty(max_blocks, dtype=torch.int32, device=device)
    num_valid_ids = torch.empty(2, dtype=torch.int32, device=device)
    if moe_buf is None:
        moe_buf = torch.empty(
            (num_tokens, model_dim), dtype=torch.bfloat16, device=device
        )
    fuse_quant = quant_input is not None
    if fuse_quant:
        # Same shapes / dtypes as aiter's fused_dynamic_mx_quant_moe_sort.
        cols = quant_input.shape[1]
        a1 = torch.empty((num_tokens, cols), dtype=torch.float8_e4m3fn, device=device)
        scale_n_pad = ((cols + 31) // 32 + 7) // 8 * 8
        a1_scale = torch.empty(
            ((max_rows + 31) // 32 * 32, scale_n_pad),
            dtype=torch.float8_e8m0fnu,
            device=device,
        )
    else:
        quant_input = a1 = a1_scale = sorted_ids  # unread placeholders
    module = _jit_route_sort_module(
        num_experts, topk, correction_bias.dtype, fuse_quant
    )
    module.run(
        router_logits,
        correction_bias,
        topk_weights,
        topk_ids,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf.view(-1),
        quant_input,
        a1.view(torch.uint8) if fuse_quant else a1,
        a1_scale.view(torch.uint8) if fuse_quant else a1_scale,
        block_m,
        float(routed_scaling_factor),
        renormalize,
    )
    return RouteSortOutput(
        topk_weights,
        topk_ids,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
        a1 if fuse_quant else None,
        a1_scale if fuse_quant else None,
    )
