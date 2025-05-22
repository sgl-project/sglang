"""Cutlass MoE kernel."""

import functools
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from sglang.srt.utils import is_cuda

_is_cuda = is_cuda()
if _is_cuda:
    import sgl_kernel
    from sgl_kernel import (
        fp8_blockwise_scaled_grouped_mm,
        prepare_moe_input,
        silu_and_mul,
    )


def cutlass_fused_experts(
    a: torch.Tensor,
    w1_q: torch.Tensor,
    w2_q: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    a1_strides: torch.Tensor,
    c1_strides: torch.Tensor,
    a2_strides: torch.Tensor,
    c2_strides: torch.Tensor,
    workspace: torch.Tensor,
    a_ptrs: torch.Tensor,
    b_ptrs: torch.Tensor,
    out_ptrs: torch.Tensor,
    a_scales_ptrs: torch.Tensor,
    b_scales_ptrs: torch.Tensor,
    expert_offsets: torch.Tensor,
    problem_sizes1: torch.Tensor,
    problem_sizes2: torch.Tensor,
    use_fp8_blockscale: bool = True,
) -> torch.Tensor:
    """Performs Fused MoE computation using CUTLASS-like kernels with FP8 weights and activations.

    This function implements a Mixture of Experts (MoE) layer with a SwiGLU/SiLU
    activation, leveraging custom kernels likely derived from CUTLASS principles
    for grouped matrix multiplication (`fp8_blockwise_scaled_grouped_mm`) and
    data preparation (`prepare_moe_input`, `silu_and_mul`).

    It handles per-token routing, quantizes input activations to FP8 with
    per-token scales, performs the expert computations using FP8 GEMMs with
    pre-quantized FP8 weights (per-block scales), applies the SiLU activation,
    and combines the results weighted by the router scores.

    Args:
        a (torch.Tensor): Input activations. Shape: `(m, k)`, where `m` is the total
            number of tokens and `k` is the hidden size. Expected dtype: `torch.half`
            or `torch.bfloat16`.
        w1_q (torch.Tensor): Pre-quantized FP8 weight tensor for the first GEMM
            (up-projection part of SwiGLU). Expected shape: `(E, k, n*2)`, where
            `E` is the number of experts, `k` is the hidden size, and `n*2` is the
            intermediate size (`I`). Expected dtype: `torch.float8_e4m3fn`.
            Note: This shape implies weights are stored as (num_experts, hidden_size, intermediate_size).
        w2_q (torch.Tensor): Pre-quantized FP8 weight tensor for the second GEMM
            (down-projection). Expected shape: `(E, n, k)`, where `n` is half the
            intermediate size (`I // 2`). Expected dtype: `torch.float8_e4m3fn`.
            Note: This shape implies weights are stored as (num_experts, intermediate_size // 2, hidden_size).
        w1_scale (torch.Tensor): Scales corresponding to `w1_q` (per-block scales).
            Shape: `(E, num_blocks_n, num_blocks_k)`. Dtype: `torch.float32`.
        w2_scale (torch.Tensor): Scales corresponding to `w2_q` (per-block scales).
             Shape: `(E, num_blocks_k, num_blocks_n)`. Dtype: `torch.float32`.
        topk_weights (torch.Tensor): Router weights for the selected top-k experts
            for each token. Shape: `(m, topk)`. Dtype should ideally match `a`.
        topk_ids (torch.Tensor): Indices of the selected top-k experts for each token.
            Shape: `(m, topk)`. Dtype: `torch.int32`.
        a1_strides (torch.Tensor): Stride information for the first GEMM's 'a' input.
            Passed directly to the underlying kernel. Expected shape `(E,)`, dtype `torch.int64`.
            Note: Its exact usage within `fp8_blockwise_scaled_grouped_mm` needs clarification
            as it's passed as both a_stride and b_stride in the first call.
        c1_strides (torch.Tensor): Stride information for the first GEMM's 'c' output.
            Passed directly to the underlying kernel. Expected shape `(E,)`, dtype `torch.int64`.
        a2_strides (torch.Tensor): Stride information for the second GEMM's 'a' input.
            Passed directly to the underlying kernel. Expected shape `(E,)`, dtype `torch.int64`.
            Note: Its exact usage within `fp8_blockwise_scaled_grouped_mm` needs clarification
            as it's passed as both a_stride and b_stride in the second call.
        c2_strides (torch.Tensor): Stride information for the second GEMM's 'c' output.
            Passed directly to the underlying kernel. Expected shape `(E,)`, dtype `torch.int64`.
        workspace (torch.Tensor): Reusable workspace for the underlying kernel.
        a_ptrs (torch.Tensor): Pointers container for calculating offsets of the input activations for each expert.
        b_ptrs (torch.Tensor): Pointers container for calculating offsets of the input weights for each expert.
        out_ptrs (torch.Tensor): Pointers container for calculating offsets of the output activations for each expert.
        a_scales_ptrs (torch.Tensor): Pointers container for calculating offsets of the input scales for each expert.
        b_scales_ptrs (torch.Tensor): Pointers container for calculating offsets of the input scales for each expert.
        use_fp8_blockscale (bool, optional): Flag indicating usage of FP8 with
            block scaling. Currently, only `True` is supported. Defaults to `True`.

    Returns:
        torch.Tensor: The computed MoE layer output. Shape: `(m, k)`, dtype matches `a`.

    Raises:
        AssertionError: If input shapes, dtypes, or flags are inconsistent or unsupported.
        NotImplementedError: If CUDA is not available or `sgl_kernel` is not properly installed.
    """
    assert use_fp8_blockscale, "Only support fp8 blockscale for now"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert w1_q.dtype == torch.float8_e4m3fn
    assert w2_q.dtype == torch.float8_e4m3fn
    assert a.shape[1] == w1_q.shape[1], "Hidden size mismatch w1"
    assert w1_q.shape[2] == w2_q.shape[1] * 2, "Hidden size mismatch w2"
    assert w1_q.shape[0] == w2_q.shape[0], "Expert number mismatch"
    assert w1_q.shape[0] == w2_q.shape[0], "Weights expert number mismatch"
    assert w1_q.shape[0] == w1_scale.shape[0], "w1 scales expert number mismatch"
    assert w1_q.shape[0] == w2_scale.shape[0], "w2 scales expert number mismatch"
    assert a.dtype in [torch.half, torch.bfloat16], "Invalid output dtype"

    if is_cuda:
        from sglang.srt.layers.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8,
        )

    out_dtype = a.dtype
    num_experts = w1_q.size(0)
    m = a.size(0)
    k = w1_q.size(1)
    n = w2_q.size(1)

    topk = topk_ids.size(1)

    a_q, a1_scale = sglang_per_token_group_quant_fp8(a, 128)
    device = a_q.device

    a_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
    c_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)

    prepare_moe_input(
        topk_ids,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        a_map,
        c_map,
        num_experts,
        n,
        k,
    )

    rep_a_q = a_q.view(dtype=torch.uint8)[a_map].view(dtype=a_q.dtype)
    rep_a1_scales = a1_scale[a_map]

    c1 = torch.empty((m * topk, n * 2), device=device, dtype=out_dtype)
    c2 = torch.empty((m * topk, k), device=device, dtype=out_dtype)

    a_sf_layout = torch.empty((num_experts, 5), device=device, dtype=torch.int)
    w_sf_layout = torch.empty((num_experts, 5), device=device, dtype=torch.int)

    fp8_blockwise_scaled_grouped_mm(
        c1,
        a_ptrs,
        b_ptrs,
        out_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        rep_a_q,
        w1_q,
        rep_a1_scales,
        w1_scale,
        a1_strides,
        a1_strides,
        c1_strides,
        a_sf_layout,
        w_sf_layout,
        problem_sizes1,
        expert_offsets[:-1],
        workspace,
    )

    intermediate = torch.empty((m * topk, n), device=device, dtype=out_dtype)
    silu_and_mul(c1, intermediate)

    intemediate_q, a2_scale = sglang_per_token_group_quant_fp8(intermediate, 128)

    fp8_blockwise_scaled_grouped_mm(
        c2,
        a_ptrs,
        b_ptrs,
        out_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        intemediate_q,
        w2_q,
        a2_scale,
        w2_scale,
        a2_strides,
        a2_strides,
        c2_strides,
        a_sf_layout,
        w_sf_layout,
        problem_sizes2,
        expert_offsets[:-1],
        workspace,
    )
    return (
        c2[c_map].view(m, topk, k) * topk_weights.view(m, topk, 1).to(out_dtype)
    ).sum(dim=1)
