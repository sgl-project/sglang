import functools
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
from sgl_kernel import (fp8_blockwise_scaled_grouped_mm, prepare_moe_input, silu_and_mul)


import torch


def cutlass_fused_experts_fp8_bs(
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
) -> torch.Tensor:
    """
    This function computes a a8w8-quantized Mixture of Experts (MoE) layer
    using two sets of quantized weights, w1_q and w2_q, and top-k gating
    mechanism. The matrix multiplications are implemented with CUTLASS
    grouped gemm.

    Parameters:
    - a (torch.Tensor): The input tensor to the MoE layer.
        Shape: [M, K]
    - w1_q (torch.Tensor): The first set of fp8-quantized expert weights.
        Shape: [num_experts, K, 2N] (the weights are passed transposed)
    - w2_q (torch.Tensor): The second set of fp8-quantized expert weights.
        Shape: [num_experts, N, K] (the weights are passed transposed)
    - w1_scale (torch.Tensor): The fp32 scale to dequantize w1_q.
        Shape: [num_experts] or [num_experts, 2N]
    - w2_scale (torch.Tensor): The fp32 scale to dequantize w2_q.
        Shape: [num_experts] or [num_experts, K]
    - topk_weights (torch.Tensor): The weights of each token->expert mapping.
    - ab_strides1 (torch.Tensor): The input and weights strides of the first
        grouped gemm.
    - c_strides1 (torch.Tensor): The output strides of the first grouped gemm.
    - ab_strides2 (torch.Tensor): The input and weights strides of the second
        grouped gemm.
    - c_strides2 (torch.Tensor): The output strides of the second grouped gemm.
    - a1_scale (torch.Tensor): The fp32 scale to dequantize a.
        Shape: [M, K / 128]
    - a2_scale (torch.Tensor): The fp32 scale to dequantize the intermediate result between the gemms.
        Shape: [M, N / 128]
    - out_dtype (torch.Tensor): The output tensor type.

    Returns:
    - torch.Tensor: The fp16 output tensor after applying the MoE layer.
    """    
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert w1_q.dtype == torch.float8_e4m3fn
    assert w2_q.dtype == torch.float8_e4m3fn
    assert a.shape[1] == w1_q.shape[1], "Hidden size mismatch w1"
    assert w1_q.shape[2] == w2_q.shape[1] * 2, "Hidden size mismatch w2"
    assert w1_q.shape[0] == w2_q.shape[0], "Expert number mismatch"
    assert a1_scale is None or a1_scale.dim(
    ) == 0 or a1_scale.shape[0] == 1 or a1_scale.shape[0] == a.shape[
        0], "Input scale shape mismatch"
    assert w1_q.shape[0] == w2_q.shape[0], "Weights expert number mismatch"
    assert w1_q.shape[0] == w1_scale.shape[
        0], "w1 scales expert number mismatch"
    assert w1_q.shape[0] == w2_scale.shape[
        0], "w2 scales expert number mismatch"
    assert a.dtype in [torch.half, torch.bfloat16], "Invalid output dtype"
    
    out_dtype = a.dtype
    num_experts = w1_q.size(0)
    m = a.size(0)
    k = w1_q.size(1)
    n = w2_q.size(1)

    topk = topk_ids.size(1)
    
    a_q, a1_scale = sglang_per_token_group_quant_fp8(
        a, 128)
    device = a_q.device

    expert_offsets = torch.empty((num_experts + 1),
                                 dtype=torch.int32,
                                 device=device)
    problem_sizes1 = torch.empty((num_experts, 3),
                                 dtype=torch.int32,
                                 device=device)
    problem_sizes2 = torch.empty((num_experts, 3),
                                 dtype=torch.int32,
                                 device=device)

    a_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
    c_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)

    prepare_moe_input(topk_ids, expert_offsets, problem_sizes1,
                                problem_sizes2, a_map, c_map, num_experts, n, k)

    rep_a_q = a_q.view(dtype=torch.uint8)[a_map].view(dtype=a_q.dtype)
    rep_a1_scales = a1_scale[a_map]

    c1 = torch.empty((m * topk, n * 2), device=device, dtype=out_dtype)
    c2 = torch.empty((m * topk, k), device=device, dtype=out_dtype)
    
    a1_sf_layout = torch.empty((num_experts, 5), device=device, dtype=torch.int)
    w1_sf_layout = torch.empty((num_experts, 5), device=device, dtype=torch.int)


    fp8_blockwise_scaled_grouped_mm(c1, rep_a_q, w1_q, rep_a1_scales, w1_scale,
                       a1_strides, a1_strides, c1_strides, a1_sf_layout, 
                       w1_sf_layout, problem_sizes1, expert_offsets[:-1])

    intermediate = torch.empty((m * topk, n), device=device, dtype=out_dtype)
    silu_and_mul(intermediate, c1)

    intemediate_q, a2_scale = sglang_per_token_group_quant_fp8(
        intermediate, 128)


    a2_sf_layout = torch.empty((num_experts, 5), device=device, dtype=torch.int)
    w2_sf_layout = torch.empty((num_experts, 5), device=device, dtype=torch.int)
    
    fp8_blockwise_scaled_grouped_mm(c2, intemediate_q, w2_q, a2_scale, w2_scale, a2_strides,
                       a2_strides, c2_strides, a2_sf_layout, w2_sf_layout, problem_sizes2,
                       expert_offsets[:-1])
    return (c2[c_map].view(m, topk, k) *
            topk_weights.view(m, topk, 1).to(out_dtype)).sum(dim=1)