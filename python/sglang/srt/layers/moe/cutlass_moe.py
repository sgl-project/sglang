"""CUTLASS based Fused MoE kernels."""

import functools
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from sglang.srt.layers.moe.cutlass_moe_params import CutlassMoEParams
from sglang.srt.utils import is_cuda

_is_cuda = is_cuda()
if _is_cuda:
    import sgl_kernel

    try:
        from grouped_gemm.ops import permute, unpermute
    except:
        logging.warning(f"import permute/unpermute op failed")
    from sgl_kernel import (
        apply_shuffle_mul_sum,
        cutlass_fp4_group_mm,
        cutlass_moe_mm,
        fp8_blockwise_scaled_grouped_mm,
        get_cutlass_moe_mm_data,
        prepare_moe_input,
        scaled_fp4_experts_quant,
        sgl_per_tensor_quant_fp8,
        shuffle_rows,
        silu_and_mul,
    )


# TODO make the grouped gemm kernel consistent with scaled gemm kernel
def cutlass_moe_fp8(
    a: torch.Tensor,
    w1_q: torch.Tensor,
    w2_q: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids_: torch.Tensor,
    ab_strides1: torch.Tensor,
    c_strides1: torch.Tensor,
    ab_strides2: torch.Tensor,
    c_strides2: torch.Tensor,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.half,
    expert_map: Optional[torch.Tensor] = None,
    apply_router_weight_on_input: bool = False,
    moe_ws=None,
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
    - gating_output (torch.Tensor): The output of the gating operation
        (before softmax).
    - topk_weights (torch.Tensor): The weights of each token->expert mapping.
    - ab_strides1 (torch.Tensor): The input and weights strides of the first
        grouped gemm.
    - c_strides1 (torch.Tensor): The output strides of the first grouped gemm.
    - ab_strides2 (torch.Tensor): The input and weights strides of the second
        grouped gemm.
    - c_strides2 (torch.Tensor): The output strides of the second grouped gemm.
    - a1_scale (Optional[torch.Tensor]): The optional fp32 scale to quantize a.
        Shape: scalar or [M]
    - a2_scale (Optional[torch.Tensor]): The optional fp32 scale to
        quantize the intermediate result between the gemms.
        Shape: scalar or [M]
    - out_dtype (torch.Tensor): The output tensor type.
    - expert_map (Optional[torch.Tensor]): In the case of Expert parallel,
        every Rank is responsible for a subset of experts. expert_map is a
        mapping from global expert-id to local expert-id. When expert_map[i]
        is -1, it means that this Rank is not responsible for global
        expert-id i.
    - apply_router_weight_on_input (bool): When true, the topk weights are
        applied directly on the inputs. This is only applicable when topk is 1.

    Returns:
    - torch.Tensor: The fp16 output tensor after applying the MoE layer.
    """

    assert topk_weights.shape == topk_ids_.shape, "topk shape mismatch"
    assert w1_q.dtype == torch.float8_e4m3fn
    assert w2_q.dtype == torch.float8_e4m3fn
    assert a.shape[1] == w1_q.shape[1], "Hidden size mismatch w1"
    assert w1_q.shape[2] == w2_q.shape[1] * 2, "Hidden size mismatch w2"
    assert w1_q.shape[0] == w2_q.shape[0], "Expert number mismatch"
    assert (
        a1_scale is None
        or a1_scale.dim() == 0
        or a1_scale.shape[0] == 1
        or a1_scale.shape[0] == a.shape[0]
    ), "Input scale shape mismatch"
    assert (
        w1_scale.dim() == 1
        or w1_scale.shape[1] == 1
        or w1_scale.shape[1] == w1_q.shape[2]
    ), "W1 scale shape mismatch"
    assert (
        w2_scale.dim() == 1
        or w2_scale.shape[1] == 1
        or w2_scale.shape[1] == w2_q.shape[2]
    ), "W2 scale shape mismatch"
    assert w1_q.shape[0] == w2_q.shape[0], "Weights expert number mismatch"
    assert w1_q.shape[0] == w1_scale.shape[0], "w1 scales expert number mismatch"
    assert w1_q.shape[0] == w2_scale.shape[0], "w2 scales expert number mismatch"
    assert (
        a2_scale is None or a1_scale is None or a2_scale.shape == a1_scale.shape
    ), "Intermediate scale shape mismatch"  # noqa: E501
    assert ab_strides1.shape[0] == w1_q.shape[0], "AB Strides 1 expert number mismatch"
    assert c_strides1.shape[0] == w1_q.shape[0], "C Strides 1 expert number mismatch"
    assert ab_strides2.shape[0] == w2_q.shape[0], "AB Strides 2 expert number  mismatch"
    assert c_strides2.shape[0] == w2_q.shape[0], "C Strides 2 expert number mismatch"
    assert out_dtype in [torch.half, torch.bfloat16], "Invalid output dtype"

    num_experts = w1_q.size(0)
    m = a.size(0)
    k = w1_q.size(1)
    n = w2_q.size(1)

    assert m <= moe_ws.a_q_fp8.shape[0], "runtime shape exceed max workspace shape"

    local_topk_ids = topk_ids_
    if expert_map is not None:
        "Translate info from expert_map to topk_ids"
        local_topk_ids = torch.where(
            expert_map[topk_ids_] != -1, expert_map[topk_ids_], -1
        )

    topk = local_topk_ids.size(1)

    per_act_token = (
        a1_scale.numel() != 1
        if a1_scale is not None
        else (a2_scale.numel() != 1 if a2_scale is not None else False)
    )
    if apply_router_weight_on_input:
        assert topk == 1, "apply_router_weight_on_input is only implemented for topk=1"
        # TODO: this only works for topK=1, will need to update for topK>1
        a = a * topk_weights.to(out_dtype)

    # a_q, a1_scale = scaled_fp8_quant(
    #     hidden_states, a1_scale, use_per_token_if_dynamic=per_act_token)
    device = a.device
    a_q = moe_ws.a_q_fp8[0:m, :]

    a1_scale = moe_ws.a1_scale
    a2_scale = moe_ws.a2_scale

    expert_offsets = moe_ws.expert_offsets
    problem_sizes1 = moe_ws.problem_sizes[0]
    problem_sizes2 = moe_ws.problem_sizes[1]

    intermediate = moe_ws.inter[0 : m * topk, :]
    a_map = moe_ws.permute_map[0][0 : m * topk]
    # a_map = torch.empty((m * topk), device = device, dtype = torch.int32)
    c_map = moe_ws.permute_map[1][0 : m * topk]

    c1 = moe_ws.c1[0 : m * topk, :]
    c2 = moe_ws.c2[0 : m * topk, :]
    intemediate_q = moe_ws.inter_q[0 : m * topk, :]

    # ops.get_cutlass_moe_mm_data(local_topk_ids, expert_offsets, problem_sizes1,
    #                             problem_sizes2, a_map, c_map, num_experts, n,
    #                             k)

    sgl_per_tensor_quant_fp8(a, a_q, a1_scale, True)
    get_cutlass_moe_mm_data(
        local_topk_ids,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        a_map,
        c_map,
        num_experts,
        n,
        k,
    )

    rep_a_q, unpermute_map = permute(a_q, topk_ids_)
    rep_a1_scales = a1_scale  # a1_scale[a_map] if per_act_token else a1_scale

    cutlass_moe_mm(
        c1,
        rep_a_q,
        w1_q,
        rep_a1_scales,
        w1_scale,
        expert_offsets[:-1],
        problem_sizes1,
        ab_strides1,
        ab_strides1,
        c_strides1,
    )

    silu_and_mul(c1, intermediate)

    sgl_per_tensor_quant_fp8(intermediate, intemediate_q, a2_scale, True)

    cutlass_moe_mm(
        c2,
        intemediate_q,
        w2_q,
        a2_scale,
        w2_scale,
        expert_offsets[:-1],
        problem_sizes2,
        ab_strides2,
        ab_strides2,
        c_strides2,
    )

    return unpermute(c2, unpermute_map, topk_weights)


def cutlass_fused_experts_fp8(
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

    rep_a_q = shuffle_rows(a_q, a_map, (m * topk, k))
    rep_a1_scales = shuffle_rows(a1_scale, a_map, (m * topk, int(k / 128)))

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

    result = torch.empty((m, k), device=device, dtype=out_dtype)
    return apply_shuffle_mul_sum(c2, result, c_map, topk_weights)


FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = 448.0


def cutlass_moe_fp4(
    a: torch.Tensor,
    a1_gscale: torch.Tensor,
    w1_fp4: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w1_alphas: torch.Tensor,
    a2_gscale: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_alphas: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    params: CutlassMoEParams,
    apply_router_weight_on_input: bool = False,
):
    """
    MoE implementation for FP4 Inputs

    # Gemm 1
    a: Input tensor: [m, k] (half/bfloat16)
    a1_gscale: Activation scale per expert: [e]  (float32)
    w1(gate up) (not an argument to cutlass_moe_fp4): [e, 2 * n, k]
    w1_fp4: [e, 2 * n, k // 2], dtype: torch.uint8 (stacked fp4: E2M1)
    (Note: `n` is the up projection output dim, `k` is the input dim in
     full precision)
    w1_blockscale: [e, 2 * n, k // block_size] (float8_e4m3)
                   (Block size = 16 for NVFP4)

    # Gemm 2
    a2_gscale: Activation scale per expert: [e]
    w2(down projection) (not an argument to cutlass_moe_fp4): [e, k, n]
    w2_fp4: [e, k, n // 2], dtype: torch.uint8 (stacked E2M1)
    w2_blockscale: [e, k, n // block_size], dtype: float8_e4m3

    Strides for activations, weights and output in logical number of elements.
    The activations & output stride is the number of elements to the next row.
    The weights stride is the number of elements to the next row per expert.
    For example, if the weight is [e, n, k], then the b_stride is a tensor of
    shape [e] with each element being k. Similarly for activations, if the
    shape is [m, k], then the a_stride has shape [e] with each value k.
    Similarly for output, if the output is [m, n], then the c_stride is a
    tensor of shape [e] with each element being k.

    Note: cutlass_fp4_group_mm is designed to accept the strides of
    activations and weights to be the same, so it is passed in as a single
    tensor.
    ab_strides_13: [e] dtype: int64 [Gemm 1: Activation / Weight strides]
    ab_strides_2: [e] dtype: int64 [Gemm 2: Activation / Weight strides]
    c_strides_13: [e] dtype: int64 [Gemm 1: Output Strides]
    c_strides_2: [e] dtype: int64 [Gemm 1: Output Strides]

    topk_weights: [m, topk] dtype: float8
    topk_ids: [m, topk] dtype: float8

    m, n, k: Unquantized weight shapes, dtype: int
    e: number of experts for the current rank, dtype: int
    assumes that topk < k < n to satisfy - up/down projection expectations.
    """
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert w1_fp4.dtype == torch.uint8, "weight 1 must be uint8"
    assert w2_fp4.dtype == torch.uint8, "weight 2 must be uint8"
    assert (
        w1_fp4.ndim == 3
        and w2_fp4.ndim == 3
        and w1_blockscale.ndim == 3
        and w2_blockscale.ndim == 3
    ), "All Weights must be of rank 3 for cutlass_moe_fp4"
    m_a, k_a = a.shape
    e_w1, nx2_w1, half_k_w1 = w1_fp4.shape
    e_w2, k_w2, half_n_w2 = w2_fp4.shape

    assert e_w1 == e_w2 and e_w1 == params.num_experts, (
        "Number of experts must match",
        " between weights.",
    )
    assert (
        k_a // 2 == half_k_w1 and params.hidden_size == k_w2
    ), "Hidden size mismatch between a, w1 and w2"
    assert (
        nx2_w1 == params.intermediate_size_per_partition * 2
        and half_n_w2 == params.intermediate_size_per_partition // 2
    ), ("mismatch in " "expected `n`")
    assert 2 * half_k_w1 == k_w2, "Hidden size mismatch w2 and w1"
    assert a.dtype in [torch.half, torch.bfloat16], "Invalid input dtype"

    out_dtype = a.dtype
    num_topk = topk_ids.shape[1]
    device = a.device
    a_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
    c_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
    prepare_moe_input(
        topk_ids,
        params.expert_offsets,
        params.problem_sizes1,
        params.problem_sizes2,
        a_map,
        c_map,
        params.num_experts,
        params.intermediate_size_per_partition,
        params.hidden_size,
        params.blockscale_offsets,
    )

    rep_a_fp4, rep_a_blockscale = scaled_fp4_experts_quant(
        a,
        a1_gscale,
        params.expert_offsets,
        params.blockscale_offsets,
        num_topk,
        expert_map=a_map,
    )
    c1 = cutlass_fp4_group_mm(
        rep_a_fp4,
        w1_fp4,
        rep_a_blockscale,
        w1_blockscale,
        w1_alphas,
        out_dtype,
        device,
        params.to_gemm1_args(),
    )
    del rep_a_fp4, rep_a_blockscale

    # hidden size dimension is split to one halfpytho sized tensor.
    intermediate = torch.empty(
        (m_a * num_topk, w1_fp4.shape[1] // 2), device=device, dtype=out_dtype
    )
    silu_and_mul(c1, intermediate)

    int_fp4, int_blockscale = scaled_fp4_experts_quant(
        intermediate,
        a2_gscale,
        params.expert_offsets,
        params.blockscale_offsets,
        num_topk,
    )
    c2 = cutlass_fp4_group_mm(
        int_fp4,
        w2_fp4,
        int_blockscale,
        w2_blockscale,
        w2_alphas,
        out_dtype,
        device,
        params.to_gemm2_args(),
    )
    del int_fp4, int_blockscale
    c2 = shuffle_rows(c2, c_map, (m_a * num_topk, params.hidden_size))
    c2 = c2.view(m_a, num_topk, params.hidden_size)
    if not apply_router_weight_on_input:
        c2 = c2 * topk_weights.view(m_a, num_topk, 1).to(out_dtype)
    return c2.sum(dim=1).to(out_dtype)
