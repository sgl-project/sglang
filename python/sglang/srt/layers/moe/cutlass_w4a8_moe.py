# SPDX-License-Identifier: Apache-2.0
"""Cutlass W4A8 MoE kernel."""
from typing import Optional

import torch
from sgl_kernel import (
    cutlass_w4a8_moe_mm,
    get_cutlass_w4a8_moe_mm_data,
    sgl_per_tensor_quant_fp8,
    silu_and_mul,
)

from sglang.srt.layers.moe.ep_moe.kernels import (
    post_reorder_triton_kernel,
    pre_reorder_triton_kernel_for_cutlass_moe,
    run_cutlass_moe_ep_preproess,
)


def cutlass_w4a8_moe(
    start_expert_id: int,
    end_expert_id: int,
    total_num_experts: int,
    a: torch.Tensor,
    w1_q: torch.Tensor,
    w2_q: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids_: torch.Tensor,
    local_topk_ids: torch.Tensor,
    a_strides1: torch.Tensor,
    b_strides1: torch.Tensor,
    c_strides1: torch.Tensor,
    a_strides2: torch.Tensor,
    b_strides2: torch.Tensor,
    c_strides2: torch.Tensor,
    s_strides13: torch.Tensor,
    s_strides2: torch.Tensor,
    expert_offsets: torch.Tensor,
    problem_sizes1: torch.Tensor,
    problem_sizes2: torch.Tensor,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:
    """
    This function computes a w4a8-quantized Mixture of Experts (MoE) layer
    using two sets of quantized weights, w1_q and w2_q, and top-k gating
    mechanism. The matrix multiplications are implemented with CUTLASS
    grouped gemm.

    Parameters:
    - a (torch.Tensor): The input tensor to the MoE layer.
        Shape: [M, K]
    - w1_q (torch.Tensor): The first set of int4-quantized expert weights.
        Shape: [num_experts, N * 2,  K // 2]
        (the weights are passed transposed and int4-packed)
    - w2_q (torch.Tensor): The second set of int4-quantized expert weights.
        Shape: [num_experts, K, N // 2]
        (the weights are passed transposed and int4-packed)
    - w1_scale (torch.Tensor): The fp32 scale to dequantize w1_q.
        Shape: [num_experts, K // 512, N * 8]
    - w2_scale (torch.Tensor): The fp32 scale to dequantize w2_q.
        Shape: [num_experts, N // 512, K * 4]
    - topk_weights (torch.Tensor): The weights of each token->expert mapping.
    - a_strides1 (torch.Tensor): The input strides of the first grouped gemm.
    - b_strides1 (torch.Tensor): The weights strides of the first grouped gemm.
    - c_strides1 (torch.Tensor): The output strides of the first grouped gemm.
    - a_strides2 (torch.Tensor): The input strides of the second grouped gemm.
    - b_strides2 (torch.Tensor): The weights strides of the second grouped gemm.
    - c_strides2 (torch.Tensor): The output strides of the second grouped gemm.
    - s_strides13 (torch.Tensor): The input and scale strides of the first grouped gemm.
    - s_strides2 (torch.Tensor): The scale strides of the second grouped gemm.
    - a1_scale (Optional[torch.Tensor]): The optional fp32 scale to quantize a.
        Shape: scalar or [1, K]
    - a2_scale (Optional[torch.Tensor]): The optional fp32 scale to
        quantize the intermediate result between the gemms.
        Shape: scalar or [1, N]
    - apply_router_weight_on_input (bool): When true, the topk weights are
        applied directly on the inputs. This is only applicable when topk is 1.

    Returns:
    - torch.Tensor: The fp8 output tensor after applying the MoE layer.
    """
    assert topk_weights.shape == topk_ids_.shape, "topk shape mismatch"
    assert w1_q.dtype == torch.int8
    assert w2_q.dtype == torch.int8
    assert a.shape[1] // 2 == w1_q.shape[2], "Hidden size mismatch w1"
    assert w1_q.shape[2] * 2 == w2_q.shape[1], "Hidden size mismatch w2"
    assert w1_q.shape[0] == w2_q.shape[0], "Expert number mismatch"
    assert w1_q.shape[0] == w1_scale.shape[0], "w1 scales expert number mismatch"
    assert w1_q.shape[0] == w2_scale.shape[0], "w2 scales expert number mismatch"
    assert (
        w1_scale.shape[1] == w1_q.shape[2] * 2 / 512
        and w1_scale.shape[2] == w1_q.shape[1] * 4
    ), "W1 scale shape mismatch"
    assert (
        w2_scale.shape[1] == w2_q.shape[2] * 2 / 512
        and w2_scale.shape[2] == w2_q.shape[1] * 4
    ), "W2 scale shape mismatch"

    assert a_strides1.shape[0] == w1_q.shape[0], "A Strides 1 expert number mismatch"
    assert b_strides1.shape[0] == w1_q.shape[0], "B Strides 1 expert number mismatch"
    assert a_strides2.shape[0] == w2_q.shape[0], "A Strides 2 expert number  mismatch"
    assert b_strides2.shape[0] == w2_q.shape[0], "B Strides 2 expert number mismatch"
    num_experts = w1_q.size(0)
    m = a.size(0)
    k = w1_q.size(2) * 2  # w1_q is transposed and packed
    n = w2_q.size(2) * 2  # w2_q is transposed and packed
    topk = topk_ids_.size(1)

    if apply_router_weight_on_input:
        assert topk == 1, "apply_router_weight_on_input is only implemented for topk=1"

    device = a.device

    _, src2dst, _ = run_cutlass_moe_ep_preproess(
        local_topk_ids,
        num_experts,
    )

    gateup_input = torch.empty(
        (m * topk, k),
        device=device,
        dtype=torch.float8_e4m3fn,
    )

    pre_reorder_triton_kernel_for_cutlass_moe[(m,)](
        a,
        gateup_input,
        src2dst,
        local_topk_ids,
        a1_scale,
        total_num_experts,
        topk,
        k,
        BLOCK_SIZE=512,
    )

    # NOTE: a_map and c_map are not used in the get_cutlass_w4a8_moe_mm_data kernel,
    # they are kept to allow for a quick switch of the permutation logic
    # from the current triton kernel implementation to the cutlass-based one if needed.
    a_map = torch.empty((local_topk_ids.numel()), dtype=torch.int32, device=device)
    c_map = torch.empty((local_topk_ids.numel()), dtype=torch.int32, device=device)
    get_cutlass_w4a8_moe_mm_data(
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

    c1 = torch.empty((m * topk, n * 2), device=device, dtype=torch.half)
    c2 = torch.zeros((m * topk, k), device=device, dtype=torch.half)

    cutlass_w4a8_moe_mm(
        c1,
        gateup_input,
        w1_q,
        a1_scale.float(),
        w1_scale,
        expert_offsets[:-1],
        problem_sizes1,
        a_strides1,
        b_strides1,
        c_strides1,
        s_strides13,
        128,
        topk,
    )

    intermediate = torch.empty((m * topk, n), device=device, dtype=torch.half)
    silu_and_mul(c1, intermediate)

    intermediate_q = torch.empty(
        intermediate.shape, dtype=torch.float8_e4m3fn, device=device
    )
    sgl_per_tensor_quant_fp8(intermediate, intermediate_q, a2_scale.float(), True)

    cutlass_w4a8_moe_mm(
        c2,
        intermediate_q,
        w2_q,
        a2_scale.float(),
        w2_scale,
        expert_offsets[:-1],
        problem_sizes2,
        a_strides2,
        b_strides2,
        c_strides2,
        s_strides2,
        128,
        topk,
    )

    output = torch.empty_like(a)
    post_reorder_triton_kernel[(m,)](
        c2,
        output,
        src2dst,
        topk_ids_,
        topk_weights,
        start_expert_id,
        end_expert_id,
        topk,
        k,
        0,
        BLOCK_SIZE=512,
    )
    return output
