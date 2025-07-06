# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# copied from https://github.com/NVIDIA/TensorRT-LLM/blob/04fa6c0cfc52303ec692a0cb70657e51d0db4781/tests/unittest/_torch/thop/test_moe.py#L918

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), "."))
from tensorrt_llm._torch.autotuner import autotune
from tensorrt_llm._torch.modules.fused_moe import RoutingMethodType
from tensorrt_llm.quantization.utils.fp4_utils import (
    reorder_rows_for_gated_act_gemm,
    shuffle_matrix_a,
    shuffle_matrix_sf_a,
)
from utils import getSMVersion


class moe_args:

    def __init__(
        self,
        num_tokens,
        num_experts,
        hidden_size,
        intermediate_size,
        top_k,
        padding,
        hidden_states,
        hidden_states_scale,
        hidden_states_scale_global,
        expert_logits,
        gemm1_weights,
        gemm1_scales,
        gemm1_scales_global,
        gemm2_weights,
        gemm2_scales,
        gemm2_scales_global,
        permute_info,
        use_routing_scales_on_input,
    ):
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.top_k = top_k
        self.padding = padding
        self.hidden_states = hidden_states
        self.hidden_states_scale = hidden_states_scale
        self.hidden_states_scale_global = hidden_states_scale_global
        self.expert_logits = expert_logits
        self.gemm1_weights = gemm1_weights
        self.gemm1_scales = gemm1_scales
        self.gemm1_scales_global = gemm1_scales_global
        self.gemm2_weights = gemm2_weights
        self.gemm2_scales = gemm2_scales
        self.gemm2_scales_global = gemm2_scales_global
        self.permute_info = permute_info
        self.use_routing_scales_on_input = use_routing_scales_on_input


class moe_args_dequant:

    def __init__(
        self,
        num_tokens,
        num_experts,
        hidden_size,
        intermediate_size,
        top_k,
        padding,
        hidden_states,
        expert_logits,
        gemm1_weights,
        gemm2_weights,
        permute_info,
        use_routing_scales_on_input,
    ):
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.top_k = top_k
        self.padding = padding
        self.hidden_states = hidden_states
        self.expert_logits = expert_logits
        self.gemm1_weights = gemm1_weights
        self.gemm2_weights = gemm2_weights
        self.permute_info = permute_info
        self.use_routing_scales_on_input = use_routing_scales_on_input


def routing_reference(expertLogits, topK, padding):
    originalDevice = expertLogits.device
    expertLogits = expertLogits.cpu()
    numTokens, numExperts = expertLogits.shape
    assert topK <= numExperts

    numTokensPerExpert = torch.zeros(numExperts, dtype=torch.int64)
    expandedTokenIdxToExpert = -torch.ones(numTokens * topK, dtype=torch.int64)
    expandedTokenIdxToIdxInExpert = -torch.ones(numTokens * topK, dtype=torch.int64)

    topKLogits, topKIndices = torch.topk(expertLogits, topK, dim=1)
    for tokenIdx in range(numTokens):
        for k in range(topK):
            expandedIdx = tokenIdx * topK + k
            expertIndex = topKIndices[tokenIdx, k]
            expandedTokenIdxToExpert[expandedIdx] = expertIndex
            expandedTokenIdxToIdxInExpert[expandedIdx] = numTokensPerExpert[expertIndex]
            numTokensPerExpert[expertIndex] += 1

    paddedTokensPerExpertPrefixSum = torch.zeros(numExperts + 1, dtype=torch.int64)
    for ii in range(numExperts):

        def divUpMul(a, b):
            return (a + b - 1) // b * b

        paddedTokensPerExpertPrefixSum[ii + 1] = paddedTokensPerExpertPrefixSum[
            ii
        ] + divUpMul(numTokensPerExpert[ii], padding)
    permutedBufferSize = paddedTokensPerExpertPrefixSum[numExperts]

    expandedTokenIdxToPermutedIdx = -torch.ones(numTokens * topK, dtype=torch.int64)
    permutedIdxToExpandedIdx = -torch.ones(permutedBufferSize, dtype=torch.int64)
    permutedIdxToTokenIdx = -torch.ones(permutedBufferSize, dtype=torch.int64)
    for tokenIdx in range(numTokens):
        for k in range(topK):
            expandedIdx = tokenIdx * topK + k
            expert = expandedTokenIdxToExpert[expandedIdx]
            offsetWithinExpert = expandedTokenIdxToIdxInExpert[expandedIdx]
            offsetForExpert = paddedTokensPerExpertPrefixSum[expert]
            permutedIdx = offsetForExpert + offsetWithinExpert

            expandedTokenIdxToPermutedIdx[expandedIdx] = permutedIdx
            permutedIdxToExpandedIdx[permutedIdx] = expandedIdx
            permutedIdxToTokenIdx[permutedIdx] = tokenIdx
    return {
        "paddedTokensPerExpertPrefixSum": paddedTokensPerExpertPrefixSum.to(
            originalDevice
        ),
        "permutedBufferSize": permutedBufferSize.item(),
        "expandedTokenIdxToPermutedIdx": expandedTokenIdxToPermutedIdx.to(
            originalDevice
        ),
        "permutedIdxToExpandedIdx": permutedIdxToExpandedIdx.to(originalDevice),
        "numTokensPerExpert": numTokensPerExpert.to(originalDevice),
        "expandedTokenIdxToExpert": expandedTokenIdxToExpert.to(originalDevice),
        "topKLogits": topKLogits.to(originalDevice),
        "permutedIdxToTokenIdx": permutedIdxToTokenIdx.to(originalDevice),
        "topKIndices": topKIndices.to(originalDevice),
    }


def noaux_tc_ref(logits, bias, n_group, topk_group, top_k, routed_scaling_factor):
    scores = F.sigmoid(logits)
    scores_with_bias = scores + bias
    if n_group > 1:
        scores_shape = list(scores_with_bias.shape)
        group_scores = torch.sum(
            torch.topk(
                scores_with_bias.view(
                    scores_shape[:-1] + [n_group, scores_shape[-1] // n_group]
                ),
                k=2,
                dim=-1,
                largest=True,
                sorted=True,
            )[0],
            dim=-1,
        )
        _, group_idx = torch.topk(
            group_scores, k=topk_group, dim=-1, largest=True, sorted=True
        )
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(-1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(scores_shape[:-1] + [n_group, scores_shape[-1] // n_group])
            .reshape(scores_shape)
        )
        scores_with_bias = scores_with_bias * score_mask

    _, topk_idx = torch.topk(
        scores_with_bias, k=top_k, dim=-1, largest=True, sorted=True
    )
    new_mask = torch.zeros_like(scores)
    new_mask.scatter_(-1, topk_idx, 1)
    scores = scores * new_mask
    score_sum = torch.sum(scores, dim=-1, keepdim=True) + 1e-20
    scores = scores / score_sum * routed_scaling_factor
    return scores


# Tiered TopK routing used by DeepSeek
def routing_reference_no_aux(
    expert_logits,
    routing_bias,
    top_k,
    n_groups,
    top_k_groups,
    routed_scaling,
    padding,
    use_routing_scales_on_input=False,
):
    routing_logits = expert_logits.to(dtype=torch.float, device="cuda")
    if use_routing_scales_on_input:
        # if using routing scales on input, topK == 1 and the score is a plain sigmoid
        scores = F.sigmoid(routing_logits)
    else:
        scores = noaux_tc_ref(
            routing_logits, routing_bias, n_groups, top_k_groups, top_k, routed_scaling
        )
    permute_info = routing_reference(scores, top_k, padding)
    return permute_info, scores


# TopK -> Softmax
def routing_reference_renormalize(expert_logits, top_k, num_experts, padding):
    topk_values, topk_idx = torch.topk(expert_logits, k=top_k, dim=-1)
    topk_values = torch.nn.functional.softmax(topk_values.float(), dim=-1)

    new_mask = torch.zeros_like(expert_logits)
    new_mask.scatter_(-1, topk_idx, 1)
    scores = expert_logits * new_mask

    for i in range(topk_idx.shape[0]):
        for j in range(topk_idx.shape[1]):
            scores[i, topk_idx[i, j]] = topk_values[i, j]
    permute_info = routing_reference(scores, top_k, padding)
    return permute_info, scores


# Softmax->TopK -> Normalize
def routing_reference_renormalize_naive(expert_logits, top_k, num_experts, padding):
    norm_topk_prob = True
    scores = torch.nn.functional.softmax(expert_logits.float(), dim=-1)
    topk_values, topk_idx = torch.topk(scores, k=top_k, dim=-1)

    if norm_topk_prob:  # only diff with mixtral sparse moe block!
        topk_values /= topk_values.sum(dim=-1, keepdim=True)
    topk_values = topk_values.to(expert_logits.dtype)
    scores = scores.to(expert_logits.dtype)

    new_mask = torch.zeros_like(expert_logits)
    new_mask.scatter_(-1, topk_idx, 1)
    scores = expert_logits * new_mask

    for i in range(topk_idx.shape[0]):
        for j in range(topk_idx.shape[1]):
            scores[i, topk_idx[i, j]] = topk_values[i, j]
    permute_info = routing_reference(scores, top_k, padding)
    return permute_info, scores


def dequant_reference_dsfp8(input, scale, transpose_scale, block_m, block_n):
    input = input.to(torch.float)
    scale = scale.to(torch.float)
    if transpose_scale:
        scale = scale.t()
    output = torch.zeros_like(input)
    m, n = input.shape
    m_tile = 128 if block_m else 1
    n_tile = 128 if block_n else 1
    assert m % m_tile == 0
    assert n % n_tile == 0
    assert scale.shape == (m // m_tile, n // n_tile)
    if m_tile == 1:
        for j in range(0, n, n_tile):
            output[:, j : j + n_tile] = (
                input[:, j : j + n_tile] * scale[:, j // n_tile][:, None]
            )
    elif n_tile == 1:
        for i in range(0, m, m_tile):
            output[i : i + m_tile] = input[i : i + m_tile] * scale[i // m_tile]
    else:
        for i in range(0, m, m_tile):
            for j in range(0, n, n_tile):
                output[i : i + m_tile, j : j + n_tile] = (
                    input[i : i + m_tile, j : j + n_tile]
                    * scale[i // m_tile, j // n_tile]
                )
    return output


def run_moe_dequant(args, quant_mode=["fp4", "dsFp8", "perTensorFp8"]):
    # Permute
    total_num_padded_tokens = args.permute_info["permutedBufferSize"]
    expanded_idx_to_permuted_idx = args.permute_info[
        "expandedTokenIdxToPermutedIdx"
    ].cpu()
    num_tokens_per_expert = args.permute_info["numTokensPerExpert"].cpu()
    permute_output = torch.full(
        (total_num_padded_tokens, args.hidden_size), float("nan"), device="cuda"
    ).to(torch.float)
    for i in range(args.num_tokens):
        for j in range(args.top_k):
            permuted_idx = expanded_idx_to_permuted_idx[i * args.top_k + j]
            permute_output[permuted_idx] = args.hidden_states[i]
    # Gemm1
    gemm1_output = torch.full(
        (total_num_padded_tokens, 2 * args.intermediate_size),
        float("nan"),
        device="cuda",
    ).to(torch.float)
    i = 0
    for expert_idx in range(args.num_experts):
        my_num_tokens = num_tokens_per_expert[expert_idx]
        if my_num_tokens == 0:
            continue
        my_a = permute_output[i : i + my_num_tokens]
        my_b = args.gemm1_weights[expert_idx]
        my_c = my_a @ my_b.t()
        gemm1_output[i : i + my_num_tokens] = my_c
        i += my_num_tokens
        i = (i + args.padding - 1) // args.padding * args.padding

    if args.use_routing_scales_on_input:
        assert args.top_k == 1
        # For each token and its top_k experts
        for token_idx in range(args.num_tokens):
            for k in range(args.top_k):
                # Get the permuted index for this token's k-th expert
                expanded_idx = token_idx * args.top_k + k
                permuted_idx = expanded_idx_to_permuted_idx[expanded_idx]
                expert_weight = args.permute_info["topKLogits"].to(torch.float)
                # Get the expert weight for this token and expert
                weight = expert_weight[token_idx, k]
                # Scale the corresponding row in gemm1_output
                gemm1_output[permuted_idx] *= weight

    # Activation
    activation_output = torch.full(
        (total_num_padded_tokens, args.intermediate_size), float("nan"), device="cuda"
    ).to(torch.float)

    i = 0
    for expert_idx in range(args.num_experts):
        my_num_tokens = num_tokens_per_expert[expert_idx]
        if my_num_tokens == 0:
            continue
        my_a = gemm1_output[i : i + my_num_tokens]
        my_x1 = my_a[:, : args.intermediate_size]
        my_x2 = my_a[:, args.intermediate_size :]
        activation_output[i : i + my_num_tokens] = F.silu(my_x2) * my_x1
        i += my_num_tokens
        i = (i + args.padding - 1) // args.padding * args.padding

    if quant_mode == "fp4":
        activation_output, c_global_sf = quant_dequant_fp4(
            activation_output.to(torch.bfloat16), False, True
        )
        activation_output = activation_output.to(torch.float)
        args.c_global_sf = c_global_sf
    elif quant_mode == "perTensorFp8":
        activation_output, c_global_sf = quant_dequant_per_tensor_fp8(
            activation_output.to(torch.bfloat16)
        )
        activation_output = activation_output.to(torch.float)
        args.c_global_sf = c_global_sf

    # Gemm2
    gemm2_output = torch.full(
        (total_num_padded_tokens, args.hidden_size), float("nan"), device="cuda"
    ).to(torch.float)
    i = 0
    for expert_idx in range(args.num_experts):
        my_num_tokens = num_tokens_per_expert[expert_idx]
        if my_num_tokens == 0:
            continue
        my_a = activation_output[i : i + my_num_tokens]
        my_b = args.gemm2_weights[expert_idx]
        my_c = my_a @ my_b.t()
        gemm2_output[i : i + my_num_tokens] = my_c
        i += my_num_tokens
        i = (i + args.padding - 1) // args.padding * args.padding
    # Finalize
    expert_weight = args.permute_info["topKLogits"].to(torch.float)
    finalize_output = torch.full(
        (args.num_tokens, args.hidden_size), float("nan"), device="cuda"
    ).to(torch.float)
    for i in range(args.num_tokens):
        acc = torch.zeros(args.hidden_size, dtype=torch.float, device="cuda")
        for top_k_idx in range(args.top_k):
            expanded_idx = i * args.top_k + top_k_idx
            permuted_idx = expanded_idx_to_permuted_idx[expanded_idx]
            original_vector = gemm2_output[permuted_idx]
            weight = (
                expert_weight[i, top_k_idx]
                if not args.use_routing_scales_on_input
                else 1.0
            )
            acc += original_vector * weight
        finalize_output[i] = acc
    return finalize_output


def e2m1_and_ufp8_scale_to_float_tensor_v2(
    e2m1_tensor: torch.Tensor,
    ufp8_scale_tensor: torch.Tensor,
    global_scale_tensor: torch.Tensor,
    sf_vec_size,
    ufp8_type: int = 1,
    is_sf_swizzled_layout: bool = True,
):
    float_tensor = torch.ops.tensorrt_llm.e2m1_and_ufp8sf_scale_to_float_v2(
        e2m1_tensor.cpu(),
        ufp8_scale_tensor.cpu().reshape(-1),
        global_scale_tensor.cpu(),
        sf_vec_size,
        ufp8_type,
        is_sf_swizzled_layout,
    )
    return float_tensor


def e2m1_and_ufp8_scale_batches(
    mat_fp4: torch.Tensor,
    scale_tensor: torch.Tensor,
    global_scale_tensor: torch.Tensor,
    sf_vec_size: int,
    ufp8_type: int = 1,
):
    num_batches = mat_fp4.size(0)

    scale_tensor = scale_tensor.view(num_batches, -1)

    tensors = [
        e2m1_and_ufp8_scale_to_float_tensor_v2(
            mat_fp4[b, :, :], scale_tensor[b, :], global_scale_tensor[b], sf_vec_size
        )
        for b in range(num_batches)
    ]

    result = torch.stack(tensors)

    return result


def run_moe_reference_fp4(args):
    sf_vec_size = 16

    hidden_states_dequant = e2m1_and_ufp8_scale_to_float_tensor_v2(
        args.hidden_states,
        args.hidden_states_scale,
        1 / args.hidden_states_scale_global,
        sf_vec_size,
    ).cuda()

    gemm1_weights_dequant = e2m1_and_ufp8_scale_batches(
        args.gemm1_weights, args.gemm1_scales, 1 / args.gemm1_scales_global, sf_vec_size
    ).cuda()

    gemm2_weights_dequant = e2m1_and_ufp8_scale_batches(
        args.gemm2_weights, args.gemm2_scales, 1 / args.gemm2_scales_global, sf_vec_size
    ).cuda()

    args_dequant = moe_args_dequant(
        args.num_tokens,
        args.num_experts,
        args.hidden_size,
        args.intermediate_size,
        args.top_k,
        args.padding,
        hidden_states_dequant,
        args.expert_logits,
        gemm1_weights_dequant,
        gemm2_weights_dequant,
        args.permute_info,
        args.use_routing_scales_on_input,
    )

    return run_moe_dequant(args_dequant, "fp4"), args_dequant


def run_moe_reference_dsfp8(args):
    hidden_states_dequant = dequant_reference_dsfp8(
        args.hidden_states, args.hidden_states_scale, True, False, True
    )

    gemm1_weights_dequant = {}
    for i in range(args.num_experts):
        gemm1_weights_dequant[i] = dequant_reference_dsfp8(
            args.gemm1_weights[i], args.gemm1_scales[i], False, True, True
        )

    gemm2_weights_dequant = {}
    for i in range(args.num_experts):
        gemm2_weights_dequant[i] = dequant_reference_dsfp8(
            args.gemm2_weights[i], args.gemm2_scales[i], False, True, True
        )

    args_dequant = moe_args_dequant(
        args.num_tokens,
        args.num_experts,
        args.hidden_size,
        args.intermediate_size,
        args.top_k,
        args.padding,
        hidden_states_dequant,
        args.expert_logits,
        gemm1_weights_dequant,
        gemm2_weights_dequant,
        args.permute_info,
        args.use_routing_scales_on_input,
    )

    return run_moe_dequant(args_dequant, "dsFp8"), args_dequant


def run_moe_reference_per_tensor_scale_fp8(args):

    hidden_states_dequant = (
        args.hidden_states.to(torch.float) / args.hidden_states_scale_global
    )

    gemm1_weights_dequant = {}
    for i in range(args.num_experts):
        gemm1_weights_dequant[i] = (
            args.gemm1_weights[i].to(torch.float) / args.gemm1_scales_global[i]
        )

    gemm2_weights_dequant = {}
    for i in range(args.num_experts):
        gemm2_weights_dequant[i] = (
            args.gemm2_weights[i].to(torch.float) / args.gemm2_scales_global[i]
        )

    args_dequant = moe_args_dequant(
        args.num_tokens,
        args.num_experts,
        args.hidden_size,
        args.intermediate_size,
        args.top_k,
        args.padding,
        hidden_states_dequant,
        args.expert_logits,
        gemm1_weights_dequant,
        gemm2_weights_dequant,
        args.permute_info,
        args.use_routing_scales_on_input,
    )

    return run_moe_dequant(args_dequant, "perTensorFp8"), args_dequant


def quant_fp4(a, use_ue8m0=False, is_sf_swizzled_layout=True):
    a_global_sf = (448 * 6) / a.float().abs().nan_to_num().max()
    sf_vec_size = 16

    a_fp4, a_sf = torch.ops.trtllm.fp4_quantize(
        a.cuda(), a_global_sf.cuda(), sf_vec_size, use_ue8m0, is_sf_swizzled_layout
    )

    return a_fp4, a_sf, a_global_sf


def quant_fp4_batches(a, num_experts, use_ue8m0=False, is_sf_swizzled_layout=True):
    quant_a = []
    sfs = []
    global_sfs = []
    for i in range(num_experts):
        a_fp4, a_sf, a_global_sf = quant_fp4(a[i], use_ue8m0, is_sf_swizzled_layout)
        quant_a.append(a_fp4)
        sfs.append(a_sf)
        global_sfs.append(a_global_sf)

    result_quant_a = torch.stack(quant_a)
    result_sfs = torch.stack(sfs)
    result_global_sfs = torch.stack(global_sfs)

    return result_quant_a, result_sfs, result_global_sfs


def quant_dequant_fp4(a, use_ue8m0=False, is_sf_swizzled_layout=True):
    a_global_sf = (448 * 6) / a.float().abs().nan_to_num().max()
    sf_vec_size = 16

    a_fp4, a_sf = torch.ops.trtllm.fp4_quantize(
        a.cuda(), a_global_sf.cuda(), sf_vec_size, use_ue8m0, is_sf_swizzled_layout
    )

    a_pt = e2m1_and_ufp8_scale_to_float_tensor_v2(
        a_fp4.cpu(), a_sf.cpu(), 1 / a_global_sf, sf_vec_size
    )

    return a_pt.cuda(), a_global_sf


def quant_fp8_per_tensor(a):
    a_global_sf = 448 / a.float().abs().nan_to_num().max()

    a_fp8 = (a * a_global_sf).to(torch.float8_e4m3fn)

    return a_fp8, a_global_sf


def quant_fp8_per_tensor_batches(a):
    num_batches = a.size(0)
    a_quant = []
    a_scales = []

    for i in range(num_batches):
        a_fp8, a_global_sf = quant_fp8_per_tensor(a[i])
        a_quant.append(a_fp8)
        a_scales.append(a_global_sf)

    result_a_quant = torch.stack(a_quant)
    result_a_scales = torch.stack(a_scales)

    return result_a_quant, result_a_scales


def quant_dequant_per_tensor_fp8(a):
    a_global_sf = 448 / a.float().abs().nan_to_num().max()
    a_fp8 = (a * a_global_sf).to(torch.float8_e4m3fn)
    a_pt = a_fp8.to(torch.float) / a_global_sf

    return a_pt.cuda(), a_global_sf


@pytest.mark.skipif(
    getSMVersion() != 100,
    reason="The kernel only supports Blackwell. Current SM is %d." % getSMVersion(),
)
@pytest.mark.parametrize("num_tokens", [16, 64, 1024, 4096])
@pytest.mark.parametrize(
    "expert_info", [(32, 8, 4, 8), (32, 1, 1, 5), (72, 1, 1, 6), (256, 8, 4, 8)]
)
@pytest.mark.parametrize("hidden_size", [512])
@pytest.mark.parametrize("intermediate_size", [512])
@pytest.mark.parametrize("use_autotune", [True, False], ids=["autotune", "no_autotune"])
def test_moe_fp8(num_tokens, expert_info, hidden_size, intermediate_size, use_autotune):
    torch.random.manual_seed(0)

    #
    # Data Generation
    #
    num_experts, n_groups, top_k_groups, top_k = expert_info
    padding = 8
    routed_scaling = 2.5
    routing_method_type = RoutingMethodType.DeepSeekV3
    tile_tokens_dim = 8 if num_tokens < 1024 else 32

    assert top_k <= num_experts
    assert top_k <= 8
    assert top_k_groups <= 4
    assert num_experts > n_groups
    assert num_experts % n_groups == 0
    assert num_experts % 4 == 0
    assert top_k < (top_k_groups * num_experts / n_groups)

    expert_logits = torch.randn((num_tokens, num_experts), device="cuda").to(
        torch.float
    )
    routing_bias = torch.randn(num_experts, device="cuda", dtype=torch.bfloat16)

    hidden_states = torch.randn((num_tokens, hidden_size), device="cuda").to(
        torch.float8_e4m3fn
    )
    hidden_states_scale = 2 * torch.rand(
        (hidden_size // 128, num_tokens), device="cuda"
    ).to(torch.float)

    gemm1_weights = torch.randn(
        (num_experts, 2 * intermediate_size, hidden_size), device="cuda"
    ).to(torch.float8_e4m3fn)
    gemm1_scales = 2 * torch.rand(
        (num_experts, 2 * intermediate_size // 128, hidden_size // 128), device="cuda"
    ).to(torch.float)
    gemm2_weights = torch.randn(
        (num_experts, hidden_size, intermediate_size), device="cuda"
    ).to(torch.float8_e4m3fn)
    gemm2_scales = 2 * torch.rand(
        (num_experts, hidden_size // 128, intermediate_size // 128), device="cuda"
    ).to(torch.float)

    permute_info, scores = routing_reference_no_aux(
        expert_logits,
        routing_bias,
        top_k,
        n_groups,
        top_k_groups,
        routed_scaling,
        padding,
    )

    args = moe_args(
        num_tokens,
        num_experts,
        hidden_size,
        intermediate_size,
        top_k,
        padding,
        hidden_states,
        hidden_states_scale,
        None,
        scores,
        gemm1_weights,
        gemm1_scales,
        None,
        gemm2_weights,
        gemm2_scales,
        None,
        permute_info,
        False,
    )

    with autotune(use_autotune):
        output = torch.ops.trtllm.fp8_block_scale_moe_runner(
            expert_logits,
            routing_bias,
            hidden_states,
            hidden_states_scale,
            gemm1_weights,
            gemm1_scales,
            gemm2_weights,
            gemm2_scales,
            num_experts,
            top_k,
            n_groups,
            top_k_groups,
            intermediate_size,
            0,
            num_experts,
            routed_scaling,
            tile_tokens_dim,
            routing_method_type,
        )

    output_dequant_actual = output.to(torch.float)
    #
    # Run the reference implementations
    #
    output_dequant_reference, _ = run_moe_reference_dsfp8(args)

    #
    # Check the results
    #
    def check_accuracy(a, b, atol, rtol, percent):
        if torch.any(torch.isnan(a)):
            raise Exception("NaN in a")
        if torch.any(torch.isnan(b)):
            raise Exception("NaN in b")
        assert a.shape == b.shape
        left = torch.abs(a - b)
        right = atol + rtol * torch.abs(b)
        count = torch.sum(left > right)
        mismatch_percent = count / a.numel()
        if mismatch_percent > 1 - percent:
            raise Exception(
                "Mismatch percentage is %f for rtol %f" % (mismatch_percent, rtol)
            )

    check_accuracy(
        output_dequant_reference,
        output_dequant_actual,
        atol=0.1,
        rtol=0.85,
        percent=0.925,
    )


@pytest.mark.skipif(
    getSMVersion() != 100,
    reason="The kernel only supports Blackwell. Current SM is %d." % getSMVersion(),
)
@pytest.mark.parametrize("num_tokens", [1, 1024, 4096])
@pytest.mark.parametrize("hidden_size", [1024])
@pytest.mark.parametrize("intermediate_size", [1024, 768, 384, 192])
@pytest.mark.parametrize(
    "routing_info",
    [
        pytest.param(
            {
                "num_experts": 256,
                "top_k": 8,
                "padding": 8,
                "n_groups": 8,
                "top_k_groups": 4,
                "routed_scaling": 2.5,
                "has_routing_bias": True,
                "routing_method_type": RoutingMethodType.DeepSeekV3,
            },
            id="RoutingDSv3",
        ),
        pytest.param(
            {
                "num_experts": 72,
                "top_k": 6,
                "padding": 8,
                "n_groups": 1,
                "top_k_groups": 1,
                "routed_scaling": 2.5,
                "has_routing_bias": True,
                "routing_method_type": RoutingMethodType.DeepSeekV3,
            },
            id="RoutingDSlite",
        ),
        pytest.param(
            {
                "num_experts": 128,
                "top_k": 8,
                "padding": 8,
                "n_groups": None,
                "top_k_groups": None,
                "routed_scaling": None,
                "has_routing_bias": False,
                "routing_method_type": RoutingMethodType.Renormalize,
            },
            id="RoutingRenormalize",
        ),
        pytest.param(
            {
                "num_experts": 128,
                "top_k": 8,
                "padding": 8,
                "n_groups": None,
                "top_k_groups": None,
                "routed_scaling": None,
                "has_routing_bias": False,
                "routing_method_type": RoutingMethodType.RenormalizeNaive,
            },
            id="RoutingRenormalizeNaive",
        ),
    ],
)
def test_moe_fp4(num_tokens, hidden_size, intermediate_size, routing_info):
    torch.random.manual_seed(0)

    #
    # Data Generation
    #

    top_k = routing_info["top_k"]
    # FIXME: set to TileN size
    padding = routing_info["padding"]
    n_groups = routing_info["n_groups"]
    top_k_groups = routing_info["top_k_groups"]
    routed_scaling = routing_info["routed_scaling"]
    num_experts = routing_info["num_experts"]
    routing_method_type = routing_info["routing_method_type"]
    tile_tokens_dim = 8

    assert top_k <= num_experts
    assert top_k <= 8
    if (top_k_groups is not None) and (n_groups is not None):
        assert top_k_groups <= 4
        assert num_experts > n_groups
        assert num_experts % n_groups == 0
        assert num_experts % 4 == 0
        assert top_k < (top_k_groups * num_experts / n_groups)

    if routing_method_type == RoutingMethodType.DeepSeekV3:
        expert_logits = torch.randn((num_tokens, num_experts), device="cuda").to(
            torch.float
        )
    elif (
        routing_method_type == RoutingMethodType.RenormalizeNaive
        or routing_method_type == RoutingMethodType.Renormalize
    ):
        expert_logits = torch.randn((num_tokens, num_experts), device="cuda").to(
            torch.bfloat16
        )

    if routing_info["has_routing_bias"]:
        routing_bias = torch.randn(num_experts, device="cuda", dtype=torch.bfloat16)
    else:
        routing_bias = None

    hidden_states = 2 * torch.randn(
        (num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16
    )
    gemm1_weights = torch.randn(
        (num_experts, 2 * intermediate_size, hidden_size),
        device="cuda",
        dtype=torch.bfloat16,
    )
    gemm2_weights = torch.randn(
        (num_experts, hidden_size, intermediate_size),
        device="cuda",
        dtype=torch.bfloat16,
    )

    use_ue8m0 = False
    # Quantize hidden states. Produces scales for activations in 128x4 layout for ref impl.
    (
        hidden_states_fp4_bytes,
        hidden_states_scale_fp4_bytes,
        hidden_states_scale_global,
    ) = quant_fp4(hidden_states, use_ue8m0, True)
    # We do it twice to get the linear layout for scales for the FP4 kernels.
    _, hidden_states_scale_linear_fp4_bytes, _ = quant_fp4(
        hidden_states, use_ue8m0, False
    )

    hidden_states_fp4 = hidden_states_fp4_bytes.reshape(
        num_tokens, hidden_size // 2
    )  # packed fp4

    hidden_states_scale_linear_fp4 = hidden_states_scale_linear_fp4_bytes.view(
        torch.float8_e4m3fn
    )  # fp8 scaling factors

    # Quantize the weights for FC1. Produces scales for weights in 128x4 layout for ref impl.
    gemm1_weights_fp4_bytes, gemm1_scales_fp4_bytes, gemm1_scales_global = (
        quant_fp4_batches(gemm1_weights, num_experts, use_ue8m0, True)
    )
    # We do it twice to get the linear layout for scales for the FP4 kernels.
    _, gemm1_scales_linear_fp4_bytes, _ = quant_fp4_batches(
        gemm1_weights, num_experts, use_ue8m0, False
    )

    gemm1_weights_fp4 = gemm1_weights_fp4_bytes.view(torch.float8_e4m3fn).reshape(
        num_experts, 2 * intermediate_size, hidden_size // 2
    )  # packed fp4
    gemm1_scales_linear_fp4 = gemm1_scales_linear_fp4_bytes.view(
        torch.float8_e4m3fn
    ).reshape(
        num_experts, 2 * intermediate_size, hidden_size // 16
    )  # fp8 scaling factors

    # Quantize the weights for FC2. Produces scales for weights in 128x4 layout for ref impl.
    gemm2_weights_fp4_bytes, gemm2_scales_fp4_bytes, gemm2_scales_global = (
        quant_fp4_batches(gemm2_weights, num_experts, use_ue8m0, True)
    )
    # We do it twice to get the linear layout for scales for the FP4 kernels.
    _, gemm2_scales_linear_fp4_bytes, _ = quant_fp4_batches(
        gemm2_weights, num_experts, use_ue8m0, False
    )

    gemm2_weights_fp4 = gemm2_weights_fp4_bytes.view(torch.float8_e4m3fn).reshape(
        num_experts, hidden_size, intermediate_size // 2
    )  # packed fp4
    gemm2_scales_linear_fp4 = gemm2_scales_linear_fp4_bytes.view(
        torch.float8_e4m3fn
    ).reshape(
        num_experts, hidden_size, intermediate_size // 16
    )  # fp8 scaling factors
    if routing_method_type == RoutingMethodType.DeepSeekV3:
        permute_info, scores = routing_reference_no_aux(
            expert_logits,
            routing_bias,
            top_k,
            n_groups,
            top_k_groups,
            routed_scaling,
            padding,
        )
    elif routing_method_type == RoutingMethodType.Renormalize:
        permute_info, scores = routing_reference_renormalize(
            expert_logits, top_k, num_experts, padding
        )
    elif routing_method_type == RoutingMethodType.RenormalizeNaive:
        permute_info, scores = routing_reference_renormalize_naive(
            expert_logits, top_k, num_experts, padding
        )

    args = moe_args(
        num_tokens,
        num_experts,
        hidden_size,
        intermediate_size,
        top_k,
        padding,
        hidden_states_fp4_bytes,
        hidden_states_scale_fp4_bytes,
        hidden_states_scale_global,
        scores,
        gemm1_weights_fp4_bytes,
        gemm1_scales_fp4_bytes,
        gemm1_scales_global,
        gemm2_weights_fp4_bytes,
        gemm2_scales_fp4_bytes,
        gemm2_scales_global,
        permute_info,
        False,
    )
    #
    # Run the reference implementations
    #
    # It is important to run the reference implementation before the TRT-LLM kernel
    # because the MoE shuffles the weights in-place.
    output_dequant_reference, args_dequant = run_moe_reference_fp4(args)

    # FIXME: this depends on the kernel internals
    epilogue_tile_m = 128

    # Reorder rows of W1 and scales for fused gated activation
    gemm1_weights_fp4_interleaved = []
    gemm1_scales_fp4_interleaved = []
    for i in range(num_experts):
        gemm1_weights_fp4_interleaved.append(
            reorder_rows_for_gated_act_gemm(gemm1_weights_fp4[i].clone())
        )
        gemm1_scales_fp4_interleaved.append(
            reorder_rows_for_gated_act_gemm(gemm1_scales_linear_fp4[i].clone())
        )

    # Stack weights and scales for all experts
    gemm1_weights_fp4_interleaved = torch.stack(gemm1_weights_fp4_interleaved).reshape(
        num_experts, 2 * intermediate_size, hidden_size // 2
    )
    gemm1_scales_fp4_interleaved = torch.stack(gemm1_scales_fp4_interleaved).reshape(
        num_experts, 2 * intermediate_size, hidden_size // 16
    )

    # Shuffle weights and scaling factors for transposed mma output
    gemm1_weights_fp4_shuffled = []
    gemm1_scales_fp4_shuffled = []
    gemm2_weights_fp4_shuffled = []
    gemm2_scales_fp4_shuffled = []
    for i in range(num_experts):
        gemm1_weights_fp4_shuffled.append(
            shuffle_matrix_a(
                gemm1_weights_fp4_interleaved[i].view(torch.uint8), epilogue_tile_m
            )
        )
        gemm1_scales_fp4_shuffled.append(
            shuffle_matrix_sf_a(
                gemm1_scales_fp4_interleaved[i].view(torch.uint8), epilogue_tile_m
            )
        )

        gemm2_weights_fp4_shuffled.append(
            shuffle_matrix_a(gemm2_weights_fp4[i].view(torch.uint8), epilogue_tile_m)
        )
        gemm2_scales_fp4_shuffled.append(
            shuffle_matrix_sf_a(
                gemm2_scales_linear_fp4[i].view(torch.uint8), epilogue_tile_m
            )
        )

    # Stack weights for all experts
    gemm1_weights_fp4_shuffled = torch.stack(gemm1_weights_fp4_shuffled)
    gemm1_scales_fp4_shuffled = (
        torch.stack(gemm1_scales_fp4_shuffled)
        .view(torch.float8_e4m3fn)
        .reshape(num_experts, 2 * intermediate_size, hidden_size // 16)
    )

    gemm2_weights_fp4_shuffled = torch.stack(gemm2_weights_fp4_shuffled)
    gemm2_scales_fp4_shuffled = (
        torch.stack(gemm2_scales_fp4_shuffled)
        .view(torch.float8_e4m3fn)
        .reshape(num_experts, hidden_size, intermediate_size // 16)
    )

    #
    # Run the TRT-LLM kernel
    #

    # c_global_sf: fc2_input_scale
    scale_c_fc1 = (
        args_dequant.c_global_sf
        * (1.0 / args.gemm1_scales_global)
        * (1.0 / args.hidden_states_scale_global)
    )

    # self.fc31_alpha
    scale_gate_fc1 = (1.0 / args.gemm1_scales_global) * (
        1.0 / args.hidden_states_scale_global
    )

    # self.fc2_alpha
    scale_c_fc2 = (1.0 / args_dequant.c_global_sf) * (1.0 / args.gemm2_scales_global)

    output = torch.ops.trtllm.fp4_block_scale_moe_runner(
        expert_logits,
        routing_bias,
        hidden_states_fp4,
        hidden_states_scale_linear_fp4,
        gemm1_weights_fp4_shuffled,
        gemm1_scales_fp4_shuffled,
        gemm2_weights_fp4_shuffled,
        gemm2_scales_fp4_shuffled,
        scale_c_fc1,
        scale_gate_fc1,
        scale_c_fc2,
        num_experts,
        top_k,
        n_groups,
        top_k_groups,
        intermediate_size,
        0,
        num_experts,
        routed_scaling,
        tile_tokens_dim,
        routing_method_type,
        do_finalize=True,
    )

    output_dequant_actual = output[0].to(torch.float)

    #
    # Check the results
    #
    def check_accuracy(a, b, atol, rtol, percent):
        if torch.any(torch.isnan(a)):
            raise Exception("NaN in a")
        if torch.any(torch.isnan(b)):
            raise Exception("NaN in b")
        assert a.shape == b.shape
        left = torch.abs(a - b)
        right = atol + rtol * torch.abs(b)
        count = torch.sum(left > right)
        mismatch_percent = count / a.numel()
        if mismatch_percent > 1 - percent:
            raise Exception(
                "Mismatch percentage is %f for rtol %f" % (mismatch_percent, rtol)
            )

    check_accuracy(
        output_dequant_reference,
        output_dequant_actual,
        atol=0.1,
        rtol=0.85,
        percent=0.925,
    )


@pytest.mark.skipif(
    getSMVersion() != 100,
    reason="The kernel only supports Blackwell. Current SM is %d." % getSMVersion(),
)
@pytest.mark.parametrize("num_tokens", [1, 2, 16, 64, 1024, 4096])
@pytest.mark.parametrize("expert_info", [(128, 0, 0, 1, True)])
@pytest.mark.parametrize("hidden_size", [2048])
@pytest.mark.parametrize("intermediate_size", [2048])
def test_moe_fp8_per_tensor_scale(
    num_tokens, expert_info, hidden_size, intermediate_size
):
    torch.random.manual_seed(0)

    #
    # Data Generation
    #
    num_experts, n_groups, top_k_groups, top_k, use_routing_scales_on_input = (
        expert_info
    )
    # FIXME: set to TileN size
    padding = 8
    routed_scaling = 2.5
    routing_method_type = RoutingMethodType.Llama4
    tile_tokens_dim = 8

    assert top_k <= num_experts
    assert top_k <= 8
    assert top_k_groups <= 4
    assert num_experts > n_groups
    assert n_groups == 0 or num_experts % n_groups == 0
    assert num_experts % 4 == 0
    assert n_groups == 0 or top_k < (top_k_groups * num_experts / n_groups)

    expert_logits = torch.randn((num_tokens, num_experts), device="cuda").to(
        torch.float
    )
    routing_bias = torch.randn(num_experts, device="cuda", dtype=torch.bfloat16)

    hidden_states = torch.randn((num_tokens, hidden_size), device="cuda").to(
        torch.bfloat16
    )

    gemm1_weights = torch.randn(
        (num_experts, 2 * intermediate_size, hidden_size), device="cuda"
    ).to(torch.bfloat16)
    gemm2_weights = torch.randn(
        (num_experts, hidden_size, intermediate_size), device="cuda"
    ).to(torch.bfloat16)

    hidden_states_quant, hidden_states_global_scale = quant_fp8_per_tensor(
        hidden_states
    )
    gemm1_weights_quant, gemm1_global_scales = quant_fp8_per_tensor_batches(
        gemm1_weights
    )
    gemm2_weights_quant, gemm2_global_scales = quant_fp8_per_tensor_batches(
        gemm2_weights
    )

    permute_info, scores = routing_reference_no_aux(
        expert_logits,
        routing_bias,
        top_k,
        n_groups,
        top_k_groups,
        routed_scaling,
        padding,
        use_routing_scales_on_input,
    )

    args = moe_args(
        num_tokens,
        num_experts,
        hidden_size,
        intermediate_size,
        top_k,
        padding,
        hidden_states_quant,
        None,
        hidden_states_global_scale,
        scores,
        gemm1_weights_quant,
        None,
        gemm1_global_scales,
        gemm2_weights_quant,
        None,
        gemm2_global_scales,
        permute_info,
        use_routing_scales_on_input,
    )
    #
    # Run the reference implementations
    #
    # It is important to run the reference implementation before the TRT-LLM kernel
    # because the MoE shuffles the weights in-place.
    output_dequant_reference, args_dequant = run_moe_reference_per_tensor_scale_fp8(
        args
    )

    # FIXME: this depends on the kernel internals
    epilogue_tile_m = 128

    # Reorder rows of W1 for fused gated activation
    gemm1_weights_fp8_interleaved = []
    for i in range(num_experts):
        gemm1_weights_fp8_interleaved.append(
            reorder_rows_for_gated_act_gemm(gemm1_weights_quant[i].clone())
        )

    # Stack weights and scales for all experts
    gemm1_weights_fp8_interleaved = torch.stack(gemm1_weights_fp8_interleaved).reshape(
        num_experts, 2 * intermediate_size, hidden_size
    )

    # Shuffle weights and scaling factors for transposed mma output
    gemm1_weights_fp8_shuffled = []
    gemm2_weights_fp8_shuffled = []
    for i in range(num_experts):
        gemm1_weights_fp8_shuffled.append(
            shuffle_matrix_a(
                gemm1_weights_fp8_interleaved[i].view(torch.uint8), epilogue_tile_m
            )
        )

        gemm2_weights_fp8_shuffled.append(
            shuffle_matrix_a(gemm2_weights_quant[i].view(torch.uint8), epilogue_tile_m)
        )

    # Stack weights for all experts
    gemm1_weights_fp8_shuffled = torch.stack(gemm1_weights_fp8_shuffled).view(
        torch.float8_e4m3fn
    )
    gemm2_weights_fp8_shuffled = torch.stack(gemm2_weights_fp8_shuffled).view(
        torch.float8_e4m3fn
    )

    # c_global_sf: fc2_input_scale
    scale_c_fc1 = (
        args_dequant.c_global_sf
        * (1.0 / args.gemm1_scales_global)
        * (1.0 / args.hidden_states_scale_global)
    )

    # self.fc31_alpha
    scale_gate_fc1 = (1.0 / args.gemm1_scales_global) * (
        1.0 / args.hidden_states_scale_global
    )

    # self.fc2_alpha
    scale_c_fc2 = (1.0 / args_dequant.c_global_sf) * (1.0 / args.gemm2_scales_global)

    output = torch.ops.trtllm.fp8_per_tensor_scale_moe_runner(
        (
            expert_logits.to(torch.bfloat16)
            if use_routing_scales_on_input
            else expert_logits
        ),
        routing_bias,
        hidden_states_quant,
        gemm1_weights_fp8_shuffled,
        scale_c_fc1,
        scale_gate_fc1,
        gemm2_weights_fp8_shuffled,
        scale_c_fc2,
        num_experts,
        top_k,
        n_groups,
        top_k_groups,
        intermediate_size,
        0,
        num_experts,
        routed_scaling,
        use_routing_scales_on_input,
        tile_tokens_dim,
        routing_method_type,
    )

    output_dequant_actual = output.to(torch.float)

    #
    # Check the results
    #
    def check_accuracy(a, b, atol, rtol, percent):
        if torch.any(torch.isnan(a)):
            raise Exception("NaN in a")
        if torch.any(torch.isnan(b)):
            raise Exception("NaN in b")
        assert a.shape == b.shape
        left = torch.abs(a - b)
        right = atol + rtol * torch.abs(b)
        count = torch.sum(left > right)
        mismatch_percent = count / a.numel()
        if mismatch_percent > 1 - percent:
            raise Exception(
                "Mismatch percentage is %f for rtol %f" % (mismatch_percent, rtol)
            )

    check_accuracy(
        output_dequant_reference,
        output_dequant_actual,
        atol=0.1,
        rtol=0.85,
        percent=0.925,
    )
