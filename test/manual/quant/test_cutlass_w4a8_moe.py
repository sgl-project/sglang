# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import pytest
import torch

from sglang.srt.layers.moe.cutlass_w4a8_moe import cutlass_w4a8_moe
from sglang.srt.layers.moe.topk import TopKConfig, select_experts


def pack_int4_values_to_int8(int4_values_interleaved: torch.Tensor) -> torch.Tensor:
    if int4_values_interleaved.shape[-1] % 2 != 0:
        raise ValueError(
            "the last dim size of int4_values_interleaved tensor must be even."
        )

    input_tensor_int8 = int4_values_interleaved.to(torch.int8)

    low_nibbles = input_tensor_int8[..., 0::2]
    high_nibbles = input_tensor_int8[..., 1::2]

    packed_tensor = (high_nibbles << 4) | (low_nibbles & 0x0F)

    return packed_tensor.to(torch.int8)


def pack_interleave(num_experts, ref_weight, ref_scale, alignment=4):
    n, k = ref_weight.shape[1], ref_weight.shape[2]

    weight = pack_int4_values_to_int8(ref_weight.cpu()).cuda()
    w_q = weight.view((num_experts, n, k // 2)).view(torch.int8)
    w_q = w_q.contiguous()

    scale_interleaved = ref_scale.reshape(
        ref_scale.shape[0],
        ref_scale.shape[1],
        (ref_scale.shape[2] // alignment),
        alignment,
    )  # [E, N, K/4, 4]
    scale_interleaved = scale_interleaved.permute(0, 2, 1, 3)  # [E, K/4, N, 4]
    scale_interleaved = scale_interleaved.reshape(
        ref_scale.shape[0],
        ref_scale.shape[2] // alignment,
        ref_scale.shape[1] * alignment,
    )  # [E, K/4, N*4]
    w_scale = scale_interleaved.contiguous()

    return w_q, w_scale


@pytest.mark.parametrize("M", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("N", [2048])
@pytest.mark.parametrize("K", [7168])
@pytest.mark.parametrize("E", [256])
@pytest.mark.parametrize("tp_size", [8])
@pytest.mark.parametrize("use_ep_moe", [True, False])
@pytest.mark.parametrize("topk", [8])
@pytest.mark.parametrize("group_size", [128])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_cutlass_w4a8_moe(M, N, K, E, tp_size, use_ep_moe, topk, group_size, dtype):
    if use_ep_moe:
        local_e = E // tp_size
    else:  # tp mode
        local_e = E
        N = N // tp_size

    debug = False
    if debug:
        a = torch.ones((M, K), dtype=dtype, device="cuda") * 0.001
        ref_weight_1 = torch.ones((local_e, N * 2, K), dtype=torch.int8, device="cuda")
        ref_weight_2 = torch.ones((local_e, K, N), dtype=torch.int8, device="cuda")
        a1_scale = torch.ones(1, dtype=torch.float32, device="cuda")
        a2_scale = torch.ones(1, dtype=torch.float32, device="cuda")
        scale_1 = torch.ones(
            (local_e, N * 2, K // group_size), dtype=dtype, device="cuda"
        )
        scale_2 = torch.ones((local_e, K, N // group_size), dtype=dtype, device="cuda")
    else:
        a = torch.randn(M, K, dtype=dtype, device="cuda")
        ref_weight_1 = torch.randint(
            -8, 8, (local_e, N * 2, K), dtype=torch.int8, device="cuda"
        )
        ref_weight_2 = torch.randint(
            -8, 8, (local_e, K, N), dtype=torch.int8, device="cuda"
        )
        affine_coeff = 0.005
        a1_scale = torch.randn(1, dtype=torch.float32, device="cuda")
        a2_scale = torch.randn(1, dtype=torch.float32, device="cuda")
        scale_1 = (
            torch.randn(local_e, N * 2, K // group_size, dtype=dtype, device="cuda")
            * affine_coeff
        )
        scale_2 = (
            torch.randn(local_e, K, N // group_size, dtype=dtype, device="cuda")
            * affine_coeff
        )

    w1_q, w1_scale = pack_interleave(local_e, ref_weight_1, scale_1)
    if use_ep_moe:
        w2_q, w2_scale = pack_interleave(local_e, ref_weight_2, scale_2)
    else:
        w2_q, w2_scale = pack_interleave(local_e, ref_weight_2, scale_2, 1)

    device = "cuda"
    a_strides1 = torch.full((local_e, 3), K, device=device, dtype=torch.int64)
    c_strides1 = torch.full((local_e, 3), 2 * N, device=device, dtype=torch.int64)
    a_strides2 = torch.full((local_e, 3), N, device=device, dtype=torch.int64)
    c_strides2 = torch.full((local_e, 3), K, device=device, dtype=torch.int64)
    b_strides1 = a_strides1
    s_strides13 = c_strides1
    b_strides2 = a_strides2
    s_strides2 = c_strides2

    score = torch.randn((M, E), dtype=dtype, device=device)
    topk_output = select_experts(
        hidden_states=a,
        router_logits=score,
        topk_config=TopKConfig(top_k=topk, renormalize=False),
    )
    topk_weights, topk_ids, _ = topk_output
    expert_map = torch.arange(E, dtype=torch.int32, device=device)
    expert_map[local_e:] = -1

    output = cutlass_moe(
        a,
        w1_q,
        w2_q,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids,
        a_strides1,
        b_strides1,
        c_strides1,
        a_strides2,
        b_strides2,
        c_strides2,
        s_strides13,
        s_strides2,
        local_e,
        a1_scale,
        a2_scale,
        expert_map,
    )

    ref_output = ref(
        a,
        local_e,
        topk_weights,
        topk_ids,
        ref_weight_1,
        ref_weight_2,
        scale_1,
        scale_2,
        has_pre_quant=True,
        has_alpha=True,
        pre_quant_scale_1=a1_scale,
        pre_quant_scale_2=a2_scale,
        alpha_1=a1_scale,
        alpha_2=a2_scale,
    )

    # compare
    torch.cuda.synchronize()

    # compare final output
    torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=0.1)
    print("SUCCESS: Final output tensors are close.")


def cutlass_moe(
    a: torch.Tensor,
    w1_q: torch.Tensor,
    w2_q: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    a_strides1: torch.Tensor,
    b_strides1: torch.Tensor,
    c_strides1: torch.Tensor,
    a_strides2: torch.Tensor,
    b_strides2: torch.Tensor,
    c_strides2: torch.Tensor,
    s_strides13: torch.Tensor,
    s_strides2: torch.Tensor,
    num_local_experts: int,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    expert_map: Optional[torch.Tensor] = None,
    apply_router_weight_on_input: bool = False,
):
    topk_ids = expert_map[topk_ids]
    device = a.device

    expert_offsets = torch.empty(
        (num_local_experts + 1), dtype=torch.int32, device=device
    )
    problem_sizes1 = torch.empty(
        (num_local_experts, 3), dtype=torch.int32, device=device
    )
    problem_sizes2 = torch.empty(
        (num_local_experts, 3), dtype=torch.int32, device=device
    )
    return cutlass_w4a8_moe(
        a,
        w1_q,
        w2_q,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids,
        a_strides1,
        b_strides1,
        c_strides1,
        a_strides2,
        b_strides2,
        c_strides2,
        s_strides13,
        s_strides2,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        a1_scale,
        a2_scale,
        apply_router_weight_on_input,
    )


def ref(
    x: torch.Tensor,
    num_experts: int,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    ref_weight_1: torch.Tensor,
    ref_weight_2: torch.Tensor,
    ref_weight_scale_1: torch.Tensor,
    ref_weight_scale_2: torch.Tensor,
    has_pre_quant: bool = False,
    has_alpha: bool = False,
    pre_quant_scale_1: Optional[torch.Tensor] = None,
    pre_quant_scale_2: Optional[torch.Tensor] = None,
    alpha_1: Optional[torch.Tensor] = None,
    alpha_2: Optional[torch.Tensor] = None,
):
    results = torch.zeros_like(x)
    dtype = x.dtype
    for e_idx in range(num_experts):
        mask = topk_ids == e_idx
        activated_tokens = mask.sum(1).bool()
        act = x[activated_tokens, :]
        if act.shape[0] == 0:
            continue
        final_scale = (topk_weights * mask).sum(1)[activated_tokens].unsqueeze(1)

        act = (
            torch.clamp((act / pre_quant_scale_1.float()), -448.0, 448.0)
            .to(torch.float8_e4m3fn)
            .to(dtype)
        )
        w3_w1 = ref_weight_1[e_idx]
        ref_w_scale_repeat = (
            ref_weight_scale_1[e_idx].repeat_interleave(128, dim=1).to(float)
        )
        w3_w1 = (w3_w1.to(float) * ref_w_scale_repeat).to(dtype)
        fc1 = ((torch.matmul(act, w3_w1.T)) * alpha_1).to(torch.float16)

        gate, fc1 = fc1.chunk(2, dim=-1)
        fc1 = fc1 * torch.nn.functional.silu(gate)
        act = torch.clamp((fc1 / pre_quant_scale_2.float()), -448.0, 448.0).to(
            torch.float8_e4m3fn
        )
        act = act.to(dtype)

        w2 = ref_weight_2[e_idx]
        ref_w_scale_repeat = (
            ref_weight_scale_2[e_idx].repeat_interleave(128, dim=1).to(float)
        )
        w2 = (w2.to(float) * ref_w_scale_repeat).to(dtype)
        fc2 = (torch.matmul(act, w2.T) * alpha_2).to(torch.float16)

        results[activated_tokens, :] += (fc2 * final_scale).to(results.dtype)

    return results
