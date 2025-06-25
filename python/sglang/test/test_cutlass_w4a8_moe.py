# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

from typing import Optional

from sglang.srt.layers.moe.topk import select_experts
from sglang.srt.layers.moe.ep_moe.kernels import (
    pre_reorder_triton_kernel,
    pre_reorder_triton_kernel_for_cutlass_moe,
    run_moe_ep_preproess,
    run_cutlass_moe_ep_preproess,
)
from sglang.srt.layers.moe.cutlass_w4a8_moe import cutlass_w4a8_moe

debug = False
# debug = False
print_info = False

def print_tensor_info(name, tensor):
    if not print_info:
        return
    print(f"\n{name}:")
    print(f"  shape: {tensor.shape}")
    print(f"  values: {tensor.flatten()[:10]}")  # Print first 10 values


def pack_int4_values_to_int8(int4_values_interleaved: torch.Tensor) -> torch.Tensor:
    if int4_values_interleaved.shape[-1] % 2 != 0:
        raise ValueError(
            "int4_values_interleaved 的最后一个维度的大小必须是偶数。"
        )

    input_tensor_int8 = int4_values_interleaved.to(torch.int8)

    # 分离低位和高位半字节的值
    # a[..., 0::2] 取最后一个维度上索引为偶数的元素
    # a[..., 1::2] 取最后一个维度上索引为奇数的元素
    low_nibbles = input_tensor_int8[..., 0::2]
    high_nibbles = input_tensor_int8[..., 1::2]

    packed_tensor = (high_nibbles << 4) | (low_nibbles & 0x0F)
    
    return packed_tensor.to(torch.int8)


def pack_interleave(num_experts, ref_weight, ref_scale):
    n, k = ref_weight.shape[1], ref_weight.shape[2]
    # packer = torch.ops.trtllm.pack_int8_tensor_to_packed_int4

    # weight = packer(ref_weight.cpu()).cuda()
    weight = pack_int4_values_to_int8(ref_weight.cpu()).cuda()
    w_q = weight.view((num_experts, n, k // 2)).view(torch.int8)
    # w_q = w_q.contiguous().transpose(1, 2)
    w_q = w_q.contiguous()

    ###############################################################
    # scale interleave, [E, K, N]
    scale = ref_scale.permute(0, 2, 1)  # [E, N, K]
    # scale = ref_scale
    scale_interleaved = scale.reshape(
        scale.shape[0], scale.shape[1], (scale.shape[2] // 4), 4
    )  # [E, N, K/4, 4]
    scale_interleaved = scale_interleaved.permute(0, 2, 1, 3)  # [E, K/4, N, 4]
    scale_interleaved = scale_interleaved.reshape(
        scale.shape[0], scale.shape[2] // 4, scale.shape[1] * 4
    )  # [E, K/4, N*4]
    w_scale = scale_interleaved.contiguous()

    return w_q, w_scale


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_fused_moe_w4afp8(dtype):

    M = 5
    K = 7168
    N = 2048
    group_size = 128
    E = 256
    local_e = 32
    # local_e = 32
    topk = 8
    dtype = torch.bfloat16

    if debug:
        a = torch.ones((M, k), dtype=dtype, device='cuda') * 0.001
        # a[1:] = 0.02
        ref_weight_1 = torch.ones((local_e, N * 2, K),
                              dtype=torch.int8,
                              device="cuda")
        ref_weight_2 = torch.ones((local_e, K, N),
                              dtype=torch.int8,
                              device="cuda")
        a1_scale = torch.ones(1, dtype=torch.float32, device="cuda")
        a2_scale = torch.ones(1, dtype=torch.float32, device="cuda")
        scale_1 = torch.ones(
            (local_e, K // group_size, N * 2), dtype=dtype,
            device="cuda")
        scale_2 = torch.ones(
            (local_e, N // group_size, K), dtype=dtype,
            device="cuda")
    else:
        a = torch.randn(M, K, dtype=dtype, device='cuda')
        ref_weight_1 = torch.randint(-8,
                                 8, (local_e, N * 2, K),
                                 dtype=torch.int8,
                                 device='cuda')
        ref_weight_2 = torch.randint(-8,
                                 8, (local_e, K, N),
                                 dtype=torch.int8,
                                 device='cuda')
        affine_coeff = 0.005
        a1_scale = torch.randn(1, dtype=torch.float32, device="cuda")
        a2_scale = torch.randn(1, dtype=torch.float32, device="cuda")
        # a1_scale = torch.ones(1, dtype=torch.float32, device="cuda")
        # a2_scale = torch.ones(1, dtype=torch.float32, device="cuda")
        scale_1 = torch.randn(
            local_e, K // group_size, N * 2, dtype=dtype,
            device="cuda") * affine_coeff
        scale_2 = torch.randn(
            local_e, N // group_size, K, dtype=dtype,
            device="cuda") * affine_coeff


    w1_q, w1_scale = pack_interleave(local_e, ref_weight_1, scale_1)
    w2_q, w2_scale = pack_interleave(local_e, ref_weight_2, scale_2)
    print("w1_q.shape", w1_q.shape)
    print("w1_scale.shape", w1_scale.shape)

    device = "cuda"
    a_strides1 = torch.full((local_e, 3),
                                    K,
                                    device=device,
                                    dtype=torch.int64)
    c_strides1 = torch.full((local_e, 3),
                                    2 * N,
                                    device=device,
                                    dtype=torch.int64)
    a_strides2 = torch.full((local_e, 3),
                                    N,
                                    device=device,
                                    dtype=torch.int64)
    c_strides2 = torch.full((local_e, 3),
                                    K,
                                    device=device,
                                    dtype=torch.int64)
    b_strides1 = a_strides1
    s_strides13 = c_strides1
    b_strides2 = a_strides2
    s_strides2 = c_strides2

    score = torch.randn((M, E), device="cuda", dtype=dtype)
    topk_weights, topk_ids = select_experts(
        hidden_states=a,
        router_logits=score,
        top_k=topk,
        use_grouped_topk=False,
        renormalize=False,
    )
    expert_map = torch.arange(E, dtype=torch.int32, device="cuda")
    expert_map[local_e:] = E

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
        0,
        local_e - 1,
        E,
        a1_scale,
        a2_scale,
        expert_map,
    )

    ref_output, ref_tensors = ref(
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

    # compare_intermediate_val(cutlass_tensors, ref_tensors)

    # compare final output
    print("\nComparing final output tensors...")
    print("output", output)
    print("ref_output", ref_output)
    torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=0.1)
    # woq_assert_near_eq(ref_output, output, 2)
    print("SUCCESS: Final output tensors are close.")


def cutlass_moe(
    a: torch.Tensor,
    w1_q: torch.Tensor,
    w2_q: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids_: torch.Tensor,
    a_strides1: torch.Tensor,
    b_strides1: torch.Tensor,
    c_strides1: torch.Tensor,
    a_strides2: torch.Tensor,
    b_strides2: torch.Tensor,
    c_strides2: torch.Tensor,
    s_strides13: torch.Tensor,
    s_strides2: torch.Tensor,
    start_expert_id: int,
    end_expert_id: int,
    E: int,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    expert_map: Optional[torch.Tensor] = None,
    apply_router_weight_on_input: bool = False,
):
    local_topk_ids = topk_ids_
    local_topk_ids = torch.where(expert_map[topk_ids_] != E,
                                    expert_map[topk_ids_], E)
    device = a.device

    local_num_experts = end_expert_id - start_expert_id + 1
    expert_offsets = torch.empty((local_num_experts + 1),
                                dtype=torch.int32,
                                device=device)
    problem_sizes1 = torch.empty((local_num_experts, 3),
                                dtype=torch.int32,
                                device=device)
    problem_sizes2 = torch.empty((local_num_experts, 3),
                                dtype=torch.int32,
                                device=device)
    return cutlass_w4a8_moe(
        start_expert_id,
        end_expert_id,
        E,
        a,
        w1_q,
        w2_q,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids_,
        local_topk_ids,
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


def ref(x: torch.Tensor,
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
            alpha_2: Optional[torch.Tensor] = None):
        results = torch.zeros_like(x)
        dtype = x.dtype
        m = x.shape[0]
        k = x.shape[1]
        n = ref_weight_2.shape[1]
        # selected_experts, final_scales = routing_method.apply(router_logits)
        # unpacker = torch.ops.trtllm.unpack_int4_packed_tensor_to_int8
        tensors_collector = []
        aggregated_tensors_lists = {
            "c1": [],
            "silu_intermediate": [],
            "intermediate_q": [],
            "c2": [],
            "delta_results":
            []  # Stores the contribution of each expert to the results
        }
        for e_idx in range(num_experts):
            print(f"==================expert {e_idx}======================")
            mask = topk_ids == e_idx
            activated_tokens = mask.sum(1).bool()
            act = x[activated_tokens, :]
            if act.shape[0] == 0:
                continue
            final_scale = (topk_weights *
                           mask).sum(1)[activated_tokens].unsqueeze(1)

            act = torch.clamp((act / pre_quant_scale_1.float()), -448.0,
                              448.0).to(torch.float8_e4m3fn).to(dtype)
            print_tensor_info("act", act)
            w3_w1 = ref_weight_1[e_idx]
            ref_w_scale_repeat = ref_weight_scale_1[e_idx].t(
            ).repeat_interleave(128, dim=1).to(float)
            print_tensor_info("ref_w_scale_repeat1", ref_w_scale_repeat)
            w3_w1 = (w3_w1.to(float) * ref_w_scale_repeat).to(dtype)
            fc1 = ((torch.matmul(act, w3_w1.T)) * alpha_1).to(
                torch.float16)
            print_tensor_info("fc1", fc1)
            aggregated_tensors_lists["c1"].append(fc1.clone().detach())
            # tensors_collector.append({
            #     "name": "c1",
            #     "tensor": fc1.clone().detach()
            # })

            gate, fc1 = fc1.chunk(2, dim=-1)
            print_tensor_info("gate", gate)
            print_tensor_info("fc1", fc1)
            fc1 = fc1 * torch.nn.functional.silu(gate)
            print_tensor_info("fc1 after silu", fc1)
            # tensors_collector.append({
            #     "name": "silu_intermediate",
            #     "tensor": fc1.clone().detach()
            # })
            aggregated_tensors_lists["silu_intermediate"].append(
                fc1.clone().detach())

            # act = torch.clamp((fc1 / pre_quant_scale_2[e_idx].float()), -448.0,
            #                   448.0).to(torch.float8_e4m3fn).to(dtype)
            print_tensor_info("pre_quant_scale_2", pre_quant_scale_2)
            act = (fc1 / pre_quant_scale_2.float()).to(
                torch.float8_e4m3fn)
            # torch.save(act, "ref_intermediate_q_fp8")
            act = act.to(dtype)
            print_tensor_info("act2", act)
            # tensors_collector.append({
            #     "name": "intermediate_q",
            #     "tensor": act.clone().detach()
            # })
            aggregated_tensors_lists["intermediate_q"].append(
                act.clone().detach())

            # act = torch.load("ref_intermediate_q_fp8").to(dtype)
            # tensors_collector.append({
            #     "name": "intermediate_q",
            #     "tensor": act.clone().detach()
            # })

            w2 = ref_weight_2[e_idx]
            ref_w_scale_repeat = ref_weight_scale_2[e_idx].t(
            ).repeat_interleave(128, dim=1).to(float)
            print_tensor_info("ref_w_scale_repeat2", ref_w_scale_repeat)
            w2 = (w2.to(float) * ref_w_scale_repeat).to(dtype)
            fc2 = (torch.matmul(act, w2.T) * alpha_2).to(torch.float16)
            print_tensor_info("fc2", fc2)
            # tensors_collector.append({
            #     "name": "c2",
            #     "tensor": fc2.clone().detach()
            # })
            aggregated_tensors_lists["c2"].append(fc2.clone().detach())

            results[activated_tokens, :] += (fc2 * final_scale).to(
                results.dtype)
            print_tensor_info("results", results)
            # tensors_collector.append({
            #     "name": "results",
            #     "tensor": results.clone().detach()
            # })

        for name, tensor_list in aggregated_tensors_lists.items():
            non_empty_tensors = [t for t in tensor_list if t.numel() > 0]
            if non_empty_tensors:
                aggregated_tensor = torch.cat(non_empty_tensors, dim=0)
                tensors_collector.append({"name": name, "tensor": aggregated_tensor})
            elif name in ["c1", "silu_intermediate", "intermediate_q", "c2", "delta_results"]:
                print(f"Warning: All tensors for step '{name}' were empty or skipped. Appending an empty tensor.")
                # Determine a representative dtype and device
                ref_dtype = x.dtype
                ref_device = x.device
                expected_k_dim = 0
                if name == "c1": expected_k_dim = n * 2
                elif name == "silu_intermediate" or name == "intermediate_q": expected_k_dim = n
                elif name == "c2" or name == "delta_results": expected_k_dim = k

                tensors_collector.append({
                    "name": name,
                    "tensor": torch.empty((0, expected_k_dim), dtype=ref_dtype, device=ref_device)
                })
        return results, tensors_collector


def compare_intermediate_val(cutlass_tensors, ref_tensors):
    cutlass_tensors_map = {}
    if cutlass_tensors: # Check if cutlass_tensors is not None and not empty
        cutlass_tensors_map = {item["name"]: item["tensor"] for item in cutlass_tensors}


    for ref_item in ref_tensors:
        ref_name = ref_item["name"]
        ref_tensor_val = ref_item["tensor"]

        print(f"\nComparing tensor: '{ref_name}'")

        if ref_name in cutlass_tensors_map:
            cutlass_tensor_val = cutlass_tensors_map[ref_name]
            try:
                if cutlass_tensor_val.device != ref_tensor_val.device:
                    print(f"  WARNING: Tensor '{ref_name}' devices differ. Ref: {ref_tensor_val.device}, Cutlass: {cutlass_tensor_val.device}. Moving Cutlass tensor to Ref tensor's device.")
                    cutlass_tensor_val = cutlass_tensor_val.to(ref_tensor_val.device)

                torch.testing.assert_close(cutlass_tensor_val, ref_tensor_val, rtol=1e-2, atol=0.1)
                print(f"  SUCCESS: '{ref_name}' tensors are close.")
            except AssertionError as e:
                # torch.set_printoptions(threshold=10_000)
                print(f"  FAILURE: '{ref_name}' tensors are NOT close.")
                print(f"    Ref tensor: {ref_tensor_val.flatten()}")
                print(f"    Cutlass tensor: {cutlass_tensor_val.flatten()}")
                print(f"    Max absolute difference: {torch.max(torch.abs(cutlass_tensor_val.to(ref_tensor_val.dtype) - ref_tensor_val))}")
                print(f"    Mean absolute difference: {torch.mean(torch.abs(cutlass_tensor_val.to(ref_tensor_val.dtype) - ref_tensor_val))}")
                print(f"    AssertionError: {e}")
                raise
        else:
            print(f"  WARNING: Tensor '{ref_name}' not found in cutlass_tensors output.")
