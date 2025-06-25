import argparse
import numpy as np
import time
from typing import Callable, List, Optional, Dict

import torch
import triton

from grouped_gemm.ops import permute

from sglang.srt.layers.moe.cutlass_w4a8_moe import cutlass_w4a8_moe
from sglang.srt.layers.moe.ep_moe.kernels import (
    grouped_gemm_triton,
    post_reorder_triton_kernel,
    pre_reorder_triton_kernel,
    pre_reorder_triton_kernel_for_cutlass_moe,
    run_moe_ep_preproess,
    run_cutlass_moe_ep_preproess,
    silu_and_mul_triton_kernel,
)
from sglang.srt.layers.moe.topk import select_experts
from sgl_kernel import (
    cutlass_w4a8_moe_mm,
    get_cutlass_w4a8_moe_mm_data,
    sgl_per_tensor_quant_fp8,
    silu_and_mul,
)

def pack_int4_values_to_int8(int4_values_interleaved: torch.Tensor) -> torch.Tensor:
    if int4_values_interleaved.shape[-1] % 2 != 0:
        raise ValueError(
            "int4_values_interleaved 的最后一个维度的大小必须是偶数。"
        )

    input_tensor_int8 = int4_values_interleaved.to(torch.int8)

    low_nibbles = input_tensor_int8[..., 0::2]
    high_nibbles = input_tensor_int8[..., 1::2]

    packed_tensor = (high_nibbles << 4) | (low_nibbles & 0x0F)

    return packed_tensor.to(torch.int8)


def pack_interleave(num_experts, ref_weight, ref_scale):
    n, k = ref_weight.shape[1], ref_weight.shape[2]

    weight = pack_int4_values_to_int8(ref_weight.cpu()).cuda()
    w_q = weight.view((num_experts, n, k // 2)).view(torch.int8)
    w_q = w_q.contiguous()

    ###############################################################
    # scale interleave, [E, K, N]
    # scale = ref_scale.permute(0, 2, 1)  # [E, N, K]
    scale = ref_scale
    scale_interleaved = scale.reshape(
        scale.shape[0], scale.shape[1], (scale.shape[2] // 4), 4
    )  # [E, N, K/4, 4]
    scale_interleaved = scale_interleaved.permute(0, 2, 1, 3)  # [E, K/4, N, 4]
    scale_interleaved = scale_interleaved.reshape(
        scale.shape[0], scale.shape[2] // 4, scale.shape[1] * 4
    )  # [E, K/4, N*4]
    w_scale = scale_interleaved.contiguous()

    return w_q, w_scale


def create_cutlass_test_data(M, E, K, N):
    device = "cuda"
    dtype = torch.bfloat16

    a_scale = torch.randn(1, dtype=torch.float32).cuda() * 0.02
    
    group_size = 128
    ref_w_1 = torch.randint(-8, 8, (E, 2 * N, K), dtype=torch.int8, device=device)
    ref_w_2 = torch.randint(-8, 8, (E, K, N), dtype=torch.int8, device=device)
    affine_coeff = 0.005
    ref_w_scale_1 = (
        torch.randn(E, 2 * N, K // group_size, dtype=dtype, device=device) * affine_coeff
    )
    ref_w_scale_2 = (
        torch.randn(E, K, N // group_size, dtype=dtype, device=device) * affine_coeff
    )
    w_1, w_scale_1 = pack_interleave(E, ref_w_1, ref_w_scale_1)
    w_2, w_scale_2 = pack_interleave(E, ref_w_2, ref_w_scale_2)

    a_strides_1 = torch.full((E, 3), K, device=device, dtype=torch.int64)
    c_strides_1 = torch.full((E, 3), N * 2, device=device, dtype=torch.int64)
    b_strides_1 = a_strides_1
    s_strides_1 = c_strides_1
    a_strides_2 = torch.full((E, 3), N, device=device, dtype=torch.int64)
    c_strides_2 = torch.full((E, 3), K, device=device, dtype=torch.int64)
    b_strides_2 = a_strides_2
    s_strides_2 = c_strides_2

    local_e = E
    expert_offsets = torch.empty((local_e + 1),
                                 dtype=torch.int32,
                                 device=device)
    problem_sizes1 = torch.empty((local_e, 3),
                                 dtype=torch.int32,
                                 device=device)
    problem_sizes2 = torch.empty((local_e, 3),
                                 dtype=torch.int32,
                                 device=device)
    a_map=torch.zeros((M * 8),
                              dtype=torch.int32,
                              device=device)
    c_map=torch.zeros((M * 8),
                              dtype=torch.int32,
                              device=device)
    
    return (
        w_1,
        w_2,
        w_scale_1,
        w_scale_2,
        a_strides_1,
        b_strides_1,
        c_strides_1,
        s_strides_1,
        a_strides_2,
        b_strides_2,
        c_strides_2,
        s_strides_2,
        a_scale,
        a_scale,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        a_map,
        c_map,
    )


def create_triton_test_data(M, E, K, N):
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min
    ref_w_1 = torch.randn(
        E, N * 2, K, dtype=torch.bfloat16, device="cuda"
    )
    ref_w_2 = torch.randn(
        E, K, N, dtype=torch.bfloat16, device="cuda"
    )
    w_1 = ref_w_1.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
    w_2 = ref_w_2.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
    block_n, block_k = 128, 128
    n_tiles_w1 = (2 * N + block_n - 1) // block_n
    n_tiles_w2 = (K + block_n - 1) // block_n
    k_tiles_w1 = (K + block_k - 1) // block_k
    k_tiles_w2 = (N + block_k - 1) // block_k
    factor_for_scale = 1e-2
    w_scale_1 = (
        torch.rand((E, n_tiles_w1, k_tiles_w1), dtype=torch.float32, device="cuda")
        * factor_for_scale
    )
    w_scale_2 = (
        torch.rand((E, n_tiles_w2, k_tiles_w2), dtype=torch.float32, device="cuda")
        * factor_for_scale
    )

    start_expert_id = 0
    end_expert_id = E - 1

    return (
        w_1,
        w_2,
        w_scale_1,
        w_scale_2,
        [block_n, block_k],
        start_expert_id,
        end_expert_id,
    )


def ep_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    router_logits: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    renormalize: bool,
    # ep config
    num_experts: int = 256,
    fp8_dtype: torch.types = torch.float8_e4m3fn,
    num_experts_per_partition: int = 128,
    start_expert_id: int = 0,
    end_expert_id: int = 127,
    use_grouped_topk: bool = False,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    custom_routing_function: Optional[Callable] = None,
    use_fp8_w8a8: bool = False,
    w1_scale_inv: Optional[torch.Tensor] = None,
    w2_scale_inv: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    cal_time: bool = False,
    triton_time_map: Optional[dict[str, float]] = None,
    print_time: bool = False,
):  
    start_time = time.time()
    use_blockwise_fp8 = block_shape is not None
    
    reorder_topk_ids, src2dst, seg_indptr = run_moe_ep_preproess(topk_ids, num_experts)
    if cal_time:
        end_preprocess = time.time()
        preprocess_duration = (end_preprocess - start_time) * 1000
        if print_time:
            print(f"preprocess_duration: {preprocess_duration} ms")
        if triton_time_map is not None:
            triton_time_map["ep_preproess"].append(preprocess_duration)

    gateup_input = torch.empty(
        (int(hidden_states.shape[0] * top_k), hidden_states.shape[1]),
        device=hidden_states.device,
        dtype=(
            fp8_dtype
            if (use_fp8_w8a8 and not use_blockwise_fp8)
            else hidden_states.dtype
        ),
    )

    if use_fp8_w8a8 and not use_blockwise_fp8:
        max_value = (
            torch.max(hidden_states).repeat(num_experts_per_partition).to(torch.float32)
        )
        w1_input_scale = max_value / torch.finfo(fp8_dtype).max
    else:
        w1_input_scale = None

    # PreReorder
    pre_reorder_triton_kernel[(hidden_states.shape[0],)](
        hidden_states,
        gateup_input,
        src2dst,
        topk_ids,
        w1_input_scale,
        start_expert_id,
        end_expert_id,
        top_k,
        hidden_states.shape[1],
        BLOCK_SIZE=512,
    )
    if cal_time:
        end_pre_reorder = time.time()
        pre_reorder_duration = (end_pre_reorder - end_preprocess) * 1000
        if print_time:
            print(f"pre_reorder_duration: {pre_reorder_duration} ms")
        if triton_time_map is not None:
            triton_time_map["pre_reorder"].append(pre_reorder_duration)

    seg_indptr_cur_rank = seg_indptr[start_expert_id : end_expert_id + 2]
    weight_indices_cur_rank = torch.arange(
        0,
        num_experts_per_partition,
        device=hidden_states.device,
        dtype=torch.int64,
    )

    # GroupGemm-0
    gateup_output = torch.empty(
        gateup_input.shape[0],
        w1.shape[1],
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    gateup_output = grouped_gemm_triton(
        a=gateup_input,
        b=w1,
        c=gateup_output,
        batch_size=num_experts_per_partition,
        weight_column_major=True,
        seg_indptr=seg_indptr_cur_rank,
        weight_indices=weight_indices_cur_rank,
        use_fp8_w8a8=use_fp8_w8a8,
        scale_a=w1_input_scale,
        scale_b=w1_scale_inv,
        block_shape=block_shape,
    )
    if cal_time:
        end_first_gemm = time.time()
        first_gemm_duration = (end_first_gemm - end_pre_reorder) * 1000
        if print_time:
            print(f"first_gemm_duration: {first_gemm_duration} ms")
        if triton_time_map is not None:
            triton_time_map["first_gemm_duration"].append(first_gemm_duration)

    # Act
    down_input = torch.empty(
        gateup_output.shape[0],
        gateup_output.shape[1] // 2,
        device=gateup_output.device,
        dtype=(
            fp8_dtype
            if (use_fp8_w8a8 and not use_blockwise_fp8)
            else hidden_states.dtype
        ),
    )
    if use_fp8_w8a8 and not use_blockwise_fp8:
        w2_input_scale = torch.ones(
            num_experts_per_partition,
            dtype=torch.float32,
            device=hidden_states.device,
        )
    else:
        w2_input_scale = None

    silu_and_mul_triton_kernel[(gateup_output.shape[0],)](
        gateup_output,
        down_input,
        gateup_output.shape[1],
        reorder_topk_ids,
        w2_input_scale,
        start_expert_id,
        end_expert_id,
        BLOCK_SIZE=512,
    )
    if cal_time:
        end_silu_and_mul = time.time()
        silu_and_mul_duration = (end_silu_and_mul - end_first_gemm) * 1000
        if print_time:
            print(f"silu_and_mul_and_scales took: {silu_and_mul_duration:.6f} ms")
        if triton_time_map is not None:
            triton_time_map["silu_and_mul_and_scale"].append(silu_and_mul_duration)

    # GroupGemm-1
    down_output = torch.empty(
        down_input.shape[0],
        w2.shape[1],
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    down_output = grouped_gemm_triton(
        a=down_input,
        b=w2,
        c=down_output,
        batch_size=num_experts_per_partition,
        weight_column_major=True,
        seg_indptr=seg_indptr_cur_rank,
        weight_indices=weight_indices_cur_rank,
        use_fp8_w8a8=use_fp8_w8a8,
        scale_a=w2_input_scale,
        scale_b=w2_scale_inv,
        block_shape=block_shape,
    )
    if cal_time:
        end_second_gemm = time.time()
        second_gemm_duration = (end_second_gemm - end_silu_and_mul) * 1000
        if print_time:
            print(f"second_gemm_duration: {second_gemm_duration} ms")
        if triton_time_map is not None:
            triton_time_map["second_gemm_duration"].append(second_gemm_duration)

    # PostReorder
    output = torch.empty_like(hidden_states)
    post_reorder_triton_kernel[(hidden_states.size(0),)](
        down_output,
        output,
        src2dst,
        topk_ids,
        topk_weights,
        start_expert_id,
        end_expert_id,
        top_k,
        hidden_states.size(1),
        BLOCK_SIZE=512,
    )
    if cal_time:
        end_post_reorder_triton = time.time()
        post_reorder_triton_duration = (end_post_reorder_triton - end_second_gemm) * 1000
        if print_time:
            print(f"post_reorder_triton took: {post_reorder_triton_duration:.6f} ms")
        if triton_time_map is not None:
            triton_time_map["post_reorder_triton"].append(post_reorder_triton_duration)

    if cal_time:
        end_time = time.time()
        duration = (end_time - start_time) * 1000
        if print_time:
            print(f"triton_moe took: {duration:.6f} ms")
        if triton_time_map is not None:
            triton_time_map["total_duration"].append(duration)
    return output


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
    local_e: int,
    expert_offsets: torch.Tensor,
    problem_sizes1: torch.Tensor,
    problem_sizes2: torch.Tensor,
    a_map: torch.Tensor,
    c_map: torch.Tensor,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    expert_map: Optional[torch.Tensor] = None,
    apply_router_weight_on_input: bool = False,
    cal_time: bool = False,
    cutlass_time_map: Optional[Dict[str, float]] = None,
    print_time: bool = False,
):  
    device = a.device
    m = a.size(0)
    k = w1_q.size(2) * 2
    n = w2_q.size(2) * 2
    topk = topk_ids_.size(1)
    per_act_token = False
    
    start_time = time.time()
    local_topk_ids = topk_ids_
    local_topk_ids = torch.where(expert_map[topk_ids_] != E,
                                    expert_map[topk_ids_], E)
    gateup_input = torch.empty(
        ((a.shape[0] * 8), a.shape[1]),
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    reorder_topk_ids, src2dst, seg_indptr = run_cutlass_moe_ep_preproess(
        local_topk_ids, end_expert_id - start_expert_id + 1,
    )
    if cal_time:
        end_preprocess = time.time()
        preprocess_duration = (end_preprocess - start_time) * 1000
        if print_time:
            print(f"run_cutlass_moe_ep_preproess took: {preprocess_duration:.6f} ms")
        if cutlass_time_map is not None:
            cutlass_time_map["ep_preproess"].append(preprocess_duration)

    pre_reorder_triton_kernel_for_cutlass_moe[(a.shape[0],)](
        a,
        gateup_input,
        src2dst,
        local_topk_ids,
        a1_scale,
        E,
        8,
        a.shape[1],
        BLOCK_SIZE=512,
    )
    if cal_time:
        end_prereorder = time.time()
        prereorder_duration = (end_prereorder - end_preprocess) * 1000
        if print_time:
            print(f"pre_reorder_triton_kernel_for_cutlass_moe took: {prereorder_duration:.6f} ms")
        if cutlass_time_map is not None:
            cutlass_time_map["pre_reorder"].append(prereorder_duration)

    get_cutlass_w4a8_moe_mm_data(local_topk_ids, expert_offsets, problem_sizes1,
                                problem_sizes2, a_map, c_map, local_e, n,
                                k)
    if cal_time:
        end_get_cutlass_w4a8_moe_mm_data = time.time()
        get_cutlass_w4a8_moe_mm_data_duration = (end_get_cutlass_w4a8_moe_mm_data -
                                        end_prereorder) * 1000
        if print_time:
            print(f"get_cutlass_w4a8_moe_mm_data took: {get_cutlass_w4a8_moe_mm_data_duration:.6f} ms")
        if cutlass_time_map is not None:
            cutlass_time_map["get_cutlass_w4a8_moe_mm_data"].append(get_cutlass_w4a8_moe_mm_data_duration)
    
    c1 = torch.empty((m * topk, n * 2), device=device, dtype=torch.half)
    c2 = torch.zeros((m * topk, k), device=device, dtype=torch.half)
    cutlass_w4a8_moe_mm(c1, gateup_input, w1_q, a1_scale.float(), w1_scale,
                            expert_offsets[:-1], problem_sizes1, a_strides1,
                            b_strides1, c_strides1, s_strides13, 128, topk)
    if cal_time:
        end_first_gemm = time.time()
        first_gemm_duration = (end_first_gemm - end_get_cutlass_w4a8_moe_mm_data) * 1000
        if print_time:
            print(f"first_gemm_duration took: {first_gemm_duration:.6f} ms")
        if cutlass_time_map is not None:
            cutlass_time_map["first_gemm_duration"].append(first_gemm_duration)

    intermediate = torch.empty((m * topk, n), device=device, dtype=torch.half)
    silu_and_mul(c1, intermediate)
    intermediate_q = torch.empty(intermediate.shape, dtype=torch.float8_e4m3fn, device=device)
    sgl_per_tensor_quant_fp8(intermediate, intermediate_q, a2_scale.float(), True)

    if cal_time:
        end_silu_and_mul_and_scale = time.time()
        silu_and_mul_and_scale_duration = (end_silu_and_mul_and_scale - end_first_gemm) * 1000
        if print_time:
            print(f"fp8 scale took: {silu_and_mul_and_scale_duration:.6f} ms")
        if cutlass_time_map is not None:
            cutlass_time_map["silu_and_mul_and_scale"].append(silu_and_mul_and_scale_duration)

    cutlass_w4a8_moe_mm(c2, intermediate_q, w2_q, a2_scale.float(), w2_scale,
                            expert_offsets[:-1], problem_sizes2, a_strides2,
                            b_strides2, c_strides2, s_strides2, 128, topk)
    if cal_time:
        end_second_gemm = time.time()
        second_gemm_duration = (end_second_gemm - end_silu_and_mul_and_scale) * 1000
        if print_time:
            print(f"second_gemm_duration took: {second_gemm_duration:.6f} ms")
        if cutlass_time_map is not None:
            cutlass_time_map["second_gemm_duration"].append(second_gemm_duration)

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
            BLOCK_SIZE=512,
        )
    if cal_time:
        end_post_reorder_triton = time.time()
        post_reorder_triton_duration = (end_post_reorder_triton - end_second_gemm) * 1000
        if print_time:
            print(f"post_reorder_triton took: {post_reorder_triton_duration:.6f} ms")
        if cutlass_time_map is not None:
            cutlass_time_map["post_reorder_triton"].append(post_reorder_triton_duration)          
    
    if cal_time:
        end_time = time.time()
        duration = (end_time - start_time) * 1000
        if print_time:
            print(f"cutlass_w4a8_moe took: {duration:.6f} ms")
        if cutlass_time_map is not None:
            cutlass_time_map["total_duration"].append(duration)
    
    return output


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
        # x_vals=[1024],
        x_log=False,
        line_arg="provider",
        line_vals=["cutlass_w4a8", "triton_ep_moe"],
        line_names=["cutlass_w4a8", "triton_ep_moe"],
        styles=[("blue", "-"), ("orange", "-")],
        ylabel="ms",
        plot_name="cutlass_w4a8_moe vs triton_ep_moe",
        args={},
    )
)
def benchmark(batch_size, provider):
    print(f"==========={provider}============")
    M, N, K = batch_size, 2048, 7168
    # E = 256
    E = 32
    local_e = 32

    dtype = torch.bfloat16
    device = "cuda"
    torch.manual_seed(0)

    a = torch.randn(M, K, dtype=dtype, device=device)
    score = torch.randn((M, E), dtype=dtype, device=device)
    topk_weights, topk_ids = select_experts(
        hidden_states=a,
        router_logits=score,
        top_k=8,
        use_grouped_topk=False,
        renormalize=False,
    )
    
    cutlass_time_map = {
        "ep_preproess": [],
        "pre_reorder": [],
        "get_cutlass_w4a8_moe_mm_data": [],
        "first_gemm_duration": [],
        "silu_and_mul_and_scale": [],
        "second_gemm_duration": [],
        "post_reorder_triton": [],
        "total_duration": [],
    }
    triton_time_map = {
        "ep_preproess": [],
        "pre_reorder": [],
        "first_gemm_duration": [],
        "silu_and_mul_and_scale": [],
        "second_gemm_duration": [],
        "post_reorder_triton": [],
        "total_duration": [],
    }
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = 0,0,0
    cal_time = False
    if provider == "cutlass_w4a8":
        (
            w1,
            w2,
            w1_scale,
            w2_scale,
            a_strides1,
            b_strides1,
            c_strides1,
            s_strides13,
            a_strides2,
            b_strides2,
            c_strides2,
            s_strides2,
            a1_scale,
            a2_scale,
            expert_offsets,
            problem_sizes1,
            problem_sizes2,
            a_map,
            c_map,
        ) = create_cutlass_test_data(M, local_e, K, N)
        expert_map = torch.arange(E, dtype=torch.int32, device="cuda")
        expert_map[local_e:] = E
        # warm up
        for _ in range(5):
            cutlass_moe(
                    a,
                    w1,
                    w2,
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
                    local_e,
                    expert_offsets,
                    problem_sizes1,
                    problem_sizes2,
                    a_map,
                    c_map,
                    a1_scale,
                    a2_scale,
                    expert_map,
                )
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: cutlass_moe(
                a,
                w1,
                w2,
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
                local_e,
                expert_offsets,
                problem_sizes1,
                problem_sizes2,
                a_map,
                c_map,
                a1_scale,
                a2_scale,
                expert_map,
                cal_time = cal_time,
                cutlass_time_map = cutlass_time_map,
                print_time = False,
            ),
            quantiles=quantiles,
        )
        if cal_time:
            for key, timings in cutlass_time_map.items():
                print(f"{key}: len {len(timings)}")            
                if timings:  # Check if the list is not empty
                    avg_duration = sum(timings) / len(timings)
                    print(f"  {key}: {avg_duration * 1000:.4f} ms") # Multiply by 1000 for ms
                else:
                    print(f"  {key}: No data collected")

    if provider == "triton_ep_moe":
        (
            w_1,
            w_2,
            w_scale_1,
            w_scale_2,
            [block_n, block_k],
            start_expert_id,
            end_expert_id,
        ) = create_triton_test_data(M, local_e, K, N)
        # warm up
        for _ in range(5):
            ep_moe(
                a,
                w_1,
                w_2,
                score,
                topk_weights,
                topk_ids,
                8,
                False,
                num_experts=E,
                fp8_dtype=torch.float8_e4m3fn,
                num_experts_per_partition=local_e,
                start_expert_id=start_expert_id,
                end_expert_id=end_expert_id,
                use_fp8_w8a8=True,
                w1_scale_inv=w_scale_1,
                w2_scale_inv=w_scale_2,
                block_shape=[block_n, block_k],
                triton_time_map=None,
            )
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: ep_moe(
                a,
                w_1,
                w_2,
                score,
                topk_weights,
                topk_ids,
                8,
                False,
                num_experts=E,
                fp8_dtype=torch.float8_e4m3fn,
                num_experts_per_partition=local_e,
                start_expert_id=start_expert_id,
                end_expert_id=end_expert_id,
                use_fp8_w8a8=True,
                w1_scale_inv=w_scale_1,
                w2_scale_inv=w_scale_2,
                block_shape=[block_n, block_k],
                cal_time=cal_time,
                triton_time_map=triton_time_map,
                print_time=False,
            ),
            quantiles=quantiles,
        )
        if cal_time:
            for key, timings in triton_time_map.items():
                print(f"{key}: len {len(timings)}")
                if timings:  # Check if the list is not empty
                    avg_duration = sum(timings) / len(timings)
                    print(f"  {key}: {avg_duration * 1000:.4f} ms") # Multiply by 1000 for ms
                else:
                    print(f"  {key}: No data collected")
    return ms, min_ms, max_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-path",
        type=str,
        default="./cutlass_w4a8_moe/",
    )
    args = parser.parse_args()

    benchmark.run(
        show_plots=True,
        print_data=True,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
