import argparse

import torch
import triton

from sglang.srt.layers.moe.ep_moe.kernels import grouped_gemm_triton, run_moe_ep_preproess, run_cutlass_moe_ep_preproess, pre_reorder_triton_kernel_for_cutlass_moe, pre_reorder_triton_kernel
from sglang.srt.layers.moe.topk import select_experts
from sgl_kernel import cutlass_w4a8_moe_mm, get_cutlass_w4a8_moe_mm_data

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


def create_cutlass_test_data(M, E, K, N, a, topk_ids):
    device = "cuda"
    dtype = torch.bfloat16

    a_scale = torch.randn(1, dtype=torch.float32).cuda() * 0.02
    a_q = torch.clamp((a / a_scale), -448.0, 448.0).to(torch.float8_e4m3fn).to(device)
    
    group_size = 128
    ref_w = torch.randint(-8, 8, (E, N * 2, K), dtype=torch.int8, device=device)
    affine_coeff = 0.005
    ref_w_scale = (
        torch.randn(E, N * 2, K // group_size, dtype=dtype, device=device) * affine_coeff
    )
    w, w_scale = pack_interleave(E, ref_w, ref_w_scale)

    # tokens_per_group = M // E
    reorder_topk_ids, src2dst, seg_indptr = run_cutlass_moe_ep_preproess(
        topk_ids, E,
    )

    gateup_input = torch.empty(
        ((a.shape[0] * 8), a.shape[1]),
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    pre_reorder_triton_kernel_for_cutlass_moe[(a.shape[0],)](
        a,
        gateup_input,
        src2dst,
        topk_ids,
        a_scale,
        E,
        8,
        a.shape[1],
        BLOCK_SIZE=512,
    )

    expert_offsets = torch.zeros(E + 1, dtype=torch.int32, device=device)
    problem_sizes1 = torch.empty((E, 3),
                                 dtype=torch.int32,
                                 device=device)
    problem_sizes2 = torch.empty((E, 3),
                                 dtype=torch.int32,
                                 device=device)
    a_map=torch.zeros((topk_ids.numel()),
                              dtype=torch.int32,
                              device=device)
    c_map=torch.zeros((topk_ids.numel()),
                              dtype=torch.int32,
                              device=device)
    get_cutlass_w4a8_moe_mm_data(topk_ids, expert_offsets, problem_sizes1,
                                problem_sizes2, a_map, c_map, E, N,
                                K)

    a_strides = torch.full((E, 3), K, device=device, dtype=torch.int64)
    c_strides = torch.full((E, 3), N * 2, device=device, dtype=torch.int64)
    b_strides = a_strides
    s_strides = c_strides

    c = torch.empty((M * 8, N * 2), dtype=torch.float16, device=device)

    return (
        c, gateup_input, w, a_scale, w_scale, expert_offsets, problem_sizes1, a_strides, b_strides, c_strides, s_strides, group_size
    )


def create_triton_test_data(M, E, K, N, a, topk_ids):
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min
    ref_w = torch.randn(
        E, N * 2, K, dtype=torch.bfloat16, device="cuda"
    )
    w = ref_w.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
    block_n, block_k = 128, 128
    n_tiles_w = (2 * N + block_n - 1) // block_n
    k_tiles_w = (K + block_k - 1) // block_k
    factor_for_scale = 1e-2
    w_scale = (
        torch.rand((E, n_tiles_w, k_tiles_w), dtype=torch.float32, device="cuda")
        * factor_for_scale
    )

    reorder_topk_ids, src2dst, seg_indptr = run_moe_ep_preproess(topk_ids, E)
    gateup_input = torch.empty(
            (int(a.shape[0] * 8), a.shape[1]),
            device=a.device,
            dtype=a.dtype,
        )
    pre_reorder_triton_kernel[(a.shape[0],)](
            a,
            gateup_input,
            src2dst,
            topk_ids,
            None,
            0,
            E - 1,
            8,
            a.shape[1],
            BLOCK_SIZE=512,
        )
    weight_indices = torch.arange(E, dtype=torch.int64, device="cuda")
    c = torch.empty(
        gateup_input.shape[0], N * 2, dtype=torch.bfloat16, device="cuda"
    )

    return (
        gateup_input,
        w,
        c,
        seg_indptr,
        weight_indices,
        w_scale,
        [block_n, block_k],
    )


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
        x_log=False,
        line_arg="provider",
        line_vals=["cutlass_w4a8_moe_mm", "grouped_gemm_triton"],
        line_names=["cutlass_w4a8_moe_mm", "grouped_gemm_triton"],
        styles=[("blue", "-"), ("orange", "-")],
        ylabel="ms",
        plot_name="cutlass_w4a8_moe_mm vs grouped_gemm_triton",
        args={},
    )
)
def benchmark(batch_size, provider):
    print(f"batch_size {batch_size}, provider {provider}")
    M, N, K = batch_size, 2048, 7168
    E = 32

    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    score = torch.randn((M, E), dtype=a.dtype, device=a.device)
    topk_weights, topk_ids = select_experts(
        hidden_states=a,
        router_logits=score,
        top_k=8,
        use_grouped_topk=False,
        renormalize=False,
    )
    quantiles = [0.5, 0.2, 0.8]
    if provider == "cutlass_w4a8_moe_mm":
        (
            c,
            a_q,
            w,
            a_scale,
            w_scale,
            expert_offsets,
            problem_sizes,
            a_strides,
            b_strides,
            c_strides,
            s_strides,
            group_size,
        ) = create_cutlass_test_data(M, E, K, N, a, topk_ids)
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: cutlass_w4a8_moe_mm(
                c,
                a_q,
                w,
                a_scale,
                w_scale,
                expert_offsets[:-1],
                problem_sizes,
                a_strides,
                b_strides,
                c_strides,
                s_strides,
                group_size,
                8,
            ),
            quantiles=quantiles,
        )
    if provider == "grouped_gemm_triton":
        (
            a,
            w,
            c,
            seg_indptr,
            weight_indices,
            w_scale,
            block_shape,
        ) = create_triton_test_data(M, E, K, N, a, topk_ids)
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: grouped_gemm_triton(
                a,
                w,
                c,
                E,
                weight_column_major=True,
                seg_indptr=seg_indptr,
                weight_indices=weight_indices,
                use_fp8_w8a8=True,
                scale_b=w_scale,
                block_shape=block_shape,
            ),
            quantiles=quantiles,
        )

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
