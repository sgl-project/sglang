from typing import Tuple, Optional

import os
import math
import torch
import triton

from sgl_kernel.scalar_type import ScalarType, scalar_types
from sgl_kernel import machete_supported_schedules, machete_mm, machete_prepack_B

from utils import group_size_valid, create_gemm_data, TypeConfig, Tensors

# None stype means scales use the same dtype as a
def machete_mm_test_helper(types: TypeConfig,
                           tensors: Tensors,
                           group_size: Optional[int] = None,
                           schedule: Optional[str] = None,
                           check_correctness: bool = False):
    output_ref = torch.matmul(tensors.a_ref, tensors.w_ref)
    output_ref_type = output_ref.dtype

    if tensors.w_ch_s is not None:
        output_ref = (output_ref.to(tensors.w_ch_s.dtype) *
                      tensors.w_ch_s.unsqueeze(0)).to(output_ref_type)
    if tensors.w_tok_s is not None:
        output_ref = (output_ref.to(tensors.w_tok_s.dtype) *
                      tensors.w_tok_s.unsqueeze(1)).to(output_ref_type)

    output = machete_mm(
        a=tensors.a,
        b_q=tensors.w_q,
        b_type=types.weight_type,
        b_group_scales=tensors.w_g_s,
        b_group_zeros=tensors.w_g_zp,
        b_group_size=group_size,
        b_channel_scales=tensors.w_ch_s,
        a_token_scales=tensors.w_tok_s,
        out_type=types.output_type,
        schedule=schedule,
    )

    # print(output)
    # print(output_ref)

    # Relax atol as our reduction dim becomes larger (more rounding error)
    # Relax atol when we have zeropoints since the way machete applies
    #  zeropoints (after scales) causes noise around 0
    if check_correctness:
        atol = 1 if tensors.w_g_zp is not None\
            else min(5e-2 * math.sqrt(tensors.a.shape[1]), 1)
        rtol = 1e-1 if tensors.a.element_size() >= 2 else 2e-1
        torch.testing.assert_close(output,
                                output_ref.to(output.dtype),
                                rtol=rtol,
                                atol=atol)

def calc_mm_diff(m, n, k, group_size, machete_type, output_dtype):
    machete_type_config = create_machete_type(machete_type, output_dtype)

    all_schedules = machete_supported_schedules(
                        machete_type_config.act_type,
                        machete_type_config.weight_type,
                        group_scales_type=machete_type_config.group_scale_type,
                        group_zeros_type=machete_type_config.group_zero_type,
                        out_type=machete_type_config.output_type)

    for schedule in all_schedules:
        if group_size > 0 and not group_size_valid([m,n,k], group_size):
            continue
        print(f"Test correctness of mm: m={m}, n={n}, k={k}, schedule={schedule}")
        tensors = create_gemm_data([m,n,k], machete_type_config, group_size)
        machete_mm_test_helper(machete_type_config, tensors, group_size, schedule, check_correctness=True)

def run_torch_mm(tensors: Tensors):
    output_ref = torch.matmul(tensors.a_ref, tensors.w_ref)
    output_ref_type = output_ref.dtype

    if tensors.w_ch_s is not None:
        output_ref = (output_ref.to(tensors.w_ch_s.dtype) *
                      tensors.w_ch_s.unsqueeze(0)).to(output_ref_type)
    if tensors.w_tok_s is not None:
        output_ref = (output_ref.to(tensors.w_tok_s.dtype) *
                      tensors.w_tok_s.unsqueeze(1)).to(output_ref_type)
    return output_ref
    
def run_machete_mm(types: TypeConfig,
                   tensors: Tensors,
                   group_size: int, 
                   schedule: Optional[str] = None):
    
    output = machete_mm(
        a=tensors.a,
        b_q=tensors.w_q,
        b_type=types.weight_type,
        b_group_scales=tensors.w_g_s,
        b_group_zeros=tensors.w_g_zp,
        b_group_size=group_size,
        b_channel_scales=tensors.w_ch_s,
        a_token_scales=tensors.w_tok_s,
        out_type=types.output_type,
        schedule=schedule,
    )

    return output

def get_weight_shapes():
    
    K_N_s = [
        (256, 7168),
        (512, 4096),
        (1536, 3072),
        (2048, 7168),
        (2304, 7168),
        (7168, 512),
        (7168, 4608)
    ]

    return K_N_s

def create_benchmark_configs(group_size, machete_type_config):
    configs = []
    weight_shapes = get_weight_shapes() # List of (K, N)
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 4096]
    all_schedules = machete_supported_schedules(
                        machete_type_config.act_type,
                        machete_type_config.weight_type,
                        group_scales_type=machete_type_config.group_scale_type,
                        group_zeros_type=machete_type_config.group_zero_type,
                        out_type=machete_type_config.output_type)

    for schedule in all_schedules:
        for m in batch_sizes:
            for k, n in weight_shapes:
                if group_size > 0 and not group_size_valid([m,n,k], group_size):
                    continue
                configs.append((m, k, n, schedule))

    print(f"{len(configs)} configs created for benchmark")
    return configs

def create_machete_type(machete_type, output_dtype):
    a_type = torch.float8_e4m3fn
    if output_dtype == "fp16":
        output_type = torch.float16
    elif output_dtype == "bf16":
        output_type = torch.bfloat16
    else:
        assert False, f"output only supports fp16/bf16 for w4a16"

    machete_type_config = TypeConfig(act_type=torch.float8_e4m3fn,
                                        weight_type=scalar_types.uint4b8,
                                        output_type=output_type,
                                        group_scale_type=output_type,
                                        group_zero_type=None,
                                        channel_scale_type=None,
                                        token_scale_type=None)
        

    return machete_type_config

def get_benchmark(group_size, machete_type, output_dtype):
    machete_type_config = create_machete_type(machete_type, output_dtype)
    all_configs = create_benchmark_configs(group_size, machete_type_config)

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["m", "k", "n", "schedule"],
            x_vals=[list(config) for config in all_configs],
            line_arg="provider",
            line_vals=["machete", "torch"],
            line_names=["machete_mm", "torch_mm"],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="ms",
            plot_name=f"machete-performance-comparison-{machete_type}-{output_dtype}",
            args={},
        )
    )
    def benchmark(m, k, n, schedule, provider):
        print(f"Shape (m={m}, n={n}, k={k}, Provider: {provider}")

        tensors = create_gemm_data([m,n,k], machete_type_config, group_size)

        quantiles = [0.5, 0.2, 0.8]
        if provider == "torch":
            for _ in range(10):
                run_torch_mm(tensors)

            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: run_torch_mm(tensors),
                quantiles=quantiles,
            )
        elif provider == "machete":
            for _ in range(10):
                run_machete_mm(machete_type_config, tensors, group_size, schedule)

            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: run_machete_mm(
                    machete_type_config,
                    tensors,
                    group_size,
                    schedule
                ),
                quantiles=quantiles,
            )
        else:
            assert False, f"Provider {provider} not supported"

        # Calculate TFLOPS
        flops = 2 * m * n * k  # multiply-adds
        tflops = flops / (ms * 1e-3) / 1e12

        # Print shape-specific results with TFLOPS
        print(f"Time: {ms*1000:.2f} ms, TFLOPS: {tflops:.2f}")
        return ms * 1000, max_ms * 1000, min_ms * 1000  # convert to ms

    return benchmark

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        default="./configs/benchmark_ops/machete_mm/",
        help="Path to save machete gemm benchmark results",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="Whether to use group-wise scales",
    )
    parser.add_argument(
        "--machete-type",
        default="awq_w4afp8",
        choices=["awq_w4afp8"],
        help="Type of machete mm",
    )
    parser.add_argument(
        "--output-dtype",
        default="bf16",
        choices=["bf16", "fp16"],
        help="Type of output",
    )
    parser.add_argument(
        "--run_correctness",
        action="store_true",
        default=True,
        help="Whether to run correctness test",
    )
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Set random seed for reproducibility
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    torch.set_default_device("cuda")
    torch.cuda.manual_seed_all(0)

    # Run correctness tests on a few examples
    if args.run_correctness:
        print("Running correctness tests...")
        calc_mm_diff(64, 256, 7168, args.group_size, args.machete_type, args.output_dtype)   # Small test
        calc_mm_diff(64, 7168, 512, args.group_size, args.machete_type, args.output_dtype)   # Medium test
        calc_mm_diff(64, 16384, 7168, args.group_size, args.machete_type, args.output_dtype)   # Large test

    # Get the benchmark function with the specified tp_size
    benchmark = get_benchmark(args.group_size, 
                              args.machete_type,
                              args.output_dtype)

    print(f"Running performance benchmark for {args.machete_type} (group size = {args.group_size}, output type = {args.output_dtype})...")
    benchmark.run(print_data=True, save_path=args.save_path)
