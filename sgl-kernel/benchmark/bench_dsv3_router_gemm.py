import argparse
import os

# CI environment detection
IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)

import torch
import torch.nn.functional as F
import triton
import triton.testing
from sgl_kernel import dsv3_router_gemm

# CI environment uses simplified parameters
if IS_CI:
    num_tokens_vals = [1]  # Only test 1 value in CI
    line_vals = ["sgl-kernel-256"]  # Only test one implementation in CI
else:
    num_tokens_vals = [i + 1 for i in range(16)]  # Test 1-16 in full mode
    line_vals = ["torch-256", "sgl-kernel-256", "torch-384", "sgl-kernel-384"]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens"],
        x_vals=num_tokens_vals,
        x_log=False,
        line_arg="impl",
        line_vals=line_vals,
        line_names=(
            [
                "torch-256",
                "dsv3_router_gemm-256",
                "torch-384",
                "dsv3_router_gemm-384",
            ]
            if not IS_CI
            else ["dsv3_router_gemm-256"]
        ),
        styles=(
            [("blue", "-"), ("orange", "-"), ("green", "-"), ("red", "-")]
            if not IS_CI
            else [("orange", "-")]
        ),
        ylabel="TFLOPs",
        plot_name="input-bf16-output-bf16 dsv3 router gemm throughput",
        args={},
    )
)
def benchmark_bf16_output(num_tokens, impl):
    # M: num_tokens, K: hidden_dim, N: num_experts
    M, K = num_tokens, 7168

    if impl == "torch-256" or impl == "sgl-kernel-256":
        N = 256
    elif impl == "torch-384" or impl == "sgl-kernel-384":
        N = 384
    else:
        raise ValueError(f"Unknown impl: {impl}")

    mat_a = torch.randn((M, K), dtype=torch.bfloat16, device="cuda").contiguous()
    mat_b = torch.randn((N, K), dtype=torch.bfloat16, device="cuda").contiguous()

    quantiles = [0.5, 0.2, 0.8]

    if impl == "torch-256" or impl == "torch-384":

        def runner():
            F.linear(mat_a, mat_b)

    elif impl == "sgl-kernel-256" or impl == "sgl-kernel-384":

        def runner():
            dsv3_router_gemm(mat_a, mat_b, out_dtype=torch.bfloat16)

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(runner, quantiles=quantiles)

    def tflops(t_ms):
        flops = 2 * M * K * N
        return flops / (t_ms * 1e-3) / 1e12

    return tflops(ms), tflops(max_ms), tflops(min_ms)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens"],
        x_vals=num_tokens_vals,
        x_log=False,
        line_arg="impl",
        line_vals=line_vals,
        line_names=(
            [
                "torch-256",
                "dsv3_router_gemm-256",
                "torch-384",
                "dsv3_router_gemm-384",
            ]
            if not IS_CI
            else ["dsv3_router_gemm-256"]
        ),
        styles=(
            [("blue", "-"), ("orange", "-"), ("green", "-"), ("red", "-")]
            if not IS_CI
            else [("orange", "-")]
        ),
        ylabel="TFLOPs",
        plot_name="input-bf16-output-fp32 dsv3 router gemm throughput",
        args={},
    )
)
def benchmark_float_output(num_tokens, impl):
    # M: num_tokens, K: hidden_dim, N: num_experts
    M, K = num_tokens, 7168

    if impl == "torch-256" or impl == "sgl-kernel-256":
        N = 256
    elif impl == "torch-384" or impl == "sgl-kernel-384":
        N = 384
    else:
        raise ValueError(f"Unknown impl: {impl}")

    mat_a = torch.randn((M, K), dtype=torch.bfloat16, device="cuda").contiguous()
    mat_b = torch.randn((N, K), dtype=torch.bfloat16, device="cuda").contiguous()

    quantiles = [0.5, 0.2, 0.8]

    if impl == "torch-256" or impl == "torch-384":

        def runner():
            F.linear(mat_a, mat_b).to(torch.float32)

    elif impl == "sgl-kernel-256" or impl == "sgl-kernel-384":

        def runner():
            dsv3_router_gemm(mat_a, mat_b, out_dtype=torch.float32)

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(runner, quantiles=quantiles)

    def tflops(t_ms):
        flops = 2 * M * K * N
        return flops / (t_ms * 1e-3) / 1e12

    return tflops(ms), tflops(max_ms), tflops(min_ms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    benchmark_bf16_output.run(print_data=True)
    benchmark_float_output.run(print_data=True)
