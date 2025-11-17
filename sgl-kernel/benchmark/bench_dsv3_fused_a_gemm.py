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
from sgl_kernel import dsv3_fused_a_gemm

# CI environment uses simplified parameters
if IS_CI:
    num_tokens_vals = [1]  # Only test 1 value in CI
    line_vals = ["sgl-kernel"]  # Only test sgl-kernel implementation in CI
else:
    num_tokens_vals = [i + 1 for i in range(16)]  # Test 1-16 in full mode
    line_vals = ["torch", "sgl-kernel"]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens"],
        x_vals=num_tokens_vals,
        x_log=False,
        line_arg="impl",
        line_vals=line_vals,
        line_names=(
            ["torch (bf16)", "dsv3_fused_a_gemm"]
            if not IS_CI
            else ["dsv3_fused_a_gemm"]
        ),
        styles=[("blue", "-"), ("orange", "-")] if not IS_CI else [("orange", "-")],
        ylabel="TFLOPs",
        plot_name="bf16 dsv3 fused a GEMM throughput",
        args={},
    )
)
def benchmark(num_tokens, impl):
    kHdIn = 7168
    kHdOut = 2112
    M, K, N = num_tokens, kHdIn, kHdOut

    mat_a = torch.randn((M, K), dtype=torch.bfloat16, device="cuda").contiguous()
    mat_b = torch.randn((N, K), dtype=torch.bfloat16, device="cuda").transpose(0, 1)

    quantiles = [0.5, 0.2, 0.8]

    if impl == "torch":

        def runner():
            F.linear(mat_a, mat_b.T)

    elif impl == "sgl-kernel":

        def runner():
            dsv3_fused_a_gemm(mat_a, mat_b)

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(runner, quantiles=quantiles)

    def tflops(t_ms):
        flops = 2 * M * K * N
        return flops / (t_ms * 1e-3) / 1e12

    return tflops(ms), tflops(max_ms), tflops(min_ms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    benchmark.run(print_data=True)
