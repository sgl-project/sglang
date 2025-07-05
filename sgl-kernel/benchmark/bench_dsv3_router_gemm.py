import argparse

import torch
import torch.nn.functional as F
import triton
import triton.testing
from sgl_kernel import dsv3_router_gemm


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens"],
        x_vals=[i + 1 for i in range(16)],
        x_log=False,
        line_arg="impl",
        line_vals=["torch", "sgl-kernel"],
        line_names=["torch", "dsv3_router_gemm"],
        styles=[("blue", "-"), ("orange", "-")],
        ylabel="TFLOPs",
        plot_name="input-bf16-output-fp32 dsv3 router gemm throughput",
        args={},
    )
)
def benchmark(num_tokens, impl):
    # M: num_tokens, K: hidden_dim, N: num_experts
    M, K, N = num_tokens, 7168, 256

    mat_a = torch.randn((M, K), dtype=torch.bfloat16, device="cuda").contiguous()
    mat_b = torch.randn((N, K), dtype=torch.bfloat16, device="cuda").contiguous()

    quantiles = [0.5, 0.2, 0.8]

    if impl == "torch":

        def runner():
            F.linear(mat_a, mat_b).to(torch.float32)

    elif impl == "sgl-kernel":

        def runner():
            dsv3_router_gemm(mat_a, mat_b)

    ms, min_ms, max_ms = triton.testing.do_bench(runner, quantiles=quantiles)

    def tflops(t_ms):
        flops = 2 * M * K * N
        return flops / (t_ms * 1e-3) / 1e12

    return tflops(ms), tflops(max_ms), tflops(min_ms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    benchmark.run(print_data=True, show_plots=True, save_path="bench_dsv3_router_gemm")
