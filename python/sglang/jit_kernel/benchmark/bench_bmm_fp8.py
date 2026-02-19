"""Benchmark: flashinfer (JIT) bmm_fp8.

Usage:
    python python/sglang/jit_kernel/benchmark/bench_bmm_fp8.py
    python python/sglang/jit_kernel/benchmark/bench_bmm_fp8.py \
        --batch-sizes 1 4 16 --m 64 128 --k 128 256 --n 256 512
"""

import argparse

import torch
import triton
import triton.testing

import flashinfer
from sglang.jit_kernel.benchmark.utils import (
    get_benchmark_range,
    run_benchmark,
)

# Parse CLI args early so x_vals is set before the decorator runs
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--batch-sizes", nargs="+", type=int)
_parser.add_argument("--m", nargs="+", type=int)
_parser.add_argument("--k", nargs="+", type=int)
_parser.add_argument("--n", nargs="+", type=int)
_cli, _ = _parser.parse_known_args()

BATCH_SIZES = get_benchmark_range(
    _cli.batch_sizes or [1, 2, 4, 8, 16],
    [2],
)
M_VALS = get_benchmark_range(_cli.m or [64, 128, 512], [64])
K_VALS = get_benchmark_range(_cli.k or [128, 512, 1024], [128])
N_VALS = get_benchmark_range(_cli.n or [256, 512, 1024], [256])


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=BATCH_SIZES,
        x_log=False,
        line_arg="provider",
        line_vals=["flashinfer"],
        line_names=["flashinfer (JIT)"],
        styles=[("orange", "-")],
        ylabel="us",
        plot_name="bmm_fp8",
        args={},
    )
)
def benchmark(batch_size, provider, M, K, N):
    A = torch.randn(batch_size, M, K, device="cuda").to(torch.float8_e4m3fn)
    B = torch.randn(batch_size, K, N, device="cuda").to(torch.float8_e4m3fn)
    A_scale = torch.ones(1, device="cuda", dtype=torch.float32)
    B_scale = torch.ones(1, device="cuda", dtype=torch.float32)
    out_dtype = torch.bfloat16

    return run_benchmark(lambda: flashinfer.bmm_fp8(A, B, A_scale, B_scale, out_dtype))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[_parser])
    parser.parse_args()

    for M in M_VALS:
        for K in K_VALS:
            for N in N_VALS:
                print(f"\nM={M}  K={K}  N={N}")
                benchmark.run(print_data=True, M=M, K=K, N=N)

    print("\nBenchmark finished!")
