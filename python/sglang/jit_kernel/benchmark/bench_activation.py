"""Benchmark: flashinfer (JIT) vs sgl-kernel (AOT) activation kernels.

Usage:
    python python/sglang/jit_kernel/benchmark/bench_activation.py
    python python/sglang/jit_kernel/benchmark/bench_activation.py \
        --batch-sizes 1 4 16 --seq-lens 1 8 --dims 1024 4096
"""

import argparse
import itertools

import flashinfer
import torch
import triton
import triton.testing
from sgl_kernel import gelu_and_mul as sgl_gelu_and_mul
from sgl_kernel import gelu_tanh_and_mul as sgl_gelu_tanh_and_mul
from sgl_kernel import silu_and_mul as sgl_silu_and_mul

from sglang.jit_kernel.benchmark.utils import (
    get_benchmark_range,
    run_benchmark,
)

# Parse CLI args early so x_vals is set before the decorator runs
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--batch-sizes", nargs="+", type=int)
_parser.add_argument("--seq-lens", nargs="+", type=int)
_parser.add_argument("--dims", nargs="+", type=int)
_cli, _ = _parser.parse_known_args()

BATCH_SIZES = get_benchmark_range(_cli.batch_sizes or [1, 4, 16], [1])
SEQ_LENS = get_benchmark_range(_cli.seq_lens or [1, 8, 64], [1])
DIMS = get_benchmark_range(_cli.dims or [1024, 4096, 11008], [1024])

KERNELS = ["silu_and_mul", "gelu_tanh_and_mul", "gelu_and_mul"]
DTYPES = [torch.float16, torch.bfloat16]

AOT_FNS = {
    "silu_and_mul": sgl_silu_and_mul,
    "gelu_tanh_and_mul": sgl_gelu_tanh_and_mul,
    "gelu_and_mul": sgl_gelu_and_mul,
}
JIT_FNS = {
    "silu_and_mul": flashinfer.silu_and_mul,
    "gelu_tanh_and_mul": flashinfer.gelu_tanh_and_mul,
    "gelu_and_mul": flashinfer.gelu_and_mul,
}

configs = list(itertools.product(KERNELS, DTYPES, BATCH_SIZES, SEQ_LENS, DIMS))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["kernel", "dtype", "batch_size", "seq_len", "dim"],
        x_vals=configs,
        x_log=False,
        line_arg="provider",
        line_vals=["aot", "fi"],
        line_names=["SGL AOT Kernel", "FlashInfer (JIT)"],
        styles=[("orange", "-"), ("blue", "--")],
        ylabel="us",
        plot_name="activation-performance",
        args={},
    )
)
def benchmark(kernel, dtype, batch_size, seq_len, dim, provider):
    x = torch.randn(batch_size, seq_len, 2 * dim, dtype=dtype, device="cuda")
    if provider == "aot":
        return run_benchmark(lambda: AOT_FNS[kernel](x))
    else:
        return run_benchmark(lambda: JIT_FNS[kernel](x))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[_parser])
    parser.parse_args()

    benchmark.run(print_data=True)
    print("\nBenchmark finished!")
