import argparse
import itertools
import math
from typing import Any, Dict, List, Optional, Tuple

import sgl_kernel
import torch
import triton
import triton.testing
from sgl_kernel import gelu_and_mul, gelu_tanh_and_mul, silu_and_mul
from vllm import _custom_ops as vllm_ops


def calculate_diff(
    kernel: str, dtype: torch.dtype, batch_size: int, seq_len: int, dim: int
):
    """Calculate difference between VLLM and SGLang implementations."""
    device = torch.device("cuda")

    x = torch.randn(batch_size, seq_len, 2 * dim, dtype=dtype, device=device)
    vllm_out_ref = torch.zeros(batch_size, seq_len, dim, dtype=dtype, device=device)

    sglang_act_kernel = getattr(sgl_kernel, kernel)
    assert sglang_act_kernel in [silu_and_mul, gelu_and_mul, gelu_tanh_and_mul]

    vllm_act_kernel = getattr(vllm_ops, kernel)

    vllm_act_kernel(vllm_out_ref, x)
    sglang_out = sglang_act_kernel(x)

    if torch.allclose(vllm_out_ref, sglang_out, rtol=1e-3, atol=1e-5):
        print("✅ All implementations match")
    else:
        print("❌ Implementations differ")


kernels = ["silu_and_mul", "gelu_and_mul", "gelu_tanh_and_mul"]
dtypes = [torch.float16, torch.bfloat16]
batch_sizes = [2**i for i in range(0, 5, 2)]
seq_lens = [2**i for i in range(0, 8, 2)]
dims = [2**i for i in range(7, 15)]


configs = list(itertools.product(kernels, dtypes, batch_sizes, seq_lens, dims))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["kernel", "dtype", "batch_size", "seq_len", "dim"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["vllm", "sglang"],
        line_names=["VLLM", "SGL Kernel"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="activation-performance",
        args={},
    )
)
def benchmark(kernel, dtype, batch_size, seq_len, dim, provider):
    device = torch.device("cuda")
    x = torch.randn(batch_size, seq_len, 2 * dim, dtype=dtype, device=device)
    vllm_out_ref = torch.zeros(batch_size, seq_len, dim, dtype=dtype, device=device)

    def get_vllm_api_func(fn):
        def wrapper(x):
            out = vllm_out_ref.clone()
            fn(out, x)
            return out

        return wrapper

    # Warm up
    api_func = (
        get_vllm_api_func(getattr(vllm_ops, kernel))
        if provider == "vllm"
        else getattr(sgl_kernel, kernel)
    )

    for _ in range(5):
        api_func(x)
    torch.cuda.synchronize()

    quantiles = [0.5, 0.2, 0.8]

    if provider == "vllm":
        vllm_act_kernel = getattr(vllm_ops, kernel)
        fn = lambda: vllm_act_kernel(vllm_out_ref.clone(), x)
    elif provider == "sglang":
        sglang_act_kernel = getattr(sgl_kernel, kernel)
        fn = lambda: sglang_act_kernel(x)

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        default="./configs/benchmark_ops/activations/",
        help="Path to save activations benchmark results",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify kernel",
    )

    args = parser.parse_args()

    if args.verify:
        calculate_diff("silu_and_mul", torch.float16, batch_size=1, seq_len=1, dim=128)
    else:
        benchmark.run(print_data=True)
