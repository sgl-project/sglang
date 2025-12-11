import argparse
import itertools
from typing import List, Tuple

import sgl_kernel
import torch
import triton
import triton.testing


def str2int_list(arg: str) -> List[int]:
    if arg in ("", None):
        return []
    return [int(x) for x in arg.split(",")]


def fused_layernorm_scale_shift_naive(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float = 1e-5,
):
    dtype = x.dtype
    x32 = x.float()
    w32 = weight.float()
    b32 = bias.float()
    s32 = scale.float()
    sh32 = shift.float()
    
    mean = x32.mean(dim=1, keepdim=True)
    var = (x32 - mean).pow(2).mean(dim=1, keepdim=True)
    inv_std = (var + eps).sqrt().reciprocal()
    
    y_ln32 = (x32 - mean) * inv_std
    y_ln32 = y_ln32 * w32 + b32
    y_out = (y_ln32 * (1.0 + s32) + sh32).to(dtype)
    return y_out


def fused_layernorm_scale_shift_sglang(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float = 1e-5,
):
    return sgl_kernel.fused_layernorm_scale_shift(x, weight, bias, scale, shift)


def make_configs(bsizes: List[int], hsizes: List[int]) -> List[Tuple]:
    return list(itertools.product(bsizes, hsizes))


# Define providers
providers = ["naive", "sglang"]
provider_names = ["PyTorch Naive", "SGL Kernel"]
styles = [("blue", "-"), ("orange", "-")]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "N"],  # Argument names to use as an x-axis for the plot
        x_vals=[],  # Will be filled in __main__
        line_arg="provider",
        line_vals=providers,
        line_names=provider_names,
        styles=styles,
        ylabel="µs (median)",
        plot_name="fused-layernorm-scale-shift-performance",
        args={},
    )
)
def benchmark(M, N, provider):
    device = "cuda"
    dtype = torch.bfloat16
    eps = 1e-5

    # Initialize tensors
    x = torch.randn(M, N, device=device, dtype=dtype)
    weight = torch.randn(N, device=device, dtype=dtype)
    bias = torch.randn(N, device=device, dtype=dtype)
    scale = torch.randn(M, N, device=device, dtype=dtype)
    shift = torch.randn(M, N, device=device, dtype=dtype)

    # Timing helper
    def timed(fn):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        ms, qmin, qmax = triton.testing.do_bench_cudagraph(
            fn, quantiles=[0.5, 0.2, 0.8]
        )
        return 1000 * ms, 1000 * qmax, 1000 * qmin

    if provider == "naive":
        return timed(lambda: fused_layernorm_scale_shift_naive(x, weight, bias, scale, shift, eps))
    elif provider == "sglang":
        return timed(lambda: fused_layernorm_scale_shift_sglang(x, weight, bias, scale, shift, eps))


if __name__ == "__main__":
    p = argparse.ArgumentParser("Fused LayerNorm Scale Shift Benchmark")
    # Default cases: (M, N)
    # M represents batch_size * seq_len or similar flattened dimension
    # N represents hidden_size
    p.add_argument("--m_sizes", type=str2int_list, default=[128, 1024, 4096])
    p.add_argument("--n_sizes", type=str2int_list, default=[1024, 3072, 4096])
    
    args = p.parse_args()

    # Patch grid
    benchmark_grid = make_configs(args.m_sizes, args.n_sizes)
    if hasattr(benchmark, "benchmarks"):
        benchmark.benchmarks.x_vals = benchmark_grid
    else:
        benchmark.benchmark.x_vals = benchmark_grid

    print(f"Benchmarking with M={args.m_sizes}, N={args.n_sizes}")
    benchmark.run(print_data=True)


