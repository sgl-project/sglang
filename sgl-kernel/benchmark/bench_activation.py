# Benchmarks SGLang kernels versus vLLM across
# (kernel, dtype, batch_size, seq_len, dim) and prints speed-up.
import argparse
import itertools
import os
import re
from typing import List, Tuple

import sgl_kernel
import torch
import torch.nn.functional as F
import triton
import triton.testing
from sgl_kernel import gelu_and_mul, gelu_tanh_and_mul, silu_and_mul

# Optional vLLM import
try:
    from vllm import _custom_ops as vllm_ops

    VLLM_AVAILABLE = True
except ImportError:
    vllm_ops = None
    VLLM_AVAILABLE = False

# CI environment detection
IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)

# gelu_quick is only available on HIP/ROCm platforms
try:
    from sgl_kernel import gelu_quick

    GELU_QUICK_AVAILABLE = True
except ImportError:
    GELU_QUICK_AVAILABLE = False
    gelu_quick = None

if VLLM_AVAILABLE and not hasattr(vllm_ops, "silu_and_mul"):
    vllm_ops = torch.ops._C


def str2int_list(arg: str) -> List[int]:
    if arg in ("", None):
        return []
    if re.fullmatch(r"\d+(,\d+)*", arg.strip()) is None:
        raise argparse.ArgumentTypeError(f"Bad int list: {arg}")
    return [int(x) for x in arg.split(",")]


def calculate_diff(
    kernel: str, dtype: torch.dtype, batch_size: int, seq_len: int, dim: int
) -> bool:
    """Compare vLLM with SGLang for one shape."""
    device = torch.device("cuda")

    if not VLLM_AVAILABLE:
        print(
            f"[{kernel:14s} | {str(dtype):9s} | B={batch_size:3d} | "
            f"L={seq_len:3d} | D={dim:5d}] ⚠️  vLLM not available, skipping comparison"
        )
        return True

    # activation-only quick GELU
    if kernel == "gelu_quick":
        if not GELU_QUICK_AVAILABLE:
            print(
                f"[{kernel:14s} | {str(dtype):9s} | B={batch_size:3d} | "
                f"L={seq_len:3d} | D={dim:5d}] ⚠️  not available on this platform"
            )
            return True
        x = torch.randn(batch_size, seq_len, dim, dtype=dtype, device=device)
        ref_out = torch.zeros_like(x)
        getattr(vllm_ops, kernel)(ref_out, x)
        test_out = getattr(sgl_kernel, kernel)(x)
    # fused activation x mul kernels
    else:
        x = torch.randn(batch_size, seq_len, 2 * dim, dtype=dtype, device=device)
        ref_out = torch.zeros(batch_size, seq_len, dim, dtype=dtype, device=device)
        getattr(vllm_ops, kernel)(ref_out, x)
        test_out = getattr(sgl_kernel, kernel)(x)

    ok = torch.allclose(ref_out, test_out, rtol=1e-3, atol=1e-5)
    tag = "✅ match" if ok else "❌ mismatch"
    print(
        f"[{kernel:14s} | {str(dtype):9s} | B={batch_size:3d} | "
        f"L={seq_len:3d} | D={dim:5d}] {tag}"
    )
    return ok


# CI environment uses simplified parameters for kernels and dtypes too
if IS_CI:
    kernels = ["silu_and_mul"]  # Only test one kernel in CI
    dtypes = [torch.float16]  # Only test one dtype in CI
else:
    kernels = ["silu_and_mul", "gelu_and_mul", "gelu_tanh_and_mul"]
    if GELU_QUICK_AVAILABLE:
        kernels.append("gelu_quick")
    dtypes = [torch.float16, torch.bfloat16]


def make_configs(bsizes: List[int], slens: List[int], dims_: List[int]) -> List[Tuple]:
    return list(itertools.product(kernels, dtypes, bsizes, slens, dims_))


# CI environment uses simplified parameters
if IS_CI:
    default_batch_sizes = [1]  # Single batch size for CI
    default_seq_lens = [1]  # Single sequence length for CI
    default_dims = [1024]  # Single dimension for CI
else:
    default_batch_sizes = [2**i for i in range(0, 5, 2)]  # 1,4,16
    default_seq_lens = [2**i for i in range(0, 8, 2)]  # 1,4,16,64
    default_dims = [2**i for i in range(10, 15)]  # 1024...16384


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["kernel", "dtype", "batch_size", "seq_len", "dim"],
        x_vals=[],
        line_arg="provider",
        line_vals=["vllm", "sglang", "speedup"],
        line_names=["vLLM", "SGL Kernel", "Speed-up (x)"],
        styles=[("blue", "-"), ("green", "-"), ("red", "--")],
        ylabel="µs (median)  or  × (speed-up)",
        plot_name="activation-performance",
        args={},
    )
)
def benchmark(kernel, dtype, batch_size, seq_len, dim, provider):
    device = torch.device("cuda")
    in_mult = 1 if kernel == "gelu_quick" else 2
    x = torch.randn(batch_size, seq_len, in_mult * dim, dtype=dtype, device=device)
    y0 = torch.zeros(batch_size, seq_len, dim, dtype=dtype, device=device)

    if not VLLM_AVAILABLE and provider in ["vllm", "speedup"]:
        # Skip vLLM-related benchmarks if vLLM is not available
        return (0, 0, 0)

    if VLLM_AVAILABLE:
        vllm_kernel = getattr(vllm_ops, kernel)
    if kernel == "gelu_quick" and not GELU_QUICK_AVAILABLE:
        # Skip benchmark for gelu_quick if not available
        return (0, 0, 0)
    sglang_kernel = getattr(sgl_kernel, kernel)

    def baseline():
        if VLLM_AVAILABLE:
            tmp = y0.clone()
            vllm_kernel(tmp, x)
            return tmp
        else:
            return torch.zeros_like(y0)

    def sglang():
        return sglang_kernel(x)

    # timing helper
    def timed(fn):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        ms, qmin, qmax = triton.testing.do_bench_cudagraph(
            fn, quantiles=[0.5, 0.2, 0.8]
        )
        return 1000 * ms, 1000 * qmax, 1000 * qmin

    if provider == "vllm":
        return timed(baseline)
    if provider == "sglang":
        return timed(sglang)

    # provider == "speedup"
    t_ref, _, _ = timed(baseline)
    t_sgl, _, _ = timed(sglang)
    spd = t_ref / t_sgl if t_ref > 0 else 1.0
    return (spd, spd, spd)


if __name__ == "__main__":
    p = argparse.ArgumentParser("Activation kernel benchmark")
    p.add_argument("--batch_sizes", type=str2int_list, default=default_batch_sizes)
    p.add_argument("--seq_lens", type=str2int_list, default=default_seq_lens)
    p.add_argument("--dims", type=str2int_list, default=default_dims)
    p.add_argument("--verify_only", action="store_true")
    args = p.parse_args()

    # coerce lists
    if isinstance(args.batch_sizes, str):
        args.batch_sizes = str2int_list(args.batch_sizes)
    if isinstance(args.seq_lens, str):
        args.seq_lens = str2int_list(args.seq_lens)
    if isinstance(args.dims, str):
        args.dims = str2int_list(args.dims)

    # patch perf_report grid
    benchmark_grid = make_configs(args.batch_sizes, args.seq_lens, args.dims)
    if hasattr(benchmark, "benchmarks"):
        benchmark.benchmarks.x_vals = benchmark_grid
    else:
        benchmark.benchmark.x_vals = benchmark_grid

    if args.verify_only:
        # Test with the first available kernel
        test_kernel = kernels[0]
        ok = calculate_diff(test_kernel, torch.float16, 1, 1, args.dims[0])
        print("✅ sanity pass" if ok else "❌ mismatch")
    else:
        benchmark.run(print_data=True)
