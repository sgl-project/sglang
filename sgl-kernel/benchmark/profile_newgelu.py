"""
NewGELU CUDA kernel profiling & benchmark script.

Usage:
  # Benchmark only (no ncu needed):
  python profile_newgelu.py --bench

  # For ncu profiling (run via ncu):
  ncu --kernel-name "new_gelu_kernel" --set full -o newgelu_report python profile_newgelu.py --ncu

  # Both:
  python profile_newgelu.py --bench
  ncu --kernel-name "new_gelu_kernel" --set full -o newgelu_report python profile_newgelu.py --ncu
"""

import argparse
import time

import torch
import torch.nn.functional as F

import sgl_kernel


def pytorch_new_gelu(x: torch.Tensor) -> torch.Tensor:
    """PyTorch reference implementation of NewGELU (gelu_tanh)."""
    return F.gelu(x, approximate="tanh")


def benchmark_one(fn, x, warmup=20, iters=100):
    """Benchmark a function, return median time in microseconds."""
    # warmup
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(x)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)  # microseconds

    times.sort()
    return times[len(times) // 2]  # median


def run_benchmark():
    """Benchmark PyTorch vs sgl_kernel new_gelu across different sizes."""
    print("=" * 90)
    print(f"{'Input Shape':>30s} | {'dtype':>8s} | {'PyTorch (µs)':>12s} | {'SGL Kernel (µs)':>15s} | {'Speedup':>8s}")
    print("-" * 90)

    dtypes = [torch.float16, torch.bfloat16]
    # (batch_size, seq_len, hidden_dim) — typical LLM shapes
    shapes = [
        (1, 1, 4096),
        (1, 128, 4096),
        (1, 512, 4096),
        (4, 128, 4096),
        (4, 512, 4096),
        (16, 128, 4096),
        (1, 1, 11008),
        (1, 128, 11008),
        (4, 128, 11008),
        (16, 128, 11008),
        (1, 1, 14336),
        (4, 128, 14336),
    ]

    for dtype in dtypes:
        for bs, sl, dim in shapes:
            x = torch.randn(bs, sl, dim, dtype=dtype, device="cuda")

            t_pytorch = benchmark_one(pytorch_new_gelu, x)
            t_sgl = benchmark_one(sgl_kernel.new_gelu, x)
            speedup = t_pytorch / t_sgl if t_sgl > 0 else float("inf")

            shape_str = f"({bs}, {sl}, {dim})"
            print(f"{shape_str:>30s} | {str(dtype):>8s} | {t_pytorch:>12.1f} | {t_sgl:>15.1f} | {speedup:>7.2f}x")

        print("-" * 90)

    print("=" * 90)


def run_ncu_target():
    """Run a single kernel invocation for ncu profiling."""
    # Use a typical LLM shape
    x = torch.randn(4, 128, 4096, dtype=torch.float16, device="cuda")

    # Warmup
    for _ in range(5):
        sgl_kernel.new_gelu(x)
    torch.cuda.synchronize()

    # Single invocation for ncu to capture
    y = sgl_kernel.new_gelu(x)
    torch.cuda.synchronize()

    # Correctness check
    ref = pytorch_new_gelu(x)
    max_diff = (y - ref).abs().max().item()
    print(f"Max diff vs PyTorch: {max_diff:.6e}")
    assert max_diff < 1e-2, f"Correctness check failed: max_diff={max_diff}"
    print("Correctness check passed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true", help="Run benchmark")
    parser.add_argument("--ncu", action="store_true", help="Run single invocation for ncu profiling")
    args = parser.parse_args()

    if not args.bench and not args.ncu:
        args.bench = True  # default to benchmark

    device_name = torch.cuda.get_device_name(0)
    print(f"GPU: {device_name}")
    print(f"CUDA: {torch.version.cuda}")
    print()

    if args.ncu:
        print("Running single kernel invocation for ncu profiling...")
        run_ncu_target()

    if args.bench:
        print("Running benchmark...")
        run_benchmark()


if __name__ == "__main__":
    main()
