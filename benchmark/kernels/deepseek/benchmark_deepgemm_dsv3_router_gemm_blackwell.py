import argparse
from typing import List

import torch
import triton
from flashinfer.gemm.routergemm_dsv3 import mm_M1_16_K7168_N256
from sgl_kernel import dsv3_router_gemm as dsv3_router_gemm

N = 256
K = 7168


def create_benchmark_configs(tp_sizes: List[int]):
    configs = []
    for launch_with_pdl in [False, True]:
        for tp_size in tp_sizes:
            for m in range(1, 17):
                configs.append((m, N, K, tp_size, launch_with_pdl))
    return configs


def dsv3_router_gemm_flashinfer(
    hidden_states: torch.Tensor,
    router_weights: torch.Tensor,
    launch_with_pdl: bool,
):
    """Flashinfer implementation of dsv3 router gemm"""
    output = torch.randn(
        hidden_states.shape[0],
        router_weights.shape[0],
        device="cuda",
        dtype=torch.float32,
    ).contiguous()

    mm_M1_16_K7168_N256(
        hidden_states, router_weights.t(), output, launch_with_pdl=launch_with_pdl
    )
    return output


def dsv3_router_gemm_sgl(
    hidden_states: torch.Tensor,
    router_weights: torch.Tensor,
):
    """SGLang implementation of dsv3 router gemm"""
    output = dsv3_router_gemm(
        hidden_states,
        router_weights,
        out_dtype=torch.float32,
    )
    return output


def check_accuracy(a, b, atol, rtol, percent):
    """Unified accuracy checking function with detailed error reporting."""
    if not torch.isfinite(a).all():
        print("Non-finite values in reference output")
        return False
    if not torch.isfinite(b).all():
        print("Non-finite values in actual output")
        return False
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"

    close = torch.isclose(a, b, atol=atol, rtol=rtol)
    match_ratio = close.float().mean()
    if match_ratio >= percent:
        return True

    mismatch_percent = 1.0 - match_ratio.item()
    if mismatch_percent > 1 - percent:
        print(
            f"Mismatch percentage is {mismatch_percent:.4f} for rtol {rtol} "
            f"(threshold: {1 - percent:.4f})"
        )
        return False


def calculate_diff(m: int, n: int, k: int, launch_with_pdl: bool):
    hidden_states = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    router_weights = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)

    out_flashinfer = dsv3_router_gemm_flashinfer(
        hidden_states.clone(memory_format=torch.contiguous_format),
        router_weights.clone(memory_format=torch.contiguous_format),
        launch_with_pdl,
    )

    out_sgl = dsv3_router_gemm_sgl(
        hidden_states.clone(memory_format=torch.contiguous_format),
        router_weights.clone(memory_format=torch.contiguous_format),
    )

    print(f"Shape m={m}, n={n}, k={k}:")
    print(f"Using launch_with_pdl={launch_with_pdl} for flashinfer")
    print(f"Flashinfer output: {out_flashinfer[0, 0:5]}")
    print(f"SGLang output: {out_sgl[0, 0:5]}")

    flashinfer_sgl_match = check_accuracy(out_flashinfer, out_sgl, 0.1, 0.6, 0.95)
    print("Correctness check:")
    print(f"  - Flashinfer vs SGLang: {'✅' if flashinfer_sgl_match else '❌'}")


def _benchmark(m, n, k, tp_size, launch_with_pdl, provider):
    print(
        f"Shape (m={m}, n={n}, k={k}, tp={tp_size}), launch_with_pdl={launch_with_pdl}, Provider: {provider}"
    )
    hidden_states = torch.randn(
        (m, k), device="cuda", dtype=torch.bfloat16
    ).contiguous()
    router_weights = torch.randn(
        (n, k), device="cuda", dtype=torch.bfloat16
    ).contiguous()

    quantiles = [0.5, 0.2, 0.8]

    if provider == "sglang":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: dsv3_router_gemm_sgl(
                hidden_states.clone(memory_format=torch.contiguous_format),
                router_weights.clone(memory_format=torch.contiguous_format),
            ),
            quantiles=quantiles,
        )
    elif provider == "flashinfer":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: dsv3_router_gemm_flashinfer(
                hidden_states.clone(memory_format=torch.contiguous_format),
                router_weights.clone(memory_format=torch.contiguous_format),
                launch_with_pdl,
            ),
            quantiles=quantiles,
        )

    # Calculate TFLOPS
    flops = 2 * m * n * k  # multiply-adds
    tflops = flops / (ms * 1e-3) / 1e12

    # Print shape-specific results with TFLOPS
    print(f"Time: {ms*1000:.2f} us, TFLOPS: {tflops:.2f}")
    return ms, max_ms, min_ms


def get_benchmark_plot_friendly(tp_sizes):
    all_configs = create_benchmark_configs(tp_sizes)
    x_vals = list(range(len(all_configs)))

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["cfg_id"],
            x_vals=x_vals,
            line_arg="provider",
            line_vals=["sglang", "flashinfer"],
            line_names=["SGLang", "Flashinfer"],
            styles=[("blue", "-"), ("red", "-")],
            ylabel="us",
            plot_name=f"fp8-gemm-performance-comparison-tp-{"-".join(str(tp) for tp in tp_sizes)}",
            args={},
        )
    )
    def benchmark(cfg_id, provider):
        m, n, k, tp_size, launch_with_pdl = all_configs[cfg_id]
        ms, min_ms, max_ms = _benchmark(m, n, k, tp_size, launch_with_pdl, provider)
        return ms * 1000, max_ms * 1000, min_ms * 1000  # convert to ms

    return benchmark


def get_benchmark(tp_sizes):
    all_configs = create_benchmark_configs(tp_sizes)

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=[
                "m",
                "n",
                "k",
                "tp_size",
                "launch_with_pdl",
            ],
            x_vals=[list(config) for config in all_configs],
            line_arg="provider",
            line_vals=["sglang", "flashinfer"],
            line_names=["SGLang", "Flashinfer"],
            styles=[("blue", "-"), ("red", "-")],
            ylabel="us",
            plot_name=f"fp8-gemm-performance-comparison-tp-{"-".join(str(tp) for tp in tp_sizes)}",
            args={},
        )
    )
    def benchmark(m, n, k, tp_size, launch_with_pdl, provider):
        ms, min_ms, max_ms = _benchmark(m, n, k, tp_size, launch_with_pdl, provider)
        return ms * 1000, max_ms * 1000, min_ms * 1000  # convert to ms

    return benchmark


if __name__ == "__main__":
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        print("Skipping benchmark because the device is not supported")
        exit(0)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-path",
        type=str,
        default="./configs/benchmark_ops/dsv3_router_gemm/",
        help="Path to save dsv3 router gemm benchmark results",
    )
    parser.add_argument(
        "--run-correctness",
        action="store_true",
        default=True,
        help="Whether to run correctness test",
    )
    parser.add_argument(
        "--tp-sizes",
        type=int,
        nargs="+",
        default=[1],
        help="List of tensor parallelism sizes to benchmark",
    )
    parser.add_argument(
        "--plot-friendly",
        action="store_true",
        default=False,
        help="Plot x axis as the config index instead of the m",
    )
    args = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # Run correctness tests on a few examples
    if args.run_correctness:
        print("Running correctness tests...")
        for m, n, k, _, launch_with_pdl in create_benchmark_configs(args.tp_sizes):
            calculate_diff(m, n, k, launch_with_pdl)

    # Get the benchmark function with the specified tp_size
    benchmark = (
        get_benchmark_plot_friendly(args.tp_sizes)
        if args.plot_friendly
        else get_benchmark(args.tp_sizes)
    )

    print(f"Running performance benchmark for TP sizes = {args.tp_sizes}...")
    benchmark.run(print_data=True, save_path=args.save_path)
