import argparse
from typing import Tuple

import torch
import triton
from deep_gemm import ceil_div
from flashinfer.gemm import gemm_fp8_nt_groupwise

from sglang.srt.layers.quantization.fp8_kernel import (
    sglang_per_token_group_quant_fp8,
    w8a8_block_fp8_matmul_deepgemm,
)
from sglang.srt.layers.quantization.fp8_utils import requant_weight_ue8m0

BLOCK_SIZE = 128


def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    assert BLOCK_SIZE == 128
    m, n = x.shape
    x_padded = torch.zeros(
        (ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(
        x_view.size(0), x_view.size(2)
    )


def get_weight_shapes(tp_size):
    # cannot TP
    total = [
        (512 + 64, 7168),
        ((128 + 64) * 128, 7168),
        (128 * (128 + 128), 512),
        (7168, 16384),
        (7168, 18432),
    ]
    # N can TP
    n_tp = [
        (18432 * 2, 7168),
        ((128 + 64) * 128, 7168),
        (128 * (128 + 128), 512),
        (24576, 1536),
        (4096, 7168),
    ]
    # K can TP
    k_tp = [(7168, 18432), (7168, 16384), (7168, 2048)]

    weight_shapes = []
    for t in total:
        weight_shapes.append(t)
    for n_t in n_tp:
        new_t = (n_t[0] // tp_size, n_t[1])
        weight_shapes.append(new_t)
    for k_t in k_tp:
        new_t = (k_t[0], k_t[1] // tp_size)
        weight_shapes.append(new_t)

    return weight_shapes


def create_benchmark_configs(tp_size):
    configs = []
    weight_shapes = get_weight_shapes(tp_size)
    batch_sizes = [8, 16, 32, 64, 128, 256, 1024, 2048, 4096]

    for n, k in weight_shapes:
        for m in batch_sizes:
            configs.append((m, n, k, tp_size))

    return configs


def fp8_gemm_flashinfer(
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    y_fp8: torch.Tensor,
    y_scale: torch.Tensor,
):
    """Flashinfer implementation of FP8 GEMM"""
    output = gemm_fp8_nt_groupwise(
        x_fp8,
        y_fp8,
        x_scale,
        y_scale,
        out_dtype=torch.bfloat16,
        backend="trtllm",
    )
    return output


def fp8_gemm_deepgemm_blackwell(
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    y_fp8: torch.Tensor,
    y_scale: torch.Tensor,
):
    """DeepGEMM implementation of FP8 GEMM"""
    block_size = [BLOCK_SIZE, BLOCK_SIZE]
    output = w8a8_block_fp8_matmul_deepgemm(
        x_fp8, y_fp8, x_scale, y_scale, block_size, output_dtype=torch.bfloat16
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


def calculate_diff(m: int, n: int, k: int):
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    y = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)

    y_fp8, y_scale = per_block_cast_to_fp8(y)
    x_fp8, x_scale = sglang_per_token_group_quant_fp8(
        x, BLOCK_SIZE, column_major_scales=True
    )
    out_flashinfer = fp8_gemm_flashinfer(
        x_fp8,
        x_scale,
        y_fp8,
        y_scale,
    )

    dg_x_fp8, dg_x_scale = sglang_per_token_group_quant_fp8(
        x,
        BLOCK_SIZE,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )
    # We can directly quantize y here, but to mimic the behavior of the actual
    # implementations, we requant it here.
    dg_y_fp8, dg_y_scale = requant_weight_ue8m0(
        y_fp8, y_scale, [BLOCK_SIZE, BLOCK_SIZE]
    )
    out_deepgemm = fp8_gemm_deepgemm_blackwell(
        dg_x_fp8, dg_x_scale, dg_y_fp8, dg_y_scale
    )

    print(f"Shape m={m}, n={n}, k={k}:")
    print(f"Flashinfer output: {out_flashinfer[0, 0:5]}")
    print(f"DeepGEMM output: {out_deepgemm[0, 0:5]}")

    flashinfer_deepgemm_match = check_accuracy(
        out_flashinfer, out_deepgemm, 0.1, 0.6, 0.95
    )
    print("Correctness check:")
    print(f"  - Flashinfer vs DeepGEMM: {'✅' if flashinfer_deepgemm_match else '❌'}")


def _benchmark(m, n, k, tp_size, provider):
    print(f"Shape (m={m}, n={n}, k={k}, tp={tp_size}), Provider: {provider}")
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    y = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)

    # Preprocess data before benchmarking
    y_fp8, y_scale = per_block_cast_to_fp8(y)
    x_fp8, x_scale = sglang_per_token_group_quant_fp8(
        x, BLOCK_SIZE, column_major_scales=True
    )
    dg_x_fp8, dg_x_scale = sglang_per_token_group_quant_fp8(
        x,
        BLOCK_SIZE,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )
    dg_y_fp8, dg_y_scale = requant_weight_ue8m0(
        y_fp8, y_scale, [BLOCK_SIZE, BLOCK_SIZE]
    )

    quantiles = [0.5, 0.2, 0.8]

    if provider == "deepgemm":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: fp8_gemm_deepgemm_blackwell(
                dg_x_fp8,
                dg_x_scale,
                dg_y_fp8,
                dg_y_scale,
            ),
            quantiles=quantiles,
        )
    elif provider == "flashinfer":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: fp8_gemm_flashinfer(
                x_fp8,
                x_scale,
                y_fp8,
                y_scale,
            ),
            quantiles=quantiles,
        )

    # Calculate TFLOPS
    flops = 2 * m * n * k  # multiply-adds
    tflops = flops / (ms * 1e-3) / 1e12

    # Print shape-specific results with TFLOPS
    print(f"Time: {ms*1000:.2f} us, TFLOPS: {tflops:.2f}")
    return ms, max_ms, min_ms


def get_benchmark_plot_friendly(tp_size):
    all_configs = create_benchmark_configs(tp_size)
    x_vals = list(range(len(all_configs)))

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["cfg_id"],
            x_vals=x_vals,
            line_arg="provider",
            line_vals=["deepgemm", "flashinfer"],
            line_names=["DeepGEMM", "Flashinfer"],
            styles=[("blue", "-"), ("red", "-")],
            ylabel="us",
            plot_name=f"fp8-gemm-performance-comparison-tp{tp_size}",
            args={},
        )
    )
    def benchmark(cfg_id, provider):
        m, n, k, tp_size = all_configs[cfg_id]
        ms, min_ms, max_ms = _benchmark(m, n, k, tp_size, provider)
        return ms * 1000, max_ms * 1000, min_ms * 1000  # convert to ms

    return benchmark


def get_benchmark(tp_size):
    all_configs = create_benchmark_configs(tp_size)

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["m", "n", "k", "tp_size"],
            x_vals=[list(config) for config in all_configs],
            line_arg="provider",
            line_vals=["deepgemm", "flashinfer"],
            line_names=["DeepGEMM", "Flashinfer"],
            styles=[("blue", "-"), ("red", "-")],
            ylabel="us",
            plot_name=f"fp8-gemm-performance-comparison-tp{tp_size}",
            args={},
        )
    )
    def benchmark(m, n, k, tp_size, provider):
        ms, min_ms, max_ms = _benchmark(m, n, k, tp_size, provider)
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
        default="./configs/benchmark_ops/fp8_gemm/",
        help="Path to save fp8 gemm benchmark results",
    )
    parser.add_argument(
        "--run-correctness",
        action="store_true",
        default=True,
        help="Whether to run correctness test",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallelism size to benchmark (default: 1)",
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
        calculate_diff(64, 512, 7168)  # Small test
        calculate_diff(64, 7168, 16384)  # Medium test
        calculate_diff(64, 18432, 7168)  # Large test

    # Get the benchmark function with the specified tp_size
    benchmark = (
        get_benchmark_plot_friendly(args.tp_size)
        if args.plot_friendly
        else get_benchmark(args.tp_size)
    )

    print(f"Running performance benchmark for TP size = {args.tp_size}...")
    benchmark.run(print_data=True, save_path=args.save_path)
