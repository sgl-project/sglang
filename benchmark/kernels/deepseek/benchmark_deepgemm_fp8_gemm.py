import itertools
from typing import Tuple

import deep_gemm
import numpy as np
import torch
import triton
import triton.language as tl
from deep_gemm import cell_div, get_col_major_tma_aligned_tensor
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    w8a8_block_fp8_matmul as vllm_w8a8_block_fp8_matmul,
)

from sglang.srt.layers.quantization.fp8_kernel import w8a8_block_fp8_matmul


def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(
        m, n
    ), (x_amax / 448.0).view(m, -1)


def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(
        (cell_div(m, 128) * 128, cell_div(n, 128) * 128), dtype=x.dtype, device=x.device
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(
        x_view.size(0), x_view.size(2)
    )


def fp8_gemm_deepgemm(x: torch.Tensor, y: torch.Tensor, m: int, n: int, k: int):
    """DeepGEMM implementation of FP8 GEMM"""
    # Convert inputs to FP8 format
    x_fp8, x_scale = per_token_cast_to_fp8(x)
    y_fp8, y_scale = per_block_cast_to_fp8(y)

    # Transpose earlier so that the testing will not trigger transposing kernels
    x_scale = get_col_major_tma_aligned_tensor(x_scale)

    # Prepare inputs for DeepGEMM
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

    # Run DeepGEMM kernel
    deep_gemm.gemm_fp8_fp8_bf16_nt((x_fp8, x_scale), (y_fp8, y_scale), out)
    return out


def fp8_gemm_sglang(x: torch.Tensor, y: torch.Tensor, m: int, n: int, k: int):
    """SGLang implementation of FP8 GEMM"""
    # Convert inputs to FP8 format
    x_fp8, x_scale = per_token_cast_to_fp8(x)
    y_fp8, y_scale = per_block_cast_to_fp8(y)

    block_size = [128, 128]  # Matches the block size in per_block_cast_to_fp8

    # Run SGLang kernel
    out = w8a8_block_fp8_matmul(
        x_fp8, y_fp8, x_scale, y_scale, block_size, torch.bfloat16
    )
    return out


def fp8_gemm_vllm(x: torch.Tensor, y: torch.Tensor, m: int, n: int, k: int):
    """vLLM implementation of FP8 GEMM"""
    # Convert inputs to FP8 format
    x_fp8, x_scale = per_token_cast_to_fp8(x)
    y_fp8, y_scale = per_block_cast_to_fp8(y)

    block_size = [128, 128]  # Matches the block size in per_block_cast_to_fp8

    # Run vLLM kernel
    out = vllm_w8a8_block_fp8_matmul(
        x_fp8, y_fp8, x_scale, y_scale, block_size, torch.bfloat16
    )
    return out


def calculate_diff(m: int, n: int, k: int):
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    y = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)

    out_deepgemm = fp8_gemm_deepgemm(x.clone(), y.clone(), m, n, k)
    out_sglang = fp8_gemm_sglang(x.clone(), y.clone(), m, n, k)
    out_vllm = fp8_gemm_vllm(x.clone(), y.clone(), m, n, k)

    diff_sglang_deepgemm = torch.abs(out_deepgemm - out_sglang).mean().item()
    diff_vllm_deepgemm = torch.abs(out_deepgemm - out_vllm).mean().item()
    diff_vllm_sglang = torch.abs(out_vllm - out_sglang).mean().item()

    print(f"Shape m={m}, n={n}, k={k}:")
    print(f"DeepGEMM output: {out_deepgemm[0, 0:5]}")
    print(f"SGLang output: {out_sglang[0, 0:5]}")
    print(f"vLLM output: {out_vllm[0, 0:5]}")
    print(f"Mean absolute difference (SGLang-DeepGEMM): {diff_sglang_deepgemm}")
    print(f"Mean absolute difference (vLLM-DeepGEMM): {diff_vllm_deepgemm}")
    print(f"Mean absolute difference (vLLM-SGLang): {diff_vllm_sglang}")

    sglang_deepgemm_match = torch.allclose(
        out_deepgemm, out_sglang, atol=1e-2, rtol=1e-2
    )
    vllm_deepgemm_match = torch.allclose(out_deepgemm, out_vllm, atol=1e-2, rtol=1e-2)
    vllm_sglang_match = torch.allclose(out_vllm, out_sglang, atol=1e-2, rtol=1e-2)

    if sglang_deepgemm_match and vllm_deepgemm_match and vllm_sglang_match:
        print("✅ All implementations match\n")
    else:
        print("❌ Some implementations differ:")
        print(f"  - SGLang vs DeepGEMM: {'✅' if sglang_deepgemm_match else '❌'}")
        print(f"  - vLLM vs DeepGEMM: {'✅' if vllm_deepgemm_match else '❌'}")
        print(f"  - vLLM vs SGLang: {'✅' if vllm_sglang_match else '❌'}\n")


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


def get_benchmark(tp_size):
    all_configs = create_benchmark_configs(tp_size)

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["m", "n", "k", "tp_size"],
            x_vals=[list(config) for config in all_configs],
            line_arg="provider",
            line_vals=["deepgemm", "sglang", "vllm"],
            line_names=["DeepGEMM", "SGLang", "vLLM"],
            styles=[("blue", "-"), ("red", "-"), ("green", "-")],
            ylabel="ms",
            plot_name=f"fp8-gemm-performance-comparison-tp{tp_size}",
            args={},
        )
    )
    def benchmark(m, n, k, tp_size, provider):
        x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        y = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)

        quantiles = [0.5, 0.2, 0.8]

        if provider == "deepgemm":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: fp8_gemm_deepgemm(x.clone(), y.clone(), m, n, k),
                quantiles=quantiles,
            )
        elif provider == "sglang":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: fp8_gemm_sglang(x.clone(), y.clone(), m, n, k),
                quantiles=quantiles,
            )
        else:  # vllm
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: fp8_gemm_vllm(x.clone(), y.clone(), m, n, k),
                quantiles=quantiles,
            )

        # Calculate TFLOPS
        flops = 2 * m * n * k  # multiply-adds
        tflops = flops / (ms * 1e-3) / 1e12

        # Print shape-specific results with TFLOPS
        print(f"Shape (m={m}, n={n}, k={k}, tp={tp_size}), Provider: {provider}")
        print(f"Time: {ms*1000:.2f} ms, TFLOPS: {tflops:.2f}")
        return ms * 1000, max_ms * 1000, min_ms * 1000  # convert to ms

    return benchmark


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        default="./configs/benchmark_ops/fp8_gemm/",
        help="Path to save fp8 gemm benchmark results",
    )
    parser.add_argument(
        "--run_correctness",
        action="store_true",
        help="Whether to run correctness test",
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=1,
        help="Tensor parallelism size to benchmark (default: 1)",
    )
    args = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # Enable TF32, adapted from https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_core.py#L148
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Run correctness tests on a few examples
    if args.run_correctness:
        print("Running correctness tests...")
        calculate_diff(64, 512, 7168)  # Small test
        calculate_diff(64, 7168, 16384)  # Medium test
        calculate_diff(64, 18432, 7168)  # Large test

    # Get the benchmark function with the specified tp_size
    benchmark = get_benchmark(args.tp_size)

    print(f"Running performance benchmark for TP size = {args.tp_size}...")
    benchmark.run(print_data=True, save_path=args.save_path)
