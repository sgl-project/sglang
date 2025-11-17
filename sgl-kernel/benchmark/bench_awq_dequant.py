import itertools
import os
from typing import List, Tuple

import torch
import triton
import triton.testing
from sgl_kernel import awq_dequantize

# Optional vLLM import
try:
    from vllm import _custom_ops as ops

    VLLM_AVAILABLE = True
except ImportError:
    ops = None
    VLLM_AVAILABLE = False

# CI environment detection
IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)


def vllm_awq_dequantize(
    qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not VLLM_AVAILABLE:
        # Fallback to SGLang implementation
        return sglang_awq_dequantize(qweight, scales, qzeros)
    return ops.awq_dequantize(qweight, scales, qzeros, 0, 0, 0)


def sglang_awq_dequantize(
    qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    return awq_dequantize(qweight, scales, qzeros)


def calculate_diff(qweight_row: int, qweight_col: int):
    """Calculate difference between VLLM and SGLang implementations."""
    device = torch.device("cuda")
    qweight = torch.randint(
        0,
        torch.iinfo(torch.int32).max,
        (qweight_row, qweight_col),
        dtype=torch.int32,
        device=device,
    )
    group_size = qweight_row
    scales_row = qweight_row // group_size
    scales_col = qweight_col * 8
    scales = torch.rand(scales_row, scales_col, dtype=torch.float16, device=device)
    qzeros = torch.randint(
        0,
        torch.iinfo(torch.int32).max,
        (scales_row, qweight_col),
        dtype=torch.int32,
        device=device,
    )

    if not VLLM_AVAILABLE:
        print("⚠️ vLLM not available, skipping comparison")
        return

    vllm_out = vllm_awq_dequantize(qweight, scales, qzeros)
    sglang_out = sglang_awq_dequantize(qweight, scales, qzeros)

    output_diff = torch.abs(vllm_out.float() - sglang_out.float()).mean().item()

    if torch.allclose(
        vllm_out.to(torch.float32), sglang_out.to(torch.float32), rtol=1e-3, atol=1e-5
    ):
        print("✅ All implementations match")
    else:
        print("❌ Implementations differ")


# CI environment uses simplified parameters
if IS_CI:
    qweight_row_range = [128]  # Single row size for CI
    qweight_cols_range = [16]  # Single column size for CI
else:
    qweight_row_range = [3584, 18944, 128, 256, 512, 1024]
    qweight_cols_range = [448, 576, 4736, 16, 32, 64, 128]

configs = list(itertools.product(qweight_row_range, qweight_cols_range))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["qweight_row", "qweight_col"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["vllm", "sglang"] if VLLM_AVAILABLE else ["sglang"],
        line_names=["VLLM", "SGL Kernel"] if VLLM_AVAILABLE else ["SGL Kernel"],
        styles=[("blue", "-"), ("green", "-")] if VLLM_AVAILABLE else [("green", "-")],
        ylabel="us",
        plot_name="awq-dequantize-performance",
        args={},
    )
)
def benchmark(qweight_row, qweight_col, provider):
    dtype = torch.float16
    device = torch.device("cuda")
    qweight = torch.randint(
        0,
        torch.iinfo(torch.int32).max,
        (qweight_row, qweight_col),
        dtype=torch.int32,
        device=device,
    )
    group_size = qweight_row
    scales_row = qweight_row // group_size
    scales_col = qweight_col * 8
    scales = torch.rand(scales_row, scales_col, dtype=torch.float16, device=device)
    qzeros = torch.randint(
        0,
        torch.iinfo(torch.int32).max,
        (scales_row, qweight_col),
        dtype=torch.int32,
        device=device,
    )

    quantiles = [0.5, 0.2, 0.8]

    if provider == "vllm":
        if not VLLM_AVAILABLE:
            return (0, 0, 0)
        fn = lambda: vllm_awq_dequantize(
            qweight.clone(), scales.clone(), qzeros.clone()
        )
    elif provider == "sglang":
        fn = lambda: sglang_awq_dequantize(
            qweight.clone(), scales.clone(), qzeros.clone()
        )

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, quantiles=quantiles)

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    # Simplify for CI environment
    if IS_CI:
        qweight_row, qweight_col = 128, 16  # Smaller values for CI
    else:
        qweight_row, qweight_col = 3584, 448

    calculate_diff(qweight_row=qweight_row, qweight_col=qweight_col)
    benchmark.run(print_data=True)
