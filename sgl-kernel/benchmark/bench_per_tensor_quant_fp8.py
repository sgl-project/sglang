import itertools
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import triton
import triton.testing
from sgl_kernel import sgl_per_tensor_quant_fp8

# Optional imports
try:
    from vllm import _custom_ops as ops

    VLLM_AVAILABLE = True
except ImportError:
    ops = None
    VLLM_AVAILABLE = False

from sglang.srt.utils import is_hip

_is_hip = is_hip()

# CI environment detection
IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)

fp8_type_ = torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn


def vllm_scaled_fp8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not VLLM_AVAILABLE:
        # Fallback to SGLang implementation
        return sglang_scaled_fp8_quant(input, scale)
    return ops.scaled_fp8_quant(input, scale)


def sglang_scaled_fp8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    fp8_type_: torch.dtype = torch.float8_e4m3fn
    output = torch.empty_like(input, device=input.device, dtype=fp8_type_)
    is_static = True
    if scale is None:
        scale = torch.zeros(1, device=input.device, dtype=torch.float32)
        is_static = False
    sgl_per_tensor_quant_fp8(input, output, scale, is_static)

    return output, scale


def calculate_diff(batch_size: int, seq_len: int):
    """Calculate difference between VLLM and SGLang implementations."""
    device = torch.device("cuda")
    x = torch.rand((batch_size, seq_len), dtype=torch.float16, device=device)

    if not VLLM_AVAILABLE:
        print("⚠️ vLLM not available, skipping comparison")
        return

    vllm_out, vllm_scale = vllm_scaled_fp8_quant(x)
    sglang_out, sglang_scale = sglang_scaled_fp8_quant(x)

    scale_diff = torch.abs(vllm_scale - sglang_scale).item()
    output_diff = torch.abs(vllm_out.float() - sglang_out.float()).mean().item()

    if torch.allclose(
        vllm_out.to(torch.float32), sglang_out.to(torch.float32), rtol=1e-3, atol=1e-5
    ) and torch.allclose(vllm_scale, sglang_scale, rtol=1e-3, atol=1e-5):
        print("✅ All implementations match")
    else:
        print("❌ Implementations differ")


# CI environment uses simplified parameters
if IS_CI:
    batch_size_range = [16]  # Single batch size for CI
    seq_len_range = [64]  # Single sequence length for CI
else:
    batch_size_range = [16, 32, 64, 128]
    seq_len_range = [64, 128, 256, 512, 1024, 2048]

configs = list(itertools.product(batch_size_range, seq_len_range))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "seq_len"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["vllm", "sglang"],
        line_names=["VLLM", "SGL Kernel"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="per-tensor-quant-fp8-performance",
        args={},
    )
)
def benchmark(batch_size, seq_len, provider):
    dtype = torch.float16
    device = torch.device("cuda")

    x = torch.randn(batch_size * seq_len, 4096, device=device, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "vllm":
        fn = lambda: vllm_scaled_fp8_quant(x.clone())
    elif provider == "sglang":
        fn = lambda: sglang_scaled_fp8_quant(x.clone())

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, quantiles=quantiles)

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    calculate_diff(batch_size=4, seq_len=4096)
    benchmark.run(print_data=True)
