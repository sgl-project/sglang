import itertools
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import triton
import triton.testing
from sgl_kernel import (
    sgl_per_tensor_quant_fp8,
    sgl_silu_and_mul_per_tensor_quant_fp8,
    silu_and_mul,
)

from sglang.srt.utils import is_hip

_is_hip = is_hip()
fp8_type_ = torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn


def sglang_silu_and_mul_scaled_fp8_quant(
    input: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    fp8_type_: torch.dtype = torch.float8_e4m3fn
    output = torch.empty(
        input.shape[0], input.shape[1] // 2, device=input.device, dtype=fp8_type_
    )

    intermediate_cache = torch.empty(
        input.shape[0], input.shape[1] // 2, device=input.device, dtype=input.dtype
    )
    silu_and_mul(input, intermediate_cache)
    scale = torch.zeros(1, device=input.device, dtype=torch.float32)
    sgl_per_tensor_quant_fp8(
        intermediate_cache.contiguous(), output, scale, is_static=False
    )

    return output, scale


def sglang_fused_silu_and_mul_scaled_fp8_quant(
    input: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    fp8_type_: torch.dtype = torch.float8_e4m3fn
    output = torch.empty(
        input.shape[0], input.shape[1] // 2, device=input.device, dtype=fp8_type_
    )
    intermediate_cache = torch.empty(
        input.shape[0], input.shape[1] // 2, device=input.device, dtype=input.dtype
    )
    scale = torch.zeros(1, device=input.device, dtype=torch.float32)
    # input_gate, input_up = input.chunk(2, dim=-1)
    # input_gate = input_gate.contiguous()
    # input_up = input_up.contiguous()
    sgl_silu_and_mul_per_tensor_quant_fp8(
        input, intermediate_cache, output, scale, is_static=False
    )

    return output, scale


def calculate_diff(batch_size: int, seq_len: int):
    """Calculate difference between VLLM and SGLang implementations."""
    device = torch.device("cuda")
    x = torch.rand((batch_size, seq_len), dtype=torch.float16, device=device)

    sglang_out, sglang_scale = sglang_silu_and_mul_scaled_fp8_quant(x)
    fused_out, fused_scale = sglang_fused_silu_and_mul_scaled_fp8_quant(x)

    scale_diff = torch.abs(fused_scale - sglang_scale).item()
    output_diff = torch.abs(fused_out.float() - sglang_out.float()).mean().item()

    if torch.allclose(
        fused_out.to(torch.float32), sglang_out.to(torch.float32), rtol=1e-3, atol=1e-5
    ) and torch.allclose(fused_scale, sglang_scale, rtol=1e-3, atol=1e-5):
        print("✅ All implementations match")
    else:
        print("❌ Implementations differ")


batch_size_range = [16, 32, 64, 128]
seq_len_range = [64, 128, 256, 512, 1024, 2048]

configs = list(itertools.product(batch_size_range, seq_len_range))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "seq_len"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["sglang", "sglang_fused"],
        line_names=["sglang", "sglang_fused"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="silu-and-mul-per-tensor-quant-fp8-performance",
        args={},
    )
)
def benchmark(batch_size, seq_len, provider):
    dtype = torch.float16
    device = torch.device("cuda")

    x = torch.randn(batch_size * seq_len, 4096, device=device, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "sglang_fused":
        fn = lambda: sglang_fused_silu_and_mul_scaled_fp8_quant(x.clone())
    elif provider == "sglang":
        fn = lambda: sglang_silu_and_mul_scaled_fp8_quant(x.clone())

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    calculate_diff(batch_size=4, seq_len=4096)
    benchmark.run(print_data=True)
