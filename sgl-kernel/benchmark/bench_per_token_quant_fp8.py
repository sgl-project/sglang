import itertools
from typing import Optional, Tuple

import torch
import triton
import triton.testing
from sgl_kernel import sgl_per_token_quant_fp8
from vllm import _custom_ops as ops

from sglang.srt.utils import is_hip

_is_hip = is_hip()
fp8_type_ = torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn


def vllm_per_token_quant_fp8(
    input: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return ops.scaled_fp8_quant(input, use_per_token_if_dynamic=True)


def sglang_per_token_quant_fp8(
    input: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    scale = torch.zeros(input.size(0), device=input.device, dtype=torch.float32)
    output = torch.empty_like(input, device=input.device, dtype=fp8_type_)
    sgl_per_token_quant_fp8(input, output, scale)

    return output, scale


def calculate_diff(batch_size: int, seq_len: int):
    """Calculate difference between VLLM and SGLang implementations."""
    device = torch.device("cuda")
    x = torch.rand((batch_size, seq_len), dtype=torch.float16, device=device)

    vllm_out, vllm_scale = vllm_per_token_quant_fp8(x)
    sglang_out, sglang_scale = sglang_per_token_quant_fp8(x)

    scale_diff = torch.abs(vllm_scale - sglang_scale).mean().item()
    output_diff = torch.abs(vllm_out.float() - sglang_out.float()).mean().item()

    if torch.allclose(
        vllm_out.to(torch.float32), sglang_out.to(torch.float32), rtol=1e-3, atol=1e-5
    ) and torch.allclose(vllm_scale, sglang_scale, rtol=1e-3, atol=1e-5):
        print("✅ All implementations match")
    else:
        print("❌ Implementations differ")


batch_size_range = [16, 32, 64, 128]
seq_len_range = [64, 128, 256, 512, 1024, 2048, 4096]

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
        plot_name="per-token-dynamic-quant-fp8-performance",
        args={},
    )
)
def benchmark_quantization(batch_size, seq_len, provider):
    dtype = torch.float16
    device = torch.device("cuda")

    x = torch.randn(batch_size * seq_len, 4096, device=device, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "vllm":
        fn = lambda: vllm_per_token_quant_fp8(x.clone())
    elif provider == "sglang":
        fn = lambda: sglang_per_token_quant_fp8(x.clone())

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    calculate_diff(batch_size=4, seq_len=4096)
    benchmark_quantization.run(print_data=True)
