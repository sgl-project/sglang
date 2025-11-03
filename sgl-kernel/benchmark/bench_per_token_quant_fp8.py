import itertools
import os
from typing import Optional, Tuple

import torch
import triton
import triton.testing
from sgl_kernel import sgl_per_token_quant_fp8

# Optional vLLM import
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

# Get correct FP8 E4M3 maximum value
if _is_hip:
    FP8_E4M3_MAX = 224.0  # ROCM uses 224.0
else:
    # For CUDA, get the actual max value from the type
    FP8_E4M3_MAX = float(torch.finfo(fp8_type_).max)


def torch_per_token_quant_fp8(
    input: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch reference implementation for per-token FP8 quantization."""
    device = input.device
    dtype = input.dtype

    # Find max absolute value per token (row) - exactly like CUDA kernel
    max_vals = torch.abs(input).max(dim=1)[0]  # [num_tokens]

    # Calculate scale per token - exactly like CUDA kernel: scale = max_value / FP8_E4M3_MAX
    scales = max_vals / FP8_E4M3_MAX  # [num_tokens]

    # No special zero handling - directly compute 1.0 / scale like CUDA kernel
    scale_inv = 1.0 / scales  # [num_tokens]

    # Quantize: input * scale_inv, then clamp to FP8 range
    quantized_float = input * scale_inv.unsqueeze(1)  # Broadcast scale_inv
    quantized_float = torch.clamp(quantized_float, -FP8_E4M3_MAX, FP8_E4M3_MAX)

    # Convert to FP8 - use more explicit conversion
    quantized_fp8 = quantized_float.to(fp8_type_)

    return quantized_fp8, scales


def vllm_per_token_quant_fp8(
    input: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not VLLM_AVAILABLE:
        # Fallback to SGLang implementation
        return sglang_per_token_quant_fp8(input)
    return ops.scaled_fp8_quant(input, use_per_token_if_dynamic=True)


def sglang_per_token_quant_fp8(
    input: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    scale = torch.zeros(input.size(0), device=input.device, dtype=torch.float32)
    output = torch.empty_like(input, device=input.device, dtype=fp8_type_)
    sgl_per_token_quant_fp8(input, output, scale)

    return output, scale


def calculate_diff(batch_size: int, seq_len: int, hidden_dim: int):
    """Compare Torch reference, VLLM, and SGLang implementations."""
    device = torch.device("cuda")
    x = torch.rand(
        (batch_size * seq_len, hidden_dim), dtype=torch.float16, device=device
    )

    # Get all three implementations
    torch_out, torch_scale = torch_per_token_quant_fp8(x)
    vllm_out, vllm_scale = vllm_per_token_quant_fp8(x)
    sglang_out, sglang_scale = sglang_per_token_quant_fp8(x)

    if not VLLM_AVAILABLE:
        print("⚠️ vLLM not available, skipping vLLM comparison")
        # Only compare Torch vs SGLang
        torch_sglang_scale_diff = torch.abs(torch_scale - sglang_scale).mean().item()
        torch_sglang_out_diff = (
            torch.abs(torch_out.float() - sglang_out.float()).mean().item()
        )
        print(f"Scale difference (Torch vs SGLang): {torch_sglang_scale_diff:.8f}")
        print(f"Output difference (Torch vs SGLang): {torch_sglang_out_diff:.8f}")
        return

    print(f"\n=== Comparison for hidden_dim={hidden_dim} ===")

    # Compare scales
    torch_vllm_scale_diff = torch.abs(torch_scale - vllm_scale).mean().item()
    torch_sglang_scale_diff = torch.abs(torch_scale - sglang_scale).mean().item()
    vllm_sglang_scale_diff = torch.abs(vllm_scale - sglang_scale).mean().item()

    print(f"Scale differences:")
    print(f"  Torch vs VLLM:   {torch_vllm_scale_diff:.8f}")
    print(f"  Torch vs SGLang: {torch_sglang_scale_diff:.8f}")
    print(f"  VLLM vs SGLang:  {vllm_sglang_scale_diff:.8f}")

    # Compare outputs
    torch_vllm_out_diff = torch.abs(torch_out.float() - vllm_out.float()).mean().item()
    torch_sglang_out_diff = (
        torch.abs(torch_out.float() - sglang_out.float()).mean().item()
    )
    vllm_sglang_out_diff = (
        torch.abs(vllm_out.float() - sglang_out.float()).mean().item()
    )

    print(f"Output differences:")
    print(f"  Torch vs VLLM:   {torch_vllm_out_diff:.8f}")
    print(f"  Torch vs SGLang: {torch_sglang_out_diff:.8f}")
    print(f"  VLLM vs SGLang:  {vllm_sglang_out_diff:.8f}")

    # Check tolerances
    rtol, atol = 1e-3, 1e-5

    torch_vllm_match = torch.allclose(
        torch_out.float(), vllm_out.float(), rtol=rtol, atol=atol
    ) and torch.allclose(torch_scale, vllm_scale, rtol=rtol, atol=atol)
    torch_sglang_match = torch.allclose(
        torch_out.float(), sglang_out.float(), rtol=rtol, atol=atol
    ) and torch.allclose(torch_scale, sglang_scale, rtol=rtol, atol=atol)

    if hidden_dim == 1368:
        rtol = 1e-2
        # we found vllm sglang has diff when hidden dim is not dividable by 16
        # and we believe SGLang is closer to Torch implementation

    vllm_sglang_match = torch.allclose(
        vllm_out.float(), sglang_out.float(), rtol=rtol, atol=atol
    ) and torch.allclose(vllm_scale, sglang_scale, rtol=rtol, atol=atol)

    print(f"Matches (rtol={rtol}, atol={atol}):")
    print(f"  Torch vs VLLM:   {'✅' if torch_vllm_match else '❌'}")
    print(f"  Torch vs SGLang: {'✅' if torch_sglang_match else '❌'}")
    print(f"  VLLM vs SGLang:  {'✅' if vllm_sglang_match else '❌'}")


# CI environment uses simplified parameters
if IS_CI:
    batch_size_range = [16]  # Single batch size for CI
    seq_len_range = [64]  # Single sequence length for CI
    hidden_dim_range = [2048]  # Single hidden dimension for CI
else:
    batch_size_range = [16, 32, 64, 128]
    seq_len_range = [64, 128, 256, 512, 1024, 2048, 4096]
    hidden_dim_range = [1368, 2048, 4096]

configs = list(itertools.product(batch_size_range, seq_len_range, hidden_dim_range))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "seq_len", "hidden_dim"],
        x_vals=configs,
        line_arg="provider",
        line_vals=(
            ["torch", "vllm", "sglang"] if VLLM_AVAILABLE else ["torch", "sglang"]
        ),
        line_names=(
            ["Torch Reference", "VLLM", "SGL Kernel"]
            if VLLM_AVAILABLE
            else ["Torch Reference", "SGL Kernel"]
        ),
        styles=(
            [("red", "-"), ("blue", "-"), ("green", "-")]
            if VLLM_AVAILABLE
            else [("red", "-"), ("green", "-")]
        ),
        ylabel="us",
        plot_name="per-token-dynamic-quant-fp8-performance",
        args={},
    )
)
def benchmark_quantization(batch_size, seq_len, hidden_dim, provider):
    dtype = torch.float16
    device = torch.device("cuda")

    x = torch.randn(batch_size * seq_len, hidden_dim, device=device, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        fn = lambda: torch_per_token_quant_fp8(x.clone())
    elif provider == "vllm":
        if not VLLM_AVAILABLE:
            return (0, 0, 0)
        fn = lambda: vllm_per_token_quant_fp8(x.clone())
    elif provider == "sglang":
        fn = lambda: sglang_per_token_quant_fp8(x.clone())

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, quantiles=quantiles)

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    # Test various hidden dimensions for correctness - simplified for CI
    if IS_CI:
        test_dims = [2048]  # Single dimension for CI
        batch_size, seq_len = 4, 64  # Smaller values for CI
    else:
        test_dims = [1368, 2048, 4096]
        batch_size, seq_len = 4, 4096

    for dim in test_dims:
        calculate_diff(batch_size=batch_size, seq_len=seq_len, hidden_dim=dim)

    print("\n" + "=" * 60)
    print("Starting performance benchmark...")
    benchmark_quantization.run(print_data=True)
