# Benchmarks SGLang fused layernorm/rmsnorm scale shift kernels
# Tests three kernel variants:
# 1. fused_norm_scale_shift - with affine parameters (gamma/beta)
# 2. fused_norm_scale_shift_no_affine - without affine parameters
# 3. fused_scale_residual_norm_scale_shift - with residual and gate
import itertools
import os
from typing import Tuple

import torch
import triton
import triton.testing

from sglang.multimodal_gen.runtime.layers.layernorm import (
    _NormScaleShift,
    _ScaleResidualNormScaleShift,
)

IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)

DTYPE = torch.bfloat16
DEVICE = "cuda"
EPS = 1e-5

if IS_CI:
    B_RANGE = [1]
    S_RANGE = [128]
    D_RANGE = [1024]
else:
    B_RANGE = [1]
    S_RANGE = [128, 1024, 4096]
    D_RANGE = [1024, 3072, 4096]

NORM_TYPE_RANGE = ["layer", "rms"]


def create_norm_scale_shift_layer(D: int, norm_type: str):
    """Create layer with affine parameters."""
    layer = _NormScaleShift(D, norm_type, eps=EPS, elementwise_affine=True, dtype=DTYPE)
    weight = torch.randn(D, dtype=DTYPE, device=DEVICE)
    bias = torch.randn(D, dtype=DTYPE, device=DEVICE)
    with torch.no_grad():
        layer.norm.weight.copy_(weight)
        if norm_type == "layer":
            layer.norm.bias.copy_(bias)
    return layer


def create_norm_scale_shift_no_affine_layer(D: int, norm_type: str):
    """Create layer without affine parameters."""
    layer = _NormScaleShift(
        D, norm_type, eps=EPS, elementwise_affine=False, dtype=DTYPE
    )
    return layer


def create_scale_residual_norm_scale_shift_layer(D: int, norm_type: str):
    """Create layer with residual, gate, and affine parameters."""
    layer = _ScaleResidualNormScaleShift(
        D, norm_type, eps=EPS, elementwise_affine=True, dtype=DTYPE
    )
    weight = torch.randn(D, dtype=DTYPE, device=DEVICE)
    bias = torch.randn(D, dtype=DTYPE, device=DEVICE)
    with torch.no_grad():
        layer.norm.weight.copy_(weight)
        if norm_type == "layer":
            layer.norm.bias.copy_(bias)
    return layer


# ============================================================================
# Benchmark 1: fused_norm_scale_shift (with affine)
# ============================================================================
LINE_VALS = ["native", "cuda"]
LINE_NAMES = ["SGLang Native", "SGLang Fused"]
STYLES = [("red", "-"), ("blue", "--")]

configs_norm_scale_shift = list(
    itertools.product(B_RANGE, S_RANGE, D_RANGE, NORM_TYPE_RANGE)
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["B", "S", "D", "norm_type"],
        x_vals=configs_norm_scale_shift,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="fused_norm_scale_shift",
        args={},
    )
)
def bench_fused_norm_scale_shift(
    B: int, S: int, D: int, norm_type: str, provider: str
) -> Tuple[float, float, float]:
    x = torch.randn(B, S, D, dtype=DTYPE, device=DEVICE)
    scale = torch.randn(B, S, D, dtype=DTYPE, device=DEVICE)
    shift = torch.randn(B, S, D, dtype=DTYPE, device=DEVICE)
    layer = create_norm_scale_shift_layer(D, norm_type)

    if provider == "native":
        fn = lambda: layer.forward_native(x, shift, scale)
    else:
        fn = lambda: layer.forward_cuda(x, shift, scale)

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms  # convert to us


# ============================================================================
# Benchmark 2: fused_norm_scale_shift_no_affine (without affine)
# ============================================================================
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["B", "S", "D", "norm_type"],
        x_vals=configs_norm_scale_shift,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="fused_norm_scale_shift_no_affine",
        args={},
    )
)
def bench_fused_norm_scale_shift_no_affine(
    B: int, S: int, D: int, norm_type: str, provider: str
) -> Tuple[float, float, float]:
    x = torch.randn(B, S, D, dtype=DTYPE, device=DEVICE)
    scale = torch.randn(B, S, D, dtype=DTYPE, device=DEVICE)
    shift = torch.randn(B, S, D, dtype=DTYPE, device=DEVICE)
    layer = create_norm_scale_shift_no_affine_layer(D, norm_type)

    if provider == "native":
        fn = lambda: layer.forward_native(x, shift, scale)
    else:
        fn = lambda: layer.forward_cuda(x, shift, scale)

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms  # convert to us


# ============================================================================
# Benchmark 3: fused_scale_residual_norm_scale_shift (with residual and gate)
# ============================================================================
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["B", "S", "D", "norm_type"],
        x_vals=configs_norm_scale_shift,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="fused_scale_residual_norm_scale_shift",
        args={},
    )
)
def bench_fused_scale_residual_norm_scale_shift(
    B: int, S: int, D: int, norm_type: str, provider: str
) -> Tuple[float, float, float]:
    residual = torch.randn(B, S, D, dtype=DTYPE, device=DEVICE)
    x = torch.randn(B, S, D, dtype=DTYPE, device=DEVICE)
    scale = torch.randn(B, S, D, dtype=DTYPE, device=DEVICE)
    shift = torch.randn(B, S, D, dtype=DTYPE, device=DEVICE)
    gate = torch.randn(B, 1, D, dtype=DTYPE, device=DEVICE)
    layer = create_scale_residual_norm_scale_shift_layer(D, norm_type)

    if provider == "native":
        fn = lambda: layer.forward_native(residual, x, gate, shift, scale)
    else:
        fn = lambda: layer.forward_cuda(residual, x, gate, shift, scale)

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms  # convert to us


if __name__ == "__main__":
    print(f"\n{'='*80}")
    print("Benchmark: fused_norm_scale_shift (with affine)")
    print(f"{'='*80}\n")
    bench_fused_norm_scale_shift.run(print_data=True)

    print(f"\n{'='*80}")
    print("Benchmark: fused_norm_scale_shift_no_affine (without affine)")
    print(f"{'='*80}\n")
    bench_fused_norm_scale_shift_no_affine.run(print_data=True)

    print(f"\n{'='*80}")
    print("Benchmark: fused_scale_residual_norm_scale_shift (with residual and gate)")
    print(f"{'='*80}\n")
    bench_fused_scale_residual_norm_scale_shift.run(print_data=True)
