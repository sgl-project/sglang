# Benchmarks SGLang fused layernorm/rmsnorm scale shift kernels
# 1. fused_norm_scale_shift
# 2. fused_scale_residual_norm_scale_shift
import itertools
from typing import Tuple

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import is_in_ci
from sglang.multimodal_gen.runtime.layers.layernorm import (
    _NormScaleShift,
    _ScaleResidualNormScaleShift,
)

if is_in_ci():
    B_RANGE, S_RANGE, D_RANGE = [1], [128], [1024]
else:
    B_RANGE, S_RANGE, D_RANGE = [1], [128, 1024, 4096], [1024, 3072, 4096]

NORM_TYPE_RANGE = ["layer", "rms"]
AFFINE_RANGE = [True, False]
DTYPE = torch.bfloat16
DEVICE = "cuda"
EPS = 1e-5
LINE_VALS = ["native", "cuda"]
LINE_NAMES = ["SGLang Native", "SGLang Fused"]
STYLES = [("red", "-"), ("blue", "--")]
configs = list(
    itertools.product(B_RANGE, S_RANGE, D_RANGE, NORM_TYPE_RANGE, AFFINE_RANGE)
)


def create_layer(D: int, norm_type: str, elementwise_affine: bool, Layer):
    """Create layer with or without affine parameters."""
    layer = Layer(
        D, norm_type, eps=EPS, elementwise_affine=elementwise_affine, dtype=DTYPE
    )
    if elementwise_affine:
        weight = torch.randn(D, dtype=DTYPE, device=DEVICE)
        bias = torch.randn(D, dtype=DTYPE, device=DEVICE)
        with torch.no_grad():
            layer.norm.weight.copy_(weight)
            if norm_type == "layer":
                layer.norm.bias.copy_(bias)
    return layer


# ============================================================================
# Benchmark 1: fused_norm_scale_shift
# ============================================================================
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["B", "S", "D", "norm_type", "affine"],
        x_vals=configs,
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
    B: int, S: int, D: int, norm_type: str, affine: bool, provider: str
) -> Tuple[float, float, float]:
    x = torch.randn(B, S, D, dtype=DTYPE, device=DEVICE)
    scale = torch.randn(B, S, D, dtype=DTYPE, device=DEVICE)
    shift = torch.randn(B, S, D, dtype=DTYPE, device=DEVICE)
    layer = create_layer(D, norm_type, affine, _NormScaleShift)

    if provider == "native":
        fn = lambda: layer.forward_native(x, shift, scale)
    else:
        fn = lambda: layer.forward_cuda(x, shift, scale)

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms  # convert to us


# ============================================================================
# Benchmark 2: fused_scale_residual_norm_scale_shift
# ============================================================================
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["B", "S", "D", "norm_type", "affine"],
        x_vals=configs,
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
    B: int, S: int, D: int, norm_type: str, affine: bool, provider: str
) -> Tuple[float, float, float]:
    residual = torch.randn(B, S, D, dtype=DTYPE, device=DEVICE)
    x = torch.randn(B, S, D, dtype=DTYPE, device=DEVICE)
    scale = torch.randn(B, S, D, dtype=DTYPE, device=DEVICE)
    shift = torch.randn(B, S, D, dtype=DTYPE, device=DEVICE)
    gate = torch.randn(B, 1, D, dtype=DTYPE, device=DEVICE)
    layer = create_layer(D, norm_type, affine, _ScaleResidualNormScaleShift)

    if provider == "native":
        fn = lambda: layer.forward_native(residual, x, gate, shift, scale)
    else:
        fn = lambda: layer.forward_cuda(residual, x, gate, shift, scale)

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms  # convert to us


if __name__ == "__main__":
    print(f"\n{'='*80}")
    print("Benchmark: fused_norm_scale_shift")
    print(f"{'='*80}\n")
    bench_fused_norm_scale_shift.run(print_data=True)

    print(f"\n{'='*80}")
    print("Benchmark: fused_scale_residual_norm_scale_shift")
    print(f"{'='*80}\n")
    bench_fused_scale_residual_norm_scale_shift.run(print_data=True)
