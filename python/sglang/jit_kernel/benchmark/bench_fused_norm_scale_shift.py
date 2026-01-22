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
    LayerNormScaleShift,
    RMSNormScaleShift,
    ScaleResidualLayerNormScaleShift,
    ScaleResidualRMSNormScaleShift,
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
config = list(
    itertools.product(B_RANGE, S_RANGE, D_RANGE, NORM_TYPE_RANGE, AFFINE_RANGE)
)


def preprocess_layer(layer, affine: bool, D: int, DTYPE: torch.dtype):
    if affine:
        weight = torch.randn(D, dtype=DTYPE, device=DEVICE)
        bias = torch.randn(D, dtype=DTYPE, device=DEVICE)
        with torch.no_grad():
            layer.norm.weight.copy_(weight)
            if hasattr(layer.norm, "bias"):
                layer.norm.bias.copy_(bias)
    layer.requires_grad_(False)
    return layer.to(DEVICE)


# ============================================================================
# Benchmark 1: fused_norm_scale_shift
# ============================================================================
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["B", "S", "D", "norm_type", "affine"],
        x_vals=config,
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
    B: int, S: int, D: int, norm_type, affine: bool, provider: str
) -> Tuple[float, float, float]:
    x = torch.randn(B, S, D, dtype=DTYPE, device=DEVICE)
    scale = torch.randn(B, S, D, dtype=DTYPE, device=DEVICE)
    shift = torch.randn(B, S, D, dtype=DTYPE, device=DEVICE)
    if norm_type == "layer":
        layer = LayerNormScaleShift(D, EPS, affine, dtype=DTYPE)
    else:
        layer = RMSNormScaleShift(D, EPS, affine, dtype=DTYPE)
    layer = preprocess_layer(layer, affine, D, DTYPE)
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
        x_vals=config,
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
    B: int, S: int, D: int, norm_type, affine: bool, provider: str
) -> Tuple[float, float, float]:
    residual = torch.randn(B, S, D, dtype=DTYPE, device=DEVICE)
    x = torch.randn(B, S, D, dtype=DTYPE, device=DEVICE)
    scale = torch.randn(B, S, D, dtype=DTYPE, device=DEVICE)
    shift = torch.randn(B, S, D, dtype=DTYPE, device=DEVICE)
    gate = torch.randn(B, 1, D, dtype=DTYPE, device=DEVICE)
    if norm_type == "layer":
        layer = ScaleResidualLayerNormScaleShift(D, EPS, affine, dtype=DTYPE).to(DEVICE)
    else:
        layer = ScaleResidualRMSNormScaleShift(D, EPS, affine, dtype=DTYPE).to(DEVICE)
    layer = preprocess_layer(layer, affine, D, DTYPE)
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
