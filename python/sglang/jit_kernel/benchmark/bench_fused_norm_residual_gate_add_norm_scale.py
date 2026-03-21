import itertools
from typing import Tuple

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import is_in_ci
from sglang.multimodal_gen.runtime.layers.layernorm import (
    AddGateLayerNorm,
    AddGateRMSNorm,
    LayerNormResidualGateAddNormScale,
    RMSNormResidualGateAddNormScale,
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
        weight1 = torch.randn(D, dtype=DTYPE, device=DEVICE)
        bias1 = torch.randn(D, dtype=DTYPE, device=DEVICE)
        weight2 = torch.randn(D, dtype=DTYPE, device=DEVICE)
        bias2 = torch.randn(D, dtype=DTYPE, device=DEVICE)
        with torch.no_grad():
            layer.norm1.weight.copy_(weight1)
            if hasattr(layer.norm1, "bias"):
                layer.norm1.bias.copy_(bias1)
            layer.norm2.weight.copy_(weight2)
            if hasattr(layer.norm2, "bias"):
                layer.norm2.bias.copy_(bias2)
    layer.requires_grad_(False)
    return layer.to(DEVICE)


def preprocess_add_gate_layer(layer, affine: bool, D: int, DTYPE: torch.dtype):
    if affine:
        weight = torch.randn(D, dtype=DTYPE, device=DEVICE)
        bias = torch.randn(D, dtype=DTYPE, device=DEVICE)
        with torch.no_grad():
            layer.norm.weight.copy_(weight)
            if hasattr(layer.norm, "bias"):
                layer.norm.bias.copy_(bias)
    layer.requires_grad_(False)
    return layer.to(DEVICE)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["B", "S", "D", "norm_type", "affine"],
        x_vals=config,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="fused_norm_residual_gate_add_norm_scale",
        args={},
    )
)
def bench_fused_norm_residual_gate_add_norm_scale(
    B: int, S: int, D: int, norm_type, affine: bool, provider: str
) -> Tuple[float, float, float]:
    residual = torch.randn(B, S, D, dtype=DTYPE, device=DEVICE)
    x = torch.randn(B, S, D, dtype=DTYPE, device=DEVICE)
    gate = torch.randn(B, 1, D, dtype=DTYPE, device=DEVICE)
    scale = torch.randn(B, S, D, dtype=DTYPE, device=DEVICE)
    if norm_type == "layer":
        layer = LayerNormResidualGateAddNormScale(D, EPS, affine, dtype=DTYPE)
    else:
        layer = RMSNormResidualGateAddNormScale(D, EPS, affine, dtype=DTYPE)
    layer = preprocess_layer(layer, affine, D, DTYPE)
    if provider == "native":
        fn = lambda: layer.forward_native(residual, x, gate, scale)
    else:
        fn = lambda: layer.forward_cuda(residual, x, gate, scale)

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["B", "S", "D", "norm_type", "affine"],
        x_vals=config,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="fused_add_gate_norm",
        args={},
    )
)
def bench_fused_add_gate_norm(
    B: int, S: int, D: int, norm_type, affine: bool, provider: str
) -> Tuple[float, float, float]:
    residual = torch.randn(B, S, D, dtype=DTYPE, device=DEVICE)
    x = torch.randn(B, S, D, dtype=DTYPE, device=DEVICE)
    gate = torch.randn(B, 1, D, dtype=DTYPE, device=DEVICE)
    if norm_type == "layer":
        layer = AddGateLayerNorm(D, EPS, affine, dtype=DTYPE)
    else:
        layer = AddGateRMSNorm(D, EPS, affine, dtype=DTYPE)
    layer = preprocess_add_gate_layer(layer, affine, D, DTYPE)
    if provider == "native":
        fn = lambda: layer.forward_native(residual, x, gate)
    else:
        fn = lambda: layer.forward_cuda(residual, x, gate)

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    bench_fused_norm_residual_gate_add_norm_scale.run(print_data=True)
    bench_fused_add_gate_norm.run(print_data=True)
