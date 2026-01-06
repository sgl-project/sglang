# Benchmarks SGLang fused layernorm/rmsnorm scale shift kernels versus PyTorch naive implementations
# Tests three kernel variants:
# 1. fused_norm_scale_shift - with affine parameters (gamma/beta)
# 2. fused_norm_scale_shift_no_affine - without affine parameters
# 3. fused_scale_residual_norm_scale_shift - with residual and gate
import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import torch

from sglang.multimodal_gen.runtime.layers.layernorm import (
    _NormScaleShift,
    _ScaleResidualNormScaleShift,
)

# CI environment detection
IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)


def bench_fused_norm_scale_shift(
    B: int,
    L: int,
    C: int,
    norm_type: str,
    num_warmup: int,
    num_run: int,
) -> Tuple[float, float]:
    device = "cuda"
    dtype = torch.bfloat16
    eps = 1e-5

    x = torch.randn(B, L, C, dtype=dtype, device=device)
    weight = torch.randn(C, dtype=dtype, device=device)
    bias = torch.randn(C, dtype=dtype, device=device)
    scale = torch.randn(B, L, C, dtype=dtype, device=device)
    shift = torch.randn(B, L, C, dtype=dtype, device=device)
    layer = _NormScaleShift(C, norm_type, eps=eps, elementwise_affine=True, dtype=dtype)
    with torch.no_grad():
        layer.norm.weight.copy_(weight)
        if norm_type == "layer":
            layer.norm.bias.copy_(bias)

    # warmup
    for _ in range(num_warmup):
        layer.forward_native(x, shift, scale)
    torch.cuda.synchronize()

    # run naive
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(num_run):
        layer.forward_native(x, shift, scale)
    end_event.record()
    end_event.synchronize()
    torch.cuda.synchronize()
    naive_time = start_event.elapsed_time(end_event) / num_run * 1000  # us

    # warmup
    for _ in range(num_warmup):
        layer.forward_cuda(x, shift, scale)
    torch.cuda.synchronize()

    # run sglang
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(num_run):
        layer.forward_cuda(x, shift, scale)
    end_event.record()
    end_event.synchronize()
    torch.cuda.synchronize()
    sglang_time = start_event.elapsed_time(end_event) / num_run * 1000  # us

    return naive_time, sglang_time


def bench_fused_norm_scale_shift_no_affine(
    B: int,
    L: int,
    C: int,
    norm_type: str,
    num_warmup: int,
    num_run: int,
) -> Tuple[float, float]:
    device = "cuda"
    dtype = torch.bfloat16
    eps = 1e-5

    x = torch.randn(B, L, C, dtype=dtype, device=device)
    scale = torch.randn(B, L, C, dtype=dtype, device=device)
    shift = torch.randn(B, L, C, dtype=dtype, device=device)

    layer = _NormScaleShift(
        C, norm_type, eps=eps, elementwise_affine=False, dtype=dtype
    )

    # warmup
    for _ in range(num_warmup):
        layer.forward_native(x, shift, scale)
    torch.cuda.synchronize()

    # run naive
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(num_run):
        layer.forward_native(x, shift, scale)
    end_event.record()
    end_event.synchronize()
    torch.cuda.synchronize()
    naive_time = start_event.elapsed_time(end_event) / num_run * 1000  # us

    # warmup
    for _ in range(num_warmup):
        layer.forward_cuda(x, shift, scale)
    torch.cuda.synchronize()

    # run sglang
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(num_run):
        layer.forward_cuda(x, shift, scale)
    end_event.record()
    end_event.synchronize()
    torch.cuda.synchronize()
    sglang_time = start_event.elapsed_time(end_event) / num_run * 1000  # us

    return naive_time, sglang_time


def bench_fused_scale_residual_norm_scale_shift(
    B: int,
    L: int,
    C: int,
    norm_type: str,
    num_warmup: int,
    num_run: int,
) -> Tuple[float, float]:
    device = "cuda"
    dtype = torch.bfloat16
    eps = 1e-5

    residual = torch.randn(B, L, C, dtype=dtype, device=device)
    x = torch.randn(B, L, C, dtype=dtype, device=device)
    weight = torch.randn(C, dtype=dtype, device=device)
    bias = torch.randn(C, dtype=dtype, device=device)
    scale = torch.randn(B, L, C, dtype=dtype, device=device)
    shift = torch.randn(B, L, C, dtype=dtype, device=device)
    gate = torch.randn(B, 1, C, dtype=dtype, device=device)
    layer = _ScaleResidualNormScaleShift(
        C, norm_type, eps=eps, elementwise_affine=True, dtype=dtype
    )
    with torch.no_grad():
        layer.norm.weight.copy_(weight)
        if norm_type == "layer":
            layer.norm.bias.copy_(bias)
    # warmup
    for _ in range(num_warmup):
        layer.forward_native(residual, x, gate, shift, scale)
    torch.cuda.synchronize()

    # run naive
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(num_run):
        layer.forward_native(residual, x, gate, shift, scale)
    end_event.record()
    end_event.synchronize()
    torch.cuda.synchronize()
    naive_time = start_event.elapsed_time(end_event) / num_run * 1000  # us

    # warmup
    for _ in range(num_warmup):
        layer.forward_cuda(residual, x, gate, shift, scale)
    torch.cuda.synchronize()

    # run sglang
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(num_run):
        layer.forward_cuda(residual, x, gate, shift, scale)
    end_event.record()
    end_event.synchronize()
    torch.cuda.synchronize()
    sglang_time = start_event.elapsed_time(end_event) / num_run * 1000  # us

    return naive_time, sglang_time


benchmark_kernels = {
    "fused_norm_scale_shift": bench_fused_norm_scale_shift,
    "fused_norm_scale_shift_no_affine": bench_fused_norm_scale_shift_no_affine,
    "fused_scale_residual_norm_scale_shift": bench_fused_scale_residual_norm_scale_shift,
}


@dataclass
class ShapeArg:
    B: int
    L: int
    C: int


def benchmark_one_shape(
    kernel_name: str,
    shape_args: List[ShapeArg],
    norm_type_args: List[str],
    num_warmup: int,
    num_run: int,
):
    print(f"\n{'='*80}")
    print(f"Kernel: {kernel_name}")
    print(f"{'='*80}")

    kernel_func = benchmark_kernels[kernel_name]

    for norm_type in norm_type_args:
        for shape in shape_args:
            naive_time, sglang_time = kernel_func(
                shape.B,
                shape.L,
                shape.C,
                norm_type,
                num_warmup,
                num_run,
            )
            speedup = naive_time / sglang_time if sglang_time > 0 else 0.0
            print(
                f"B={shape.B:1d}, L={shape.L:4d}, C={shape.C:4d} norm={norm_type} | "
                f"Naive: {naive_time:8.2f} us | "
                f"SGLang: {sglang_time:8.2f} us | "
                f"Speedup: {speedup:5.2f}x"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-warmup", type=int, default=3)
    parser.add_argument("--num-run", type=int, default=10)

    # CI environment uses simplified parameters
    if IS_CI:
        shape_args = [
            # Only test one simple shape in CI
            ShapeArg(B=1, L=128, C=1024),
        ]
    else:
        shape_args = [
            # Small shapes
            ShapeArg(B=1, L=128, C=1024),
            ShapeArg(B=1, L=128, C=3072),
            ShapeArg(B=1, L=128, C=4096),
            # Medium shapes
            ShapeArg(B=1, L=1024, C=1024),
            ShapeArg(B=1, L=1024, C=3072),
            ShapeArg(B=1, L=1024, C=4096),
            # Large shapes
            ShapeArg(B=1, L=4096, C=1024),
            ShapeArg(B=1, L=4096, C=3072),
            ShapeArg(B=1, L=4096, C=4096),
        ]

    norm_type_args = ["layer", "rms"]

    args = parser.parse_args()

    for kernel_name in benchmark_kernels.keys():
        benchmark_one_shape(
            kernel_name, shape_args, norm_type_args, args.num_warmup, args.num_run
        )


if __name__ == "__main__":
    main()
