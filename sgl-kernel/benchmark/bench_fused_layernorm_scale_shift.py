# Benchmarks SGLang fused layernorm scale shift kernels versus PyTorch naive implementations
# Tests three kernel variants:
# 1. fused_layernorm_scale_shift - with affine parameters (gamma/beta)
# 2. fused_layernorm_scale_shift_no_affine - without affine parameters
# 3. fused_scale_residual_layernorm_scale_shift - with residual and gate
import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import sgl_kernel
import torch

# CI environment detection
IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)


# ========== fused_layernorm_scale_shift ==========
def fused_layernorm_scale_shift_naive(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float = 1e-5,
):
    dtype = x.dtype
    x32 = x.float()
    w32 = weight.float()
    b32 = bias.float()
    s32 = scale.float()
    sh32 = shift.float()

    mean = x32.mean(dim=1, keepdim=True)
    var = (x32 - mean).pow(2).mean(dim=1, keepdim=True)
    inv_std = (var + eps).sqrt().reciprocal()

    y_ln32 = (x32 - mean) * inv_std
    y_ln32 = y_ln32 * w32 + b32
    y_out = (y_ln32 * (1.0 + s32) + sh32).to(dtype)
    return y_out


def fused_layernorm_scale_shift_sglang(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float = 1e-5,
):
    return sgl_kernel.fused_layernorm_scale_shift(x, weight, bias, scale, shift)


# ========== fused_layernorm_scale_shift_no_affine ==========
def fused_layernorm_scale_shift_no_affine_naive(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float = 1e-5,
):
    dtype = x.dtype
    x32 = x.float()
    s32 = scale.float()
    sh32 = shift.float()

    mean = x32.mean(dim=1, keepdim=True)
    var = (x32 - mean).pow(2).mean(dim=1, keepdim=True)
    inv_std = (var + eps).sqrt().reciprocal()

    y_ln32 = (x32 - mean) * inv_std
    y_out = (y_ln32 * (1.0 + s32) + sh32).to(dtype)
    return y_out


def fused_layernorm_scale_shift_no_affine_sglang(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float = 1e-5,
):
    return sgl_kernel.fused_layernorm_scale_shift_no_affine(x, scale, shift)


# ========== fused_scale_residual_layernorm_scale_shift ==========
def fused_scale_residual_layernorm_scale_shift_naive(
    residual: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    gate: torch.Tensor,
    eps: float = 1e-5,
):
    dtype = x.dtype
    # Residual connection with gate
    res32 = residual.float() + x.float() * gate.float()
    residual_out = res32.to(dtype)

    # LayerNorm
    w32 = weight.float()
    b32 = bias.float()
    s32 = scale.float()
    sh32 = shift.float()

    mean = res32.mean(dim=1, keepdim=True)
    var = (res32 - mean).pow(2).mean(dim=1, keepdim=True)
    inv_std = (var + eps).sqrt().reciprocal()

    y_ln32 = (res32 - mean) * inv_std
    y_ln32 = y_ln32 * w32 + b32
    y_out = (y_ln32 * (1.0 + s32) + sh32).to(dtype)
    return y_out, residual_out


def fused_scale_residual_layernorm_scale_shift_sglang(
    residual: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    gate: torch.Tensor,
    eps: float = 1e-5,
):
    return sgl_kernel.fused_scale_residual_layernorm_scale_shift(
        residual, x, weight, bias, scale, shift, gate
    )


def bench_fused_layernorm_scale_shift(
    M: int,
    N: int,
    num_warmup: int,
    num_run: int,
) -> Tuple[float, float]:
    device = "cuda"
    dtype = torch.bfloat16
    eps = 1e-5

    x = torch.randn(M, N, dtype=dtype, device=device)
    weight = torch.randn(N, dtype=dtype, device=device)
    bias = torch.randn(N, dtype=dtype, device=device)
    scale = torch.randn(M, N, dtype=dtype, device=device)
    shift = torch.randn(M, N, dtype=dtype, device=device)

    def run_naive():
        fused_layernorm_scale_shift_naive(x, weight, bias, scale, shift, eps)

    def run_sglang():
        fused_layernorm_scale_shift_sglang(x, weight, bias, scale, shift, eps)

    # warmup
    for _ in range(num_warmup):
        run_naive()
    torch.cuda.synchronize()

    # run naive
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(num_run):
        run_naive()
    end_event.record()
    end_event.synchronize()
    torch.cuda.synchronize()
    naive_time = start_event.elapsed_time(end_event) / num_run * 1000  # us

    # warmup
    for _ in range(num_warmup):
        run_sglang()
    torch.cuda.synchronize()

    # run sglang
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(num_run):
        run_sglang()
    end_event.record()
    end_event.synchronize()
    torch.cuda.synchronize()
    sglang_time = start_event.elapsed_time(end_event) / num_run * 1000  # us

    return naive_time, sglang_time


def bench_fused_layernorm_scale_shift_no_affine(
    M: int,
    N: int,
    num_warmup: int,
    num_run: int,
) -> Tuple[float, float]:
    device = "cuda"
    dtype = torch.bfloat16
    eps = 1e-5

    x = torch.randn(M, N, dtype=dtype, device=device)
    scale = torch.randn(M, N, dtype=dtype, device=device)
    shift = torch.randn(M, N, dtype=dtype, device=device)

    def run_naive():
        fused_layernorm_scale_shift_no_affine_naive(x, scale, shift, eps)

    def run_sglang():
        fused_layernorm_scale_shift_no_affine_sglang(x, scale, shift, eps)

    # warmup
    for _ in range(num_warmup):
        run_naive()
    torch.cuda.synchronize()

    # run naive
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(num_run):
        run_naive()
    end_event.record()
    end_event.synchronize()
    torch.cuda.synchronize()
    naive_time = start_event.elapsed_time(end_event) / num_run * 1000  # us

    # warmup
    for _ in range(num_warmup):
        run_sglang()
    torch.cuda.synchronize()

    # run sglang
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(num_run):
        run_sglang()
    end_event.record()
    end_event.synchronize()
    torch.cuda.synchronize()
    sglang_time = start_event.elapsed_time(end_event) / num_run * 1000  # us

    return naive_time, sglang_time


def bench_fused_scale_residual_layernorm_scale_shift(
    M: int,
    N: int,
    num_warmup: int,
    num_run: int,
) -> Tuple[float, float]:
    device = "cuda"
    dtype = torch.bfloat16
    eps = 1e-5

    residual = torch.randn(M, N, dtype=dtype, device=device)
    x = torch.randn(M, N, dtype=dtype, device=device)
    weight = torch.randn(N, dtype=dtype, device=device)
    bias = torch.randn(N, dtype=dtype, device=device)
    scale = torch.randn(M, N, dtype=dtype, device=device)
    shift = torch.randn(M, N, dtype=dtype, device=device)
    gate = torch.randn(M, N, dtype=dtype, device=device)

    def run_naive():
        fused_scale_residual_layernorm_scale_shift_naive(
            residual, x, weight, bias, scale, shift, gate, eps
        )

    def run_sglang():
        fused_scale_residual_layernorm_scale_shift_sglang(
            residual, x, weight, bias, scale, shift, gate, eps
        )

    # warmup
    for _ in range(num_warmup):
        run_naive()
    torch.cuda.synchronize()

    # run naive
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(num_run):
        run_naive()
    end_event.record()
    end_event.synchronize()
    torch.cuda.synchronize()
    naive_time = start_event.elapsed_time(end_event) / num_run * 1000  # us

    # warmup
    for _ in range(num_warmup):
        run_sglang()
    torch.cuda.synchronize()

    # run sglang
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(num_run):
        run_sglang()
    end_event.record()
    end_event.synchronize()
    torch.cuda.synchronize()
    sglang_time = start_event.elapsed_time(end_event) / num_run * 1000  # us

    return naive_time, sglang_time


benchmark_kernels = {
    "fused_layernorm_scale_shift": bench_fused_layernorm_scale_shift,
    "fused_layernorm_scale_shift_no_affine": bench_fused_layernorm_scale_shift_no_affine,
    "fused_scale_residual_layernorm_scale_shift": bench_fused_scale_residual_layernorm_scale_shift,
}


@dataclass
class ShapeArg:
    M: int
    N: int


def benchmark_one_shape(
    kernel_name: str,
    shape_args: List[ShapeArg],
    num_warmup: int,
    num_run: int,
):
    print(f"\n{'='*80}")
    print(f"Kernel: {kernel_name}")
    print(f"{'='*80}")

    kernel_func = benchmark_kernels[kernel_name]

    for shape in shape_args:
        naive_time, sglang_time = kernel_func(
            shape.M,
            shape.N,
            num_warmup,
            num_run,
        )
        speedup = naive_time / sglang_time if sglang_time > 0 else 0.0
        print(
            f"M={shape.M:5d}, N={shape.N:5d} | "
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
            ShapeArg(M=128, N=1024),
        ]
    else:
        shape_args = [
            # Small shapes
            ShapeArg(M=128, N=1024),
            ShapeArg(M=128, N=3072),
            ShapeArg(M=128, N=4096),
            # Medium shapes
            ShapeArg(M=1024, N=1024),
            ShapeArg(M=1024, N=3072),
            ShapeArg(M=1024, N=4096),
            # Large shapes
            ShapeArg(M=4096, N=1024),
            ShapeArg(M=4096, N=3072),
            ShapeArg(M=4096, N=4096),
        ]

    args = parser.parse_args()

    for kernel_name in benchmark_kernels.keys():
        benchmark_one_shape(kernel_name, shape_args, args.num_warmup, args.num_run)


if __name__ == "__main__":
    main()
