# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Benchmark for MXFP8 block-scaled matrix multiplication.

This benchmarks the triton MXFP8 kernels used for Blackwell GPUs (SM100+).
Tests both the optimized TensorDescriptor kernel (N >= 128) and the simple
tl.load kernel (N < 128).

Usage:
    python bench_mxfp8_gemm.py
"""

import numpy as np
import torch
from flashinfer.testing import bench_gpu_time

from sglang.srt.utils import is_sm100_supported


def check_sm100():
    if not is_sm100_supported():
        print("MXFP8 requires Blackwell GPUs (SM100+). Skipping benchmark.")
        return False
    return True


def get_mxfp8_functions():
    from sglang.srt.layers.quantization.fp8_kernel import (
        mxfp8_block_scaled_matmul_triton,
        mxfp8_block_scaled_matmul_triton_simple,
    )
    from sglang.srt.layers.quantization.fp8_utils import (
        _pack_mxfp8_scales,
        mxfp8_group_quantize,
    )

    return (
        mxfp8_block_scaled_matmul_triton,
        mxfp8_block_scaled_matmul_triton_simple,
        _pack_mxfp8_scales,
        mxfp8_group_quantize,
    )


def ceil_div(a, b):
    return (a + b - 1) // b


def bench_mxfp8_optimized(M, N, K):
    """Benchmark optimized MXFP8 GEMM using TensorDescriptor (requires N >= 128)."""
    (
        mxfp8_block_scaled_matmul_triton,
        _,
        _pack_mxfp8_scales,
        mxfp8_group_quantize,
    ) = get_mxfp8_functions()

    x = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((N, K), device="cuda", dtype=torch.bfloat16)
    x_fp8, x_scale = mxfp8_group_quantize(x)
    w_fp8, w_scale = mxfp8_group_quantize(w)

    # Pad M if needed
    block_m = 128
    if M % block_m != 0:
        pad_rows = ceil_div(M, block_m) * block_m - M
        x_fp8 = torch.cat(
            [x_fp8, torch.zeros((pad_rows, K), device="cuda", dtype=x_fp8.dtype)],
            dim=0,
        )
        x_scale = torch.cat(
            [x_scale, torch.full((pad_rows, K // 32), 127, device="cuda", dtype=x_scale.dtype)],
            dim=0,
        )

    a_scale_packed = _pack_mxfp8_scales(x_scale)
    b_scale_packed = _pack_mxfp8_scales(w_scale)
    w_fp8_c = w_fp8.contiguous()
    block_n = 256 if N % 256 == 0 else 128

    def run(x_fp8, a_scale_packed, w_fp8, b_scale_packed):
        return mxfp8_block_scaled_matmul_triton(
            x_fp8, a_scale_packed, w_fp8, b_scale_packed,
            output_dtype=torch.bfloat16, block_m=128, block_n=block_n, block_k=128,
        )

    measurements = bench_gpu_time(
        run,
        input_args=(x_fp8, a_scale_packed, w_fp8_c, b_scale_packed),
        use_cuda_graph=True,
    )
    ms = np.median(measurements)
    tflops = 2 * M * N * K * 1e-9 / ms
    return ms, tflops


def bench_mxfp8_simple(M, N, K):
    """Benchmark simple MXFP8 GEMM using tl.load (works with any N)."""
    (
        _,
        mxfp8_block_scaled_matmul_triton_simple,
        _,
        mxfp8_group_quantize,
    ) = get_mxfp8_functions()

    x = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((N, K), device="cuda", dtype=torch.bfloat16)
    x_fp8, x_scale = mxfp8_group_quantize(x)
    w_fp8, w_scale = mxfp8_group_quantize(w)

    # Pad M if needed
    block_m = 128
    if M % block_m != 0:
        pad_rows = ceil_div(M, block_m) * block_m - M
        x_fp8 = torch.cat(
            [x_fp8, torch.zeros((pad_rows, K), device="cuda", dtype=x_fp8.dtype)],
            dim=0,
        )
        x_scale = torch.cat(
            [x_scale, torch.full((pad_rows, K // 32), 127, device="cuda", dtype=x_scale.dtype)],
            dim=0,
        )

    if N <= 64:
        block_n = N if N in [16, 32, 64] else 64
    else:
        block_n = 64

    w_fp8_c = w_fp8.contiguous()
    w_scale_c = w_scale.contiguous()

    def run(x_fp8, x_scale, w_fp8, w_scale):
        return mxfp8_block_scaled_matmul_triton_simple(
            x_fp8, x_scale, w_fp8, w_scale,
            output_dtype=torch.bfloat16, block_m=128, block_n=block_n, block_k=128,
        )

    measurements = bench_gpu_time(
        run,
        input_args=(x_fp8, x_scale, w_fp8_c, w_scale_c),
        use_cuda_graph=True,
    )
    ms = np.median(measurements)
    tflops = 2 * M * N * K * 1e-9 / ms
    return ms, tflops


def bench_bf16(M, N, K):
    """Benchmark BF16 matmul reference."""
    x = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((N, K), device="cuda", dtype=torch.bfloat16)
    w_t = w.T.contiguous()

    def run(x, w_t):
        return torch.matmul(x, w_t)

    measurements = bench_gpu_time(
        run,
        input_args=(x, w_t),
        use_cuda_graph=True,
    )
    ms = np.median(measurements)
    tflops = 2 * M * N * K * 1e-9 / ms
    return ms, tflops


def test_accuracy():
    """Test accuracy of MXFP8 kernels."""
    (
        mxfp8_block_scaled_matmul_triton,
        mxfp8_block_scaled_matmul_triton_simple,
        _pack_mxfp8_scales,
        mxfp8_group_quantize,
    ) = get_mxfp8_functions()

    shapes = [(1, 128, 4096), (128, 128, 4096), (128, 4096, 4096), (256, 4096, 14336)]

    print("Running accuracy tests...")
    for M, N, K in shapes:
        x = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
        w = torch.randn((N, K), device="cuda", dtype=torch.bfloat16)
        x_fp8, x_scale = mxfp8_group_quantize(x)
        w_fp8, w_scale = mxfp8_group_quantize(w)

        # Pad M
        block_m = 128
        if M % block_m != 0:
            pad_rows = ceil_div(M, block_m) * block_m - M
            x_fp8_pad = torch.cat([x_fp8, torch.zeros((pad_rows, K), device="cuda", dtype=x_fp8.dtype)], dim=0)
            x_scale_pad = torch.cat([x_scale, torch.full((pad_rows, K // 32), 127, device="cuda", dtype=x_scale.dtype)], dim=0)
        else:
            x_fp8_pad, x_scale_pad = x_fp8, x_scale

        # Simple kernel
        block_n = 64
        out_simple = mxfp8_block_scaled_matmul_triton_simple(
            x_fp8_pad, x_scale_pad, w_fp8.contiguous(), w_scale.contiguous(),
            output_dtype=torch.bfloat16, block_m=128, block_n=block_n, block_k=128,
        )[:M, :]

        # Optimized kernel
        a_scale_packed = _pack_mxfp8_scales(x_scale_pad)
        b_scale_packed = _pack_mxfp8_scales(w_scale)
        block_n_opt = 256 if N % 256 == 0 else 128
        out_opt = mxfp8_block_scaled_matmul_triton(
            x_fp8_pad, a_scale_packed, w_fp8.contiguous(), b_scale_packed,
            output_dtype=torch.bfloat16, block_m=128, block_n=block_n_opt, block_k=128,
        )[:M, :]

        # They should be identical
        torch.testing.assert_close(out_opt, out_simple, atol=0, rtol=0)
        print(f"  M={M}, N={N}, K={K}: OK")

    print("All accuracy tests passed!\n")


# Benchmark shapes: (M, N, K, description)
SHAPES = [
    # Small N (simple kernel only)
    (1, 64, 4096, "decode bs=1, small N"),
    (128, 64, 4096, "decode bs=128, small N"),
    # Large N (both kernels)
    (1, 4096, 4096, "decode bs=1"),
    (32, 4096, 4096, "decode bs=32"),
    (128, 4096, 4096, "decode bs=128"),
    (512, 4096, 4096, "prefill"),
    (1, 14336, 4096, "MLP up proj bs=1"),
    (128, 14336, 4096, "MLP up proj bs=128"),
]


if __name__ == "__main__":
    if not check_sm100():
        exit(0)

    test_accuracy()

    print(f"{'Shape':<35} {'Kernel':<12} {'Time (us)':<12} {'TFLOPS':<10}")
    print("=" * 75)

    for M, N, K, desc in SHAPES:
        shape_str = f"M={M}, N={N}, K={K}"

        if N >= 128:
            ms_opt, tflops_opt = bench_mxfp8_optimized(M, N, K)
            print(f"{shape_str:<35} {'optimized':<12} {ms_opt*1000:<12.2f} {tflops_opt:<10.2f}")

        ms_simple, tflops_simple = bench_mxfp8_simple(M, N, K)
        print(f"{shape_str:<35} {'simple':<12} {ms_simple*1000:<12.2f} {tflops_simple:<10.2f}")

        ms_bf16, tflops_bf16 = bench_bf16(M, N, K)
        print(f"{shape_str:<35} {'bf16':<12} {ms_bf16*1000:<12.2f} {tflops_bf16:<10.2f}")
        print()
