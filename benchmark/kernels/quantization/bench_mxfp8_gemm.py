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

Strategies tested:
  - Baseline: Tutorial kernel from sglang (simple linear pid mapping)
  - Split-K: Split K dimension across blocks for increased SM occupancy

Usage:
    python bench_mxfp8_gemm.py
"""

import sys

import numpy as np
import torch
import triton
import triton.language as tl

try:
    from triton.tools.tensor_descriptor import TensorDescriptor
except Exception:
    pass

from flashinfer.testing import bench_gpu_time

from sglang.srt.utils import is_sm100_supported


def check_sm100():
    if not is_sm100_supported():
        print("MXFP8 requires Blackwell GPUs (SM100+).", file=sys.stderr)
        return False
    return True


def get_mxfp8_functions():
    from sglang.srt.layers.quantization.fp8_kernel import (
        mxfp8_block_scaled_matmul_triton,
    )
    from sglang.srt.layers.quantization.fp8_utils import (
        _interleave_mxfp8_scales_for_cublas,
        _pack_mxfp8_scales,
        cublas_mxfp8_blockscaled_linear,
        mxfp8_group_quantize,
        prepare_mxfp8_weight_for_cublas,
    )
    return (
        mxfp8_block_scaled_matmul_triton,
        _pack_mxfp8_scales,
        mxfp8_group_quantize,
        _interleave_mxfp8_scales_for_cublas,
        cublas_mxfp8_blockscaled_linear,
        prepare_mxfp8_weight_for_cublas,
    )


def ceil_div(a, b):
    return (a + b - 1) // b


# ---------------------------------------------------------------------------
# Split-K MXFP8 kernel
# ---------------------------------------------------------------------------

@triton.jit
def _mxfp8_matmul_splitk_kernel(
    a_desc,
    a_scale_desc,
    b_desc,
    b_scale_desc,
    partial_ptr,
    stride_partial_sk,
    stride_partial_m,
    num_pid_m,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    rep_m: tl.constexpr,
    rep_n: tl.constexpr,
    rep_k: tl.constexpr,
    K_ITERS_PER_SPLIT: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    SWAP_AB: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    split_k_id = tl.program_id(axis=1)

    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_am = pid_m * BLOCK_M
    offs_bn = pid_n * BLOCK_N
    offs_scale_m = pid_m * rep_m
    offs_scale_n = pid_n * rep_n

    VEC_SIZE: tl.constexpr = 32
    k_start = split_k_id * K_ITERS_PER_SPLIT

    if SWAP_AB:
        accumulator = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    else:
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_offset in tl.range(0, K_ITERS_PER_SPLIT, num_stages=NUM_STAGES):
        k = k_start + k_offset
        a = a_desc.load([offs_am, k * BLOCK_K])
        b = b_desc.load([offs_bn, k * BLOCK_K])
        scale_a = a_scale_desc.load([0, offs_scale_m, k * rep_k, 0, 0])
        scale_b = b_scale_desc.load([0, offs_scale_n, k * rep_k, 0, 0])

        scale_a = (
            scale_a.reshape(rep_m, rep_k, 32, 4, 4)
            .trans(0, 3, 2, 1, 4)
            .reshape(BLOCK_M, BLOCK_K // VEC_SIZE)
        )
        scale_b = (
            scale_b.reshape(rep_n, rep_k, 32, 4, 4)
            .trans(0, 3, 2, 1, 4)
            .reshape(BLOCK_N, BLOCK_K // VEC_SIZE)
        )

        if SWAP_AB:
            # b (BLOCK_N, BLOCK_K) as LHS, a.T (BLOCK_K, BLOCK_M) as RHS
            accumulator = tl.dot_scaled(
                b, scale_b, "e4m3", a.T, scale_a, "e4m3", accumulator
            )
        else:
            accumulator = tl.dot_scaled(
                a, scale_a, "e4m3", b.T, scale_b, "e4m3", accumulator
            )

    if SWAP_AB:
        accumulator = tl.trans(accumulator, (1, 0))

    # Pointer-based store to fp32 partial buffer
    offs_m = offs_am + tl.arange(0, BLOCK_M)
    offs_n = offs_bn + tl.arange(0, BLOCK_N)
    partial_ptrs = (
        partial_ptr
        + split_k_id * stride_partial_sk
        + offs_m[:, None] * stride_partial_m
        + offs_n[None, :]
    )
    tl.store(partial_ptrs, accumulator)


def should_swap_ab_mxfp8(M, N, K, block_m=128, block_n=256, block_k=128,
                          target_sms=192):
    """Swap A/B when BLOCK_N > BLOCK_M and there are few tiles (small M, large K).

    Only beneficial when split-K is also used, on shapes with low SM occupancy.
    """
    if block_n <= block_m:
        return False
    tiles = ceil_div(M, block_m) * ceil_div(N, block_n)
    k_iters = K // block_k
    # Only swap when: few tiles (need split-K) and enough K to split
    return tiles < target_sms and k_iters >= 16


def mxfp8_matmul_splitk(
    a, a_scale, b, b_scale, output_dtype,
    *, block_m=128, block_n=256, block_k=128,
    num_stages=4, num_warps=4, split_k=2, swap_ab=False,
):
    M, K = a.shape
    N, K_b = b.shape
    assert K == K_b

    k_iters = K // block_k
    assert k_iters % split_k == 0, \
        f"K iters ({k_iters}) not divisible by split_k ({split_k})"
    k_iters_per_split = k_iters // split_k

    rep_m = block_m // 128
    rep_n = block_n // 128
    rep_k = block_k // 32 // 4

    a_desc = TensorDescriptor.from_tensor(a, [block_m, block_k])
    b_desc = TensorDescriptor.from_tensor(b, [block_n, block_k])
    a_scale_desc = TensorDescriptor.from_tensor(
        a_scale, block_shape=[1, rep_m, rep_k, 2, 256]
    )
    b_scale_desc = TensorDescriptor.from_tensor(
        b_scale, block_shape=[1, rep_n, rep_k, 2, 256]
    )

    num_pid_m = triton.cdiv(M, block_m)
    num_pid_n = triton.cdiv(N, block_n)

    partial = torch.empty((split_k, M, N), dtype=torch.float32, device=a.device)

    grid = (num_pid_m * num_pid_n, split_k)
    _mxfp8_matmul_splitk_kernel[grid](
        a_desc, a_scale_desc, b_desc, b_scale_desc,
        partial, partial.stride(0), partial.stride(1),
        num_pid_m,
        block_m, block_n, block_k, rep_m, rep_n, rep_k,
        k_iters_per_split, num_stages, swap_ab,
        num_warps=num_warps,
    )

    output = partial.sum(dim=0).to(output_dtype)
    return output


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def prepare_mxfp8_inputs(M, N, K):
    (_, _pack_mxfp8_scales, mxfp8_group_quantize, _, _, _) = get_mxfp8_functions()
    x = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((N, K), device="cuda", dtype=torch.bfloat16)
    x_fp8, x_scale = mxfp8_group_quantize(x)
    w_fp8, w_scale = mxfp8_group_quantize(w)

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
    return x_fp8, a_scale_packed, w_fp8.contiguous(), b_scale_packed


def bench_kernel(M, N, K, kernel_fn, **kernel_kwargs):
    x_fp8, a_scale_packed, w_fp8_c, b_scale_packed = prepare_mxfp8_inputs(M, N, K)

    def run(x_fp8, a_scale_packed, w_fp8, b_scale_packed):
        return kernel_fn(
            x_fp8, a_scale_packed, w_fp8, b_scale_packed,
            output_dtype=torch.bfloat16, **kernel_kwargs,
        )

    measurements = bench_gpu_time(
        run, input_args=(x_fp8, a_scale_packed, w_fp8_c, b_scale_packed),
        use_cuda_graph=True,
    )
    ms = np.median(measurements)
    tflops = 2 * M * N * K * 1e-9 / ms
    return ms, tflops


def bench_bf16(M, N, K):
    x = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((N, K), device="cuda", dtype=torch.bfloat16)
    w_t = w.T.contiguous()

    def run(x, w_t):
        return torch.matmul(x, w_t)

    measurements = bench_gpu_time(run, input_args=(x, w_t), use_cuda_graph=True)
    ms = np.median(measurements)
    tflops = 2 * M * N * K * 1e-9 / ms
    return ms, tflops


def bench_torch_scaled_mm(M, N, K):
    """Benchmark torch._scaled_mm with MXFP8 (cuBLAS path)."""
    (_, _pack_mxfp8_scales, mxfp8_group_quantize, _, _, _) = get_mxfp8_functions()
    x = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((N, K), device="cuda", dtype=torch.bfloat16)
    x_fp8, x_scale = mxfp8_group_quantize(x)
    w_fp8, w_scale = mxfp8_group_quantize(w)

    # Pack scales into cuBLAS interleaved layout (same as tl.dot_scaled layout)
    sa = _pack_mxfp8_scales(x_scale).reshape(M, K // 32).view(torch.float8_e8m0fnu)
    sb = _pack_mxfp8_scales(w_scale).reshape(N, K // 32).view(torch.float8_e8m0fnu)
    w_fp8_t = w_fp8.t()  # column-major for cuBLAS

    def run(x_fp8, w_fp8_t, sa, sb):
        return torch._scaled_mm(
            x_fp8, w_fp8_t, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16,
        )

    measurements = bench_gpu_time(
        run, input_args=(x_fp8, w_fp8_t, sa, sb), use_cuda_graph=True,
    )
    ms = np.median(measurements)
    tflops = 2 * M * N * K * 1e-9 / ms
    return ms, tflops


def bench_cublas_fused(M, N, K):
    """Benchmark the full fused cuBLAS MXFP8 pipeline.

    Simulates production: weight pre-computed at load time, input quantized
    + scale-interleaved at runtime, then torch._scaled_mm.
    """
    (_, _, mxfp8_group_quantize, _interleave_mxfp8_scales_for_cublas,
     _, prepare_mxfp8_weight_for_cublas) = get_mxfp8_functions()

    x = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((N, K), device="cuda", dtype=torch.bfloat16)
    w_fp8, w_scale = mxfp8_group_quantize(w)

    # Pre-compute weight data (simulates model load time)
    from sglang.srt.layers.quantization.fp8_utils import _pack_mxfp8_scales
    weight_t, weight_scale_cublas = prepare_mxfp8_weight_for_cublas(w_fp8, w_scale)

    # Pre-quantize input (this would happen at runtime)
    x_fp8, x_scale = mxfp8_group_quantize(x)
    m_padded = ceil_div(M, 128) * 128
    if M % 128 != 0:
        pad_rows = m_padded - M
        x_fp8 = torch.cat(
            [x_fp8, torch.zeros((pad_rows, K), device="cuda", dtype=x_fp8.dtype)],
            dim=0,
        )
        x_scale = torch.cat(
            [x_scale, torch.full((pad_rows, K // 32), 127, device="cuda", dtype=x_scale.dtype)],
            dim=0,
        )
    sa = _interleave_mxfp8_scales_for_cublas(x_scale, m_padded, K)

    def run(x_fp8, weight_t, sa, weight_scale_cublas):
        return torch._scaled_mm(
            x_fp8, weight_t, scale_a=sa, scale_b=weight_scale_cublas,
            out_dtype=torch.bfloat16,
        )

    measurements = bench_gpu_time(
        run, input_args=(x_fp8, weight_t, sa, weight_scale_cublas),
        use_cuda_graph=True,
    )
    ms = np.median(measurements)
    tflops = 2 * M * N * K * 1e-9 / ms
    return ms, tflops


def test_accuracy_cublas_fused():
    """Test accuracy of cuBLAS fused MXFP8 path vs Triton baseline."""
    (mxfp8_block_scaled_matmul_triton, _pack_mxfp8_scales, mxfp8_group_quantize,
     _, cublas_mxfp8_blockscaled_linear, prepare_mxfp8_weight_for_cublas,
     ) = get_mxfp8_functions()

    shapes = [
        (1, 256, 512), (127, 256, 512), (128, 128, 4096),
        (128, 4096, 4096), (129, 384, 1024), (255, 512, 2048),
        (256, 4096, 8192), (512, 8192, 8192),
    ]
    print("Running cuBLAS fused MXFP8 accuracy tests...", file=sys.stderr)
    for M, N, K in shapes:
        x = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
        w = torch.randn((N, K), device="cuda", dtype=torch.bfloat16)
        w_fp8, w_scale = mxfp8_group_quantize(w)

        # Triton baseline
        x_fp8, x_scale = mxfp8_group_quantize(x)
        block_m = 128
        m_padded = ceil_div(M, block_m) * block_m
        if M % block_m != 0:
            pad_rows = m_padded - M
            x_fp8_pad = torch.cat(
                [x_fp8, torch.zeros((pad_rows, K), device="cuda", dtype=x_fp8.dtype)],
                dim=0,
            )
            x_scale_pad = torch.cat(
                [x_scale, torch.full((pad_rows, K // 32), 127, device="cuda", dtype=x_scale.dtype)],
                dim=0,
            )
        else:
            x_fp8_pad, x_scale_pad = x_fp8, x_scale
        block_n = 256 if N % 256 == 0 else 128
        a_scale_packed = _pack_mxfp8_scales(x_scale_pad)
        b_scale_packed = _pack_mxfp8_scales(w_scale)
        out_triton = mxfp8_block_scaled_matmul_triton(
            x_fp8_pad, a_scale_packed, w_fp8.contiguous(), b_scale_packed,
            output_dtype=torch.bfloat16, block_m=128, block_n=block_n, block_k=128,
        )[:M, :]

        # cuBLAS fused (using the production function)
        weight_t, weight_scale_cublas = prepare_mxfp8_weight_for_cublas(w_fp8, w_scale)
        out_cublas = cublas_mxfp8_blockscaled_linear(
            input=x,
            weight_t=weight_t,
            weight_scale_cublas=weight_scale_cublas,
            input_scale=None,
        )

        # Also test pre-quantized input path
        out_cublas_prequant = cublas_mxfp8_blockscaled_linear(
            input=x_fp8,
            weight_t=weight_t,
            weight_scale_cublas=weight_scale_cublas,
            input_scale=x_scale,
            output_dtype=torch.bfloat16,
        )

        for label, out_test in [("dynamic", out_cublas), ("prequant", out_cublas_prequant)]:
            abs_diff = (out_triton.float() - out_test.float()).abs()
            max_val = out_triton.float().abs().max().item()
            rel_diff = abs_diff.max().item() / max(max_val, 1e-6)
            assert rel_diff < 0.02, \
                f"cuBLAS {label} mismatch M={M},N={N},K={K}: rel={rel_diff:.4f}"

        print(f"  M={M},N={N},K={K}: OK", file=sys.stderr)
    print("All cuBLAS fused accuracy tests passed!\n", file=sys.stderr)


def valid_split_k_values(K, block_k=128, max_split_k=16):
    k_iters = K // block_k
    return [sk for sk in [1, 2, 3, 4, 6, 8, 16] if sk <= max_split_k and k_iters % sk == 0]


def pick_split_k(M, N, K, block_m=128, block_n=256, block_k=128,
                  target_sms=192, min_k_iters_per_split=8):
    """Heuristic: pick smallest split_k so tiles * split_k >= target_sms.

    Only splits when each partition has enough K iterations (≥min_k_iters_per_split)
    to amortize the reduction overhead.
    """
    tiles = ceil_div(M, block_m) * ceil_div(N, block_n)
    k_iters = K // block_k
    if tiles >= target_sms or k_iters < min_k_iters_per_split * 2:
        return 1
    for sk in [1, 2, 3, 4, 6, 8, 16]:
        if k_iters % sk != 0:
            continue
        if k_iters // sk < min_k_iters_per_split:
            continue
        if tiles * sk >= target_sms:
            return sk
    # Return max valid split_k that respects min_k_iters
    valid = [sk for sk in [1, 2, 3, 4, 6, 8, 16]
             if k_iters % sk == 0 and k_iters // sk >= min_k_iters_per_split]
    return valid[-1] if valid else 1


def test_accuracy_splitk():
    (mxfp8_block_scaled_matmul_triton, _pack_mxfp8_scales, mxfp8_group_quantize,
     _, _, _) = get_mxfp8_functions()

    shapes = [
        (128, 128, 4096), (128, 4096, 4096),
        (256, 4096, 8192), (512, 8192, 8192),
    ]
    print("Running split-K + swap_ab accuracy tests...", file=sys.stderr)
    for M, N, K in shapes:
        x = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
        w = torch.randn((N, K), device="cuda", dtype=torch.bfloat16)
        x_fp8, x_scale = mxfp8_group_quantize(x)
        w_fp8, w_scale = mxfp8_group_quantize(w)
        a_scale_packed = _pack_mxfp8_scales(x_scale)
        b_scale_packed = _pack_mxfp8_scales(w_scale)
        w_fp8_c = w_fp8.contiguous()
        block_n = 256 if N % 256 == 0 else 128

        out_baseline = mxfp8_block_scaled_matmul_triton(
            x_fp8, a_scale_packed, w_fp8_c, b_scale_packed,
            output_dtype=torch.bfloat16, block_m=128, block_n=block_n, block_k=128,
        )
        for sab in [False, True]:
            for sk in valid_split_k_values(K):
                if sk == 1 and not sab:
                    continue
                out_sk = mxfp8_matmul_splitk(
                    x_fp8, a_scale_packed, w_fp8_c, b_scale_packed,
                    output_dtype=torch.bfloat16,
                    block_m=128, block_n=block_n, block_k=128,
                    split_k=sk, swap_ab=sab,
                )
                assert torch.isfinite(out_sk).all(), \
                    f"NaN/Inf at M={M},N={N},K={K},sk={sk},sab={sab}"
                abs_diff = (out_baseline.float() - out_sk.float()).abs()
                max_val = out_baseline.float().abs().max().item()
                rel_diff = abs_diff.max().item() / max(max_val, 1e-6)
                assert rel_diff < 0.02, \
                    f"Mismatch M={M},N={N},K={K},sk={sk},sab={sab}: rel={rel_diff:.4f}"
        print(f"  M={M},N={N},K={K}: OK", file=sys.stderr)
    print("All accuracy tests passed!\n", file=sys.stderr)


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------
Ms = [128, 256, 512, 1024, 2048, 4096]
Ns = [128, 256, 512, 1024, 1536, 2048, 4096, 8192]
Ks = [768, 2048, 4096, 8192]


if __name__ == "__main__":
    if not check_sm100():
        exit(0)

    torch_only = "--torch-only" in sys.argv
    cublas_fused = "--cublas-fused" in sys.argv

    if cublas_fused:
        # Benchmark cuBLAS fused MXFP8 (pre-computed weights) vs Triton baseline vs BF16
        test_accuracy_cublas_fused()
        print("# cuBLAS fused MXFP8 benchmark", file=sys.stderr)
        print(
            "M,N,K,triton_us,cublas_us,bf16_us,"
            "triton_tflops,cublas_tflops,bf16_tflops,"
            "cublas_vs_triton,cublas_vs_bf16"
        )
        for K in Ks:
            for N in Ns:
                for M in Ms:
                    block_n = 256 if N % 256 == 0 else 128
                    ms_triton, tflops_triton = bench_kernel(
                        M, N, K,
                        get_mxfp8_functions()[0],  # mxfp8_block_scaled_matmul_triton
                        block_m=128, block_n=block_n, block_k=128,
                    )
                    ms_cublas, tflops_cublas = bench_cublas_fused(M, N, K)
                    ms_bf16, tflops_bf16 = bench_bf16(M, N, K)
                    cublas_vs_triton = ms_triton / ms_cublas
                    cublas_vs_bf16 = ms_bf16 / ms_cublas
                    print(
                        f"{M},{N},{K},"
                        f"{ms_triton*1000:.2f},{ms_cublas*1000:.2f},{ms_bf16*1000:.2f},"
                        f"{tflops_triton:.2f},{tflops_cublas:.2f},{tflops_bf16:.2f},"
                        f"{cublas_vs_triton:.3f},{cublas_vs_bf16:.3f}"
                    )
                print(f"  K={K},N={N} done", file=sys.stderr)
        exit(0)

    if torch_only:
        # Only benchmark torch._scaled_mm (cuBLAS MXFP8 path)
        print("# torch._scaled_mm MXFP8 benchmark (cuBLAS)", file=sys.stderr)
        print(
            "M,N,K,torch_us,torch_tflops,bf16_us,bf16_tflops,"
            "torch_vs_bf16"
        )
        for K in Ks:
            for N in Ns:
                for M in Ms:
                    ms_torch, tflops_torch = bench_torch_scaled_mm(M, N, K)
                    ms_bf16, tflops_bf16 = bench_bf16(M, N, K)
                    torch_vs_bf16 = ms_bf16 / ms_torch
                    print(
                        f"{M},{N},{K},"
                        f"{ms_torch*1000:.2f},{tflops_torch:.2f},"
                        f"{ms_bf16*1000:.2f},{tflops_bf16:.2f},"
                        f"{torch_vs_bf16:.3f}"
                    )
                print(f"  K={K},N={N} done", file=sys.stderr)
        exit(0)

    test_accuracy_splitk()

    (mxfp8_block_scaled_matmul_triton, _, _, _, _, _) = get_mxfp8_functions()

    # Phase 1: Split-K + swap_ab sweep on representative shapes
    print("# Phase 1: Split-K + swap_ab sweep", file=sys.stderr)
    print("M,N,K,baseline_us,best_us,speedup,best_sk,best_sab,tiles_base,tiles_best")

    sweep_shapes = [
        (128, 4096, 8192), (128, 8192, 8192),
        (256, 4096, 8192), (256, 8192, 8192),
        (512, 4096, 4096), (512, 8192, 8192),
        (1024, 4096, 4096), (1024, 8192, 8192),
        (2048, 4096, 4096), (2048, 8192, 8192),
        (4096, 4096, 4096), (4096, 8192, 8192),
    ]

    # (split_k, swap_ab) -> best config per shape
    best_configs = {}
    for M, N, K in sweep_shapes:
        block_n = 256 if N % 256 == 0 else 128
        tiles_base = ceil_div(M, 128) * ceil_div(N, block_n)
        ms_base, _ = bench_kernel(
            M, N, K, mxfp8_block_scaled_matmul_triton,
            block_m=128, block_n=block_n, block_k=128,
        )

        best_ms = ms_base
        best_sk = 1
        best_sab = False
        for sab in [False, True]:
            for sk in valid_split_k_values(K):
                if sk == 1 and not sab:
                    continue
                ms_sk, _ = bench_kernel(
                    M, N, K, mxfp8_matmul_splitk,
                    block_m=128, block_n=block_n, block_k=128,
                    split_k=sk, swap_ab=sab,
                )
                if ms_sk < best_ms:
                    best_ms = ms_sk
                    best_sk = sk
                    best_sab = sab

        tiles_best = tiles_base * best_sk
        speedup = ms_base / best_ms
        sab_str = "T" if best_sab else "F"
        print(
            f"{M},{N},{K},{ms_base*1000:.2f},{best_ms*1000:.2f},"
            f"{speedup:.3f},{best_sk},{sab_str},{tiles_base},{tiles_best}"
        )
        best_configs[(M, N, K)] = (best_sk, best_sab)
        print(
            f"  {M}x{N}x{K} done (sk={best_sk},sab={best_sab})",
            file=sys.stderr,
        )

    # Phase 2: Full comparison using heuristic split-K + swap_ab
    print(file=sys.stderr)
    print("# Phase 2: Full comparison", file=sys.stderr)
    print()
    print(
        "M,N,K,baseline_us,best_us,bf16_us,"
        "baseline_tflops,best_tflops,bf16_tflops,"
        "best_vs_base,best_vs_bf16,split_k,swap_ab"
    )

    for K in Ks:
        for N in Ns:
            for M in Ms:
                block_n = 256 if N % 256 == 0 else 128

                ms_base, tflops_base = bench_kernel(
                    M, N, K, mxfp8_block_scaled_matmul_triton,
                    block_m=128, block_n=block_n, block_k=128,
                )
                ms_bf16, tflops_bf16 = bench_bf16(M, N, K)

                # Use Phase 1 result if available, else heuristic
                if (M, N, K) in best_configs:
                    sk, sab = best_configs[(M, N, K)]
                else:
                    sk = pick_split_k(M, N, K, block_n=block_n)
                    sab = should_swap_ab_mxfp8(M, N, K, block_n=block_n)

                if sk <= 1 and not sab:
                    ms_best, tflops_best = ms_base, tflops_base
                else:
                    ms_best, tflops_best = bench_kernel(
                        M, N, K, mxfp8_matmul_splitk,
                        block_m=128, block_n=block_n, block_k=128,
                        split_k=max(sk, 1), swap_ab=sab,
                    )

                best_vs_base = ms_base / ms_best
                best_vs_bf16 = ms_bf16 / ms_best
                sab_str = "T" if sab else "F"

                print(
                    f"{M},{N},{K},{ms_base*1000:.2f},{ms_best*1000:.2f},{ms_bf16*1000:.2f},"
                    f"{tflops_base:.2f},{tflops_best:.2f},{tflops_bf16:.2f},"
                    f"{best_vs_base:.3f},{best_vs_bf16:.2f},{sk},{sab_str}"
                )
            print(f"  K={K},N={N} done", file=sys.stderr)
