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

Compares baseline tutorial kernel vs improved kernel with GROUP_SIZE_M
swizzling and tuned num_warps/num_stages.

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
        print("MXFP8 requires Blackwell GPUs (SM100+). Skipping benchmark.", file=sys.stderr)
        return False
    return True


def get_mxfp8_functions():
    from sglang.srt.layers.quantization.fp8_kernel import (
        mxfp8_block_scaled_matmul_triton,
    )
    from sglang.srt.layers.quantization.fp8_utils import (
        _pack_mxfp8_scales,
        mxfp8_group_quantize,
    )

    return (
        mxfp8_block_scaled_matmul_triton,
        _pack_mxfp8_scales,
        mxfp8_group_quantize,
    )


def ceil_div(a, b):
    return (a + b - 1) // b


# ---------------------------------------------------------------------------
# Improved MXFP8 kernel with GROUP_SIZE_M swizzling
# ---------------------------------------------------------------------------

@triton.jit
def _mxfp8_matmul_kernel_v2(
    a_desc,
    a_scale_desc,
    b_desc,
    b_scale_desc,
    c_desc,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    output_type: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    rep_m: tl.constexpr,
    rep_n: tl.constexpr,
    rep_k: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    if output_type == 0:
        output_dtype = tl.float32
    elif output_type == 1:
        output_dtype = tl.float16
    elif output_type == 2:
        output_dtype = tl.bfloat16

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # GROUP_SIZE_M swizzling for L2 cache locality
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_M
    offs_bn = pid_n * BLOCK_N
    offs_scale_m = pid_m * rep_m
    offs_scale_n = pid_n * rep_n

    VEC_SIZE: tl.constexpr = 32

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):
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

        accumulator = tl.dot_scaled(
            a, scale_a, "e4m3", b.T, scale_b, "e4m3", accumulator
        )

    c_desc.store([offs_am, offs_bn], accumulator.to(output_dtype))


def mxfp8_matmul_v2(
    a, a_scale, b, b_scale, output_dtype,
    *, block_m=128, block_n=256, block_k=128,
    num_stages=4, group_size_m=8, num_warps=4,
):
    M, K = a.shape
    N, K_b = b.shape
    assert K == K_b

    if output_dtype == torch.float32:
        output_type = 0
    elif output_dtype == torch.float16:
        output_type = 1
    elif output_dtype == torch.bfloat16:
        output_type = 2
    else:
        raise ValueError(f"Unsupported output dtype: {output_dtype}")

    rep_m = block_m // 128
    rep_n = block_n // 128
    rep_k = block_k // 32 // 4

    a_desc = TensorDescriptor.from_tensor(a, [block_m, block_k])
    b_desc = TensorDescriptor.from_tensor(b, [block_n, block_k])
    a_scale_desc = TensorDescriptor.from_tensor(a_scale, block_shape=[1, rep_m, rep_k, 2, 256])
    b_scale_desc = TensorDescriptor.from_tensor(b_scale, block_shape=[1, rep_n, rep_k, 2, 256])

    output = torch.empty((M, N), dtype=output_dtype, device=a.device)
    c_desc = TensorDescriptor.from_tensor(output, [block_m, block_n])

    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n), 1)
    _mxfp8_matmul_kernel_v2[grid](
        a_desc, a_scale_desc, b_desc, b_scale_desc, c_desc,
        M, N, K, output_type,
        block_m, block_n, block_k, rep_m, rep_n, rep_k,
        group_size_m, num_stages, num_warps=num_warps,
    )
    return output


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def prepare_mxfp8_inputs(M, N, K):
    (_, _pack_mxfp8_scales, mxfp8_group_quantize) = get_mxfp8_functions()

    x = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((N, K), device="cuda", dtype=torch.bfloat16)
    x_fp8, x_scale = mxfp8_group_quantize(x)
    w_fp8, w_scale = mxfp8_group_quantize(w)

    block_m = 128
    if M % block_m != 0:
        pad_rows = ceil_div(M, block_m) * block_m - M
        x_fp8 = torch.cat(
            [x_fp8, torch.zeros((pad_rows, K), device="cuda", dtype=x_fp8.dtype)], dim=0,
        )
        x_scale = torch.cat(
            [x_scale, torch.full((pad_rows, K // 32), 127, device="cuda", dtype=x_scale.dtype)], dim=0,
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


def test_accuracy_v2():
    (mxfp8_block_scaled_matmul_triton, _pack_mxfp8_scales, mxfp8_group_quantize) = get_mxfp8_functions()

    shapes = [(128, 128, 4096), (128, 4096, 4096), (256, 4096, 14336), (512, 8192, 8192)]
    print("Running v2 accuracy tests...", file=sys.stderr)
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
        for gsm in [4, 8]:
            for nw in [4, 8]:
                out_v2 = mxfp8_matmul_v2(
                    x_fp8, a_scale_packed, w_fp8_c, b_scale_packed,
                    output_dtype=torch.bfloat16,
                    block_m=128, block_n=block_n, block_k=128,
                    group_size_m=gsm, num_warps=nw,
                )
                assert torch.isfinite(out_v2).all(), f"NaN/Inf at M={M},N={N},K={K},gsm={gsm},nw={nw}"
                assert torch.equal(out_baseline, out_v2), \
                    f"Mismatch at M={M},N={N},K={K},gsm={gsm},nw={nw}: max_diff={(out_baseline-out_v2).abs().max().item()}"
        print(f"  M={M},N={N},K={K}: OK", file=sys.stderr)
    print("All v2 accuracy tests passed!\n", file=sys.stderr)


# Configs to sweep
V2_CONFIGS = [
    (gsm, ns, nw)
    for gsm in [4, 8, 16]
    for ns in [2, 3, 4]
    for nw in [4, 8]
]

Ms = [128, 256, 512, 1024, 2048, 4096]
Ns = [128, 256, 512, 1024, 1536, 2048, 4096, 8192]
Ks = [768, 2048, 4096, 8192]


if __name__ == "__main__":
    if not check_sm100():
        exit(0)

    test_accuracy_v2()

    (mxfp8_block_scaled_matmul_triton, _, _) = get_mxfp8_functions()

    # Phase 1: Config sweep
    print("# Phase 1: Config sweep", file=sys.stderr)
    print("M,N,K,baseline_us,best_v2_us,v2_vs_base,best_gsm,best_stages,best_warps")

    sweep_shapes = [
        (128, 4096, 8192), (256, 4096, 8192), (512, 4096, 4096),
        (512, 8192, 8192), (1024, 2048, 4096), (1024, 4096, 4096),
        (2048, 4096, 4096), (2048, 8192, 8192), (4096, 4096, 4096),
        (4096, 8192, 8192),
    ]

    best_configs = {}
    for M, N, K in sweep_shapes:
        block_n = 256 if N % 256 == 0 else 128
        ms_base, _ = bench_kernel(M, N, K, mxfp8_block_scaled_matmul_triton,
                                   block_m=128, block_n=block_n, block_k=128)

        best_ms = float('inf')
        best_cfg = None
        for gsm, ns, nw in V2_CONFIGS:
            ms_v2, _ = bench_kernel(M, N, K, mxfp8_matmul_v2,
                                     block_m=128, block_n=block_n, block_k=128,
                                     num_stages=ns, group_size_m=gsm, num_warps=nw)
            if ms_v2 < best_ms:
                best_ms = ms_v2
                best_cfg = (gsm, ns, nw)

        ratio = best_ms / ms_base
        print(f"{M},{N},{K},{ms_base*1000:.2f},{best_ms*1000:.2f},{ratio:.3f},{best_cfg[0]},{best_cfg[1]},{best_cfg[2]}")
        best_configs[(M, N, K)] = best_cfg
        print(f"  {M}x{N}x{K} done", file=sys.stderr)

    # Phase 2: Full comparison
    print(file=sys.stderr)
    print("# Phase 2: Full comparison", file=sys.stderr)
    print()
    print("M,N,K,baseline_us,v2_us,bf16_us,baseline_tflops,v2_tflops,bf16_tflops,v2_vs_base,v2_speedup_vs_bf16")

    default_cfg = (8, 3, 4)

    for K in Ks:
        for N in Ns:
            for M in Ms:
                block_n = 256 if N % 256 == 0 else 128
                ms_base, tflops_base = bench_kernel(
                    M, N, K, mxfp8_block_scaled_matmul_triton,
                    block_m=128, block_n=block_n, block_k=128)
                ms_bf16, tflops_bf16 = bench_bf16(M, N, K)

                cfg = best_configs.get((M, N, K), default_cfg)
                gsm, ns, nw = cfg
                ms_v2, tflops_v2 = bench_kernel(
                    M, N, K, mxfp8_matmul_v2,
                    block_m=128, block_n=block_n, block_k=128,
                    num_stages=ns, group_size_m=gsm, num_warps=nw)

                v2_vs_base = ms_v2 / ms_base
                v2_vs_bf16 = ms_bf16 / ms_v2

                print(f"{M},{N},{K},{ms_base*1000:.2f},{ms_v2*1000:.2f},{ms_bf16*1000:.2f},"
                      f"{tflops_base:.2f},{tflops_v2:.2f},{tflops_bf16:.2f},"
                      f"{v2_vs_base:.3f},{v2_vs_bf16:.2f}")
            print(f"  K={K},N={N} done", file=sys.stderr)
