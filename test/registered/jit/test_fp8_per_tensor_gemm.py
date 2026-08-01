"""Accuracy tests for the CUTLASS FP8 per-row/per-column scaled GEMM.

Ported from the deleted AOT sgl-kernel test (python/sglang/kernels/aot/tests/
test_fp8_gemm.py); the SM90 swap-AB shape list is kept verbatim and the SM120
M-bucket boundaries are added, since the JIT kernel's SM120 path now dispatches
on M (16 / 32 / 256 / default) instead of always using one 128x128x128 tile.
"""

import sys

import pytest
import torch

from sglang.kernels.ops.gemm.fp8_per_tensor_gemm import fp8_per_tensor_scaled_mm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def torch_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias):
    o = torch.matmul(a.to(torch.float32), b.to(torch.float32))
    temp1 = o * scale_a.view(-1, 1)
    temp2 = temp1 * scale_b.view(1, -1)
    final = temp2.to(out_dtype)
    if bias is not None:
        final = final + bias.view(1, -1)
    return final


def _test_accuracy_once(M, N, K, with_bias, out_dtype, device):
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min
    a_fp32 = (torch.rand(M, K, dtype=torch.float32, device=device) - 0.5) * 2 * fp8_max
    a_fp8 = a_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
    b_fp32 = (torch.rand(N, K, dtype=torch.float32, device=device) - 0.5) * 2 * fp8_max
    b_fp8 = b_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
    scale_a = torch.randn((M,), device=device, dtype=torch.float32) * 0.001
    scale_b = torch.randn((N,), device=device, dtype=torch.float32) * 0.001
    bias = torch.randn((N,), device=device, dtype=out_dtype) if with_bias else None
    b_fp8 = b_fp8.t()
    o = torch_scaled_mm(a_fp8, b_fp8, scale_a, scale_b, out_dtype, bias)
    o1 = fp8_per_tensor_scaled_mm(a_fp8, b_fp8, scale_a, scale_b, out_dtype, bias)
    torch.testing.assert_close(o, o1, rtol=0.02, atol=1)


@pytest.mark.parametrize("M", [1, 128, 512, 1024, 4096])
@pytest.mark.parametrize("N", [16, 128, 512, 1024, 4096])
@pytest.mark.parametrize("K", [512, 1024, 4096, 8192, 16384])
@pytest.mark.parametrize("with_bias", [True, False])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_accuracy(M, N, K, with_bias, out_dtype):
    _test_accuracy_once(M, N, K, with_bias, out_dtype, "cuda")


# (M, N) shapes that exercise each dispatch bucket / boundary. K is varied
# separately below so every (M, N) is tested across multiple K values.
SM90_SWAP_AB_MN_SHAPES = [
    (1, 128),
    (1, 4096),
    (8, 1024),
    (8, 8192),
    (16, 1280),
    (16, 8192),
    (17, 128),
    (17, 4096),
    (32, 1024),
    (32, 8192),
    (64, 1280),
    (64, 8192),
    (65, 4096),
    (96, 4096),
    (128, 4096),
    # Cluster-misaligned M_orig in the M64_smallN bucket (TileN=16, cluster_N=4).
    # For M_orig in {17, 20, 33, 48}, grid_N = ceil(M_orig/16) in {2, 2, 3, 3},
    # not a multiple of cluster_N=4. Explicit coverage so any can_implement
    # failure or silent miscompute surfaces here.
    (20, 128),
    (20, 1024),
    (20, 1280),
    (33, 128),
    (33, 1024),
    (33, 1280),
    (48, 128),
    (48, 1024),
    (48, 1280),
]


@pytest.mark.parametrize(
    "shape_mn", SM90_SWAP_AB_MN_SHAPES, ids=lambda s: f"M{s[0]}_N{s[1]}"
)
@pytest.mark.parametrize("K", [2048, 4096, 8192])
@pytest.mark.parametrize("with_bias", [True, False])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_accuracy_sm90_swap_ab(shape_mn, K, with_bias, out_dtype):
    M, N = shape_mn
    _test_accuracy_once(M, N, K, with_bias, out_dtype, "cuda")


# Both sides of every SM120 M-bucket edge. M<=16 and M<=32 use a custom small
# EpilogueTile, so an off-by-one in the bucketing shows up as a wrong result or
# a can_implement failure rather than just a slowdown.
SM120_M_BUCKET_EDGES = [1, 2, 15, 16, 17, 31, 32, 33, 63, 255, 256, 257, 512]


@pytest.mark.parametrize("M", SM120_M_BUCKET_EDGES)
@pytest.mark.parametrize("N", [256, 4608])
@pytest.mark.parametrize("K", [4096, 6656])
@pytest.mark.parametrize("with_bias", [True, False])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_accuracy_sm120_m_buckets(M, N, K, with_bias, out_dtype):
    _test_accuracy_once(M, N, K, with_bias, out_dtype, "cuda")


PRODUCTION_LIKE_FP8_GEMM_CASES = [
    (189, 4608, 8192, False, torch.bfloat16),
    (3330, 256, 8192, False, torch.bfloat16),
    (17, 9216, 2048, False, torch.bfloat16),
]


@pytest.mark.parametrize(
    "M,N,K,with_bias,out_dtype",
    PRODUCTION_LIKE_FP8_GEMM_CASES,
)
def test_accuracy_production_like_shapes(M, N, K, with_bias, out_dtype):
    _test_accuracy_once(M, N, K, with_bias, out_dtype, "cuda")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
