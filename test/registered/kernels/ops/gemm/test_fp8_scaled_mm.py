import itertools
import sys

import pytest
import torch

from sglang.kernels.jit.utils import get_ci_test_range, get_jit_cuda_arch
from sglang.kernels.ops.gemm.fp8_scaled_mm import fp8_scaled_mm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
# Nightly expands the shape sweep through SGLANG_JIT_KERNEL_RUN_FULL_TESTS.
register_cuda_ci(est_time=300, stage="nightly", runner_config="1-gpu-large")


def _supported_arch() -> bool:
    if not torch.cuda.is_available():
        return False
    arch = get_jit_cuda_arch()
    return (arch.major, arch.minor) == (8, 9) or arch.major in (9, 10, 12)


pytestmark = pytest.mark.skipif(
    not _supported_arch(),
    reason="fp8_scaled_mm requires SM89, SM90, SM100/103, or SM120",
)


def _run_case(m, n, k, with_bias, out_dtype, scalar_a=False):
    finfo = torch.finfo(torch.float8_e4m3fn)
    a = (
        torch.randn(m, k, device="cuda", dtype=torch.float32)
        .clamp(finfo.min, finfo.max)
        .to(torch.float8_e4m3fn)
    )
    b = (
        torch.randn(n, k, device="cuda", dtype=torch.float32)
        .clamp(finfo.min, finfo.max)
        .to(torch.float8_e4m3fn)
        .t()
    )
    scales_a = torch.full((1 if scalar_a else m,), 0.03125, device="cuda")
    scales_b = torch.rand(n, device="cuda", dtype=torch.float32) * 0.05
    bias = torch.randn(n, device="cuda", dtype=out_dtype) if with_bias else None

    expected = (a.float() @ b.float()) * scales_a.reshape(-1, 1) * scales_b
    expected = expected.to(out_dtype)
    if bias is not None:
        expected = expected + bias

    actual = fp8_scaled_mm(a, b, scales_a, scales_b, out_dtype, bias)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=1)
    return actual, (a, b, scales_a, scales_b, bias)


GENERAL_CASES = get_ci_test_range(
    list(
        itertools.product(
            [1, 17, 128, 512, 1024, 4096],
            [16, 128, 512, 1024, 4096],
            [512, 1024, 4096, 8192, 16384],
            [False, True],
            [torch.bfloat16, torch.float16],
        )
    ),
    [
        (1, 128, 512, False, torch.bfloat16),
        (17, 1024, 2048, True, torch.float16),
        (128, 4096, 4096, False, torch.bfloat16),
        (512, 512, 8192, True, torch.bfloat16),
        (1024, 128, 1024, False, torch.float16),
    ],
)


@pytest.mark.parametrize("m,n,k,with_bias,out_dtype", GENERAL_CASES)
def test_fp8_scaled_mm(m, n, k, with_bias, out_dtype):
    _run_case(m, n, k, with_bias, out_dtype)


SM90_BOUNDARY_MN_SHAPES = [
    (1, 128),
    (1, 4096),
    (8, 1024),
    (8, 8192),
    (16, 1280),
    (16, 8192),
    (17, 128),
    (17, 4096),
    (20, 128),
    (20, 1024),
    (20, 1280),
    (32, 1024),
    (32, 8192),
    (33, 128),
    (33, 1024),
    (33, 1280),
    (48, 128),
    (48, 1024),
    (48, 1280),
    (64, 1280),
    (64, 8192),
    (65, 4096),
    (96, 4096),
    (128, 4096),
]

SM90_BOUNDARY_CASES = get_ci_test_range(
    list(
        itertools.product(
            SM90_BOUNDARY_MN_SHAPES,
            [2048, 4096, 8192],
        )
    ),
    [((1, 4096), 2048), ((20, 1024), 4096), ((33, 1280), 2048), ((65, 4096), 4096)],
)


@pytest.mark.parametrize("shape_mn,k", SM90_BOUNDARY_CASES)
def test_fp8_scaled_mm_sm90_dispatch_boundaries(shape_mn, k):
    if get_jit_cuda_arch().major != 9:
        pytest.skip("SM90-specific dispatch coverage")
    _run_case(*shape_mn, k, True, torch.bfloat16)


@pytest.mark.parametrize("m", [1, 2, 16, 64, 189])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_fp8_scaled_mm_scalar_a_scale(m, with_bias, out_dtype):
    if get_jit_cuda_arch().major not in (9, 10, 12):
        pytest.skip("scalar A scales with M > 1 require SM90+")
    actual, inputs = _run_case(m, 6144, 4096, with_bias, out_dtype, scalar_a=True)
    a, b, scale_a, scale_b, bias = inputs
    repeated = fp8_scaled_mm(a, b, scale_a.repeat(m), scale_b, out_dtype, bias)
    torch.testing.assert_close(actual, repeated, rtol=0, atol=0)


@pytest.mark.parametrize(
    "m,n,k",
    [(189, 4608, 8192), (3330, 256, 8192), (17, 9216, 2048)],
)
def test_fp8_scaled_mm_production_shapes(m, n, k):
    _run_case(m, n, k, False, torch.bfloat16)


def test_fp8_scaled_mm_rejects_invalid_a_scale_count():
    m, n, k = 8, 128, 512
    a = torch.randn(m, k, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn).t()
    with pytest.raises(RuntimeError, match="scales_a must contain either"):
        fp8_scaled_mm(
            a,
            b,
            torch.ones(2, device="cuda"),
            torch.ones(n, device="cuda"),
            torch.bfloat16,
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
