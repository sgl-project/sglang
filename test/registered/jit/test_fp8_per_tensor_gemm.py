import sys

import pytest
import torch

from sglang.kernels.ops.gemm.fp8_per_tensor_gemm import fp8_per_tensor_scaled_mm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def _cuda_version_at_least(major, minor):
    if torch.version.cuda is None:
        return False
    version = tuple(int(component) for component in torch.version.cuda.split(".")[:2])
    return version >= (major, minor)


def _native_scalar_a_supported():
    if not torch.cuda.is_available():
        return False
    capability = torch.cuda.get_device_capability()
    if capability == (9, 0):
        return _cuda_version_at_least(12, 0)
    return capability[0] in (10, 12) and _cuda_version_at_least(12, 8)


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
    torch.testing.assert_close(o, o1, rtol=0.02, atol=2)


def _test_scalar_a_accuracy_once(M, N, K, with_bias, out_dtype, device):
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    a_fp8 = (
        torch.randn(M, K, dtype=torch.float32, device=device)
        .clamp(min=fp8_info.min, max=fp8_info.max)
        .to(torch.float8_e4m3fn)
    )
    b_fp8 = (
        torch.randn(N, K, dtype=torch.float32, device=device)
        .clamp(min=fp8_info.min, max=fp8_info.max)
        .to(torch.float8_e4m3fn)
        .t()
    )
    scale_a = torch.tensor([0.03125], device=device, dtype=torch.float32)
    scale_a_repeated = scale_a.repeat(M)

    # Resemble merged projections whose component matrices were quantized with
    # different tensorwise scales before concatenation.
    scale_b = torch.empty(N, device=device, dtype=torch.float32)
    first_boundary = N // 3
    second_boundary = 2 * N // 3
    scale_b[:first_boundary] = 0.015625
    scale_b[first_boundary:second_boundary] = 0.03125
    scale_b[second_boundary:] = 0.0625

    bias = torch.randn(N, device=device, dtype=out_dtype) if with_bias else None
    expected = torch_scaled_mm(a_fp8, b_fp8, scale_a, scale_b, out_dtype, bias)
    actual = fp8_per_tensor_scaled_mm(a_fp8, b_fp8, scale_a, scale_b, out_dtype, bias)
    repeated = fp8_per_tensor_scaled_mm(
        a_fp8, b_fp8, scale_a_repeated, scale_b, out_dtype, bias
    )

    torch.testing.assert_close(expected, actual, rtol=0.02, atol=1)
    torch.testing.assert_close(repeated, actual, rtol=0, atol=0)


@pytest.mark.parametrize("M", [1, 128, 512, 1024, 4096])
@pytest.mark.parametrize("N", [16, 128, 512, 1024, 4096])
@pytest.mark.parametrize("K", [512, 1024, 4096, 8192, 16384])
@pytest.mark.parametrize("with_bias", [True, False])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_accuracy(M, N, K, with_bias, out_dtype):
    _test_accuracy_once(M, N, K, with_bias, out_dtype, "cuda")


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


SM120_M_BUCKET_EDGES = [1, 2, 15, 16, 17, 31, 32, 33, 63, 255, 256, 257, 512]


@pytest.mark.parametrize("M", SM120_M_BUCKET_EDGES)
@pytest.mark.parametrize("N", [256, 4608])
@pytest.mark.parametrize("K", [4096, 6656])
@pytest.mark.parametrize("with_bias", [True, False])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_accuracy_sm120_m_buckets(M, N, K, with_bias, out_dtype):
    _test_accuracy_once(M, N, K, with_bias, out_dtype, "cuda")


@pytest.mark.skipif(
    not _native_scalar_a_supported(),
    reason="native scalar A scales require a compatible SM90, SM100, or SM120 build",
)
@pytest.mark.parametrize("M", [1, 2, 8, 16, 64, 189])
@pytest.mark.parametrize("with_bias", [True, False])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_scalar_a_channelwise_b(M, with_bias, out_dtype):
    _test_scalar_a_accuracy_once(M, 6144, 4096, with_bias, out_dtype, "cuda")


def test_rejects_invalid_a_scale_count():
    M, N, K = 8, 128, 512
    a = torch.randn(M, K, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(N, K, device="cuda").to(torch.float8_e4m3fn).t()
    scale_a = torch.ones(2, device="cuda", dtype=torch.float32)
    scale_b = torch.ones(N, device="cuda", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="scales_a must contain either"):
        fp8_per_tensor_scaled_mm(a, b, scale_a, scale_b, torch.bfloat16, None)


@pytest.mark.parametrize("off_device_arg", ["scales_a", "scales_b", "bias"])
def test_rejects_off_device_operands(off_device_arg):
    M, N, K = 8, 128, 512
    a = torch.randn(M, K, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(N, K, device="cuda").to(torch.float8_e4m3fn).t()
    operands = {
        "scales_a": torch.ones(M, device="cuda", dtype=torch.float32),
        "scales_b": torch.ones(N, device="cuda", dtype=torch.float32),
        "bias": torch.zeros(N, device="cuda", dtype=torch.bfloat16),
    }
    operands[off_device_arg] = operands[off_device_arg].cpu()

    with pytest.raises(
        RuntimeError, match=f"{off_device_arg} must be a CUDA tensor on"
    ):
        fp8_per_tensor_scaled_mm(
            a,
            b,
            operands["scales_a"],
            operands["scales_b"],
            torch.bfloat16,
            operands["bias"],
        )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9),
    reason="SM89-specific scalar A validation",
)
def test_rejects_scalar_a_with_multiple_rows_on_sm89():
    M, N, K = 8, 128, 512
    a = torch.randn(M, K, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(N, K, device="cuda").to(torch.float8_e4m3fn).t()
    scale_a = torch.ones(1, device="cuda", dtype=torch.float32)
    scale_b = torch.ones(N, device="cuda", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="scalar scales_a with M > 1 is unsupported"):
        fp8_per_tensor_scaled_mm(a, b, scale_a, scale_b, torch.bfloat16, None)


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
