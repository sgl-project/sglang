import sys

import pytest
import torch

from sglang.jit_kernel.nvfp4 import (
    scaled_fp4_grouped_quant,
    scaled_fp4_quant,
    silu_and_mul_scaled_fp4_grouped_quant,
)

try:
    from sgl_kernel import silu_and_mul as _sgl_silu_and_mul
except Exception:
    _sgl_silu_and_mul = None

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="stage-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=120, suite="nightly-kernel-1-gpu", nightly=True)


def _nvfp4_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (10, 0)


def _silu_and_mul_reference(x: torch.Tensor) -> torch.Tensor:
    if _sgl_silu_and_mul is not None:
        return _sgl_silu_and_mul(x)
    k = x.shape[-1] // 2
    return torch.nn.functional.silu(x[:, :, :k]) * x[:, :, k:]


DTYPES = [torch.float16, torch.bfloat16]
SHAPES = [(128, 64), (128, 128), (256, 64), (256, 128)]
PAD_SHAPES = [
    (90, 64),
    (150, 64),
    (128, 48),
    (128, 80),
]

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
BLOCK_SIZE = 16

E2M1_TO_FLOAT32 = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def cast_from_fp4(x: torch.Tensor, m: int, n: int) -> torch.Tensor:
    v_2nd = (x & 0xF).to(torch.long)
    v_1st = ((x >> 4) & 0xF).to(torch.long)
    c = torch.stack((v_2nd, v_1st), dim=-1).flatten()
    lut = torch.tensor(E2M1_TO_FLOAT32, device=x.device, dtype=torch.float32)
    return lut[c].reshape(m, n)


def cast_to_fp4(x: torch.Tensor) -> torch.Tensor:
    sign = torch.sign(x)
    x = torch.abs(x)
    x[(x >= 0.0) & (x <= 0.25)] = 0.0
    x[(x > 0.25) & (x < 0.75)] = 0.5
    x[(x >= 0.75) & (x <= 1.25)] = 1.0
    x[(x > 1.25) & (x < 1.75)] = 1.5
    x[(x >= 1.75) & (x <= 2.5)] = 2.0
    x[(x > 2.5) & (x < 3.5)] = 3.0
    x[(x >= 3.5) & (x <= 5.0)] = 4.0
    x[x > 5.0] = 6.0
    return x * sign


def get_reciprocal(x):
    if isinstance(x, torch.Tensor):
        return torch.where(x == 0, torch.tensor(0.0, dtype=x.dtype), 1.0 / x)
    return 0.0 if x == 0 else 1.0 / x


def ref_nvfp4_quant(x: torch.Tensor, global_scale: torch.Tensor):
    assert global_scale.dtype == torch.float32
    assert x.ndim == 2
    m, n = x.shape
    x = torch.reshape(x, (m, n // BLOCK_SIZE, BLOCK_SIZE))
    vec_max = torch.max(torch.abs(x), dim=-1, keepdim=True)[0].to(torch.float32)
    scale = global_scale * (vec_max * get_reciprocal(FLOAT4_E2M1_MAX))
    scale = scale.to(torch.float8_e4m3fn).to(torch.float32)
    output_scale = get_reciprocal(scale * get_reciprocal(global_scale))

    scaled_x = x.to(torch.float32) * output_scale
    clipped_x = torch.clamp(scaled_x, -6.0, 6.0).reshape(m, n)
    return cast_to_fp4(clipped_x), scale.squeeze(-1)


def recover_swizzled_scales(scale: torch.Tensor, m: int, n: int) -> torch.Tensor:
    rounded_m = ((m + 128 - 1) // 128) * 128
    scale_n = n // BLOCK_SIZE
    rounded_n = ((scale_n + 4 - 1) // 4) * 4
    tmp = torch.reshape(scale, (1, rounded_m // 128, rounded_n // 4, 32, 4, 4))
    tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
    result = torch.reshape(tmp, (rounded_m, rounded_n)).to(torch.float32)
    return result[:m, :scale_n]


@pytest.mark.skipif(
    not _nvfp4_supported(), reason="NVFP4 requires compute capability >= 10.0"
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_quantize_to_fp4(dtype: torch.dtype, shape: tuple[int, int]) -> None:
    torch.manual_seed(42)
    m, n = shape

    x = torch.randn((m, n), dtype=dtype, device="cuda")
    tensor_amax = torch.abs(x).max().to(torch.float32)
    global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax
    out_ref, scale_ref = ref_nvfp4_quant(x, global_scale)

    out, out_scale = scaled_fp4_quant(x, global_scale)
    scale_ans = recover_swizzled_scales(out_scale, m, n)
    out_ans = cast_from_fp4(out, m, n)

    torch.testing.assert_close(out_ans, out_ref)
    torch.testing.assert_close(scale_ans, scale_ref)


@pytest.mark.skipif(
    not _nvfp4_supported(), reason="NVFP4 requires compute capability >= 10.0"
)
@pytest.mark.parametrize("shape", PAD_SHAPES)
def test_quantize_to_fp4_padded(shape: tuple[int, int]) -> None:
    torch.manual_seed(42)
    m, n = shape
    x = torch.randn((m, n), dtype=torch.float16, device="cuda")

    tensor_amax = torch.abs(x).max().to(torch.float32)
    global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax
    out_ref, scale_ref = ref_nvfp4_quant(x, global_scale)

    out, out_scale = scaled_fp4_quant(x, global_scale)
    scale_ans = recover_swizzled_scales(out_scale, m, n)
    out_ans = cast_from_fp4(out, m, n)

    torch.testing.assert_close(out_ans, out_ref)
    torch.testing.assert_close(scale_ans, scale_ref)


@pytest.mark.skipif(
    not _nvfp4_supported(), reason="NVFP4 requires compute capability >= 10.0"
)
@pytest.mark.parametrize("shape", [(2, 128, 512), (2, 100, 128)])
def test_quantize_to_fp4_grouped(shape: tuple[int, int, int]) -> None:
    torch.manual_seed(42)
    l, m, k = shape

    x = torch.randn((l, m, k), dtype=torch.bfloat16, device="cuda")
    mask = torch.randint(1, max(2, m // 2), (l,), dtype=torch.int32, device="cuda")
    tensor_amax = x.abs().amax(dim=(1, 2)).to(torch.float32)
    x_sf_global = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax

    output, output_scales = scaled_fp4_grouped_quant(x, x_sf_global, mask)
    output = output.permute(2, 0, 1)
    padded_m = ((m + 128 - 1) // 128) * 128
    output_scales = output_scales.permute(5, 2, 4, 0, 1, 3).view(l, padded_m, -1)

    for i in range(l):
        a_fp4, a_scale_interleaved = scaled_fp4_quant(x[i], x_sf_global[i])
        torch.testing.assert_close(a_fp4[: mask[i]], output[i][: mask[i]])
        scale_ref = recover_swizzled_scales(a_scale_interleaved, m, k)
        scale_ans = recover_swizzled_scales(output_scales[i], m, k)
        torch.testing.assert_close(scale_ref[: mask[i]], scale_ans[: mask[i]])


@pytest.mark.skipif(
    not _nvfp4_supported(), reason="NVFP4 requires compute capability >= 10.0"
)
@pytest.mark.parametrize("shape", [(4, 96, 256), (8, 128, 512)])
def test_silu_and_mul_quantize_to_fp4_grouped(shape: tuple[int, int, int]) -> None:
    torch.manual_seed(42)
    l, m, k = shape

    x = torch.randn((l, m, k * 2), dtype=torch.bfloat16, device="cuda")
    mask = torch.randint(1, max(2, m // 2), (l,), dtype=torch.int32, device="cuda")

    ref_y = _silu_and_mul_reference(x)

    tensor_amax = ref_y.abs().amax(dim=(1, 2)).to(torch.float32)
    y_sf_global = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax

    ref_output, ref_output_scales = scaled_fp4_grouped_quant(ref_y, y_sf_global, mask)
    output, output_scales = silu_and_mul_scaled_fp4_grouped_quant(x, y_sf_global, mask)

    output = output.permute(2, 0, 1)
    ref_output = ref_output.permute(2, 0, 1)

    padded_m = ((m + 128 - 1) // 128) * 128
    output_scales = output_scales.permute(5, 2, 4, 0, 1, 3).view(l, padded_m, -1)
    ref_output_scales = ref_output_scales.permute(5, 2, 4, 0, 1, 3).view(
        l, padded_m, -1
    )

    for i in range(l):
        torch.testing.assert_close(ref_output[i, : mask[i]], output[i, : mask[i]])
        scale_ref = recover_swizzled_scales(ref_output_scales[i], m, k)
        scale_ans = recover_swizzled_scales(output_scales[i], m, k)
        torch.testing.assert_close(scale_ref[: mask[i]], scale_ans[: mask[i]])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
