import pytest
import torch
from sgl_kernel import (
    scaled_fp4_grouped_quant,
    scaled_fp4_quant,
    silu_and_mul,
    silu_and_mul_scaled_fp4_grouped_quant,
)

skip_condition = torch.cuda.get_device_capability() < (10, 0)

DTYPES = [torch.float16, torch.bfloat16]
SHAPES = [(128, 64), (128, 128), (256, 64), (256, 128)]
PAD_SHAPES = [
    (90, 64),
    (150, 64),
    (128, 48),
    (128, 80),
    (150, 80),
    (90, 48),
    (90, 128),
    (150, 128),
    (150, 48),
    (90, 80),
]

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

# E2M1 to float
# 0111 -> 6
# 0110 -> 4
# 0101 -> 3
# 0100 -> 2
# 0011 -> 1.5
# 0010 -> 1
# 0001 -> 0.5
# 0000 -> 0
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
BLOCK_SIZE = 16


def cast_from_fp4(x, m, n):
    # The fp4 values are packed in uint8 as [v_1st | v_2nd]
    v_2nd = x & 0xF
    v_1st = (x >> 4) & 0xF
    c = torch.stack((v_2nd, v_1st), dim=-1)
    out = torch.tensor([E2M1_TO_FLOAT32[x] for x in c.flatten()])
    out = out.reshape(m, n).to(torch.float32)
    return out


def cast_to_fp4(x):
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
    elif isinstance(x, (float, int)):
        return 0.0 if x == 0 else 1.0 / x
    else:
        raise TypeError("Input must be a float, int, or a torch.Tensor.")


def ref_nvfp4_quant(x, global_scale):
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


def recover_swizzled_scales(scale, m, n):
    rounded_m = ((m + 128 - 1) // 128) * 128
    scale_n = n // BLOCK_SIZE
    rounded_n = ((scale_n + 4 - 1) // 4) * 4
    # Recover the swizzled scaling factor to linear layout
    tmp = torch.reshape(scale, (1, rounded_m // 128, rounded_n // 4, 32, 4, 4))
    tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
    result = torch.reshape(tmp, (rounded_m, rounded_n)).to(torch.float32)
    return result[:m, :scale_n]


@pytest.mark.skipif(
    skip_condition, reason="Nvfp4 Requires compute capability of 10 or above."
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@torch.inference_mode()
def test_quantize_to_fp4(
    dtype: torch.dtype,
    shape: tuple[int, int],
) -> None:
    torch.manual_seed(42)
    torch.set_default_device("cuda:0")

    m, n = shape

    x = torch.randn((m, n), dtype=dtype)
    tensor_amax = torch.abs(x).max().to(torch.float32)
    global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax
    out_ref, scale_ref = ref_nvfp4_quant(x, global_scale)

    out, out_scale = scaled_fp4_quant(x, global_scale)
    scale_ans = recover_swizzled_scales(out_scale, m, n)
    out_ans = cast_from_fp4(out, m, n)

    torch.testing.assert_close(out_ans, out_ref)
    torch.testing.assert_close(scale_ans, scale_ref)


@pytest.mark.skipif(
    skip_condition, reason="Nvfp4 Requires compute capability of 10 or above."
)
@pytest.mark.parametrize("pad_shape", PAD_SHAPES)
@torch.inference_mode()
def test_quantize_to_fp4_padded(pad_shape: tuple[int, int]) -> None:
    torch.manual_seed(42)
    dtype = torch.float16
    torch.set_default_device("cuda:0")

    m, n = pad_shape

    x = torch.randn((m, n), dtype=dtype)

    tensor_amax = torch.abs(x).max().to(torch.float32)
    global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax
    out_ref, scale_ref = ref_nvfp4_quant(x, global_scale)

    out, out_scale = scaled_fp4_quant(x, global_scale)

    scale_ans = recover_swizzled_scales(out_scale, m, n)
    out_ans = cast_from_fp4(out, m, n)

    torch.testing.assert_close(out_ans, out_ref)
    torch.testing.assert_close(scale_ans, scale_ref)


@pytest.mark.skipif(
    skip_condition, reason="Nvfp4 Requires compute capability of 10 or above."
)
@pytest.mark.parametrize("shape", [(2, 512, 2048), (2, 100, 128), (2, 128, 96)])
def test_quantize_to_fp4_grouped(shape):
    torch.manual_seed(42)
    torch.set_default_device("cuda:0")

    l, m, k = shape
    x = torch.randn((l, m, k), dtype=torch.bfloat16)
    max_m = m // 2
    assert max_m <= m
    mask = torch.randint(1, max_m, (l,), dtype=torch.int32)
    tensor_amax = x.abs().amax(dim=(1, 2)).to(torch.float32)
    x_sf_global = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax
    output, output_scales = scaled_fp4_grouped_quant(
        x,
        x_sf_global,
        mask,
    )
    # output in logical (m, k, l), but its physical layout is (l, m, k).
    # So permute first to (l, m, k).
    output = output.permute(2, 0, 1)
    # output_scale in logical (32, 4, rm, 4, rk, l), but its physical layout is (l, rm, rk, 32, 4, 4).
    # So permute first to (l, rm, rk, 32, 4, 4).
    padded_m = ((m + 128 - 1) // 128) * 128
    output_scales = output_scales.permute(5, 2, 4, 0, 1, 3).view(l, padded_m, -1)
    for i in range(l):
        a_fp4, a_scale_interleaved = scaled_fp4_quant(x[i], x_sf_global[i])
        torch.testing.assert_close(a_fp4[: mask[i]], output[i][: mask[i]])
        # Recover swizzled scales to linear layout and drop padded values, so
        # no extra checks on padding are needed.
        scale_ref = recover_swizzled_scales(a_scale_interleaved, m, k)
        scale_ans = recover_swizzled_scales(output_scales[i], m, k)
        torch.testing.assert_close(scale_ref[: mask[i]], scale_ans[: mask[i]])


@pytest.mark.skipif(
    skip_condition, reason="Nvfp4 Requires compute capability of 10 or above."
)
@pytest.mark.parametrize("shape", [(32, 100, 2048), (32, 512, 2048), (6, 6144, 2048)])
def test_silu_and_mul_quantize_to_fp4_grouped(shape):
    torch.manual_seed(42)
    torch.set_default_device("cuda:0")

    l, m, k = shape
    x = torch.randn((l, m, k * 2), dtype=torch.bfloat16)
    max_m = m // 2
    assert max_m <= m
    mask = torch.randint(1, max_m, (l,), dtype=torch.int32)

    ref_y = silu_and_mul(x)
    tensor_amax = ref_y.abs().amax(dim=(1, 2)).to(torch.float32)
    y_sf_global = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax
    ref_output, ref_output_scales = scaled_fp4_grouped_quant(
        ref_y,
        y_sf_global,
        mask,
    )
    output, output_scales = silu_and_mul_scaled_fp4_grouped_quant(
        x,
        y_sf_global,
        mask,
    )

    # output in logical (m, k, l), but its physical layout is (l, m, k).
    # So permute first to (l, m, k).
    output = output.permute(2, 0, 1)
    ref_output = ref_output.permute(2, 0, 1)

    # output_scale in logical (32, 4, rm, 4, rk, l), but its physical layout is (l, rm, rk, 32, 4, 4).
    # So permute first to (l, rm, rk, 32, 4, 4).
    padded_m = ((m + 128 - 1) // 128) * 128
    output_scales = output_scales.permute(5, 2, 4, 0, 1, 3).view(l, padded_m, -1)
    ref_output_scales = ref_output_scales.permute(5, 2, 4, 0, 1, 3).view(
        l, padded_m, -1
    )

    for i in range(l):
        torch.testing.assert_close(ref_output[i, : mask[i]], output[i, : mask[i]])
        # We need to recover the swizzled scales to linear layout before applying mask slice.
        scale_ref = recover_swizzled_scales(ref_output_scales[i], m, k)
        scale_ans = recover_swizzled_scales(output_scales[i], m, k)
        torch.testing.assert_close(scale_ref[: mask[i]], scale_ans[: mask[i]])


if __name__ == "__main__":
    pytest.main([__file__])
