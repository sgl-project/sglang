import pytest
import torch
from sgl_kernel import cutlass_scaled_fp4_mm, scaled_fp4_quant

if torch.cuda.get_device_capability() < (10, 0):
    pytest.skip(
        reason="Nvfp4 Requires compute capability of 10 or above.",
        allow_module_level=True,
    )

DTYPES = [torch.float16, torch.bfloat16]
# m, n, k
SHAPES = [(128, 128, 64), (128, 128, 128), (256, 128, 64), (128, 256, 128)]
PAD_SHAPES = [(150, 128, 64), (128, 128, 96)]
SHAPES.extend(PAD_SHAPES)

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

kE2M1ToFloatArray = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
]


def e2m1_to_fp32(int4_value):
    signBit = int4_value & 0x8
    int4_absValue = int4_value & 0x7
    float_result = kE2M1ToFloatArray[int4_absValue]
    if signBit:
        float_result = -float_result
    return float_result


def break_fp4_bytes(a, dtype):
    assert a.dtype == torch.uint8
    m, n = a.shape
    a = a.flatten()
    # Get upper 4 bits
    highHalfByte = (a & 0xF0) >> 4
    # Get lower 4 bits
    lowHalfByte = a & 0x0F
    fH = torch.tensor([e2m1_to_fp32(x) for x in highHalfByte]).to(a.device)
    fL = torch.tensor([e2m1_to_fp32(x) for x in lowHalfByte]).to(a.device)
    # [0xAB, 0xCD] -> [0xB, 0xA, 0xD, 0xC]
    out = torch.stack((fL, fH), dim=-1).reshape(m, n * 2)
    return out


def convert_swizzled_to_linear(a_sf_swizzled: torch.Tensor, m, k, block_size):
    sf_m, sf_k = a_sf_swizzled.shape
    m_tiles = (m + 128 - 1) // 128
    f = block_size * 4
    k_tiles = (k + f - 1) // f
    tmp = torch.reshape(a_sf_swizzled, (1, m_tiles, k_tiles, 32, 4, 4))
    tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
    out = tmp.reshape(m_tiles * 128, k_tiles * f // block_size)
    return out[0:m, 0:k]


def dequantize_to_dtype(
    tensor_fp4, tensor_sf, global_scale, dtype, device, block_size=16
):
    """Dequantize the fp4 tensor back to high precision."""
    # Two fp4 values are packed into one uint8.
    assert tensor_fp4.dtype == torch.uint8
    m, packed_k = tensor_fp4.shape
    k = packed_k * 2
    tensor_f32 = break_fp4_bytes(tensor_fp4, dtype)
    tensor_f32 = tensor_f32.reshape(m, k // block_size, block_size)
    tensor_sf = tensor_sf.view(torch.float8_e4m3fn)
    tensor_sf = convert_swizzled_to_linear(tensor_sf, m, k, block_size)
    tensor_sf_dtype = tensor_sf.to(torch.float32) / global_scale

    # scale the tensor
    out = (tensor_f32 * tensor_sf_dtype.unsqueeze(-1)).reshape(m, k)
    return out


def get_ref_results(
    a_fp4,
    b_fp4,
    a_sf,
    b_sf,
    a_global_scale,
    b_global_scale,
    m,
    n,
    dtype,
    block_size,
    device,
):
    _, m_k = a_fp4.shape
    _, n_k = b_fp4.shape
    assert m_k == n_k
    a_in_dtype = dequantize_to_dtype(
        a_fp4, a_sf, a_global_scale, dtype=dtype, device=device, block_size=block_size
    )
    b_in_dtype = dequantize_to_dtype(
        b_fp4, b_sf, b_global_scale, dtype=dtype, device=device, block_size=block_size
    )
    return torch.matmul(a_in_dtype, b_in_dtype.t())


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@torch.inference_mode()
def test_nvfp4_gemm(
    dtype: torch.dtype,
    shape: tuple[int, int],
) -> None:
    m, n, packed_k = shape
    k = packed_k * 2
    block_size = 16
    a_dtype = torch.randn((m, k), dtype=dtype, device="cuda")
    b_dtype = torch.randn((n, k), dtype=dtype, device="cuda")

    a_global_scale = (
        (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(a_dtype.flatten(), dim=-1)
    ).to(torch.float32)
    b_global_scale = (
        (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(b_dtype.flatten(), dim=-1)
    ).to(torch.float32)
    alpha = 1.0 / (a_global_scale * b_global_scale)
    a_fp4, a_scale_interleaved = scaled_fp4_quant(a_dtype, a_global_scale)
    b_fp4, b_scale_interleaved = scaled_fp4_quant(b_dtype, b_global_scale)

    expected_out = get_ref_results(
        a_fp4,
        b_fp4,
        a_scale_interleaved,
        b_scale_interleaved,
        a_global_scale,
        b_global_scale,
        m,
        n,
        dtype,
        block_size,
        "cuda",
    )
    out = cutlass_scaled_fp4_mm(
        a_fp4, b_fp4, a_scale_interleaved, b_scale_interleaved, alpha, dtype
    )

    torch.testing.assert_close(out, expected_out.to(dtype=dtype), atol=1e-1, rtol=1e-1)
