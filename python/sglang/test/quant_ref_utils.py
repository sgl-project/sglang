"""Hand-written quantization-format references for backend parity UTs.

Deliberately independent of sglang.srt -- never replace with srt imports;
the tests use these to check srt.
"""

import torch

FLOAT8_E4M3_MAX = 448.0
FLOAT4_E2M1_MAX = 6.0

kE2M1ToFloat = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


def convert_swizzled_to_linear(a_sf_swizzled: torch.Tensor, m, k, block_size=16):
    m_tiles = (m + 128 - 1) // 128
    f = block_size * 4
    k_tiles = (k + f - 1) // f
    tmp = torch.reshape(a_sf_swizzled, (1, m_tiles, k_tiles, 32, 4, 4))
    tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
    out = tmp.reshape(m_tiles * 128, k_tiles * f // block_size)
    # Crop the K-tile padding too: k // block_size scale columns, not k.
    return out[0:m, 0 : k // block_size]


def break_fp4_bytes(a, dtype=torch.float32):
    assert a.dtype == torch.uint8
    m, n = a.shape
    a_flat = a.flatten()
    high = (a_flat & 0xF0) >> 4
    low = a_flat & 0x0F
    combined = torch.stack((low, high), dim=1).flatten()
    signs = (combined & 0x08).to(torch.bool)
    abs_vals = (combined & 0x07).to(torch.long)
    kE2M1 = kE2M1ToFloat.to(device=a.device)
    values = kE2M1[abs_vals] * torch.where(signs, -1.0, 1.0)
    return values.reshape(m, n * 2).to(dtype=dtype)


def dequantize_nvfp4_to_dtype(
    tensor_fp4, tensor_sf, global_scale, dtype, block_size=16
):
    assert tensor_fp4.dtype == torch.uint8
    m, packed_k = tensor_fp4.shape
    k = packed_k * 2
    tensor_f32 = break_fp4_bytes(tensor_fp4, torch.float32)
    tensor_f32 = tensor_f32.reshape(m, k // block_size, block_size)
    tensor_sf = tensor_sf.view(torch.float8_e4m3fn)
    tensor_sf = convert_swizzled_to_linear(tensor_sf, m, k, block_size)
    tensor_sf_dtype = tensor_sf.to(torch.float32) / global_scale
    out = (tensor_f32 * tensor_sf_dtype.unsqueeze(-1)).reshape(m, k)
    return out.to(dtype=dtype)


def quantize_nvfp4_shard(w: torch.Tensor, gs=None):
    """NVFP4-quantize one checkpoint shard; returns (packed, linear sf,
    global scale, fp32 dequant reference)."""
    from flashinfer import fp4_quantize

    n, k = w.shape
    if gs is None:
        gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w.abs().max().to(torch.float32)
    w_q, w_sf_swizzled = fp4_quantize(w, gs)
    sf_linear = convert_swizzled_to_linear(
        w_sf_swizzled.view(torch.float8_e4m3fn), n, k, 16
    )
    w_dequant = dequantize_nvfp4_to_dtype(w_q, w_sf_swizzled, gs, torch.float32)
    return w_q, sf_linear, gs, w_dequant


# e2m1 magnitude code = how many of these a value passes: round half to even, saturating
# at FLOAT4_E2M1_MAX
_E2M1_ABOVE = (0.25, 1.25, 2.5, 5.0)
_E2M1_AT_OR_ABOVE = (0.75, 1.75, 3.5)


def nvfp4_kv_quantize(
    x: torch.Tensor, global_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """NVFP4 KV quantize in `NVFP4KVQuantizeUtil`'s layout without FlashInfer: ``[R, D]``
    bf16/fp16 -> (``[R, D//2]`` uint8 payload, even element in the low nibble,
    ``[R, D//16]`` uint8 E4M3 scales)."""
    rows, dim = x.shape
    assert dim % 16 == 0
    gs = global_scale.float().reshape(())
    blocks = x.float().view(rows, dim // 16, 16)
    amax = blocks.abs().amax(-1)
    # saturate at E4M3's max like the hardware conversion; torch's cast overflows to NaN
    scale = (
        (amax / (FLOAT4_E2M1_MAX * gs))
        .clamp(max=FLOAT8_E4M3_MAX)
        .to(torch.float8_e4m3fn)
    )
    inv = torch.where(amax != 0, 1.0 / (scale.float() * gs), 0.0)
    v = blocks * inv[..., None]
    a = v.abs()
    mag = sum((a > t).int() for t in _E2M1_ABOVE) + sum(
        (a >= t).int() for t in _E2M1_AT_OR_ABOVE
    )
    codes = (mag | torch.where(v < 0, 8, 0)).to(torch.uint8).view(rows, dim // 2, 2)
    return codes[..., 0] | (codes[..., 1] << 4), scale.view(torch.uint8)
