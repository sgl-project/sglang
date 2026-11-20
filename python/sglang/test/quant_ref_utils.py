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


# An e2m1 magnitude code counts the thresholds a value passes: round half to even,
# saturating at FLOAT4_E2M1_MAX.
_E2M1_ABOVE = (0.25, 1.25, 2.5, 5.0)
_E2M1_AT_OR_ABOVE = (0.75, 1.75, 3.5)


def nvfp4_kv_quantize(
    x: torch.Tensor, global_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """NVFP4 KV quantize in ``NVFP4KVQuantizeUtil``'s layout without FlashInfer.

    ``[R, D]`` bf16/fp16 -> (``[R, D//2]`` uint8 payload, even element in the low
    nibble, ``[R, D//16]`` uint8 E4M3 scales).
    """
    rows, dim = x.shape
    assert dim % 16 == 0
    gs = global_scale.float().reshape(())
    blocks = x.float().view(rows, dim // 16, 16)
    amax = blocks.abs().amax(-1)
    # Saturate at E4M3's max like the hardware conversion; torch's cast overflows to NaN.
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


# The block-scaled (MX and NVFP4) row quantizers of the MoE LoRA kernels.
# Rounding midpoints between consecutive kE2M1ToFloat values.
_E2M1_MIDPOINTS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def e2m1_codes(magnitude: torch.Tensor) -> torch.Tensor:
    """Nearest E2M1 code; ties go to the even code, as the hardware cvt.rn does."""
    mid = _E2M1_MIDPOINTS.to(magnitude.device)
    idx = torch.bucketize(magnitude, mid)  # a tie lands on the lower code
    tie = (idx < 7) & (magnitude == mid[idx.clamp(max=6)]) & (idx % 2 == 1)
    return idx + tie.long()


def quantize_mx_ref(x: torch.Tensor, vec: int, elem: str):
    """[..., K] -> (data, scale bytes [..., K/vec] uint8, exact dequant fp32)."""
    *lead, k = x.shape
    groups = x.float().reshape(*lead, k // vec, vec)
    amax = groups.abs().amax(dim=-1)
    elem_max, elem_max_log2 = (6.0, 2) if elem == "e2m1" else (448.0, 8)
    # Smallest power of two with amax <= elem_max * 2**e, as the row quantizer.
    floor_log2 = torch.frexp(amax)[1].float() - 1
    exponent = floor_log2 - elem_max_log2
    exponent = exponent + (amax > elem_max * torch.exp2(exponent)).float()
    exponent = torch.where(amax > 0, exponent, torch.zeros_like(amax))
    exponent = exponent.clamp(-126.0, 127.0)
    scale = torch.exp2(exponent)
    scaled = groups / scale[..., None]
    if elem == "e2m1":
        idx = e2m1_codes(scaled.abs())
        sign = scaled < 0
        table = kE2M1ToFloat.to(x.device)
        values = torch.where(sign, -table[idx], table[idx])
        codes = (idx | (sign.long() << 3)).to(torch.uint8).reshape(*lead, k)
        data = codes[..., 0::2] | (codes[..., 1::2] << 4)
    else:
        q = scaled.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        values = q.float()
        data = q.reshape(*lead, k)
    dequant = (values * scale[..., None]).reshape(*lead, k)
    return data, (exponent + 127).to(torch.uint8), dequant


def quantize_nvfp4_static_ref(
    x: torch.Tensor,
    global_scale: float | torch.Tensor,
    dequant_dtype: torch.dtype = torch.float32,
):
    """Return packed codes, E4M3 scale bytes, and dequant for scalar or per-slab scales."""
    *lead, k = x.shape
    groups = x.float().reshape(*lead, k // 16, 16)
    amax = groups.abs().amax(dim=-1)
    gs = (
        global_scale.view(-1, *([1] * (len(lead) - 1)), 1).float()
        if isinstance(global_scale, torch.Tensor)
        else global_scale
    )
    block_scale = (
        (amax / (6.0 * gs)).clamp(min=2.0**-6, max=448.0).to(torch.float8_e4m3fn)
    )
    denom = block_scale.float() * gs
    scaled = (groups * (1.0 / denom)[..., None]).clamp(-6.0, 6.0)
    idx = e2m1_codes(scaled.abs())
    sign = scaled < 0
    table = kE2M1ToFloat.to(x.device)
    values = torch.where(sign, -table[idx], table[idx])
    codes = (idx | (sign.long() << 3)).to(torch.uint8).reshape(*lead, k)
    data = codes[..., 0::2] | (codes[..., 1::2] << 4)
    denom = block_scale.to(dequant_dtype) * gs
    dequant = (values.to(dequant_dtype) * denom[..., None]).reshape(*lead, k)
    return data, block_scale.view(torch.uint8), dequant


def nvfp4_per_token_denominator(recipe=None) -> float:
    """Return D for the per-token row scale row_amax / D."""
    return 6.0 * (448 if recipe is None else recipe.e4m3_max)


def quantize_nvfp4_per_token_ref(x: torch.Tensor, recipe=None):
    """Use FP32 tensor division and lane-order 4over6 sums; ties keep the 6-based scale.
    Zero-scale blocks have zero codes. Return packed codes, E4M3 scale bytes
    [..., K/16], FP32 row scales, FP64 dequantization, and choice details."""
    *lead, k = x.shape
    device = x.device
    xf = x.float()
    denominator = torch.tensor(nvfp4_per_token_denominator(recipe), device=device)
    row_amax = xf.abs().amax(dim=-1)
    s = row_amax / denominator
    s_safe = torch.where(s > 0, s, torch.ones_like(s))
    groups = xf.reshape(*lead, k // 16, 16)
    amax = groups.abs().amax(dim=-1)
    scaled = groups / s_safe[..., None, None]
    sf6 = amax / (6.0 * s_safe)[..., None]
    table = kE2M1ToFloat.to(device)

    def candidate(sf_float):
        sf = sf_float.clamp(max=448.0).to(torch.float8_e4m3fn)
        decoded = sf.float()
        live = decoded > 0
        denom = torch.where(live, decoded, torch.ones_like(decoded))
        q = scaled / denom[..., None]
        q = torch.where(live[..., None], q, torch.zeros_like(q)).clamp(-6.0, 6.0)
        idx = e2m1_codes(q.abs())
        # Preserve signed zero as the hardware E2M1 conversion does.
        sign = torch.signbit(q)
        values = torch.where(sign, -table[idx], table[idx])
        codes = (idx | (sign.long() << 3)).to(torch.uint8)
        return sf.view(torch.uint8), decoded, values, codes

    sf_bytes, decoded, values, codes = candidate(sf6)
    detail = dict(sf6=sf_bytes)
    if recipe is not None:
        sf4_bytes, decoded4, values4, codes4 = candidate(sf6 * 1.5)
        # Use the source amax as the reference for the kernel's reconstructed BF16 amax.
        amax_eff = torch.where(row_amax > 0, row_amax, denominator.expand_as(row_amax))
        encode = torch.minimum(
            denominator / amax_eff,
            torch.full_like(amax_eff, torch.finfo(torch.float32).max),
        )

        def error(values_c, decoded_c):
            cand = values_c * decoded_c[..., None]
            if recipe.err_use_fast_math:
                orig = groups * encode[..., None, None]
            else:
                cand = (cand * amax_eff[..., None, None]) / denominator
                orig = groups
            diff = cand - orig
            term = diff * diff if recipe.err_mode == "MSE" else diff.abs()
            err = term[..., 0]
            for lane in range(1, 16):
                err = err + term[..., lane]
            return err

        err6, err4 = error(values, decoded), error(values4, decoded4)
        use4 = err4 < err6
        sf_bytes = torch.where(use4, sf4_bytes, sf_bytes)
        decoded = torch.where(use4, decoded4, decoded)
        values = torch.where(use4[..., None], values4, values)
        codes = torch.where(use4[..., None], codes4, codes)
        detail.update(sf4=sf4_bytes, err6=err6, err4=err4, use4=use4)
    codes = codes.reshape(*lead, k)
    data = codes[..., 0::2] | (codes[..., 1::2] << 4)
    dequant = (
        values.double() * decoded.double()[..., None] * s.double()[..., None, None]
    ).reshape(*lead, k)
    return data, sf_bytes, s, dequant, detail
