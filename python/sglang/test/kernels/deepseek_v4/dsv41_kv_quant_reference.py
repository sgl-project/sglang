"""Independent packed-FP4 (V41_FP4 layout) cache oracle for the V4.1 store and reader tests."""

from typing import Optional

import torch

KV_DIM = 512
FP4_GROUP = 16  # values per e4m3 scale
E2M1_BYTES_PER_TOKEN = KV_DIM // 2
E4M3_SCALE_BYTES_PER_TOKEN = KV_DIM // FP4_GROUP
FP4_TOKEN_BYTES = E2M1_BYTES_PER_TOKEN + E4M3_SCALE_BYTES_PER_TOKEN
E2M1_MAX = 6.0
E4M3_MAX = 448.0
MIN_SCALE = 2.0**-9
_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def quantize_to_e2m1_codes(x: torch.Tensor) -> torch.Tensor:
    """Round to the nearest e2m1 value with the semantics of
    cvt.rn.satfinite.e2m1x2.f32 (ties to even, saturating to +-6) and return
    the 4-bit codes as uint8. The sign is kept for values that round to zero
    (-0.0 and small negatives give the code 0x8); NaN maps to code 0."""
    x = x.float()
    mags = torch.tensor(_E2M1_MAGNITUDES, dtype=torch.float32, device=x.device)
    sign = torch.signbit(x).to(torch.uint8) << 3
    a = torch.nan_to_num(x.abs(), nan=0.0, posinf=6.0).clamp_max(6.0)
    mids = (mags[:-1] + mags[1:]) / 2
    code = torch.bucketize(a, mids, right=True)
    on_tie = (a.unsqueeze(-1) == mids).any(dim=-1)
    tie_code = torch.bucketize(a, mids, right=False)
    code = torch.where(on_tie, tie_code + (tie_code & 1), code)
    return sign | code.to(torch.uint8)


def dequantize_e2m1_codes(codes: torch.Tensor) -> torch.Tensor:
    mags = torch.tensor(_E2M1_MAGNITUDES, dtype=torch.float32, device=codes.device)
    val = mags[(codes & 7).long()]
    return torch.where((codes & 8) != 0, -val, val)


def quantize_k_cache_v41_fp4(
    k: torch.Tensor, page_bytes: Optional[int] = None
) -> torch.Tensor:
    """k [num_pages, page_size, 512] -> uint8 [num_pages, page_bytes]
    pages of the V41_FP4 layout: 256 B of e2m1 codes per token (even index in
    the low nibble), then 32 e4m3 scales per token (one per 16 values),
    scale = e4m3(clamp(amax / 6, 2**-9, 448)). A NaN element poisons its
    tile: NaN scale, zero codes."""
    num_pages, page_size, d = k.shape
    assert d == KV_DIM
    x = k.float().view(num_pages, page_size, KV_DIM // FP4_GROUP, FP4_GROUP)
    amax = torch.nan_to_num(x.abs(), nan=float("inf")).amax(dim=-1)
    scale = torch.clamp(amax / E2M1_MAX, MIN_SCALE, E4M3_MAX).to(torch.float8_e4m3fn)
    scale = torch.where(torch.isinf(amax), torch.full_like(scale, float("nan")), scale)
    codes = quantize_to_e2m1_codes(x / scale.float().unsqueeze(-1))
    codes = codes.view(num_pages, page_size, KV_DIM)
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    raw = page_size * FP4_TOKEN_BYTES
    if page_bytes is None:
        page_bytes = -(-raw // 256) * 256  # pages start on a 256-byte boundary
    assert page_bytes >= raw
    data_bytes = page_size * E2M1_BYTES_PER_TOKEN
    out = torch.zeros((num_pages, page_bytes), dtype=torch.uint8, device=k.device)
    out[:, :data_bytes] = packed.reshape(num_pages, data_bytes)
    out[:, data_bytes:raw] = scale.view(torch.uint8).reshape(
        num_pages, raw - data_bytes
    )
    return out


def dequantize_k_cache_v41_fp4(pages: torch.Tensor, page_size: int) -> torch.Tensor:
    """Inverse of quantize_k_cache_v41_fp4: [num_pages, page_size, 512]
    bf16. e2m1 * e4m3 has at most 2 + 4 significant bits, so the product is
    exact in bf16, as in the kernel."""
    num_pages = pages.shape[0]
    pages = pages.view(torch.uint8)
    data_bytes = page_size * E2M1_BYTES_PER_TOKEN
    data = pages[:, :data_bytes].reshape(num_pages, page_size, E2M1_BYTES_PER_TOKEN)
    scale = pages[:, data_bytes : page_size * FP4_TOKEN_BYTES].reshape(
        num_pages, page_size, E4M3_SCALE_BYTES_PER_TOKEN
    )
    codes = torch.empty(
        (num_pages, page_size, KV_DIM), dtype=torch.uint8, device=pages.device
    )
    codes[..., 0::2] = data & 0xF
    codes[..., 1::2] = data >> 4
    values = dequantize_e2m1_codes(codes).view(
        num_pages, page_size, KV_DIM // FP4_GROUP, FP4_GROUP
    )
    out = values * scale.view(torch.float8_e4m3fn).float().unsqueeze(-1)
    return out.view(num_pages, page_size, KV_DIM).to(torch.bfloat16)
