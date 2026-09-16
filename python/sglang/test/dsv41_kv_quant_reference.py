"""Independent packed-FP4 cache oracle for the AMD store and reader tests."""

from typing import Optional

import torch

from sglang.kernels.ops.attention.dsv4.torch_quant import (
    dequantize_k_cache_v41 as dequantize_k_cache_v41,
)
from sglang.kernels.ops.attention.dsv4.torch_quant import (
    fake_quant_compressed_kv as fake_quant_compressed_kv,
)
from sglang.kernels.ops.attention.dsv4.torch_quant import (
    quantize_k_cache_v41 as quantize_k_cache_v41,
)

_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def quantize_to_e2m1_codes(x: torch.Tensor) -> torch.Tensor:
    """Round to the nearest e2m1 value with the semantics of
    ``cvt.rn.satfinite.e2m1x2.f32`` (ties to even, saturating to +-6) and return
    the 4-bit codes as uint8. The sign is kept for values that round to zero
    (``-0.0`` and small negatives give the code ``0x8``); NaN maps to code 0."""
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
    """``k`` ``[num_pages, page_size, 512]`` -> uint8 ``[num_pages, page_bytes]``
    pages of the V41_FP4 layout: 256 B of e2m1 codes per token (even index in
    the low nibble), then 32 e4m3 scales per token (one per 16 values),
    ``scale = e4m3(clamp(amax / 6, 2**-9, 448))``. A NaN element poisons its
    tile: NaN scale, zero codes."""
    num_pages, page_size, d = k.shape
    assert d == 512
    x = k.float().view(num_pages, page_size, 32, 16)
    amax = torch.nan_to_num(x.abs(), nan=float("inf")).amax(dim=-1)
    scale = torch.clamp(amax / 6.0, 2.0**-9, 448.0).to(torch.float8_e4m3fn)
    scale = torch.where(torch.isinf(amax), torch.full_like(scale, float("nan")), scale)
    codes = quantize_to_e2m1_codes(x / scale.float().unsqueeze(-1))
    codes = codes.view(num_pages, page_size, 512)
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    raw = page_size * 288
    if page_bytes is None:
        page_bytes = -(-raw // 256) * 256
    assert page_bytes >= raw
    out = torch.zeros((num_pages, page_bytes), dtype=torch.uint8, device=k.device)
    out[:, : page_size * 256] = packed.reshape(num_pages, page_size * 256)
    out[:, page_size * 256 : raw] = scale.view(torch.uint8).reshape(
        num_pages, page_size * 32
    )
    return out


def dequantize_k_cache_v41_fp4(pages: torch.Tensor, page_size: int) -> torch.Tensor:
    """Inverse of :func:`quantize_k_cache_v41_fp4`: ``[num_pages, page_size, 512]``
    bf16. ``e2m1 * e4m3`` has at most 2 + 4 significant bits, so the product is
    exact in bf16, as in the kernel."""
    num_pages = pages.shape[0]
    pages = pages.view(torch.uint8)
    data = pages[:, : page_size * 256].reshape(num_pages, page_size, 256)
    scale = pages[:, page_size * 256 : page_size * 288].reshape(
        num_pages, page_size, 32
    )
    codes = torch.empty(
        (num_pages, page_size, 512), dtype=torch.uint8, device=pages.device
    )
    codes[..., 0::2] = data & 0xF
    codes[..., 1::2] = data >> 4
    values = dequantize_e2m1_codes(codes).view(num_pages, page_size, 32, 16)
    out = values * scale.view(torch.float8_e4m3fn).float().unsqueeze(-1)
    return out.view(num_pages, page_size, 512).to(torch.bfloat16)
