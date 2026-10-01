"""Pure-torch FP4 fake quantization for DeepSeek-V4.1.

Indexer values use per-32 UE8M0 scales; compressed KV uses per-16 E4M3 scales.
Both paths round to the E2M1 grid with ties to even.
"""

from typing import Optional

import torch

FP8_MAX = 448.0
FP4_MAX = 6.0
FP8_BLOCK_SIZE = 32
FP4_BLOCK_SIZE = 32
FP4_AMAX_FLOOR = 6 * 2.0**-126


def ceil_pow2(x: torch.Tensor) -> torch.Tensor:
    """2 ** ceil(log2(x)) for positive fp32 x, computed on the IEEE bits so the
    result is exact at powers of two."""
    bits = x.contiguous().view(torch.int32)
    exponent = ((bits >> 23) & 0xFF) - 127
    has_mantissa = (bits & 0x7FFFFF) != 0
    exponent = exponent + has_mantissa.to(torch.int32)
    return ((exponent + 127) << 23).view(torch.float32)


def block_scale(x: torch.Tensor, block_size: int, fmax: float, amax_floor: float):
    """Per-block ue8m0 scale, as fp32 powers of two, shape [..., N // block_size]."""
    amax = x.float().unflatten(-1, (-1, block_size)).abs().amax(dim=-1)
    amax = amax.clamp_min(amax_floor)
    # The kernel multiplies by the fp32 reciprocal rather than dividing. A Python
    # scalar keeps this free of host tensors, so it can run under CUDA graph capture.
    return ceil_pow2(amax * (1.0 / fmax))


def round_fp4(x: torch.Tensor) -> torch.Tensor:
    """Round fp32 values in [-6, 6] onto the e2m1 grid with round-to-nearest-even."""
    magnitude = x.abs()
    step = torch.where(magnitude < 2.0, 0.5, torch.where(magnitude < 4.0, 1.0, 2.0))
    return torch.round(magnitude / step) * step * torch.sign(x)


def fake_quant_fp4(x: torch.Tensor, block_size: int = FP4_BLOCK_SIZE) -> torch.Tensor:
    """Quantize to fp4 (per-block ue8m0 scale) and back, in x's dtype."""
    scale = block_scale(x, block_size, FP4_MAX, FP4_AMAX_FLOOR)
    scaled = x.float().unflatten(-1, (-1, block_size)) / scale.unsqueeze(-1)
    deq = round_fp4(scaled.clamp(-FP4_MAX, FP4_MAX)) * scale.unsqueeze(-1)
    return deq.flatten(-2).to(x.dtype)


def fake_quant_compressed_kv(x: torch.Tensor) -> torch.Tensor:
    """FP4 round-trip with one E4M3FN scale per 16 compressed-KV elements.

    Round amax / 6 to E4M3 with ties to even, clamping the scale to its
    positive finite range [2**-9, 448]. Zero blocks remain zero.
    """
    blocks = x.float().unflatten(-1, (-1, 16))
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    scale = (amax * (1.0 / FP4_MAX)).clamp(min=2**-9, max=FP8_MAX)
    scale = scale.to(torch.float8_e4m3fn).float()
    scaled = (blocks / scale).clamp(-FP4_MAX, FP4_MAX)
    deq = round_fp4(scaled) * scale
    return deq.flatten(-2).to(x.dtype)


# ---------------------------------------------------------------------------
# Pure-torch reference of the paged V4.1 fp8 KV cache format read by the sparse
# decode kernel (528 B/token, "V41").
# ---------------------------------------------------------------------------


def ceil_pow2_scale(x: torch.Tensor) -> torch.Tensor:
    """``2 ** ceil(log2(max(x, 1e-4)))`` as fp32, computed on the IEEE bits so
    that it is exact at (and just above) powers of two."""
    x = x.float()
    scale = ceil_pow2(torch.clamp_min(x, 1e-4))
    # ceil_pow2 works on the bits of a finite value; a NaN or inf amax passes
    # through (both become the ue8m0 NaN byte, but only the NaN one turns the
    # whole tile's payload into NaN).
    return torch.where(torch.isfinite(x), scale, x)


def quantize_k_cache_v41(
    k: torch.Tensor, page_bytes: Optional[int] = None
) -> torch.Tensor:
    """``k`` ``[num_pages, page_size, 512]`` -> uint8 ``[num_pages, page_bytes]``
    pages of the V41 layout: 512 e4m3 per token, then 16 ue8m0 scales per token
    (one per 32 values), ``scale = 2 ** ceil(log2(max(amax / 448, 1e-4)))``."""
    num_pages, page_size, d = k.shape
    assert d == 512
    x = k.float().view(num_pages, page_size, 16, 32)
    scale = ceil_pow2_scale(x.abs().amax(dim=-1) / 448.0)
    data = (x / scale.unsqueeze(-1)).to(torch.float8_e4m3fn).view(torch.uint8)
    scale_u8 = scale.to(torch.float8_e8m0fnu).view(torch.uint8)
    raw = page_size * 528
    if page_bytes is None:
        page_bytes = -(-raw // 512) * 512
    assert page_bytes >= raw
    out = torch.zeros((num_pages, page_bytes), dtype=torch.uint8, device=k.device)
    out[:, : page_size * 512] = data.reshape(num_pages, page_size * 512)
    out[:, page_size * 512 : raw] = scale_u8.reshape(num_pages, page_size * 16)
    return out


def dequantize_k_cache_v41(pages: torch.Tensor, page_size: int) -> torch.Tensor:
    """Inverse of :func:`quantize_k_cache_v41`: ``[num_pages, page_size, 512]`` bf16."""
    num_pages = pages.shape[0]
    pages = pages.view(torch.uint8)
    data = pages[:, : page_size * 512].reshape(num_pages, page_size, 512)
    scale = pages[:, page_size * 512 : page_size * 528].reshape(
        num_pages, page_size, 16
    )
    values = data.view(torch.float8_e4m3fn).to(torch.bfloat16)
    scale_bf16 = scale.view(torch.float8_e8m0fnu).to(torch.bfloat16)
    return (values.view(num_pages, page_size, 16, 32) * scale_bf16.unsqueeze(-1)).view(
        num_pages, page_size, 512
    )
