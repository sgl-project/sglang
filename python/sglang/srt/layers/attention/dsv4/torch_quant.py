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
# Pure-torch references of the paged V4.1 KV cache formats read by the sparse
# decode kernel (528 B/token fp8 "V41", 288 B/token fp4 "V41_FP4"). They follow
# the kernel's own reference quantizer and are what the store / dequant kernels
# and the tests are checked against, byte for byte.
# ---------------------------------------------------------------------------

_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def cast_scale_inv_to_ue8m0(scale_inv: torch.Tensor) -> torch.Tensor:
    """``2 ** ceil(log2(max(scale_inv, 1e-4)))`` as fp32, computed on the IEEE
    bits so that it is exact at (and just above) powers of two."""
    scale_inv = scale_inv.float()
    scale = ceil_pow2(torch.clamp_min(scale_inv, 1e-4))
    # ceil_pow2 works on the bits of a finite value; a NaN or inf amax passes
    # through (both become the ue8m0 NaN byte, but only the NaN one turns the
    # whole tile's payload into NaN).
    return torch.where(torch.isfinite(scale_inv), scale, scale_inv)


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


def quantize_k_cache_v41(
    k: torch.Tensor, page_bytes: Optional[int] = None
) -> torch.Tensor:
    """``k`` ``[num_pages, page_size, 512]`` -> uint8 ``[num_pages, page_bytes]``
    pages of the V41 layout: 512 e4m3 per token, then 16 ue8m0 scales per token
    (one per 32 values), ``scale = 2 ** ceil(log2(max(amax / 448, 1e-4)))``."""
    num_pages, page_size, d = k.shape
    assert d == 512
    x = k.float().view(num_pages, page_size, 16, 32)
    scale = cast_scale_inv_to_ue8m0(x.abs().amax(dim=-1) / 448.0)
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


def quantize_dsv41_packed_main_kv(k: torch.Tensor) -> torch.Tensor:
    """Pack finite ``[num_pages, page_size, 512]`` values into the 384-byte
    DSV4.1 Main-KV layout.

    The 448 noPE values keep their E2M1 payload and 28 E4M3 block scales. The
    64 RoPE values undergo the same block-16 fake quantization but persist as
    dequantized BF16. Four reserved bytes complete every scale row.
    """
    if k.ndim != 3 or k.shape[-1] != 512:
        raise ValueError(
            "packed Main KV input must have shape [num_pages, page_size, 512], "
            f"got {tuple(k.shape)}"
        )
    if k.shape[1] not in (128, 256):
        raise ValueError(
            f"packed Main KV pages must contain 128 or 256 slots, got {k.shape[1]}"
        )
    if not bool(torch.isfinite(k).all().item()):
        raise ValueError("packed Main KV only supports finite input")

    num_pages, page_size, _ = k.shape
    blocks = k.float().view(num_pages, page_size, 32, 16)
    amax = blocks.abs().amax(dim=-1)
    scale = (amax * (1.0 / FP4_MAX)).clamp(min=2**-9, max=FP8_MAX)
    scale_e4m3 = scale.to(torch.float8_e4m3fn)
    # Adding positive zero clears an input -0.0, while a negative non-zero value
    # that rounds to zero still contributes the E2M1 sign bit.
    normalized = (blocks + 0.0) / scale_e4m3.float().unsqueeze(-1)
    codes = quantize_to_e2m1_codes(normalized)

    nope_codes = codes[..., :28, :].reshape(num_pages, page_size, 448)
    payload = nope_codes[..., 0::2] | (nope_codes[..., 1::2] << 4)

    scale_rows = torch.zeros(
        (num_pages, page_size, 32), dtype=torch.uint8, device=k.device
    )
    scale_rows[..., :28] = scale_e4m3[..., :28].view(torch.uint8)

    rope_values = (
        dequantize_e2m1_codes(codes[..., 28:, :])
        * scale_e4m3[..., 28:].float().unsqueeze(-1)
    ).reshape(num_pages, page_size, 64)
    rope = rope_values.to(torch.bfloat16).view(torch.uint8)

    payload_bytes = page_size * 224
    scale_offset = payload_bytes
    rope_offset = page_size * 256
    page_bytes = page_size * 384
    pages = torch.zeros((num_pages, page_bytes), dtype=torch.uint8, device=k.device)
    pages[:, :payload_bytes] = payload.reshape(num_pages, payload_bytes)
    pages[:, scale_offset:rope_offset] = scale_rows.reshape(num_pages, page_size * 32)
    pages[:, rope_offset:] = rope.reshape(num_pages, page_size * 128)
    return pages


def dequantize_dsv41_packed_main_kv(
    pages: torch.Tensor, page_size: int
) -> torch.Tensor:
    """Decode the 384-byte DSV4.1 Main-KV layout to BF16
    ``[num_pages, page_size, 512]`` values."""
    if pages.dtype is not torch.uint8 or pages.ndim != 2:
        raise ValueError(
            "packed Main KV pages must be 2D torch.uint8, "
            f"got dtype={pages.dtype}, shape={tuple(pages.shape)}"
        )
    if page_size not in (128, 256):
        raise ValueError(
            f"packed Main KV page_size must be 128 or 256, got {page_size}"
        )
    page_bytes = page_size * 384
    if pages.shape[1] != page_bytes:
        raise ValueError(
            "packed Main KV page width does not match page_size: "
            f"{pages.shape[1]} != {page_bytes}"
        )

    num_pages = pages.shape[0]
    payload_bytes = page_size * 224
    rope_offset = page_size * 256
    payload = pages[:, :payload_bytes].reshape(num_pages, page_size, 224)
    scale_rows = pages[:, payload_bytes:rope_offset].reshape(num_pages, page_size, 32)

    codes = torch.empty(
        (num_pages, page_size, 448), dtype=torch.uint8, device=pages.device
    )
    codes[..., 0::2] = payload & 0xF
    codes[..., 1::2] = payload >> 4
    values = dequantize_e2m1_codes(codes).view(num_pages, page_size, 28, 16)
    scales = scale_rows[..., :28].view(torch.float8_e4m3fn).float()
    nope = (
        (values * scales.unsqueeze(-1))
        .reshape(num_pages, page_size, 448)
        .to(torch.bfloat16)
    )

    rope = (
        pages[:, rope_offset:]
        .contiguous()
        .view(torch.bfloat16)
        .reshape(num_pages, page_size, 64)
    )
    return torch.cat((nope, rope), dim=-1)
