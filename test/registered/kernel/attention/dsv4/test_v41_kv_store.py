"""Byte-exactness of the V4.1 (fp8 / fp4) FlashMLA KV cache store kernels.

Every store kernel is compared byte for byte with the pure-torch quantizers of
the two formats (``torch_quant.quantize_k_cache_v41`` / ``_v41_fp4``), which
follow the decode kernel's own reference quantizer.
"""

import unittest

import torch

from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


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


def rope_tail(
    x: torch.Tensor, freqs: torch.Tensor, rope_dim: int, inverse: bool = False
) -> torch.Tensor:
    """Rotate the last rope_dim features of x [T, ..., D] with complex freqs [T, rope_dim // 2]."""
    head, tail = x[..., :-rope_dim], x[..., -rope_dim:]
    tc = torch.view_as_complex(tail.float().unflatten(-1, (-1, 2)).contiguous())
    f = freqs.conj() if inverse else freqs
    f = f.view(x.shape[0], *([1] * (x.ndim - 2)), rope_dim // 2)
    rotated = torch.view_as_real(tc * f).flatten(-2).to(x.dtype)
    return torch.cat([head, rotated], dim=-1)


REFERENCE = {
    KVLayout.V41: quantize_k_cache_v41,
    KVLayout.V41_FP4: quantize_k_cache_v41_fp4,
}


def _sm100():
    return (
        torch.cuda.is_available()
        and torch.version.cuda is not None
        and torch.cuda.get_device_capability()[0] >= 10
    )


def token_rows(pages, layout, page_size, locs):
    """The (data row, scale row) bytes of the tokens at ``locs``."""
    locs = locs.long()
    page, offset = locs // page_size, locs % page_size
    data_cols = torch.arange(layout.data_bytes, device=pages.device)
    scale_cols = torch.arange(layout.scale_bytes, device=pages.device)
    data = pages[page[:, None], offset[:, None] * layout.data_bytes + data_cols]
    scale = pages[
        page[:, None],
        layout.scale_offset(page_size)
        + offset[:, None] * layout.scale_bytes
        + scale_cols,
    ]
    return data, scale


def reference_pages(layout, page_size, num_pages, locs, values, page_bytes):
    full = torch.zeros(
        num_pages, page_size, 512, device=values.device, dtype=values.dtype
    )
    full.view(-1, 512)[locs.long()] = values
    return REFERENCE[layout](full, page_bytes=page_bytes)


@unittest.skipUnless(_sm100(), "the V4.1 KV layouts are SM100 kernels")
class TestV41KVStore(CustomTestCase):
    def assert_tokens_equal(self, cache, ref, layout, page_size, locs):
        got_data, got_scale = token_rows(cache, layout, page_size, locs)
        exp_data, exp_scale = token_rows(ref, layout, page_size, locs)
        self.assertTrue(torch.equal(got_scale, exp_scale), "scale rows differ")
        self.assertTrue(torch.equal(got_data, exp_data), "data rows differ")

    def assert_untouched_zero(self, cache, layout, page_size, locs):
        num_slots = cache.shape[0] * page_size
        written = torch.zeros(num_slots, dtype=torch.bool, device=cache.device)
        written[locs.long()] = True
        others = torch.arange(num_slots, device=cache.device)[~written]
        data, scale = token_rows(cache, layout, page_size, others)
        self.assertEqual(int(data.sum()) + int(scale.sum()), 0)

    def test_c1_c2_decode_store(self):
        """The ratio-1 / ratio-2 decode compressors write the V4.1 layouts: the cache
        holds the quantized rope_tail of the pre-RoPE latent the kernel publishes
        (bitwise; the fp8 layout after the model's fp4 fake quantization), and the
        latent is the torch RMSNorm to within an fp32-reduction-order bf16 ulp."""

        from sglang.kernels.ops.attention.dsv4.c1 import c1_decode_norm_rope_store
        from sglang.kernels.ops.attention.dsv4.c2 import (
            c2_decode_or_verify_norm_rope_store,
        )

        g = torch.Generator(device="cuda").manual_seed(4)
        eps = 1e-6
        angles = torch.randn(4096, 32, generator=g, device="cuda")
        freqs = torch.polar(torch.ones_like(angles), angles)
        freqs_real = torch.view_as_real(freqs).flatten(-2)
        w = (torch.randn(512, generator=g, device="cuda") * 0.3 + 1).to(torch.bfloat16)

        def torch_norm(x):
            xf = x.float()
            return (
                xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * w.float()
            ).to(torch.bfloat16)

        def stored_reference(layout, rotated, page_size, num_pages, locs, page_bytes):
            values = (
                rotated
                if layout is KVLayout.V41_FP4
                else fake_quant_compressed_kv(rotated)
            )
            return reference_pages(
                layout, page_size, num_pages, locs, values, page_bytes
            )

        n = 200
        for layout in (KVLayout.V41, KVLayout.V41_FP4):
            for page_size, num_pages in ((256, 4), (128, 8)):
                with self.subTest(kernel="c1", layout=layout.name, page_size=page_size):
                    x = (torch.randn(n, 512, generator=g, device="cuda") * 2).to(
                        torch.bfloat16
                    )
                    pos = torch.randint(
                        0, 4096, (n,), generator=g, device="cuda", dtype=torch.int64
                    )
                    out_loc = (
                        torch.randperm(
                            num_pages * page_size - 1, generator=g, device="cuda"
                        )[:n].to(torch.int32)
                        + 1
                    )
                    out_loc[3] = 0  # a padded graph row publishes nothing
                    cache = torch.zeros(
                        num_pages,
                        layout.page_bytes(page_size),
                        dtype=torch.uint8,
                        device="cuda",
                    )
                    latent = c1_decode_norm_rope_store(
                        x,
                        w,
                        pos,
                        out_loc,
                        eps,
                        freqs_real,
                        cache,
                        page_size=page_size,
                        layout=layout,
                    )
                    torch.testing.assert_close(
                        latent, torch_norm(x), rtol=2**-7, atol=2**-14
                    )
                    valid = out_loc > 0
                    rotated = rope_tail(latent, freqs[pos], 64)
                    ref = stored_reference(
                        layout,
                        rotated[valid],
                        page_size,
                        num_pages,
                        out_loc[valid],
                        cache.shape[1],
                    )
                    self.assert_tokens_equal(
                        cache, ref, layout, page_size, out_loc[valid]
                    )
                    self.assert_untouched_zero(cache, layout, page_size, out_loc[valid])
                with self.subTest(kernel="c2", layout=layout.name, page_size=page_size):
                    ring = 2
                    kv_new = torch.randn(n, 512, generator=g, device="cuda") * 2
                    score = torch.randn(n, 512, generator=g, device="cuda")
                    kv_old = torch.randn(n, 512, generator=g, device="cuda") * 2
                    kv_input = torch.cat([kv_new, score], dim=-1).contiguous()
                    req = torch.arange(n, device="cuda", dtype=torch.int64)
                    # Odd positions complete a pair; one even (pending) row and one padded row.
                    pos = (
                        2
                        * torch.randint(
                            0, 2000, (n,), generator=g, device="cuda", dtype=torch.int64
                        )
                        + 1
                    )
                    pos[5] = 4
                    state = torch.randn(n * ring + 4, 1024, generator=g, device="cuda")
                    read_rows = req * ring + (pos - 1) % ring
                    # Equal scores make the pair pool the exact mean.
                    state[read_rows, :512] = kv_old
                    state[read_rows, 512:] = score
                    raw_out_loc = (
                        torch.randperm(
                            num_pages * page_size - 1, generator=g, device="cuda"
                        )[:n].to(torch.int32)
                        + 1
                    ) * 2
                    raw_out_loc[7] = 0
                    cache = torch.zeros(
                        num_pages,
                        layout.page_bytes(page_size),
                        dtype=torch.uint8,
                        device="cuda",
                    )
                    latent = c2_decode_or_verify_norm_rope_store(
                        kv_input,
                        state,
                        w,
                        pos,
                        req,
                        raw_out_loc,
                        eps,
                        freqs_real,
                        cache,
                        page_size=page_size,
                        ring_size=ring,
                        layout=layout,
                    )
                    valid = (raw_out_loc != 0) & (pos % 2 == 1)
                    pooled = ((kv_old + kv_new) / 2).to(torch.bfloat16)
                    torch.testing.assert_close(
                        latent[valid],
                        torch_norm(pooled)[valid],
                        rtol=2**-7,
                        atol=2**-14,
                    )
                    rotated = rope_tail(latent, freqs[(pos - 1).clamp_min(0)], 64)
                    slots = raw_out_loc >> 1
                    ref = stored_reference(
                        layout,
                        rotated[valid],
                        page_size,
                        num_pages,
                        slots[valid],
                        cache.shape[1],
                    )
                    self.assert_tokens_equal(
                        cache, ref, layout, page_size, slots[valid]
                    )
                    self.assert_untouched_zero(cache, layout, page_size, slots[valid])


if __name__ == "__main__":
    unittest.main()
