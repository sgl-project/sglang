"""Triton chunked FlashMLA sparse decode unit tests.

Validates the specific chunked FlashMLA implementation that replaces the upstream
`flash_mla` CUDA kernel (unavailable on XPU):

- ``_sm120_sparse_decode_fwd``: pure-PyTorch reference path.

DSv4 cache layout (per page):
    data section:  page_size * 576 bytes   = 64 tokens * (448 nope + 128 rope)
    scale section: page_size * 8 bytes     = 64 tokens * (7 UE8M0 scales + 1 pad)
    total bytes:   page_size * 584         = stride(0) of k_cache
"""

from __future__ import annotations

import unittest

import torch

from sglang.srt.layers.attention.dsv4.triton_flashmla import (
    flash_mla_with_kvcache_triton,
)
from sglang.srt.layers.attention.flash_mla_sm120 import (
    _D,
    _NOPE_DIM,
    _NOPE_ROPE_STRIDE,
    _NUM_TILES,
    _ROPE_DIM,
    _SCALE_STRIDE,
    _TILE_SIZE,
    _sm120_sparse_decode_fwd,
)
from sglang.srt.utils import get_device
from sglang.test.test_utils import CustomTestCase

# Per-token byte layout
_BYTES_PER_TOKEN = _NOPE_ROPE_STRIDE + _SCALE_STRIDE  # 576 + 8 = 584


def _build_kvcache(
    num_pages: int,
    page_size: int,
    *,
    device: torch.device,
    seed: int = 0,
):
    """Build a synthetic FP8/BF16/UE8M0 KV cache.

    Returns the (num_pages, page_size, 1, bpt) FP8-viewed cache plus the raw
    nope FP8 (float), rope BF16 (float), and UE8M0 scale (float) reference
    tensors for verification.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    bpt = _BYTES_PER_TOKEN  # 584 — also satisfies k_cache.stride(0) requirement
    raw = torch.zeros(num_pages, page_size, bpt, dtype=torch.uint8, device=device)

    # ---- Nope FP8 region: per-token bytes [0:448] ----
    # Stay in [0, 0x6F] to avoid NaN/Inf in float8_e4m3fn.
    nope_bytes = torch.randint(
        0, 0x70, (num_pages, page_size, _NOPE_DIM), generator=g, dtype=torch.uint8
    ).to(device)
    raw[:, :, :_NOPE_DIM] = nope_bytes

    # ---- Rope BF16 region: per-token bytes [448:576] = 64 bf16 values ----
    rope_bf16_vals = (
        torch.randn((num_pages, page_size, _ROPE_DIM), generator=g, dtype=torch.float32)
        .clamp(-2.0, 2.0)
        .to(torch.bfloat16)
        .to(device)
    )
    # View bf16 as 2 bytes per value -> 128 bytes per token rope region
    rope_as_uint8 = rope_bf16_vals.contiguous().view(
        torch.uint8
    )  # (num_pages, page_size, 128)
    raw[:, :, _NOPE_DIM : _NOPE_DIM + _ROPE_DIM * 2] = rope_as_uint8

    # ---- Scale section: starts at page_size * 576 ----
    # 7 UE8M0 bytes per token + 1 pad.  Keep exponents in a sane range
    # (UE8M0 byte 'b' decodes to 2**(b-127)), so 120..130 gives ~[1/128, 8].
    scale_bytes = torch.randint(
        120,
        131,
        (num_pages, page_size, _NUM_TILES),
        generator=g,
        dtype=torch.uint8,
    ).to(device)
    # raw is (num_pages, page_size, bpt) with bpt=584; data ends at 576, scale region 576..584
    # Per-token scale offset relative to scale section: token_idx * 8.
    # Note: the scale region in raw memory is at the END of the page after the
    # data section. Since raw shape is (num_pages, page_size, bpt), where
    # bpt=584, scale lives in raw[:, :, _NOPE_ROPE_STRIDE : _NOPE_ROPE_STRIDE+8].
    # BUT the gather code expects layout
    #   raw_pages: (num_pages, page_bytes)
    #   scale_section_offset = page_size * 576
    # i.e. data for ALL tokens first, then scale for ALL tokens. So we need
    # the flat per-page layout: [tok0_data(576), tok1_data, ..., tokN-1_data,
    # tok0_scale(8), tok1_scale, ...].
    # We constructed raw as (num_pages, page_size, bpt) which interleaves data
    # and scales per token.  Build a fresh buffer with the correct flat order.
    flat = torch.zeros(num_pages, page_size * bpt, dtype=torch.uint8, device=device)
    # Data: per-token 576 bytes contiguous
    flat_data = raw[:, :, :_NOPE_ROPE_STRIDE].reshape(
        num_pages, page_size * _NOPE_ROPE_STRIDE
    )
    flat[:, : page_size * _NOPE_ROPE_STRIDE] = flat_data
    # Scales: per-token 8 bytes (only first 7 written)
    scale_block = torch.zeros(
        num_pages, page_size, _SCALE_STRIDE, dtype=torch.uint8, device=device
    )
    scale_block[:, :, :_NUM_TILES] = scale_bytes
    flat[:, page_size * _NOPE_ROPE_STRIDE :] = scale_block.reshape(
        num_pages, page_size * _SCALE_STRIDE
    )

    # View as (num_pages, page_size, 1, bpt) float8_e4m3fn
    k_cache = flat.view(num_pages, page_size, 1, bpt).view(torch.float8_e4m3fn)
    # The Triton kernel expects k_cache.stride(0) == page_size * bpt.
    assert k_cache.stride(0) == page_size * bpt

    # Reference dequant per token (matches what _gather_and_dequant should produce):
    nope_fp8 = nope_bytes.view(
        torch.float8_e4m3fn
    ).float()  # (num_pages, page_size, 448)
    scale_e8m0 = scale_bytes.view(
        torch.float8_e8m0fnu
    ).float()  # (num_pages, page_size, 7)
    nope_dequant = (
        nope_fp8.view(num_pages, page_size, _NUM_TILES, _TILE_SIZE)
        * scale_e8m0.view(num_pages, page_size, _NUM_TILES, 1)
    ).view(
        num_pages, page_size, _NOPE_DIM
    )  # float32
    ref_per_token = torch.cat(
        [nope_dequant.to(torch.bfloat16), rope_bf16_vals], dim=-1
    )  # (num_pages, page_size, 512) bf16
    return k_cache, ref_per_token


def _build_q_indices(
    batch_size: int,
    num_heads: int,
    topk: int,
    num_pages: int,
    page_size: int,
    *,
    device: torch.device,
    seed: int = 1,
):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = (
        torch.randn((batch_size, 1, num_heads, _D), generator=g, dtype=torch.float32)
        .clamp(-1.5, 1.5)
        .to(torch.bfloat16)
        .to(device)
    )
    # Each batch picks `topk` random valid token-level indices into the pool
    pool_size = num_pages * page_size
    indices = torch.zeros((batch_size, 1, topk), dtype=torch.int32, device=device)
    for b in range(batch_size):
        perm = torch.randperm(pool_size, generator=g)[:topk]
        indices[b, 0] = perm.to(device=device, dtype=torch.int32)
    return q, indices


class TestSparseDecodeTritonVsTorch(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not (torch.cuda.is_available() or torch.xpu.is_available()):
            raise unittest.SkipTest("CUDA or XPU required")
        cls.device = torch.device(get_device())

    def _run(
        self,
        batch_size: int = 2,
        num_heads: int = 4,
        topk: int = 64,
        page_size: int = 64,
        num_pages: int = 4,
        seed: int = 11,
        with_topk_length: bool = True,
        with_attn_sink: bool = False,
    ):
        k_cache, _ = _build_kvcache(num_pages, page_size, device=self.device, seed=seed)
        q, indices = _build_q_indices(
            batch_size,
            num_heads,
            topk,
            num_pages,
            page_size,
            device=self.device,
            seed=seed + 100,
        )
        topk_length = None
        if with_topk_length:
            topk_length = torch.tensor(
                [topk // 2, topk] if batch_size == 2 else [topk] * batch_size,
                dtype=torch.int32,
                device=self.device,
            )
        attn_sink = None
        if with_attn_sink:
            attn_sink = torch.full(
                (num_heads,), -1.5, dtype=torch.float32, device=self.device
            )
        softmax_scale = _D ** (-0.5)
        # Production passes config.v_head_dim == _D (512). Both impls return
        # (B, 1, H, _D); the Triton kernel always computes the full nope+rope.
        head_dim_v = _D

        # PyTorch reference
        ref_out, ref_lse = _sm120_sparse_decode_fwd(
            q,
            k_cache,
            indices,
            topk_length,
            attn_sink,
            head_dim_v=head_dim_v,
            softmax_scale=softmax_scale,
        )

        # Triton
        tri_out, tri_lse = flash_mla_with_kvcache_triton(
            q,
            k_cache,
            head_dim_v=head_dim_v,
            softmax_scale=softmax_scale,
            indices=indices,
            topk_length=topk_length,
            attn_sink=attn_sink,
        )
        # Outputs both bf16, allow drift from online softmax base-2 vs base-e
        # math + bf16 mantissa precision (~7 bits).  5e-2 abs/rel is generous
        # but accounts for accumulated rounding across topk tokens.
        torch.testing.assert_close(
            tri_out.to(torch.float32),
            ref_out.to(torch.float32),
            atol=5e-2,
            rtol=5e-2,
        )
        return ref_out, tri_out, ref_lse, tri_lse

    def test_triton_vs_torch_basic(self):
        self._run()

    def test_triton_vs_torch_no_topk_length(self):
        self._run(with_topk_length=False)

    def test_triton_vs_torch_with_attn_sink(self):
        self._run(with_attn_sink=True)

    def test_triton_vs_torch_small_topk(self):
        self._run(topk=32, num_heads=2)

    def test_triton_vs_torch_negative_indices_are_masked(self):
        """Indices < 0 are 'invalid' and must contribute zero to output."""
        k_cache, _ = _build_kvcache(4, 64, device=self.device, seed=3)
        q, indices = _build_q_indices(2, 4, 32, 4, 64, device=self.device, seed=99)
        # Half of each batch's indices set to -1
        indices[:, :, 16:] = -1
        topk_length = torch.tensor([32, 32], dtype=torch.int32, device=self.device)

        ref_out, _ = _sm120_sparse_decode_fwd(
            q, k_cache, indices, topk_length, None, _D, _D**-0.5
        )
        tri_out, _ = flash_mla_with_kvcache_triton(
            q,
            k_cache,
            _D,
            softmax_scale=_D**-0.5,
            indices=indices,
            topk_length=topk_length,
            attn_sink=None,
        )
        torch.testing.assert_close(
            tri_out.to(torch.float32),
            ref_out.to(torch.float32),
            atol=5e-2,
            rtol=5e-2,
        )


if __name__ == "__main__":
    import sys

    sys.exit(unittest.main())
