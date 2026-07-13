"""SM120 FlashMLA sparse decode unit tests.

Validates the SM120-specific FlashMLA implementation that replaces the upstream
`flash_mla` CUDA kernel (unavailable on SM120 / RTX PRO 6000):

- ``_gather_and_dequant``: byte-precise dequant of paged FP8 + BF16 + UE8M0 KV
  cache. Covers the dtype-reinterpretation surface that caused the historic
  uint8 garbled-output regression (see progress doc §4).
- ``_sm120_sparse_decode_fwd``: pure-PyTorch reference path.
- ``flash_mla_sparse_decode_triton``: tiled Triton kernel.
- ``_apply_attn_sink`` / ``_merge_partial_attn``: post-processing helpers.
- ``flash_mla_with_kvcache_sm120``: entry-point dispatch on
  ``SGLANG_SM120_TRITON_FLASHMLA`` selects torch/triton paths and both yield
  matching output.

DSv4 cache layout (per page):
    data section:  page_size * 576 bytes   = 64 tokens * (448 nope + 128 rope)
    scale section: page_size * 8 bytes     = 64 tokens * (7 UE8M0 scales + 1 pad)
    total bytes:   page_size * 584         = stride(0) of k_cache
"""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from sglang.srt.layers.attention import flash_mla_sm120 as fmod
from sglang.srt.layers.attention.flash_mla_sm120 import (
    _D,
    _NOPE_DIM,
    _NOPE_ROPE_STRIDE,
    _NUM_TILES,
    _ROPE_DIM,
    _SCALE_STRIDE,
    _TILE_SIZE,
    _gather_and_dequant,
    _sm120_sparse_decode_fwd,
    flash_mla_with_kvcache_sm120,
)
from sglang.srt.layers.attention.flash_mla_sm120_triton import (
    _apply_attn_sink,
    _merge_partial_attn,
    flash_mla_sparse_decode_triton,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=8, stage="base-b", runner_config="1-gpu-large")


# Per-token byte layout
_BYTES_PER_TOKEN = _NOPE_ROPE_STRIDE + _SCALE_STRIDE  # 576 + 8 = 584

_IS_SM120 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


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


@unittest.skipUnless(_IS_SM120, "SM120 (compute capability 12.0) required")
class TestGatherAndDequant(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        cls.device = torch.device("cuda")

    def test_basic_dequant_matches_manual(self):
        """Per-token output equals nope_fp8 * scale concatenated with rope_bf16."""
        num_pages, page_size = 3, 64
        k_cache, ref_per_token = _build_kvcache(
            num_pages, page_size, device=self.device, seed=42
        )
        # Pick 1 index per page (positions 0, 32, 63)
        token_ids = torch.tensor(
            [[0, 1 * page_size + 32, 2 * page_size + 63]],
            dtype=torch.int32,
            device=self.device,
        )
        out = _gather_and_dequant(k_cache, token_ids, page_size)
        self.assertEqual(out.shape, (1, 3, _D))
        # Expected entries
        expected = torch.stack(
            [
                ref_per_token[0, 0],
                ref_per_token[1, 32],
                ref_per_token[2, 63],
            ],
            dim=0,
        ).unsqueeze(0)
        # bf16 dequant: allow up to 1 ULP per element.  Scales are integral
        # powers of 2 so the only loss is the fp8 -> bf16 mantissa rounding.
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)

    def test_dequant_handles_full_page_range(self):
        """All token positions in a single page produce correct output."""
        num_pages, page_size = 2, 64
        k_cache, ref_per_token = _build_kvcache(
            num_pages, page_size, device=self.device, seed=7
        )
        token_ids = torch.arange(page_size, dtype=torch.int32, device=self.device).view(
            1, page_size
        )
        out = _gather_and_dequant(k_cache, token_ids, page_size)
        torch.testing.assert_close(
            out, ref_per_token[0].unsqueeze(0), atol=1e-2, rtol=1e-2
        )


@unittest.skipUnless(_IS_SM120, "SM120 (compute capability 12.0) required")
class TestSparseDecodeTritonVsTorch(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        cls.device = torch.device("cuda")

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
        tri_out, tri_lse = flash_mla_sparse_decode_triton(
            q,
            k_cache,
            indices,
            topk_length,
            attn_sink,
            head_dim_v=head_dim_v,
            softmax_scale=softmax_scale,
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
        tri_out, _ = flash_mla_sparse_decode_triton(
            q, k_cache, indices, topk_length, None, _D, _D**-0.5
        )
        torch.testing.assert_close(
            tri_out.to(torch.float32),
            ref_out.to(torch.float32),
            atol=5e-2,
            rtol=5e-2,
        )


@unittest.skipUnless(_IS_SM120, "SM120 (compute capability 12.0) required")
class TestApplyAttnSink(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        cls.device = torch.device("cuda")

    def test_sink_zero_means_full_dampen_to_half(self):
        """attn_sink == lse implies sink contributes equal weight: output halves."""
        B, H, D = 1, 2, 8
        lse = torch.zeros(B, 1, H, device=self.device, dtype=torch.float32)
        attn_sink = torch.zeros(H, device=self.device, dtype=torch.float32)
        out = torch.ones(B, 1, H, D, device=self.device, dtype=torch.bfloat16)
        new_out, new_lse = _apply_attn_sink(out, lse, attn_sink)
        # combined_lse = logaddexp(0, 0) = log(2). w = exp(-log2) = 0.5.
        torch.testing.assert_close(
            new_out.float(),
            torch.full_like(new_out.float(), 0.5),
            atol=2e-3,
            rtol=2e-3,
        )
        torch.testing.assert_close(
            new_lse,
            torch.full_like(new_lse, 0.6931472),  # log(2)
            atol=1e-5,
            rtol=1e-5,
        )

    def test_sink_dead_lse_stays_zero(self):
        """lse == -inf (no valid tokens) -> output stays zero (weight=0)."""
        B, H, D = 1, 2, 4
        lse = torch.full(
            (B, 1, H), float("-inf"), device=self.device, dtype=torch.float32
        )
        attn_sink = torch.zeros(H, device=self.device, dtype=torch.float32)
        out = torch.ones(B, 1, H, D, device=self.device, dtype=torch.bfloat16)
        new_out, _ = _apply_attn_sink(out, lse, attn_sink)
        torch.testing.assert_close(
            new_out.float(), torch.zeros_like(new_out.float()), atol=0, rtol=0
        )


@unittest.skipUnless(_IS_SM120, "SM120 (compute capability 12.0) required")
class TestMergePartialAttn(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        cls.device = torch.device("cuda")

    def test_equal_lse_arithmetic_mean(self):
        """lse1 == lse2 -> merged output is the arithmetic mean of out1, out2."""
        B, H, D = 1, 2, 4
        lse = torch.zeros(B, 1, H, device=self.device, dtype=torch.float32)
        out1 = torch.ones(B, 1, H, D, device=self.device, dtype=torch.bfloat16)
        out2 = torch.full((B, 1, H, D), 3.0, device=self.device, dtype=torch.bfloat16)
        merged, merged_lse = _merge_partial_attn(out1, lse, out2, lse)
        torch.testing.assert_close(
            merged.float(),
            torch.full_like(merged.float(), 2.0),
            atol=1e-2,
            rtol=1e-2,
        )
        torch.testing.assert_close(
            merged_lse,
            torch.full_like(merged_lse, 0.6931472),  # log(2)
            atol=1e-5,
            rtol=1e-5,
        )

    def test_one_dead_branch_passes_through(self):
        """If lse2 == -inf, merged equals out1."""
        B, H, D = 1, 1, 4
        lse1 = torch.zeros(B, 1, H, device=self.device, dtype=torch.float32)
        lse2 = torch.full(
            (B, 1, H), float("-inf"), device=self.device, dtype=torch.float32
        )
        out1 = torch.full((B, 1, H, D), 2.5, device=self.device, dtype=torch.bfloat16)
        out2 = torch.full((B, 1, H, D), 99.0, device=self.device, dtype=torch.bfloat16)
        merged, merged_lse = _merge_partial_attn(out1, lse1, out2, lse2)
        torch.testing.assert_close(merged.float(), out1.float(), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(merged_lse, lse1, atol=1e-5, rtol=1e-5)


@unittest.skipUnless(_IS_SM120, "SM120 (compute capability 12.0) required")
class TestEntryPointDispatch(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        cls.device = torch.device("cuda")

    def test_torch_backend_matches_triton_backend(self):
        """SGLANG_SM120_TRITON_FLASHMLA toggles backend; both return matching out."""
        k_cache, _ = _build_kvcache(4, 64, device=self.device, seed=5)
        q, indices = _build_q_indices(1, 4, 32, 4, 64, device=self.device, seed=13)
        topk_length = torch.tensor([32], dtype=torch.int32, device=self.device)

        kwargs = dict(
            q=q,
            k_cache=k_cache,
            indices=indices,
            topk_length=topk_length,
            attn_sink=None,
            head_dim_v=_D,
            softmax_scale=_D**-0.5,
        )

        with mock.patch.object(fmod, "_sm120_default_backend", "torch"):
            out_torch, _ = flash_mla_with_kvcache_sm120(**kwargs)
        with mock.patch.object(fmod, "_sm120_default_backend", "triton"):
            out_triton, _ = flash_mla_with_kvcache_sm120(**kwargs)

        torch.testing.assert_close(
            out_torch.to(torch.float32),
            out_triton.to(torch.float32),
            atol=5e-2,
            rtol=5e-2,
        )

    def test_flashinfer_backend_matches_triton(self):
        """FlashInfer SM120 sparse MLA decode matches Triton reference."""
        import importlib

        if importlib.util.find_spec("flashinfer.sparse_mla_sm120") is None:
            self.skipTest("FlashInfer SM120 sparse MLA not available")

        k_cache, _ = _build_kvcache(4, 64, device=self.device, seed=5)
        q, indices = _build_q_indices(1, 4, 32, 4, 64, device=self.device, seed=13)
        topk_length = torch.tensor([32], dtype=torch.int32, device=self.device)

        kwargs = dict(
            q=q,
            k_cache=k_cache,
            indices=indices,
            topk_length=topk_length,
            attn_sink=None,
            head_dim_v=_D,
            softmax_scale=_D**-0.5,
        )

        with mock.patch.object(fmod, "_sm120_default_backend", "triton"):
            out_triton, _ = flash_mla_with_kvcache_sm120(**kwargs)
        with mock.patch.object(fmod, "_sm120_default_backend", "flashinfer"):
            out_fi, _ = flash_mla_with_kvcache_sm120(**kwargs)

        torch.testing.assert_close(
            out_fi.to(torch.float32),
            out_triton.to(torch.float32),
            atol=5e-2,
            rtol=5e-2,
        )


if __name__ == "__main__":
    import sys

    sys.exit(unittest.main())
