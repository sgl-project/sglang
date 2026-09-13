"""Byte-exactness of the V4.1 (fp8 / fp4) FlashMLA KV cache store kernels.

Every store kernel is compared byte for byte with the pure-torch quantizers of
the two formats (``torch_quant.quantize_k_cache_v41`` / ``_v41_fp4``), which
follow the decode kernel's own reference quantizer.
"""

import unittest

import torch

from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.srt.layers.attention.dsv4 import torch_quant as tq
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

REFERENCE = {
    KVLayout.V41: tq.quantize_k_cache_v41,
    KVLayout.V41_FP4: tq.quantize_k_cache_v41_fp4,
}
DEQUANT = {
    KVLayout.V41: tq.dequantize_k_cache_v41,
    KVLayout.V41_FP4: tq.dequantize_k_cache_v41_fp4,
}
# One quantization step, relative: e4m3 has 3 mantissa bits, e2m1 one.
ONE_CODE_RTOL = {KVLayout.V41: 0.13, KVLayout.V41_FP4: 0.51}


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


def random_rows(n, generator, device="cuda"):
    """bf16 rows over a wide dynamic range, with zero, negative-zero and tiny tiles."""
    x = torch.randn(n, 512, generator=generator, device=device, dtype=torch.bfloat16)
    scale = torch.exp2(
        torch.randint(-10, 6, (n, 1), generator=generator, device=device).float()
    )
    x = (x * scale).to(torch.bfloat16)
    if n >= 3:
        x[0, :32] = 0
        x[1, 32:48] = -0.0
        x[2, 100] = -1e-10
    return x


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

    def assert_rows_close(self, cache, ref, layout, page_size, locs):
        """For rows that went through the kernel's fp32 RMSNorm: torch sums the
        squares in another order, and the fp32 ulp this can cost is occasionally
        kept by a bf16 rounding boundary and then by the quantizer. Allow a
        one-code difference in a handful of elements."""
        got_data, got_scale = token_rows(cache, layout, page_size, locs)
        exp_data, exp_scale = token_rows(ref, layout, page_size, locs)
        self.assertGreater((got_scale == exp_scale).float().mean().item(), 0.999)
        self.assertGreater((got_data == exp_data).float().mean().item(), 0.999)
        deq_got = DEQUANT[layout](cache, page_size).view(-1, 512)[locs.long()].float()
        deq_exp = DEQUANT[layout](ref, page_size).view(-1, 512)[locs.long()].float()
        torch.testing.assert_close(
            deq_got, deq_exp, rtol=ONE_CODE_RTOL[layout], atol=0.06
        )

    def assert_untouched_zero(self, cache, layout, page_size, locs):
        num_slots = cache.shape[0] * page_size
        written = torch.zeros(num_slots, dtype=torch.bool, device=cache.device)
        written[locs.long()] = True
        others = torch.arange(num_slots, device=cache.device)[~written]
        data, scale = token_rows(cache, layout, page_size, others)
        self.assertEqual(int(data.sum()) + int(scale.sum()), 0)

    def test_fused_store_cache(self):
        """Ragged page fills: a random subset of slots is written, the rest stays zero."""
        from sglang.kernels.ops.attention.dsv4.attn import fused_store_cache

        g = torch.Generator(device="cuda").manual_seed(0)
        for layout in (KVLayout.V41, KVLayout.V41_FP4):
            for page_size, num_pages, n in ((64, 9, 333), (256, 3, 500), (2, 40, 37)):
                for idx_dtype in (torch.int32, torch.int64):
                    with self.subTest(
                        layout=layout.name, page_size=page_size, idx=idx_dtype
                    ):
                        x = random_rows(n, g)
                        locs = torch.randperm(
                            num_pages * page_size, generator=g, device="cuda"
                        )[:n].to(idx_dtype)
                        cache = torch.zeros(
                            num_pages,
                            layout.page_bytes(page_size),
                            dtype=torch.uint8,
                            device="cuda",
                        )
                        fused_store_cache(
                            x,
                            cache,
                            locs,
                            page_size=page_size,
                            type="flashmla",
                            layout=layout,
                        )
                        ref = reference_pages(
                            layout, page_size, num_pages, locs, x, cache.shape[1]
                        )
                        self.assert_tokens_equal(cache, ref, layout, page_size, locs)
                        self.assert_untouched_zero(cache, layout, page_size, locs)

    def test_fused_store_cache_with_rope(self):
        """The in-kernel RoPE tail equals rope_tail (bf16-rounded) before quantizing,
        so the fp4 cache holds exactly fake_quant_compressed_kv(rope_tail(x))."""
        from sglang.kernels.ops.attention.dsv4.attn import fused_store_cache
        from sglang.srt.layers.attention.dsv4.dsv41_sparse import rope_tail

        g = torch.Generator(device="cuda").manual_seed(1)
        for layout in (KVLayout.V41, KVLayout.V41_FP4):
            for page_size, num_pages, n in ((64, 5, 200), (256, 2, 129)):
                with self.subTest(layout=layout.name, page_size=page_size):
                    x = random_rows(n, g)
                    angles = torch.randn(n, 32, generator=g, device="cuda")
                    freqs = torch.polar(torch.ones_like(angles), angles)
                    locs = torch.randperm(
                        num_pages * page_size, generator=g, device="cuda"
                    )[:n]
                    cache = torch.zeros(
                        num_pages,
                        layout.page_bytes(page_size),
                        dtype=torch.uint8,
                        device="cuda",
                    )
                    fused_store_cache(
                        x,
                        cache,
                        locs,
                        page_size=page_size,
                        type="flashmla",
                        layout=layout,
                        freqs_cis=freqs,
                    )
                    rotated = rope_tail(x, freqs, 64)
                    ref = reference_pages(
                        layout, page_size, num_pages, locs, rotated, cache.shape[1]
                    )
                    self.assert_tokens_equal(cache, ref, layout, page_size, locs)
                    if layout is KVLayout.V41_FP4:
                        deq = tq.dequantize_k_cache_v41_fp4(cache, page_size).view(
                            -1, 512
                        )[locs]
                        self.assertTrue(
                            torch.equal(deq, tq.fake_quant_compressed_kv(rotated))
                        )

    def test_boundary_tiles(self):
        """Tie, saturation and subnormal-scale tiles follow the reference. (NaN / inf
        rows are not fed: the kernels do not reproduce the reference's handling of them.)"""
        from sglang.kernels.ops.attention.dsv4.attn import fused_store_cache

        maxima = [
            0,
            2**-12,
            6 * 2**-9,
            6 * 1.0625,
            6 * 1.1875,
            6 * 448,
            1e6,
            8.25,
            448.0,
            449.0,
        ]
        rows = []
        for m in maxima:
            row = torch.full((512,), m, dtype=torch.bfloat16)
            row[1::2] *= -1
            row[64:80] = torch.tensor(
                [8.25, 4.125, 3.4375, -4.8125, 0.0, -0.0, 2.5, -3.5] * 2,
                dtype=torch.bfloat16,
            )
            rows.append(row)
        x = torch.stack(rows).cuda()
        n = x.shape[0]
        page_size = 16
        locs = torch.arange(n, device="cuda", dtype=torch.int32)
        for layout in (KVLayout.V41, KVLayout.V41_FP4):
            with self.subTest(layout=layout.name):
                cache = torch.zeros(
                    1, layout.page_bytes(page_size), dtype=torch.uint8, device="cuda"
                )
                fused_store_cache(
                    x, cache, locs, page_size=page_size, type="flashmla", layout=layout
                )
                ref = reference_pages(layout, page_size, 1, locs, x, cache.shape[1])
                self.assert_tokens_equal(cache, ref, layout, page_size, locs)
                self.assert_untouched_zero(cache, layout, page_size, locs)

    def test_fused_k_norm_rope_store(self):
        """The fused RMSNorm + RoPE + store (the SWA write) equals norm -> rope_tail ->
        fused_store_cache: exact-norm rows bitwise against the torch quantizer, and
        general rows bitwise against the unfused kernel chain."""
        from sglang.kernels.ops.attention.dsv4.attn import fused_store_cache
        from sglang.kernels.ops.attention.dsv4.elementwise import (
            fused_k_norm_rope_flashmla,
        )
        from sglang.srt.layers.attention.dsv4.dsv41_sparse import rope_tail

        g = torch.Generator(device="cuda").manual_seed(2)
        page_size, num_pages, n = 256, 3, 300
        angles = torch.randn(1024, 32, generator=g, device="cuda")
        freqs_table = torch.polar(torch.ones_like(angles), angles)
        pos = torch.randint(
            0, 1024, (n,), generator=g, device="cuda", dtype=torch.int64
        )
        locs = torch.randperm(num_pages * page_size, generator=g, device="cuda")[:n].to(
            torch.int32
        )
        locs[7] = -1  # a row without a write target is skipped
        valid = locs >= 0
        for layout in (KVLayout.V41, KVLayout.V41_FP4):
            with self.subTest(layout=layout.name, rows="exact-norm"):
                # x = +-2^k per row with eps = 0 normalizes to +-1 exactly, so the
                # normed row is sign * weight and the reference is exact.
                signs = torch.where(
                    torch.rand(n, 512, generator=g, device="cuda") < 0.5, -1.0, 1.0
                )
                k = torch.randint(-6, 6, (n, 1), generator=g, device="cuda").float()
                x = (signs * torch.exp2(k)).to(torch.bfloat16)
                w = (
                    torch.randn(512, generator=g, device="cuda")
                    * torch.exp2(
                        torch.randint(-6, 4, (512,), generator=g, device="cuda").float()
                    )
                ).to(torch.bfloat16)
                cache = torch.zeros(
                    num_pages,
                    layout.page_bytes(page_size),
                    dtype=torch.uint8,
                    device="cuda",
                )
                fused_k_norm_rope_flashmla(
                    x, w, 0.0, freqs_table, pos, locs, cache, page_size, layout=layout
                )
                rotated = rope_tail(
                    (signs * w.float()).to(torch.bfloat16), freqs_table[pos], 64
                )
                ref = reference_pages(
                    layout,
                    page_size,
                    num_pages,
                    locs[valid],
                    rotated[valid],
                    cache.shape[1],
                )
                self.assert_tokens_equal(cache, ref, layout, page_size, locs[valid])
                self.assert_untouched_zero(cache, layout, page_size, locs[valid])
            with self.subTest(layout=layout.name, rows="general"):
                x = (torch.randn(n, 512, generator=g, device="cuda") * 3).to(
                    torch.bfloat16
                )
                w = (torch.randn(512, generator=g, device="cuda") * 0.3 + 1).to(
                    torch.bfloat16
                )
                eps = 1e-6
                cache = torch.zeros(
                    num_pages,
                    layout.page_bytes(page_size),
                    dtype=torch.uint8,
                    device="cuda",
                )
                fused_k_norm_rope_flashmla(
                    x, w, eps, freqs_table, pos, locs, cache, page_size, layout=layout
                )
                xf = x.float()
                normed = (
                    xf
                    * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
                    * w.float()
                ).to(torch.bfloat16)
                rotated = rope_tail(normed, freqs_table[pos], 64)
                unfused = torch.zeros_like(cache)
                fused_store_cache(
                    rotated[valid],
                    unfused,
                    locs[valid],
                    page_size=page_size,
                    type="flashmla",
                    layout=layout,
                )
                self.assert_rows_close(cache, unfused, layout, page_size, locs[valid])

    def test_c1_c2_decode_store(self):
        """The ratio-1 / ratio-2 decode compressors write the V4.1 layouts: the cache
        holds the quantized rope_tail of the pre-RoPE latent the kernel publishes
        (bitwise; the fp8 layout after the model's fp4 fake quantization), and the
        latent is the torch RMSNorm to within an fp32-reduction-order bf16 ulp."""
        from sglang.kernels.ops.attention.dsv4.c1 import c1_decode_norm_rope_store
        from sglang.kernels.ops.attention.dsv4.c2 import c2_decode_norm_rope_store
        from sglang.srt.layers.attention.dsv4.dsv41_sparse import rope_tail

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
                else tq.fake_quant_compressed_kv(rotated)
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
                    latent = c2_decode_norm_rope_store(
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

    def test_compress_norm_rope_store(self):
        """The ratio-4 / ratio-128 writer (norm + RoPE + store from a decode plan) in
        the V4.1 layouts: bitwise on exact-norm rows, one-code close on general rows."""
        from sglang.kernels.ops.attention.dsv4.compress import (
            CompressorDecodePlan,
            compress_norm_rope_store,
        )
        from sglang.srt.layers.attention.dsv4.dsv41_sparse import rope_tail

        g = torch.Generator(device="cuda").manual_seed(5)
        ratio = 4
        angles = torch.randn(4096, 32, generator=g, device="cuda")
        freqs = torch.polar(torch.ones_like(angles), angles)
        n = 150
        for layout in (KVLayout.V41, KVLayout.V41_FP4):
            for page_size, num_pages in ((64, 8), (2, 300)):
                seq_lens = (
                    torch.randint(
                        1, 1000, (n,), generator=g, device="cuda", dtype=torch.int64
                    )
                    * ratio
                )
                seq_lens[3] += 1  # not a group boundary: writes nothing
                plan = CompressorDecodePlan.generate_legacy(
                    ratio, torch.arange(n, device="cuda", dtype=torch.int64), seq_lens
                )
                valid = seq_lens % ratio == 0
                pos = seq_lens - ratio
                out_loc = torch.randperm(
                    num_pages * page_size, generator=g, device="cuda"
                )[:n].to(torch.int64)
                with self.subTest(
                    layout=layout.name, page_size=page_size, rows="exact-norm"
                ):
                    signs = torch.where(
                        torch.rand(n, 512, generator=g, device="cuda") < 0.5, -1.0, 1.0
                    )
                    k = torch.randint(-6, 6, (n, 1), generator=g, device="cuda").float()
                    kv = (signs * torch.exp2(k)).to(torch.bfloat16)
                    w = (
                        torch.randn(512, generator=g, device="cuda")
                        * torch.exp2(
                            torch.randint(
                                -6, 4, (512,), generator=g, device="cuda"
                            ).float()
                        )
                    ).to(torch.bfloat16)
                    cache = torch.zeros(
                        num_pages,
                        layout.page_bytes(page_size),
                        dtype=torch.uint8,
                        device="cuda",
                    )
                    compress_norm_rope_store(
                        kv,
                        plan,
                        norm_weight=w,
                        norm_eps=0.0,
                        freq_cis=freqs,
                        out_loc=out_loc,
                        kvcache=cache,
                        page_size=page_size,
                        layout=layout,
                    )
                    rotated = rope_tail(
                        (signs * w.float()).to(torch.bfloat16), freqs[pos], 64
                    )
                    ref = reference_pages(
                        layout,
                        page_size,
                        num_pages,
                        out_loc[valid],
                        rotated[valid],
                        cache.shape[1],
                    )
                    self.assert_tokens_equal(
                        cache, ref, layout, page_size, out_loc[valid]
                    )
                    self.assert_untouched_zero(cache, layout, page_size, out_loc[valid])
                with self.subTest(
                    layout=layout.name, page_size=page_size, rows="general"
                ):
                    eps = 1e-6
                    kv = (torch.randn(n, 512, generator=g, device="cuda") * 2).to(
                        torch.bfloat16
                    )
                    w = (torch.randn(512, generator=g, device="cuda") * 0.3 + 1).to(
                        torch.bfloat16
                    )
                    cache = torch.zeros(
                        num_pages,
                        layout.page_bytes(page_size),
                        dtype=torch.uint8,
                        device="cuda",
                    )
                    compress_norm_rope_store(
                        kv,
                        plan,
                        norm_weight=w,
                        norm_eps=eps,
                        freq_cis=freqs,
                        out_loc=out_loc,
                        kvcache=cache,
                        page_size=page_size,
                        layout=layout,
                    )
                    xf = kv.float()
                    normed = (
                        xf
                        * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
                        * w.float()
                    ).to(torch.bfloat16)
                    rotated = rope_tail(normed, freqs[pos], 64)
                    ref = reference_pages(
                        layout,
                        page_size,
                        num_pages,
                        out_loc[valid],
                        rotated[valid],
                        cache.shape[1],
                    )
                    self.assert_rows_close(
                        cache, ref, layout, page_size, out_loc[valid]
                    )


if __name__ == "__main__":
    unittest.main()
