"""Byte-exactness of the V4.1 (fp8 / fp4) FlashMLA KV cache store kernels.

Every store kernel is compared byte for byte with the pure-torch quantizers of
the two formats (torch_quant.quantize_k_cache_v41 / _v41_fp4), which
follow the decode kernel's own reference quantizer.
"""

import math
import unittest
from itertools import product

import torch

from sglang.kernels.ops.attention.deepseek_v4_rope import set_batched_rope
from sglang.kernels.ops.attention.dsv4.dequant_k_cache import dequantize_k_cache_paged
from sglang.kernels.ops.attention.dsv4.elementwise import (
    fused_k_norm_rope_flashmla,
    fused_rope_inplace,
)
from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.kernels.ops.attention.dsv4.torch_quant import (
    dequantize_k_cache_v41,
    fake_quant_compressed_kv,
    quantize_k_cache_v41,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4TokenToKVPool,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test import dsv41_kv_quant_reference as tq
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# the V4.1 store kernels are the SM100 / gfx950 JIT kernels; the only Blackwell runner
# config is the four-GPU one
register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
register_amd_ci(est_time=60, suite="stage-b-kernel-test-1-gpu-amd-mi35x")

REFERENCE = {
    KVLayout.V41: quantize_k_cache_v41,
    KVLayout.V41_FP4: tq.quantize_k_cache_v41_fp4,
}
DEQUANT = {
    KVLayout.V41: dequantize_k_cache_v41,
    KVLayout.V41_FP4: tq.dequantize_k_cache_v41_fp4,
}
# One quantization step, relative: e4m3 has 3 mantissa bits, e2m1 one.
ONE_CODE_RTOL = {KVLayout.V41: 0.13, KVLayout.V41_FP4: 0.51}


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


def token_rows(pages, layout, page_size, locs):
    """The (data row, scale row) bytes of the tokens at locs."""
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


def _make_pool(ratios, kv_source_layers, kv_layout, compressed_kv_layout=None, **sizes):
    return DeepSeekV4TokenToKVPool(
        max_num_reqs=16,
        swa_size=FULL_SIZE,
        c4_size=sizes.get("c4_size", 0),
        c128_size=sizes.get("c128_size", 0),
        c4_state_pool_size=sizes.get("c4_state_pool_size", 0),
        c128_state_pool_size=sizes.get("c128_state_pool_size", 0),
        page_size=PAGE_SIZE,
        swa_page_size=PAGE_SIZE,
        dtype=torch.float8_e4m3fn,
        c4_state_dtype=torch.float32,
        c128_state_dtype=torch.float32,
        qk_nope_head_dim=HEAD_DIM - ROPE_DIM,
        qk_rope_head_dim=ROPE_DIM,
        indexer_head_dim=128,
        layer_num=len(ratios),
        device="cuda",
        enable_memory_saver=False,
        compression_ratios=ratios,
        kv_source_layers=kv_source_layers,
        full_size=FULL_SIZE,
        kv_layout=kv_layout,
        compressed_kv_layout=compressed_kv_layout,
    )


def _v41_store_kernels_available() -> bool:
    if not torch.cuda.is_available():
        return False
    if is_hip():
        return is_gfx95_supported()
    return torch.cuda.get_device_capability()[0] >= 10


@unittest.skipUnless(
    _v41_store_kernels_available(), "the V4.1 store kernels need gfx950 or SM100"
)
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
            # A full page and a padded 2-token page, each with one index dtype
            # (the layout, the pad and the dtype are the three template axes).
            for page_size, num_pages, n, idx_dtype in (
                (64, 9, 333, torch.int32),
                (2, 40, 37, torch.int64),
            ):
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

        g = torch.Generator(device="cuda").manual_seed(1)
        for layout in (KVLayout.V41, KVLayout.V41_FP4):
            for page_size, num_pages, n in ((64, 5, 200),):
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
                            torch.equal(deq, fake_quant_compressed_kv(rotated))
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

    def test_compress_norm_rope_store(self):
        """The ratio-4 / ratio-128 writer (norm + RoPE + store from a decode plan) in
        the V4.1 layouts: bitwise on exact-norm rows, one-code close on general rows."""
        from sglang.kernels.ops.attention.dsv4.compress import (
            CompressorDecodePlan,
            compress_norm_rope_store,
        )

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


CASES = {
    KVLayout.V41: (quantize_k_cache_v41, dequantize_k_cache_v41),
    KVLayout.V41_FP4: (tq.quantize_k_cache_v41_fp4, tq.dequantize_k_cache_v41_fp4),
}


def bits(t: torch.Tensor) -> torch.Tensor:
    """bf16 as int16, so that -0.0 and NaN payloads compare exactly."""
    return t.contiguous().view(torch.int16)


@unittest.skipUnless(is_hip() and is_gfx95_supported(), "requires gfx950")
class TestV41KVDequant(CustomTestCase):
    def _gather_ref(self, dequant, pages, page_size, ids):
        return dequant(pages, page_size).view(-1, 512)[ids.long()].unsqueeze(1)

    def test_quantized_pages(self):
        g = torch.Generator(device="cuda").manual_seed(0)
        for layout, (quant, dequant) in CASES.items():
            for page_size, num_pages in ((64, 9), (2, 50)):
                with self.subTest(layout=layout.name, page_size=page_size):
                    k = torch.randn(
                        num_pages,
                        page_size,
                        512,
                        generator=g,
                        device="cuda",
                        dtype=torch.bfloat16,
                    )
                    k = (
                        k
                        * torch.exp2(
                            torch.randint(
                                -12,
                                6,
                                (num_pages, page_size, 1),
                                generator=g,
                                device="cuda",
                            ).float()
                        )
                    ).to(torch.bfloat16)
                    k[0, 0, :32] = 0
                    k[0, 0, 32:48] = -0.0
                    pages = quant(k, page_bytes=layout.page_bytes(page_size))
                    ids = torch.randint(
                        0,
                        num_pages * page_size,
                        (777,),
                        generator=g,
                        device="cuda",
                        dtype=torch.int32,
                    )
                    got = dequantize_k_cache_paged(pages, ids, page_size, layout=layout)
                    self.assertEqual(got.shape, (777, 1, 512))
                    self.assertTrue(
                        torch.equal(
                            bits(got),
                            bits(self._gather_ref(dequant, pages, page_size, ids)),
                        )
                    )
                    # The fp4 cache dequantizes to the model's fake-quantized value
                    # (compared by value: the fake quant maps an exact -0.0 to +0.0).
                    if layout is KVLayout.V41_FP4:
                        expect = fake_quant_compressed_kv(
                            k.view(-1, 512)[ids.long()]
                        ).unsqueeze(1)
                        self.assertTrue(torch.equal(got, expect))

    def test_random_bytes_and_workspace_slice(self):
        """Arbitrary payload bytes (scales in the quantizer's range) and an
        out that is a strided slice of a larger workspace."""
        g = torch.Generator(device="cuda").manual_seed(1)
        for layout, (_, dequant) in CASES.items():
            page_size, num_pages = 64, 7
            with self.subTest(layout=layout.name):
                pages = torch.randint(
                    0,
                    256,
                    (num_pages, layout.page_bytes(page_size)),
                    generator=g,
                    dtype=torch.uint8,
                    device="cuda",
                )
                if layout is KVLayout.V41:
                    lo = layout.scale_offset(page_size)
                    hi = lo + page_size * layout.scale_bytes
                    pages[:, lo:hi] = torch.randint(
                        100,
                        140,
                        (num_pages, hi - lo),
                        generator=g,
                        dtype=torch.uint8,
                        device="cuda",
                    )
                ids = torch.randint(
                    0,
                    num_pages * page_size,
                    (300,),
                    generator=g,
                    device="cuda",
                    dtype=torch.int64,
                )
                ref = self._gather_ref(dequant, pages, page_size, ids)
                workspace = torch.zeros(
                    305, 1, 512, dtype=torch.bfloat16, device="cuda"
                )
                out = dequantize_k_cache_paged(
                    pages, ids, page_size, out=workspace[5:], layout=layout
                )
                # NaN payloads (fp8 0x7F / e4m3 NaN scales) compare through their bits.
                self.assertTrue(torch.equal(bits(workspace[5:]), bits(ref)))
                self.assertEqual(int(workspace[:5].abs().sum()), 0)


HEAD_DIM, ROPE_DIM, NOPE_DIM = 512, 64, 448
# the pool under test: one page size for the SWA and the compressed caches
PAGE_SIZE = 256
FULL_SIZE = 4 * PAGE_SIZE


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestFusedKNormRopeFlashMLA(CustomTestCase):
    @unittest.skipUnless(
        is_hip() and is_gfx95_supported(),
        "the query rope rides the HIP K launch; its bitwise parity with the flat rope"
        " kernel is claimed on gfx950 only",
    )
    def test_query_rope_in_the_k_launch(self):
        """With q the K launch must rope every query head's trailing ROPE_DIM bitwise
        like the flat rope kernel, leave the cache bytes and the nope part untouched,
        and rope rows without a slot."""
        dev = "cuda"
        page_size = 256
        for layout, (num_tokens, heads, pos_dtype, seed) in product(
            (KVLayout.V4, KVLayout.V41),
            ((1, 16, torch.int64, 0), (300, 16, torch.int32, 2)),
        ):
            with self.subTest(layout=layout, num_tokens=num_tokens, heads=heads):
                torch.manual_seed(seed)
                kv = torch.randn(num_tokens, HEAD_DIM, device=dev, dtype=torch.bfloat16)
                weight = (1 + 0.1 * torch.randn(HEAD_DIM, device=dev)).to(
                    torch.bfloat16
                )
                angles = torch.rand(8192, ROPE_DIM // 2, device=dev) * 2 * math.pi
                freqs_cis = torch.polar(torch.ones_like(angles), angles)
                positions = torch.randint(0, 8192, (num_tokens,), device=dev).to(
                    pos_dtype
                )
                out_loc = torch.randperm(4 * page_size, device=dev)[:num_tokens]
                out_loc = out_loc.to(torch.int32)
                if num_tokens > 2:
                    out_loc[1] = -1
                page_bytes = layout.page_bytes(page_size)
                cache = torch.zeros(4, page_bytes, device=dev, dtype=torch.uint8)
                cache_q = cache.clone()
                q = (torch.randn(num_tokens, heads, HEAD_DIM, device=dev) * 3).to(
                    torch.bfloat16
                )
                expected = q.clone()
                # The model's standalone query rope (batched flat kernel).
                set_batched_rope(True)
                fused_rope_inplace(
                    expected[..., -ROPE_DIM:], None, freqs_cis, positions
                )
                got = q.clone()
                fused_k_norm_rope_flashmla(
                    kv,
                    weight,
                    1e-6,
                    freqs_cis,
                    positions,
                    out_loc,
                    cache,
                    page_size,
                    layout=layout,
                )
                fused_k_norm_rope_flashmla(
                    kv,
                    weight,
                    1e-6,
                    freqs_cis,
                    positions,
                    out_loc,
                    cache_q,
                    page_size,
                    q=got,
                    layout=layout,
                )
                self.assertTrue(torch.equal(got, expected))
                self.assertTrue(torch.equal(got[..., :NOPE_DIM], q[..., :NOPE_DIM]))
                self.assertTrue(torch.equal(cache_q, cache))


@unittest.skipUnless(
    _v41_store_kernels_available(), "the V4.1 store kernels need gfx950 or SM100"
)
class TestV41KVPoolWriters(CustomTestCase):
    """The pool hands its fused writers the layout, page size and slots: a wrong
    hand-off escapes every kernel-level case above."""

    @classmethod
    def setUpClass(cls):
        set_global_server_args_for_scheduler(
            ServerArgs(model_path="dummy", page_size=PAGE_SIZE)
        )

    def test_fused_writers_round_trip(self):
        """SWA write (fp8) and compressed write with in-kernel RoPE (fp4) read back
        through the layout-aware dequant as the reference values."""
        from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
            dequantize_k_cache_paged,
        )
        from sglang.srt.layers.attention.dsv4.dsv41_sparse import rope_tail

        pool = _make_pool([0, 0, 2, 1], [2, 3], KVLayout.V41)
        g = torch.Generator(device="cuda").manual_seed(3)
        n = 100
        # SWA: finished (normed, rotated) bf16 rows.
        x = torch.randn(n, HEAD_DIM, generator=g, device="cuda", dtype=torch.bfloat16)
        swa_loc = torch.randperm(FULL_SIZE, generator=g, device="cuda")[:n].to(
            torch.int32
        )
        pool.set_swa_key_buffer_radix_fused(layer_id=0, swa_loc=swa_loc, cache_k=x)
        got = dequantize_k_cache_paged(
            pool.get_swa_key_buffer_radix(0),
            swa_loc,
            pool.swa_page_size,
            layout=pool.get_swa_key_layout(),
        )
        ref = dequantize_k_cache_v41(
            quantize_k_cache_v41(x.view(1, n, HEAD_DIM)), n
        ).view(n, 1, HEAD_DIM)
        self.assertTrue(torch.equal(got, ref))
        # Compressed (fp4): the un-rotated latent plus its freqs; the cache holds
        # exactly fake_quant_compressed_kv(rope_tail(latent)).
        layer_id = pool.sources_by_ratio[1][0]
        latent = torch.randn(
            n, HEAD_DIM, generator=g, device="cuda", dtype=torch.bfloat16
        )
        angles = torch.randn(n, ROPE_DIM // 2, generator=g, device="cuda")
        freqs = torch.polar(torch.ones_like(angles), angles)
        loc = torch.randperm(FULL_SIZE, generator=g, device="cuda")[:n].to(torch.int64)
        pool.set_extra_key_buffer_fused(
            layer_id=layer_id, loc=loc, cache_k=latent, freqs_cis=freqs
        )
        got = dequantize_k_cache_paged(
            pool.get_extra_key_buffer(layer_id),
            loc,
            pool.get_extra_key_page_size(layer_id),
            layout=pool.get_extra_key_layout(layer_id),
        )
        self.assertTrue(
            torch.equal(
                got.squeeze(1),
                fake_quant_compressed_kv(rope_tail(latent, freqs, ROPE_DIM)),
            )
        )
        # The (fp8 nope, bf16 rope) pack writer is the V4 layout only.
        with self.assertRaises(AssertionError):
            pool.set_swa_key_buffer(0, swa_loc, None)


if __name__ == "__main__":
    unittest.main()
