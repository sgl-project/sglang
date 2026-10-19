"""SM80 V4.1 real writers -> real paged reader, against CPU-only oracles.

Run directly with unittest on an SM80 GPU. This deliberately does not inherit
TestV41KVStore (which is gated to SM100/gfx950), or register/change CI gates.
All FP8 casts in the reference quantizers/dequantizers happen on the CPU.
Only finite inputs are tested, including both signs of zero; arbitrary NaN
payloads and hardware-specific NaN canonicalization are outside this contract.
"""

import unittest

import torch
from sglang.kernels.ops.attention.dsv4.attn import fused_store_cache
from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
    dequantize_k_cache_paged_v41,
)
from sglang.kernels.ops.attention.dsv4.elementwise import (
    fused_k_norm_rope_flashmla,
    fused_q_norm_rope,
)
from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.kernels.ops.attention.dsv4.torch_quant import (
    dequantize_k_cache_v41,
    quantize_k_cache_v41,
)
from sglang.test.kernels.deepseek_v4.dsv41_kv_quant_reference import (
    dequantize_k_cache_v41_fp4,
    quantize_k_cache_v41_fp4,
)

CASES = {
    KVLayout.V41: (quantize_k_cache_v41, dequantize_k_cache_v41),
    KVLayout.V41_FP4: (quantize_k_cache_v41_fp4, dequantize_k_cache_v41_fp4),
}
NUM_PAGES = 3


def _is_sm80():
    return (
        torch.version.hip is None
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability() == (8, 0)
    )


def _bits(x):
    """Compare BF16 bits, not values: +0 and -0 are distinct."""
    assert x.dtype == torch.bfloat16
    return x.contiguous().view(torch.int16)


def _signs(n, heads=None):
    shape = (n, 512) if heads is None else (n, heads, 512)
    ids = torch.arange(torch.tensor(shape).prod().item()).reshape(shape)
    return torch.where((ids + ids // 7) % 3 == 0, -1.0, 1.0).to(torch.bfloat16)


def _weight():
    # BF16-exact, varied across tiles and within each tile. No zero entries:
    # eps=0 on +/-1 gives exactly sign * weight before BF16 rounding.
    ids = torch.arange(512)
    return ((1.0 + (ids % 7).float() / 8) * torch.exp2((ids % 9 - 4).float())).to(
        torch.bfloat16
    )


def _freqs_table(max_pos=13):
    # Exactly identity or i, avoiding sin/cos and reduction-order tolerances.
    # Vary by both position and complex pair so passing a table instead of
    # gathered per-token frequencies to standalone store cannot pass by chance.
    ids = torch.arange(max_pos)[:, None] + torch.arange(32)[None, :]
    use_i = ids % 3 == 1
    return torch.complex((~use_i).float(), use_i.float())


def _rope_cpu(x, per_token_freqs):
    """Independent real-arithmetic RoPE, with explicit BF16 output rounding."""
    assert x.device.type == per_token_freqs.device.type == "cpu"
    assert per_token_freqs.shape == (x.shape[0], 32)
    pairs = x[..., -64:].float().reshape(*x.shape[:-1], 32, 2)
    shape = (x.shape[0],) + (1,) * (x.ndim - 2) + (32,)
    c, s = per_token_freqs.real.reshape(shape), per_token_freqs.imag.reshape(shape)
    a, b = pairs[..., 0], pairs[..., 1]
    tail = torch.stack((a * c - b * s, b * c + a * s), dim=-1)
    return torch.cat((x[..., :-64], tail.flatten(-2).to(torch.bfloat16)), dim=-1)


def _locs(page_size, dtype=torch.int32):
    # 0, both sides of a page crossing, page ends, and the final slot;
    # all unique for P=2 as well. One slot/page subset remains unwritten.
    return torch.tensor(
        [0, page_size - 1, page_size, 2 * page_size - 1, 3 * page_size - 1],
        dtype=dtype,
    )


def _reference_pages(layout, page_size, locs, values):
    """CPU quantization, then restore zero-initialized unwritten cache rows."""
    assert locs.device.type == values.device.type == "cpu"
    assert torch.all(locs >= 0)
    full = torch.zeros(NUM_PAGES, page_size, 512, dtype=torch.bfloat16)
    full.view(-1, 512)[locs.long()] = values
    quant, _ = CASES[layout]
    pages = quant(full, page_bytes=layout.page_bytes(page_size))
    written = torch.zeros(NUM_PAGES * page_size, dtype=torch.bool)
    written[locs.long()] = True
    written = written.reshape(NUM_PAGES, page_size, 1)
    # Q(full zeros) has NONZERO scale bytes. Real sparse writes leave their
    # unwritten footer bytes zero, so merely quantizing full is not an oracle.
    lo = layout.scale_offset(page_size)
    hi = lo + page_size * layout.scale_bytes
    footer = pages[:, lo:hi].reshape(NUM_PAGES, page_size, layout.scale_bytes)
    footer = footer.masked_fill(~written, 0)
    pages[:, lo:hi] = footer.reshape(NUM_PAGES, -1)  # explicit footer writeback
    payload = pages[:, :lo].reshape(NUM_PAGES, page_size, layout.data_bytes)
    assert torch.count_nonzero(payload.masked_select(~written)) == 0
    assert torch.count_nonzero(pages[:, hi:]) == 0  # page padding stays zero
    return pages


@unittest.skipUnless(_is_sm80(), "requires CUDA SM80 (8, 0)")
class TestV41SM80Store(unittest.TestCase):
    def _cache(self, layout, page_size):
        return torch.zeros(
            NUM_PAGES, layout.page_bytes(page_size), dtype=torch.uint8, device="cuda"
        )

    def _assert_writer_reader(self, cache, layout, page_size, locs, values):
        ref = _reference_pages(layout, page_size, locs, values)
        self.assertTrue(torch.equal(cache.cpu(), ref), "whole-page writer bytes differ")
        # Read every slot, including untouched rows, in reverse order, and repeat
        # the first/last written locations. Cover both index specializations.
        for dtype in (torch.int32, torch.int64):
            ids = torch.cat(
                (torch.arange(NUM_PAGES * page_size - 1, -1, -1), locs.long())
            ).to(dtype)
            with self.subTest(reader_index_dtype=dtype):
                got = dequantize_k_cache_paged_v41(
                    cache, ids.cuda(), page_size, layout=layout
                )
                _, dequant = CASES[layout]
                expected = dequant(ref, page_size).reshape(-1, 512)[ids.long()]
                expected = expected.unsqueeze(1)
                self.assertEqual(got.shape, (ids.numel(), 1, 512))
                self.assertEqual(got.dtype, torch.bfloat16)
                self.assertTrue(torch.isfinite(expected).all())
                self.assertTrue(torch.equal(_bits(got.cpu()), _bits(expected)))

    def test_fused_store_cache(self):
        """Standalone real writer, no negative locations (it has no -1 guard)."""
        for layout in CASES:
            for page_size in (2, 64):
                for dtype in (torch.int32, torch.int64):
                    with self.subTest(
                        layout=layout.name, page_size=page_size, idx=dtype
                    ):
                        locs = _locs(page_size, dtype)
                        x = (_signs(locs.numel()).float() * _weight().float()).to(
                            torch.bfloat16
                        )
                        x[0, :32] = 0.0
                        x[1, 32:64] = -0.0
                        x[2, 64:80:2] = -(2**-16)
                        cache = self._cache(layout, page_size)
                        fused_store_cache(
                            x.cuda(),
                            cache,
                            locs.cuda(),
                            page_size=page_size,
                            type="flashmla",
                            layout=layout,
                        )
                        self._assert_writer_reader(cache, layout, page_size, locs, x)

    def test_fused_store_cache_with_rope(self):
        """True in-kernel RoPE takes per-token frequencies, not a full table."""
        table = _freqs_table()
        pos = torch.tensor([12, 1, 7, 0, 5], dtype=torch.int64)
        for layout in CASES:
            for page_size in (2, 64):
                with self.subTest(layout=layout.name, page_size=page_size):
                    locs = _locs(page_size, torch.int64)
                    x = (_signs(5).float() * _weight().float()).to(torch.bfloat16)
                    freqs = table[pos].contiguous()
                    cache = self._cache(layout, page_size)
                    fused_store_cache(
                        x.cuda(),
                        cache,
                        locs.cuda(),
                        page_size=page_size,
                        type="flashmla",
                        layout=layout,
                        freqs_cis=freqs.cuda(),
                    )
                    self._assert_writer_reader(
                        cache, layout, page_size, locs, _rope_cpu(x, freqs)
                    )

    def test_fused_k_norm_rope_flashmla(self):
        """Exact RMSNorm, strided qkv backing, full RoPE table, and -1 skip."""
        table = _freqs_table()
        pos = torch.tensor([12, 1, 7, 0, 5, 11], dtype=torch.int64)
        signs, weight = _signs(6), _weight()
        normed = (signs.float() * weight.float()).to(torch.bfloat16)
        rotated = _rope_cpu(normed, table[pos])
        for layout in CASES:
            for page_size in (2, 64):
                with self.subTest(layout=layout.name, page_size=page_size):
                    locs = torch.cat(
                        (_locs(page_size), torch.tensor([-1], dtype=torch.int32))
                    )
                    backing = torch.full(
                        (6, 768), 17.0, dtype=torch.bfloat16, device="cuda"
                    )
                    backing[:, 256:] = signs.cuda()
                    kv = backing[:, 256:]
                    self.assertEqual(kv.stride(), (768, 1))
                    self.assertFalse(kv.is_contiguous())
                    before = backing.clone()
                    cache = self._cache(layout, page_size)
                    fused_k_norm_rope_flashmla(
                        kv,
                        weight.cuda(),
                        0.0,
                        table.cuda(),
                        pos.cuda(),
                        locs.cuda(),
                        cache,
                        page_size,
                        layout=layout,
                    )
                    self.assertTrue(
                        torch.equal(backing, before), "KNORM mutated its input"
                    )
                    valid = locs >= 0
                    self._assert_writer_reader(
                        cache, layout, page_size, locs[valid], rotated[valid]
                    )

    def test_fused_q_norm_rope(self):
        """Real unweighted Q RMSNorm+RoPE is BF16-bit-exact on +/-1 rows."""
        table = _freqs_table()
        pos = torch.tensor([12, 1, 7, 0, 5, 11], dtype=torch.int64)
        for heads in (1, 3, 9):
            with self.subTest(heads=heads):
                q_cpu = _signs(6, heads)
                # mean(square(+/-1))=1, eps=0: no CPU/GPU reduction ambiguity.
                expected = _rope_cpu(q_cpu, table[pos])
                q = q_cpu.cuda()
                out = torch.empty_like(q)
                fused_q_norm_rope(q, out, 0.0, table.cuda(), pos.cuda())
                self.assertEqual(out.shape, (6, heads, 512))
                self.assertTrue(torch.equal(_bits(q.cpu()), _bits(q_cpu)))
                self.assertTrue(torch.equal(_bits(out.cpu()), _bits(expected)))

    def test_fixed_scale_boundary_tiles(self):
        """Unit scales: FP8 ties/subnormals and FP4 ties-to-even, signed zero."""
        tiles = {
            KVLayout.V41: [
                448.0,
                -448.0,
                1.0625,
                1.1875,
                -1.0625,
                -1.1875,
                2**-10,
                3 * 2**-10,
                -(2**-10),
                -3 * 2**-10,
                0.0,
                -0.0,
                8.5,
                9.5,
                240.0,
                -240.0,
            ]
            * 2,
            KVLayout.V41_FP4: [
                6.0,
                -6.0,
                0.25,
                0.75,
                1.25,
                1.75,
                2.5,
                3.5,
                5.0,
                -0.25,
                -0.75,
                -2.5,
                -3.5,
                -5.0,
                0.0,
                -0.0,
            ],
        }
        for layout, tile in tiles.items():
            for page_size in (2, 64):
                with self.subTest(layout=layout.name, page_size=page_size):
                    locs = _locs(page_size)
                    x = torch.tensor(tile, dtype=torch.bfloat16).repeat(
                        locs.numel(), 512 // len(tile)
                    )
                    cache = self._cache(layout, page_size)
                    fused_store_cache(
                        x.cuda(),
                        cache,
                        locs.cuda(),
                        page_size=page_size,
                        type="flashmla",
                        layout=layout,
                    )
                    self._assert_writer_reader(cache, layout, page_size, locs, x)


if __name__ == "__main__":
    unittest.main()
