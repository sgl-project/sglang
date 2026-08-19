"""Unit test for the page-aware ``IndexKeyCache.move`` (DSA index-K + scale).

Regression for the DSA index-K move bug: the index buffer is PAGE-indexed (dim-0 is the
page; within a page row the ``page_size`` fp8 keys form one block followed by a block of
``page_size`` fp32 scales), but ``move`` receives per-TOKEN locations. A plain per-token
row copy is correct only for ``page_size == 1``; for ``page_size == 64`` (all CUDA DSA)
it indexes the page dim with token locations -> wrong rows / out of bounds. The fix maps
each token to its ``(page, offset)`` and moves both the fp8-key and fp32-scale
sub-slices. These tests pin that per-token sub-slice move without a GPU by hand-building
the buffer behind a stub pool.
"""

import unittest

import torch

from sglang.srt.mem_cache.index_key_cache import IndexKeyCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

HD = 128  # index_head_dim (DSA asserts 128)


class _StubPool:
    """Only the attributes ``IndexKeyCache.move`` reads."""

    def __init__(self, page_size, quant_block_size=HD):
        self.page_size = page_size
        self.index_head_dim = HD
        self.quant_block_size = quant_block_size


def _views(buf, page_size, num_pages, sc=4):
    off = buf.storage_offset()
    fp8 = buf.as_strided((num_pages, page_size, HD), (buf.stride(0), HD, 1), off)
    scale = buf.as_strided(
        (num_pages, page_size, sc), (buf.stride(0), sc, 1), off + page_size * HD
    )
    return fp8, scale


class TestIndexKeyCacheMovePageAware(CustomTestCase):
    def _run(self, page_size, src, tgt, quant_block_size=HD):
        # Scale bytes per token follow _buffer_shape: index_head_dim // quant_block_size
        # fp32 values. quant_block_size < HD makes that wider than 4, which is what
        # catches a hard-coded 4 in the as_strided view.
        sc = HD // quant_block_size * 4
        src, tgt = torch.tensor(src), torch.tensor(tgt)
        touched = set(src.tolist()) | set(tgt.tolist())
        assert (
            max(touched) < 251
        ), "checked locations must be collision-free under loc % 251"
        num_tokens = max(touched) + 2  # room for one untouched token past the max
        num_pages = (num_tokens + page_size - 1) // page_size
        num_tokens = num_pages * page_size

        row = page_size * (HD + sc)  # fp8 block + fp32-scale block
        buf = torch.zeros(num_pages, row, dtype=torch.uint8)
        cache = IndexKeyCache.__new__(IndexKeyCache)
        cache.pool = _StubPool(page_size, quant_block_size)
        # A 0-row placeholder layer (skip-topk layer) must be skipped, not indexed.
        cache.buffer = [torch.zeros(0, row, dtype=torch.uint8), buf]

        fp8, scale = _views(buf, page_size, num_pages, sc)
        for loc in range(
            num_tokens
        ):  # unique signature per token in both key and scale
            pg, off = divmod(loc, page_size)
            fp8[pg, off] = loc % 251
            scale[pg, off, 0] = loc % 251

        want = {int(t): int(s) % 251 for t, s in zip(tgt, src)}
        cache.move(tgt, src)

        fp8, scale = _views(buf, page_size, num_pages, sc)
        for t in tgt.tolist():
            pg, off = divmod(t, page_size)
            self.assertEqual(
                int(fp8[pg, off, 0]), want[t], f"ps={page_size} tgt={t}: fp8 key"
            )
            self.assertEqual(
                int(scale[pg, off, 0]), want[t], f"ps={page_size} tgt={t}: scale"
            )
        untouched = next(l for l in range(num_tokens) if l not in touched)
        pg, off = divmod(untouched, page_size)
        self.assertEqual(
            int(fp8[pg, off, 0]),
            untouched % 251,
            f"ps={page_size}: token {untouched} clobbered",
        )

    def test_page_size_64_moves_correct_subslices(self):
        # Cross-page + varied intra-page offsets. Fails on the pre-fix whole-row copy
        # (wrong page / OOB) and on the raising guard.
        self._run(64, src=[1, 66, 158, 249], tgt=[0, 70, 130, 200])

    def test_page_size_1_still_correct(self):
        self._run(1, src=[0, 1, 2, 3], tgt=[4, 5, 6, 7])

    def test_scale_width_follows_quant_block_size(self):
        # quant_block_size=64 over index_head_dim=128 => 2 fp32 scales (8 bytes) per
        # token, not 4. Fails if the scale view hard-codes a 4-byte stride.
        self._run(64, src=[1, 66, 158, 249], tgt=[0, 70, 130, 200], quant_block_size=64)


if __name__ == "__main__":
    unittest.main()
