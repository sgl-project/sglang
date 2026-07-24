"""Unit test for the page-aware ``DSATokenToKVPool.move_kv_cache`` (index-K + scale).

Regression for the DSA index-K move bug: ``index_k_with_scale_buffer`` is PAGE-indexed
(dim-0 is the page; within a page row the ``page_size`` fp8 keys form one block followed
by a block of ``page_size`` fp32 scales), but ``move_kv_cache`` receives per-TOKEN
locations. A plain per-token row copy is correct only for ``page_size == 1``; for
``page_size == 64`` (all CUDA DSA) it indexes the page dim with token locations -> wrong
rows / out of bounds. The fix maps each token to its ``(page, offset)`` and moves both the
fp8-key and fp32-scale sub-slices. These tests pin that per-token sub-slice move without a
GPU by hand-building the buffer and stubbing the (token-indexed, correct) base MLA move.
"""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, MLATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

HD = 128  # index_head_dim (DSA asserts 128)


def _no_base_move(self, tgt_loc, src_loc):
    return None


def _views(buf, page_size, num_pages):
    off = buf.storage_offset()
    fp8 = buf.as_strided((num_pages, page_size, HD), (buf.stride(0), HD, 1), off)
    scale = buf.as_strided(
        (num_pages, page_size, 4), (buf.stride(0), 4, 1), off + page_size * HD
    )
    return fp8, scale


class TestDSAMoveKVCachePageAware(CustomTestCase):
    def _run(self, page_size, src, tgt):
        src, tgt = torch.tensor(src), torch.tensor(tgt)
        touched = set(src.tolist()) | set(tgt.tolist())
        assert (
            max(touched) < 251
        ), "checked locations must be collision-free under loc % 251"
        num_tokens = max(touched) + 2  # room for one untouched token past the max
        num_pages = (num_tokens + page_size - 1) // page_size
        num_tokens = num_pages * page_size

        pool = DSATokenToKVPool.__new__(DSATokenToKVPool)
        pool.page_size = page_size
        pool.index_head_dim = HD
        row = page_size * (HD + 4)  # fp8 block + fp32-scale block
        buf = torch.zeros(num_pages, row, dtype=torch.uint8)
        pool.index_k_with_scale_buffer = [buf]

        fp8, scale = _views(buf, page_size, num_pages)
        for loc in range(
            num_tokens
        ):  # unique signature per token in both key and scale
            pg, off = divmod(loc, page_size)
            fp8[pg, off] = loc % 251
            scale[pg, off, 0] = loc % 251

        want = {int(t): int(s) % 251 for t, s in zip(tgt, src)}
        with patch.object(MLATokenToKVPool, "move_kv_cache", _no_base_move):
            pool.move_kv_cache(tgt, src)

        fp8, scale = _views(buf, page_size, num_pages)
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


if __name__ == "__main__":
    unittest.main()
