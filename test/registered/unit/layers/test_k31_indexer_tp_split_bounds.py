"""Row bounds of the Kimi-K3.1 prefill indexer TP split at small extends (>= 64 rows on ROCm)."""

import random
import unittest

import torch

from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _bounds(seq_lens, extend_lens, tp):
    return Indexer._indexer_tp_split_bounds(
        seq_lens_cpu=torch.tensor(seq_lens),
        extend_lens=extend_lens,
        q_offset=sum(extend_lens),
        tp=tp,
    )


class TestIndexerTpSplitBounds(CustomTestCase):
    def test_every_rank_owns_a_contiguous_nonempty_slice(self):
        # all_gatherv rebuilds rows [0, q_offset) in rank order, so the bounds must
        # tile the rows exactly with no empty rank, down to q_offset == tp.
        rng = random.Random(0)
        cases = [([143424], [64]), ([100], [8]), ([200000, 40, 40], [6, 1, 1])]
        for _ in range(200):
            n = rng.randint(1, 6)
            ext = [rng.randint(1, 300) for _ in range(n)]
            cases.append(([e + rng.randint(0, 300000) for e in ext], ext))
        for seq, ext in cases:
            if sum(ext) < 8:
                continue
            bounds = _bounds(seq, ext, tp=8)
            with self.subTest(seq=seq, ext=ext):
                self.assertEqual(len(bounds), 9)
                self.assertEqual(bounds[0], 0)
                self.assertEqual(bounds[-1], sum(ext))
                self.assertTrue(all(a < b for a, b in zip(bounds, bounds[1:])))

    def test_slices_balance_kv_length_not_row_count(self):
        # One long-context request next to short ones: the long rows cost the most
        # logits and top-k work, so they must be spread over more ranks.
        seq, ext = [140000, 600, 600], [64, 64, 64]
        bounds = _bounds(seq, ext, tp=8)
        rows_on_long_request = sum(1 for b in bounds[1:-1] if b < 64)
        self.assertGreaterEqual(rows_on_long_request, 6)


if __name__ == "__main__":
    unittest.main()
