"""Producer grouping for the TileLang sparse MLA decode, on gfx950.

``_pick_inner_iter`` used to size the grid to a CTA count, on the assumption
that more CTAs is better. Each producer CTA is one wavefront, so once the grid
is a few waves deep the cost is set by how the last wave quantizes onto the
CUs instead: at seq 72 the threshold picks inner_iter 4, which leaves the last
wave three quarters full, while inner_iter 2 fills it to 0.90 and measures
7.15% faster. The threshold is wrong at four of the seven row counts below.

Pure host-side selection, so no GPU is required.
"""

import unittest

from sglang.kernels.ops.attention.dsa.tilelang_kernel import _pick_inner_iter
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-a", runner_config="cpu")
register_amd_ci(est_time=5, suite="stage-a-test-cpu-amd")

# GLM-5.2 TP4/EP4: 16 heads (one head block), topk 2048 over 64-key tiles, on
# a 256-CU MI355X with the deployed block_per_cu of 2.
NI = 2048 // 64
CU = 256
BPC = 2
HEAD_BLOCKS = 1

# (seq, inner_iter that measured faster). Measured with a HIP graph over 20
# calls, event timing, same-process ABBA, 45-75s warmup; spread across
# repetitions 0.08-0.28%.
MEASURED = [
    (48, 2),  # 2 is 2.94% faster
    (60, 4),  # 2 is 9.33% slower
    (66, 2),  # 2 is 12.66% faster
    (72, 2),  # 2 is 7.15% faster
    (80, 2),  # 2 is 0.88% faster
    (84, 4),  # 2 is 7.31% slower
    (96, 4),  # 2 is 3.23% slower
]


class TestSparseMlaInnerIter(CustomTestCase):
    def test_matches_measured_grouping(self):
        for seq, want in MEASURED:
            with self.subTest(seq=seq):
                got = _pick_inner_iter(seq, NI, CU, BPC, HEAD_BLOCKS)
                self.assertEqual(got, want)

    def test_cta_threshold_alone_is_wrong_at_four_of_seven(self):
        # The guard for the regression this replaces: without a head-block
        # count the old CTA threshold stands, and it misses these four.
        wrong = [
            seq for seq, want in MEASURED if _pick_inner_iter(seq, NI, CU, BPC) != want
        ]
        self.assertEqual(wrong, [60, 66, 72, 80])

    def test_confined_to_the_two_and_four_rungs(self):
        # A pure-occupancy pick lands on inner_iter 1 at seq 72 and 84, which
        # measures 20-24% slower: each extra split adds a combine the
        # occupancy model does not price. So the choice may only move between
        # 2 and 4, and must leave every other rung the threshold picks alone.
        for seq in range(1, 129):
            with self.subTest(seq=seq):
                base = _pick_inner_iter(seq, NI, CU, BPC)
                got = _pick_inner_iter(seq, NI, CU, BPC, HEAD_BLOCKS)
                if base in (2, 4):
                    self.assertIn(got, (2, 4))
                else:
                    self.assertEqual(got, base)

    def test_unstated_head_extent_keeps_the_threshold(self):
        # The DSv4 Pro call sites do not state their grid's head extent and
        # were never measured, so they must stay on the old pick.
        for seq in range(1, 129):
            with self.subTest(seq=seq):
                self.assertEqual(
                    _pick_inner_iter(seq, NI, CU, BPC, 0),
                    _pick_inner_iter(seq, NI, CU, BPC),
                )


if __name__ == "__main__":
    unittest.main()
