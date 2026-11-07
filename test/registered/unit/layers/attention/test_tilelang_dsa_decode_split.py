"""Split selection for the TileLang DSA sparse decode (partial + combine).

The decode kernel runs `ni // inner_iter` partial groups per query and the
combine kernel merges them, so `inner_iter` must divide `ni`. A top-k width
extended by the KPool tail (2051, padded to 2112: 66 tiles at block_I 32) is not
a power of two, and trying only `ni, ni // 2, ni // 4, ...` leaves it at 2 (33
groups) for 4 to 65 query rows on 132 SMs. With `any_divisor`, a larger divisor
is taken only when its blocks fill the waves they need. These tests pin that
choice, keep the power-of-two widths (top-k 2048) unchanged, and keep the
default search for callers that do not opt in.

    python -m pytest test/registered/unit/layers/attention/test_tilelang_dsa_decode_split.py -v
"""

import unittest

from sglang.kernels.ops.attention.dsa.tilelang_kernel import _pick_inner_iter
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

# (cu, block_per_cu) of the tilelang_sparse_fwd branches, passed explicitly so
# the tests are device-independent: 132 and 148 SMs, then ROCm gfx942 and gfx950.
DEVICE_SHAPES = ((132, 1), (148, 1), (304, 1), (256, 2))


class TestTileLangDSADecodeSplit(CustomTestCase):
    def test_kpool_width_on_132_sms(self):
        """66 tiles on 132 SMs: with one head block, 24 and 48 rows pick 6 and 44
        rows pick 11, since 11 or 22 there launch 144 or 132 blocks, about one per
        SM, which ran slower than 33 groups; with 4 head blocks (64 heads), 32
        rows pick 11."""
        self.assertEqual(
            _pick_inner_iter(24, 66, 132, 1, any_divisor=True, head_blocks=1), 6
        )
        self.assertEqual(
            _pick_inner_iter(48, 66, 132, 1, any_divisor=True, head_blocks=1), 6
        )
        self.assertEqual(
            _pick_inner_iter(44, 66, 132, 1, any_divisor=True, head_blocks=1), 11
        )
        self.assertEqual(
            _pick_inner_iter(32, 66, 132, 1, any_divisor=True, head_blocks=4), 11
        )
        self.assertEqual(_pick_inner_iter(32, 66, 132, 1), 2)

    def test_power_of_two_width_picks_are_unchanged(self):
        """Top-k 2048: every divisor is a halving of ni, so both searches agree."""
        for ni in (32, 64):
            for cu, block_per_cu in DEVICE_SHAPES:
                for head_blocks in (1, 2, 4, 8):
                    for seq in range(1, cu * block_per_cu + 2):
                        with self.subTest(ni=ni, cu=cu, hb=head_blocks, seq=seq):
                            self.assertEqual(
                                _pick_inner_iter(
                                    seq,
                                    ni,
                                    cu,
                                    block_per_cu,
                                    any_divisor=True,
                                    head_blocks=head_blocks,
                                ),
                                _pick_inner_iter(seq, ni, cu, block_per_cu),
                            )

    def test_default_search_tries_only_halvings(self):
        """The default search (ROCm callers) tries only ni, ni // 2, ...: on gfx950,
        33 tiles split into 33 groups below 512 query rows and 1 group from 512."""
        for seq in (47, 171, 511):
            with self.subTest(seq=seq):
                self.assertEqual(_pick_inner_iter(seq, 33, 256, 2), 1)
        self.assertEqual(_pick_inner_iter(512, 33, 256, 2), 33)


if __name__ == "__main__":
    unittest.main()
