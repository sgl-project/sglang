import unittest

from sglang.srt.mem_cache.kv_cache_configurator import dcp_virtual_loc_extent
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

MAX_TOTAL = 1_000_000


class TestDcpVirtualLocExtent(CustomTestCase):
    """How wide a replicated DCP buffer has to be.

    The LightningIndexer's index-K and a draft worker's pools are addressed at a
    raw, untranslated ``loc``, so they must span every location the allocator can
    issue. Sharded buffers translate ``// dcp_size`` and stay per-rank.

    This is arithmetic worth pinning rather than inlining, because every
    configuration CI runs has ``dcp_size == 1``, where all the interesting terms
    collapse to 1 and any of several wrong expressions look right.
    """

    def test_without_dcp_nothing_is_scaled(self):
        self.assertEqual(dcp_virtual_loc_extent(MAX_TOTAL, 1, 1), MAX_TOTAL)

    def test_the_target_worker_scales_up_to_the_virtual_range(self):
        # Its own pools arrive unscaled, so the replicated buffer needs the
        # full factor.
        for dcp_size in (2, 4, 8, 16):
            with self.subTest(dcp_size=dcp_size):
                self.assertEqual(
                    dcp_virtual_loc_extent(MAX_TOTAL, dcp_size, 1),
                    MAX_TOTAL * dcp_size,
                )

    def test_the_draft_worker_is_not_scaled_twice(self):
        """The regression this file exists for. A draft worker's sizes arrive
        already multiplied by loc_space_scale, so a bare ``* attn_dcp_size``
        asks for ``max_total * dcp_size**2`` -- 16x the intended buffer at
        DCP16, and silent, because the pool still allocates a valid shape."""
        for dcp_size in (2, 4, 8, 16):
            with self.subTest(dcp_size=dcp_size):
                already_scaled = MAX_TOTAL * dcp_size
                self.assertEqual(
                    dcp_virtual_loc_extent(already_scaled, dcp_size, dcp_size),
                    MAX_TOTAL * dcp_size,
                )

    def test_both_workers_land_on_the_same_extent(self):
        """The invariant underneath: whatever scaling a worker's own pools got,
        the replicated buffer covers the same virtual space on both."""
        for dcp_size in (2, 4, 8, 16):
            with self.subTest(dcp_size=dcp_size):
                target = dcp_virtual_loc_extent(MAX_TOTAL, dcp_size, 1)
                draft = dcp_virtual_loc_extent(MAX_TOTAL * dcp_size, dcp_size, dcp_size)
                self.assertEqual(target, draft)

    def test_an_unexpected_loc_space_scale_is_rejected(self):
        # loc_space_scale is 1 or attn_dcp_size by construction; anything else
        # means the assumption above has been broken elsewhere and the division
        # would silently produce a wrong extent.
        with self.assertRaises(AssertionError):
            dcp_virtual_loc_extent(MAX_TOTAL, 8, 4)


if __name__ == "__main__":
    unittest.main()
