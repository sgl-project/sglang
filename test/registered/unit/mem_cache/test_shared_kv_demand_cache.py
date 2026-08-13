import unittest

import torch

from sglang.srt.mem_cache.shared_kv.demand_cache import TransientRowDemandCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestTransientRowDemandCache(unittest.TestCase):
    def test_one_way_cache_has_power_of_two_sets_and_exact_accounting(self):
        cache = TransientRowDemandCache(
            rows=8,
            row_bytes=592,
            ways=1,
            device="cpu",
            collect_stats=False,
        )

        view = cache.next_view()

        self.assertEqual(view.rows.shape, (8, 592))
        self.assertEqual(view.tags.shape, (8, 1))
        self.assertEqual(view.ways, 1)
        self.assertEqual(cache.allocated_bytes, 8 * (592 + 8))

    def test_each_view_gets_a_fresh_epoch_without_clearing_rows(self):
        cache = TransientRowDemandCache(
            rows=8,
            row_bytes=592,
            ways=1,
            device="cpu",
            collect_stats=True,
        )
        cache.rows.fill_(7)

        first = cache.next_view()
        second = cache.next_view()

        self.assertEqual(second.epoch, first.epoch + 1)
        self.assertTrue(torch.all(cache.rows == 7))
        self.assertEqual(cache.stats.shape, (5,))

    def test_storage_rows_can_reserve_untagged_kernel_scratch(self):
        cache = TransientRowDemandCache(
            rows=16,
            tagged_rows=8,
            row_bytes=592,
            ways=1,
            device="cpu",
            collect_stats=False,
        )

        view = cache.next_view()

        self.assertEqual(view.rows.shape, (16, 592))
        self.assertEqual(view.tags.shape, (8, 1))
        self.assertEqual(cache.num_rows, 16)
        self.assertEqual(cache.num_tagged_rows, 8)
        self.assertEqual(cache.allocated_bytes, 16 * 592 + 8 * 8)

    def test_rejects_geometry_that_cannot_use_masked_set_indexing(self):
        with self.assertRaisesRegex(ValueError, "power-of-two"):
            TransientRowDemandCache(
                rows=7,
                row_bytes=592,
                ways=1,
                device="cpu",
                collect_stats=False,
            )


if __name__ == "__main__":
    unittest.main()
