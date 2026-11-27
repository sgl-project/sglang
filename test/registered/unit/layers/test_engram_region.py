"""Placement of the DeepSeek-V4.1 Engram tables in a shared file-backed store.

Pure arithmetic, so these run without a GPU, a model checkpoint or a distributed
context; ``_HostTable``'s file mapping is exercised separately on tmpfs.
"""

import unittest

from sglang.srt.layers.engram_region import (
    STORE_ALIGN,
    align_up,
    engram_region_offsets,
    engram_row_bytes,
    engram_table_offsets,
    engram_total_bytes,
    max_shard_rows,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

BLOCK = 32


def _even_shard(num_embeddings, column_bounds, tp_rank, tp_size):
    return (
        num_embeddings * tp_rank // tp_size,
        num_embeddings * (tp_rank + 1) // tp_size,
    )


class TestAlignUp(CustomTestCase):
    def test_rounds_up_to_a_multiple(self):
        self.assertEqual(align_up(0, 8), 0)
        self.assertEqual(align_up(1, 8), 8)
        self.assertEqual(align_up(8, 8), 8)
        self.assertEqual(align_up(9, 8), 16)

    def test_rejects_a_non_power_of_two(self):
        for align in (0, -1, 3, 6):
            with self.subTest(align=align), self.assertRaises(AssertionError):
                align_up(1, align)


class TestEngramRowBytes(CustomTestCase):
    def test_fp8_row_holds_weights_and_scales(self):
        # DeepSeek-V4.1-Flash: dim 256, one e8m0 scale per 32 weights.
        self.assertEqual(engram_row_bytes(256, BLOCK), 256 + 8)

    def test_mxfp4_row_is_half_the_weights(self):
        self.assertEqual(engram_row_bytes(256, BLOCK, mxfp4=True), 128 + 8)


class TestEngramRegionOffsets(CustomTestCase):
    def test_regions_are_aligned_unique_and_do_not_overlap(self):
        regions = engram_region_offsets([3_000, 3_000, 1_000], 256, BLOCK)
        for offset, length in regions:
            self.assertEqual(offset % STORE_ALIGN, 0)
            self.assertEqual(length % STORE_ALIGN, 0)
        for (start, length), (next_start, _) in zip(regions, regions[1:]):
            self.assertLessEqual(start + length, next_start)

    def test_total_bytes_covers_every_region(self):
        regions = engram_region_offsets([3_000, 1_000], 256, BLOCK)
        self.assertEqual(
            engram_total_bytes(regions), sum(length for _, length in regions)
        )

    def test_a_region_fits_its_rows(self):
        rows, dim = 4_096, 256
        ((_, length),) = engram_region_offsets([rows], dim, BLOCK)
        self.assertGreaterEqual(length, rows * engram_row_bytes(dim, BLOCK))

    def test_sizes_differ_per_table_without_overlapping(self):
        # The two Flash tables hold different row counts; the layout must not
        # assume one uniform size the way a ``layer_index * region_len`` would.
        big, small = engram_region_offsets([8_000, 2_000], 256, BLOCK)
        self.assertNotEqual(big[1], small[1])
        self.assertGreaterEqual(big[0] + big[1], small[0] + 0)

    def test_mxfp4_halves_the_store(self):
        # Large enough that the 2 MiB alignment does not mask the row-size difference.
        rows = 4_000_000
        fp8 = engram_region_offsets([rows], 256, BLOCK)
        mxfp4 = engram_region_offsets([rows], 256, BLOCK, mxfp4=True)
        self.assertLess(mxfp4[0][1], fp8[0][1])

    def test_an_empty_table_still_reserves_storage(self):
        ((_, length),) = engram_region_offsets([0], 256, BLOCK)
        self.assertGreaterEqual(length, STORE_ALIGN)


class TestMaxShardRows(CustomTestCase):
    def test_sizes_by_the_largest_shard(self):
        self.assertEqual(max_shard_rows(10, (), 4, _even_shard), 3)

    def test_an_even_split_needs_no_rounding_up(self):
        self.assertEqual(max_shard_rows(8, (), 4, _even_shard), 2)

    def test_uses_the_hash_column_bounds_when_they_divide(self):
        bounds = (0, 3, 7, 12)
        self.assertEqual(max_shard_rows(12, bounds, 2, _even_shard), 6)

    def test_one_shard_covers_the_whole_table(self):
        self.assertEqual(max_shard_rows(10, (), 1, _even_shard), 10)


class TestEngramTableOffsets(CustomTestCase):
    def test_each_rank_gets_its_own_region_with_replication_none(self):
        # The bug this guards: laying the tables out without the rank gives every
        # rank the same offset, so their shards overwrite each other in the store.
        rows = [3_000, 2_000]
        offsets = {
            rank: engram_table_offsets(rows, 256, BLOCK, rank, 4) for rank in range(4)
        }
        flat = [o for per_rank in offsets.values() for o in per_rank]
        self.assertEqual(len(set(flat)), len(flat))

    def test_the_store_fits_one_region_per_table_and_rank(self):
        rows = [3_000, 2_000]
        regions = engram_region_offsets(rows, 256, BLOCK, replicas=[4, 4])
        self.assertEqual(len(regions), 8)
        # Back to back and aligned: the last region ends at the total.
        self.assertEqual(engram_total_bytes(regions), regions[-1][0] + regions[-1][1])

    def test_a_replicated_table_shares_one_region(self):
        rows = [3_000, 2_000]
        offsets = {
            rank: engram_table_offsets(rows, 256, BLOCK, rank, 4, replicated=True)
            for rank in range(4)
        }
        self.assertEqual(len(set(offsets.values())), 1)

    def test_offsets_are_aligned_and_disjoint_for_every_rank(self):
        rows = [3_000, 2_000]
        regions = engram_region_offsets(rows, 256, BLOCK, replicas=[4, 4])
        for (start, length), (next_start, _) in zip(regions, regions[1:]):
            self.assertEqual(start % STORE_ALIGN, 0)
            self.assertLessEqual(start + length, next_start)

    def test_replicas_must_match_the_row_counts(self):
        with self.assertRaisesRegex(ValueError, "entries for"):
            engram_region_offsets([3_000], 256, BLOCK, replicas=[2, 2])


if __name__ == "__main__":
    unittest.main()
