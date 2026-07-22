import unittest

import torch

from sglang.srt.mem_cache.shared_kv.layout import OwnerShardedLayout
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestOwnerShardedLayout(CustomTestCase):
    def test_kv_rows_map_to_rank_major_owner_segments(self):
        layout = OwnerShardedLayout(
            cp_size=4,
            ownership_granule=256,
            logical_rows=10 * 256,
        )
        logical_rows = torch.arange(10, dtype=torch.int64) * 256

        self.assertEqual(layout.logical_blocks, 10)
        self.assertEqual(layout.blocks_per_rank, 3)
        self.assertEqual(layout.physical_rows_per_rank, 3 * 256)
        self.assertEqual(
            layout.physical_rows(logical_rows).tolist(),
            [0, 768, 1536, 2304, 256, 1024, 1792, 2560, 512, 1280],
        )

    def test_c4_state_ring_uses_four_row_ownership_blocks(self):
        layout = OwnerShardedLayout(
            cp_size=2,
            ownership_granule=4,
            logical_rows=8,
        )
        logical_rows = torch.arange(8, dtype=torch.int64)

        self.assertEqual(
            layout.owner_rank(logical_rows).tolist(),
            [0, 0, 0, 0, 1, 1, 1, 1],
        )
        self.assertEqual(
            layout.owner_local_rows(logical_rows).tolist(),
            [0, 1, 2, 3, 0, 1, 2, 3],
        )

    def test_offline_c128_state_uses_one_native_ownership_block(self):
        layout = OwnerShardedLayout(
            cp_size=8,
            ownership_granule=128,
            logical_rows=128,
        )
        logical_rows = torch.tensor([0, 63, 127], dtype=torch.int64)

        self.assertEqual(layout.logical_blocks, 1)
        self.assertEqual(layout.blocks_per_rank, 1)
        self.assertEqual(layout.owner_rank(logical_rows).tolist(), [0, 0, 0])
        self.assertEqual(layout.owner_local_rows(logical_rows).tolist(), [0, 63, 127])

    def test_rank_relative_rows_rotate_the_current_rank_to_segment_zero(self):
        layout = OwnerShardedLayout(
            cp_size=4,
            ownership_granule=4,
            logical_rows=10 * 4,
        )
        logical_rows = torch.arange(10, dtype=torch.int64) * 4

        self.assertEqual(
            layout.rank_relative_rows(logical_rows, rank=2).tolist(),
            [24, 36, 0, 12, 28, 40, 4, 16, 32, 44],
        )

    def test_physical_mapping_uses_the_aligned_owner_segment_stride(self):
        layout = OwnerShardedLayout(
            cp_size=4,
            ownership_granule=4,
            logical_rows=10 * 4,
            physical_blocks_per_rank=8,
        )
        logical_rows = torch.tensor([0, 4, 8, 12, 16], dtype=torch.int64)

        self.assertEqual(layout.minimum_blocks_per_rank, 3)
        self.assertEqual(layout.blocks_per_rank, 8)
        self.assertEqual(layout.physical_rows_per_rank, 32)
        self.assertEqual(
            layout.physical_rows(logical_rows).tolist(),
            [0, 32, 64, 96, 4],
        )

        with self.assertRaisesRegex(ValueError, "physical_blocks_per_rank"):
            OwnerShardedLayout(
                cp_size=4,
                ownership_granule=4,
                logical_rows=10 * 4,
                physical_blocks_per_rank=2,
            )

    def test_negative_sentinels_are_preserved_by_every_translation(self):
        layout = OwnerShardedLayout(
            cp_size=4,
            ownership_granule=4,
            logical_rows=32,
        )
        logical_rows = torch.tensor([-1, -7, 0, 5], dtype=torch.int32)

        self.assertEqual(layout.owner_rank(logical_rows).tolist(), [-1, -7, 0, 1])
        self.assertEqual(layout.owner_local_rows(logical_rows).tolist(), [-1, -7, 0, 1])
        self.assertEqual(layout.physical_rows(logical_rows).tolist(), [-1, -7, 0, 9])
        self.assertEqual(
            layout.rank_relative_rows(logical_rows, rank=1).tolist(),
            [-1, -7, 24, 1],
        )

    def test_owned_row_mask_rejects_sentinels(self):
        layout = OwnerShardedLayout(
            cp_size=2,
            ownership_granule=4,
            logical_rows=16,
        )
        logical_rows = torch.tensor([-1, 0, 4, 8, 12], dtype=torch.int64)

        self.assertEqual(
            layout.owned_row_mask(logical_rows, rank=1).tolist(),
            [False, False, True, False, True],
        )

    def test_tensor_translations_preserve_shape_device_and_integer_dtype(self):
        layout = OwnerShardedLayout(
            cp_size=2,
            ownership_granule=4,
            logical_rows=16,
        )
        logical_rows = torch.tensor([[0, 4], [8, -1]], dtype=torch.int32)

        for translated in (
            layout.owner_rank(logical_rows),
            layout.owner_local_rows(logical_rows),
            layout.physical_rows(logical_rows),
            layout.rank_relative_rows(logical_rows, rank=1),
        ):
            self.assertEqual(translated.shape, logical_rows.shape)
            self.assertEqual(translated.dtype, logical_rows.dtype)
            self.assertEqual(translated.device, logical_rows.device)

    def test_invalid_layout_and_rank_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "cp_size"):
            OwnerShardedLayout(cp_size=0, ownership_granule=4, logical_rows=8)
        with self.assertRaisesRegex(ValueError, "ownership_granule"):
            OwnerShardedLayout(cp_size=2, ownership_granule=0, logical_rows=8)
        with self.assertRaisesRegex(ValueError, "logical_rows"):
            OwnerShardedLayout(cp_size=2, ownership_granule=4, logical_rows=-1)

        layout = OwnerShardedLayout(
            cp_size=2,
            ownership_granule=4,
            logical_rows=8,
        )
        with self.assertRaisesRegex(ValueError, "rank"):
            layout.rank_relative_rows(torch.tensor([0]), rank=2)
        with self.assertRaisesRegex(ValueError, "rank"):
            layout.owned_row_mask(torch.tensor([0]), rank=-1)


if __name__ == "__main__":
    unittest.main()
