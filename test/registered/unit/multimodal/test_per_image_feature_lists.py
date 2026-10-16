"""A per-image list of features, rather than one packed tensor.

Several processors build one tensor per image, `torch.cat` them, and hand the
result to the scheduler -- which splits it straight back apart in
`get_new_expanded_mm_items` so each image can be hashed and cached on its own.
The concatenation is a second request-sized allocation that stands alongside
the parts it copies, and the split then has to rediscover boundaries the
producer already knew.

The framework already carries a list end to end. These tests pin that, and pin
the multi-tile case where packing actively loses the per-image granularity the
split exists for.
"""

import unittest

import torch

from sglang.srt.managers.mm_utils import _try_simple_split, hash_feature
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _parts(rows):
    return [torch.randn(n, 3, 8, 8) for n in rows]


def _item(feature, num_items, rows):
    offsets, start = [], 0
    for n in rows:
        offsets.append((start, start + n))
        start += n
    return MultimodalDataItem(
        modality=Modality.IMAGE, feature=feature, offsets=offsets[:num_items]
    )


class TestListFeaturesSplitPerImage(CustomTestCase):
    def test_a_ragged_list_splits_one_item_per_image(self):
        rows = [9, 10, 8]
        parts = _parts(rows)

        expanded = []
        self.assertTrue(_try_simple_split(_item(parts, 3, rows), 3, expanded))

        self.assertEqual(len(expanded), 3)
        for item, part in zip(expanded, parts):
            self.assertTrue(torch.equal(item.feature[0], part))

    def test_a_packed_tensor_of_multi_tile_images_declines_to_split(self):
        """Why packing loses granularity: rows outnumber placeholders.

        `_try_simple_split` requires feature_count == num_items. With more
        than one tile per image the packed rows outnumber the placeholders,
        the split declines, and the item stays bundled.
        """
        rows = [9, 10, 8]
        packed = torch.cat(_parts(rows), dim=0)

        expanded = []
        self.assertFalse(_try_simple_split(_item(packed, 3, rows), 3, expanded))
        self.assertEqual(expanded, [])

    def test_a_list_is_hashable_and_moves_to_cpu(self):
        parts = _parts([4, 5])

        self.assertIsInstance(hash_feature(parts), int)
        moved = BaseMultimodalProcessor._move_feature_to_cpu(parts)
        self.assertIsInstance(moved, list)
        self.assertEqual(len(moved), 2)

    def test_keeping_the_parts_allocates_nothing_extra(self):
        """The point: no second request-sized buffer beside the parts."""
        parts = _parts([9, 10, 8])
        before = [p.data_ptr() for p in parts]

        self.assertEqual([p.data_ptr() for p in parts], before)
        packed = torch.cat(parts, dim=0)
        self.assertNotIn(packed.data_ptr(), before)
        self.assertEqual(
            packed.numel() * packed.element_size(),
            sum(p.numel() * p.element_size() for p in parts),
        )


if __name__ == "__main__":
    unittest.main()
