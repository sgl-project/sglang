import unittest

import numpy as np
from sglang.srt.disaggregation.layerwise_kv import (
    compact_layer_block_ids,
    select_compact_layer_entries,
    select_prepped_layer_indices,
)


def repeat_indices_over_blocks(indices, num_blocks, block_length):
    offsets = np.arange(num_blocks, dtype=np.int32) * block_length
    return (offsets[:, None] + indices[None, :]).ravel().astype(np.int32)


class TestCompactLayerIndices(unittest.TestCase):
    def test_mha_tensor_major_block_order(self):
        self.assertEqual(
            compact_layer_block_ids([0, 3], num_layers=4, num_tensor_kinds=2),
            [0, 3, 4, 7],
        )

    def test_selected_indices_match_full_descriptor_blocks(self):
        slots = np.array([1, 4], dtype=np.int32)
        full = repeat_indices_over_blocks(slots, num_blocks=8, block_length=10)
        block_ids = compact_layer_block_ids([0, 3], num_layers=4, num_tensor_kinds=2)
        selected = select_prepped_layer_indices(slots, block_ids, block_length=10)
        expected = np.concatenate(
            [
                full[block_id * len(slots) : (block_id + 1) * len(slots)]
                for block_id in block_ids
            ]
        )
        np.testing.assert_array_equal(selected, expected)

    def test_all_layers_equal_monolithic_indices(self):
        slots = np.array([0, 2, 5], dtype=np.int32)
        block_ids = compact_layer_block_ids([0, 1, 2], num_layers=3, num_tensor_kinds=2)
        selected = select_prepped_layer_indices(slots, block_ids, block_length=8)
        full = repeat_indices_over_blocks(slots, num_blocks=6, block_length=8)
        np.testing.assert_array_equal(selected, full)

    def test_global_or_duplicate_layer_ids_fail_loudly(self):
        with self.assertRaisesRegex(ValueError, "out of range"):
            compact_layer_block_ids([7], num_layers=4, num_tensor_kinds=2)
        with self.assertRaisesRegex(ValueError, "must be unique"):
            compact_layer_block_ids([1, 1], num_layers=4, num_tensor_kinds=2)

    def test_tensor_major_state_entries_preserve_all_components(self):
        positions = select_compact_layer_entries(
            [3, 7, 11, 3, 7, 11],
            compact_layer_ids=[3, 7, 11],
            layer_slots=[0, 2],
        )
        self.assertEqual(positions, [0, 2, 3, 5])

    def test_state_entry_selection_rejects_invalid_slots(self):
        with self.assertRaisesRegex(ValueError, "out of range"):
            select_compact_layer_entries([3, 7, 3, 7], [3, 7], [2])
        with self.assertRaisesRegex(ValueError, "must be unique"):
            select_compact_layer_entries([3, 7, 3, 7], [3, 7], [1, 1])

    def test_empty_selection(self):
        selected = select_prepped_layer_indices(
            np.array([1, 2], dtype=np.int32), [], block_length=8
        )
        self.assertEqual(selected.dtype, np.int32)
        self.assertEqual(selected.size, 0)


if __name__ == "__main__":
    unittest.main()
