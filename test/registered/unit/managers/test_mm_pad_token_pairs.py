"""Unit tests for pad_mm_token_pairs when no explicit token pairs are given."""

import unittest
from array import array

from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternTokenPairs,
    pad_mm_token_pairs,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

START, END = 10, 11


def _item() -> MultimodalDataItem:
    item = MultimodalDataItem(modality=Modality.IMAGE)
    item.set_hash(0xABCD)
    return item


class TestPadMMTokenPairsFallback(CustomTestCase):
    def test_none_pairs_without_ids_returns_input_unchanged(self):
        ids = array("q", [1, START, 5, 5, END, 2])
        mm_inputs = MultimodalInputs(mm_items=[_item()])

        out = pad_mm_token_pairs(ids, mm_inputs, data_token_pairs=None)

        self.assertEqual(list(out), list(ids))
        self.assertEqual(mm_inputs.data_offsets, [])

    def test_none_pairs_falls_back_to_mm_inputs_start_end_ids(self):
        item = _item()
        ids = array("q", [1, START, 5, 5, END, 2])
        mm_inputs = MultimodalInputs(
            mm_items=[item], im_start_id=START, im_end_id=END
        )

        out = pad_mm_token_pairs(ids, mm_inputs, data_token_pairs=None)

        self.assertEqual(list(out[:2]), [1, START])
        self.assertEqual(list(out[2:4]), item.padding_sequence(2))
        self.assertEqual(list(out[4:]), [END, 2])
        self.assertEqual(mm_inputs.data_offsets, [1])

    def test_empty_pairs_returns_input_unchanged(self):
        ids = array("q", [1, 2, 3])
        mm_inputs = MultimodalInputs(mm_items=[_item()])

        out = pad_mm_token_pairs(ids, mm_inputs, data_token_pairs=[])

        self.assertEqual(list(out), [1, 2, 3])

    def test_legacy_pattern_class_accepts_none_pairs(self):
        ids = array("q", [1, 2, 3])
        mm_inputs = MultimodalInputs(mm_items=[_item()])

        pattern = MultiModalityDataPaddingPatternTokenPairs(data_token_pairs=None)

        self.assertEqual(list(pattern.pad_input_tokens(ids, mm_inputs)), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
