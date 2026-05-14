"""Unit-level invariants for caller-supplied mm_hashes.

Pins:
  1. GenerateReqInput.mm_hashes is an optional list of hex strings.
  2. set_pad_value() honors a pre-set hash without calling hash_feature().
  3. pad_value is deterministic across items with identical hashes.

The tokenizer_manager wiring that consumes this field is covered by e2e.
"""

import unittest
from unittest.mock import patch

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    _compute_pad_value,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=2, suite="stage-b-test-1-gpu-large")
register_amd_ci(est_time=2, suite="stage-b-test-1-gpu-small-amd")


class TestMmHashesContract(CustomTestCase):
    def test_generate_req_input_accepts_mm_hashes(self):
        """GenerateReqInput exposes mm_hashes as an optional field."""
        req = GenerateReqInput(
            text="hi",
            image_data=["http://example.com/img.png"],
            mm_hashes=["deadbeefcafe1234"],
        )
        self.assertEqual(req.mm_hashes, ["deadbeefcafe1234"])

    def test_generate_req_input_defaults_mm_hashes_to_none(self):
        """Absent mm_hashes preserves existing (None) behavior."""
        req = GenerateReqInput(text="hi")
        self.assertIsNone(req.mm_hashes)

    def test_set_pad_value_honors_preset_hash(self):
        """set_pad_value() must use a pre-set hash without recomputing."""
        item = MultimodalDataItem(modality=Modality.IMAGE, hash=0xDEADBEEF)
        # Patch hash_feature to raise so any accidental recompute is loud.
        with patch(
            "sglang.srt.managers.mm_utils.hash_feature",
            side_effect=AssertionError(
                "hash_feature must NOT be called when hash is preset"
            ),
        ):
            item.set_pad_value()
        self.assertEqual(item.hash, 0xDEADBEEF)
        self.assertEqual(item.pad_value, _compute_pad_value(0xDEADBEEF))

    def test_set_pad_value_is_deterministic_across_items(self):
        """Two items with the same preset hash must derive the same pad_value."""
        a = MultimodalDataItem(modality=Modality.IMAGE, hash=0x123456789ABCDEF0)
        b = MultimodalDataItem(modality=Modality.IMAGE, hash=0x123456789ABCDEF0)
        a.set_pad_value()
        b.set_pad_value()
        self.assertEqual(a.pad_value, b.pad_value)
        self.assertEqual(a.hash, b.hash)

    def test_set_pad_value_distinguishes_different_preset_hashes(self):
        """Distinct preset hashes must produce distinct pad_values."""
        a = MultimodalDataItem(modality=Modality.IMAGE, hash=0xAAAA)
        b = MultimodalDataItem(modality=Modality.IMAGE, hash=0xBBBB)
        a.set_pad_value()
        b.set_pad_value()
        self.assertNotEqual(a.pad_value, b.pad_value)


if __name__ == "__main__":
    unittest.main()
