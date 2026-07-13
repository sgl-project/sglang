"""Tests for caller-supplied mm_hashes plumbing.

Verifies the contract that:
  1. GenerateReqInput.mm_hashes is an optional list of hex strings.
  2. MultimodalDataItem.set_pad_value() honors a pre-set hash and does NOT
     overwrite it via hash_feature().
  3. The derived pad_value is deterministic across requests with identical
     mm_hashes — the property external KV routers depend on.

The wiring step that copies GenerateReqInput.mm_hashes into per-item
MultimodalDataItem.hash lives in tokenizer_manager.py and is exercised by
the e2e serve tests; this file pins the unit-level invariants the wiring
relies on.
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

register_cuda_ci(est_time=9, stage="base-b", runner_config="1-gpu-small")
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
        # If hash_feature is invoked, the test fails — we patch it to
        # raise so any accidental recompute is loud.
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
        # No feature payload — set_pad_value uses the preset hash.
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
