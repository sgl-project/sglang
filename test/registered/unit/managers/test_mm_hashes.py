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
from sglang.srt.managers.tokenizer_manager import _apply_caller_mm_hashes
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMmHashesContract(CustomTestCase):
    def test_generate_req_input_defaults_mm_hashes_to_none(self):
        """Absent mm_hashes preserves existing (None) behavior."""
        req = GenerateReqInput(text="hi")
        self.assertIsNone(req.mm_hashes)

    def test_content_hashes_are_distinct_from_feature_hashes(self):
        content_hash = "sha256:" + "ab" * 32
        req = GenerateReqInput(
            text="hi",
            image_data=["http://example.com/img.png"],
            mm_hashes=["deadbeef"],
            mm_content_hashes=[content_hash],
        )
        self.assertEqual(req.mm_hashes, ["deadbeef"])
        self.assertEqual(req.mm_content_hashes, [content_hash])

    def test_batched_hashes_follow_each_request(self):
        req = GenerateReqInput(
            text=["one", "two"],
            image_data=[["a"], ["b", "c"]],
            mm_hashes=["01", ["02", "03"]],
            mm_content_hashes=[
                ["sha256:" + "11" * 32],
                ["sha256:" + "22" * 32, "sha256:" + "33" * 32],
            ],
        )
        req.normalize_batch_and_arguments()
        self.assertEqual(req[0].mm_hashes, ["01"])
        self.assertEqual(req[1].mm_hashes, ["02", "03"])
        self.assertEqual(len(req[1].mm_content_hashes), 2)

    def test_set_pad_value_honors_preset_hash(self):
        """Disabling automatic hashing must not replace a producer hash."""
        item = MultimodalDataItem(modality=Modality.IMAGE, hash=0xDEADBEEF)
        # If hash_feature is invoked, the test fails — we patch it to
        # raise so any accidental recompute is loud.
        with (
            patch(
                "sglang.srt.environ.envs.SGLANG_MM_SKIP_COMPUTE_HASH.get",
                return_value=True,
            ),
            patch(
                "sglang.srt.managers.mm_utils.hash_feature",
                side_effect=AssertionError(
                    "hash_feature must NOT be called when hash is preset"
                ),
            ),
        ):
            item.set_pad_value()
        self.assertEqual(item.hash, 0xDEADBEEF)
        self.assertEqual(item.pad_value, _compute_pad_value(0xDEADBEEF))

    def test_set_pad_value_distinguishes_different_preset_hashes(self):
        """Distinct preset hashes must produce distinct pad_values."""
        a = MultimodalDataItem(modality=Modality.IMAGE, hash=0xAAAA)
        b = MultimodalDataItem(modality=Modality.IMAGE, hash=0xBBBB)
        a.set_pad_value()
        b.set_pad_value()
        self.assertNotEqual(a.pad_value, b.pad_value)

    def test_set_hash_updates_an_existing_pad_value(self):
        item = MultimodalDataItem(modality=Modality.IMAGE, hash=0xAAAA)
        item.set_pad_value()

        item.set_hash(0xBBBB)

        self.assertEqual(item.hash, 0xBBBB)
        self.assertEqual(item.pad_value, _compute_pad_value(0xBBBB))

    def test_incompatible_duplicate_hashes_fall_back_to_internal_identity(self):
        items = [
            MultimodalDataItem(
                modality=Modality.IMAGE,
                offsets=[offset],
            )
            for offset in ((0, 0), (1, 2))
        ]

        with self.assertLogs(level="WARNING"):
            _apply_caller_mm_hashes(items, ["deadbeef", "deadbeef"])

        self.assertIsNone(items[0].hash)
        self.assertIsNone(items[1].hash)

    def test_compatible_duplicate_hashes_remain_shareable(self):
        items = [
            MultimodalDataItem(
                modality=Modality.IMAGE,
                offsets=[offset],
            )
            for offset in ((0, 0), (1, 1))
        ]

        _apply_caller_mm_hashes(items, ["deadbeef", "deadbeef"])

        self.assertEqual(items[0].hash, 0xDEADBEEF)
        self.assertEqual(items[1].hash, 0xDEADBEEF)


if __name__ == "__main__":
    unittest.main()
