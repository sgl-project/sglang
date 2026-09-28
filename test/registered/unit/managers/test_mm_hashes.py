"""Tests for caller-supplied mm_hashes plumbing.

Verifies the contract that:
  1. GenerateReqInput.mm_hashes is an optional list of hex strings.
  2. MultimodalDataItem.set_pad_value() honors a pre-set hash and does NOT
     overwrite it via hash_feature().
  3. The derived pad_value is deterministic across requests with identical
     mm_hashes — the property external KV routers depend on.
  4. TokenizerManager rebuilds processor-built padded_input_ids after it
     applies the caller hashes, so the ids carry the caller-derived pads.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, Mock, patch

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
    _compute_pad_value,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestMmHashesContract(CustomTestCase):
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

    def test_set_hash_updates_an_existing_pad_value(self):
        item = MultimodalDataItem(modality=Modality.IMAGE, hash=0xAAAA)
        item.set_pad_value()

        item.set_hash(0xBBBB)

        self.assertEqual(item.hash, 0xBBBB)
        self.assertEqual(item.pad_value, _compute_pad_value(0xBBBB))

    def test_caller_hashes_repad_processor_padded_input_ids(self):
        """Caller mm_hashes must reach the padded ids a processor precomputed;
        the scheduler reuses them, so a stale pad leaves no slot for the embedding."""
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)

        expanded_ids = [1, 2, 9, 9, 9, 4]
        item = MultimodalDataItem(modality=Modality.IMAGE, offsets=[(2, 4)])
        item.set_hash(0xAAAA)
        tm = TokenizerManager.__new__(TokenizerManager)
        tm.model_config = Mock()
        tm.model_config.hf_config.architectures = []
        tm.max_req_input_len = 128
        tm.mm_processor = Mock(prefer_tokenized_input=True)
        tm.mm_processor.process_mm_data_async = AsyncMock(
            return_value=MultimodalProcessorOutput(
                mm_items=[item],
                input_ids=expanded_ids,
                padded_input_ids=MultimodalProcessorOutput.build_padded_input_ids(
                    expanded_ids, [item]
                ),
            )
        )
        tm._validate_one_request = Mock()
        tm._create_tokenized_object = Mock()
        obj = GenerateReqInput(
            input_ids=[1, 2, 9, 4], image_data=["image"], mm_hashes=["bbbb"]
        )

        asyncio.run(tm._tokenize_one_request(obj))

        mm_inputs = tm._create_tokenized_object.call_args.args[4]
        pad = _compute_pad_value(0xBBBB)
        self.assertEqual(mm_inputs.padded_input_ids, [1, 2, pad, pad, pad, 4])


if __name__ == "__main__":
    unittest.main()
