"""External image hashes must not leave processor-precomputed padding stale."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import GenerateReqInput  # noqa: E402
from sglang.srt.managers.schedule_batch import (  # noqa: E402
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestExternalHashPadding(CustomTestCase):
    def tokenize(self, hashes):
        item = MultimodalDataItem(modality=Modality.IMAGE, offsets=[(1, 2)])
        item.set_hash(0xAAAA)
        ids = [10, 151655, 151655, 20]
        original_padding = MultimodalProcessorOutput.build_padded_input_ids(ids, [item])
        output = MultimodalProcessorOutput(
            mm_items=[item], input_ids=ids, padded_input_ids=original_padding
        )
        manager = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    architectures=["Qwen3VLForConditionalGeneration"]
                )
            ),
            mm_processor=SimpleNamespace(
                prefer_tokenized_input=True,
                process_mm_data_async=AsyncMock(return_value=output),
            ),
            max_req_input_len=2048,
            _validate_mm_limits=Mock(),
            _normalize_mm_content_hashes=Mock(),
            _validate_one_request=Mock(),
            _create_tokenized_object=lambda obj, text, ids, embeds, mm, types: mm,
        )
        req = GenerateReqInput(
            input_ids=ids, image_data=["image.png"], mm_hashes=hashes
        )
        with patch(
            "sglang.srt.managers.tokenizer_manager.get_disagg",
            return_value=SimpleNamespace(
                language_model_only=False, language_only=False
            ),
        ):
            result = asyncio.run(TokenizerManager._tokenize_one_request(manager, req))
        return result, original_padding

    def test_external_hash_does_not_leave_stale_padding(self):
        result, original = self.tokenize(["bbbb"])
        item = result.mm_items[0]
        self.assertEqual(item.hash, 0xBBBB)
        self.assertNotEqual(original[1], item.pad_value)
        # Either rebuild the cached IDs or invalidate them for scheduler padding.
        if result.padded_input_ids is not None:
            self.assertEqual(result.padded_input_ids[1:3], [item.pad_value] * 2)

    def test_absent_hash_preserves_padding(self):
        result, original = self.tokenize(None)
        self.assertEqual(result.mm_items[0].hash, 0xAAAA)
        self.assertEqual(result.padded_input_ids, original)

    def test_ignored_hash_preserves_padding(self):
        for hashes in (["not-hex"], [None], ["bbbb", "cccc"]):
            with self.subTest(hashes=hashes):
                result, original = self.tokenize(hashes)
                self.assertEqual(result.mm_items[0].hash, 0xAAAA)
                self.assertEqual(result.padded_input_ids, original)


if __name__ == "__main__":
    unittest.main()
