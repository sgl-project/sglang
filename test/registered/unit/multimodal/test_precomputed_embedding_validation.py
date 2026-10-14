"""Tests for the common EPD precomputed-embedding boundary."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    BaseMultiModalProcessorOutput,
)


class _StubProcessor(BaseMultimodalProcessor):
    async def process_mm_data_async(self, *args, **kwargs):
        raise NotImplementedError

    def get_mm_data(self, prompt, embeddings, **kwargs):
        return self.output


def _item(modality, rows, offsets):
    return MultimodalDataItem(
        modality=modality,
        offsets=offsets,
        precomputed_embeddings=torch.zeros(rows, 4),
    )


class TestPrecomputedEmbeddingValidation(unittest.TestCase):
    def setUp(self):
        self.processor = object.__new__(_StubProcessor)

    def _validate(self, items, embeddings):
        self.processor.output = MultimodalProcessorOutput(
            input_ids=[1, 2, 3],
            mm_items=items,
        )
        return self.processor.get_validated_mm_data([], embeddings)

    def test_precomputed_input_preserves_caller_hash_and_offsets(self):
        feature = torch.zeros(2, 4)
        prompt = torch.tensor([7, 99, 99, 8])
        payload = {
            "format": "precomputed_embedding",
            "feature": feature,
            "offsets": [(1, 2)],
            "hash": 0x123456789ABCDEF,
        }
        self.processor.use_cuda_ipc = False
        self.processor._resolve_processor = Mock(return_value=(None, Mock()))
        self.processor.get_mm_item_offsets = Mock(
            side_effect=AssertionError("rediscovered supplied offsets")
        )
        with patch(
            "sglang.srt.managers.mm_utils.hash_feature",
            side_effect=AssertionError("recomputed supplied hash"),
        ):
            items, tokens, _ = self.processor.process_and_combine_mm_data(
                BaseMultiModalProcessorOutput(
                    input_text="", input_ids=prompt, images=[payload]
                ),
                None,
            )
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].hash, payload["hash"])
        self.assertEqual(items[0].offsets, payload["offsets"])
        self.assertIs(items[0].feature, feature)
        self.assertNotIn("hash", items[0].model_specific_data)
        self.assertNotIn("offsets", items[0].model_specific_data)
        self.assertEqual(payload["hash"], 0x123456789ABCDEF)
        torch.testing.assert_close(tokens, prompt)

    def test_accepts_exact_multi_item_layout(self):
        image_embedding = torch.zeros(5, 4)
        audio_embedding = torch.zeros(2, 4)
        output = self._validate(
            [
                _item(Modality.IMAGE, 2, [(1, 2)]),
                _item(Modality.IMAGE, 3, [(4, 6)]),
                _item(Modality.AUDIO, 2, [(8, 9)]),
            ],
            {
                Modality.IMAGE: image_embedding,
                Modality.AUDIO: audio_embedding,
            },
        )

        self.assertEqual(len(output.mm_items), 3)

    def test_rejects_item_shorter_than_prompt_offsets(self):
        with self.assertRaisesRegex(RuntimeError, "expected 3 rows.*got 2"):
            self._validate(
                [_item(Modality.IMAGE, 2, [(1, 3)])],
                {Modality.IMAGE: torch.zeros(2, 4)},
            )

    def test_rejects_unconsumed_trailing_rows(self):
        with self.assertRaisesRegex(RuntimeError, "received 3 rows, consumed 2"):
            self._validate(
                [_item(Modality.IMAGE, 2, [(1, 2)])],
                {Modality.IMAGE: torch.zeros(3, 4)},
            )

    def test_rejects_missing_modality(self):
        with self.assertRaisesRegex(RuntimeError, "received 2 rows, consumed 0"):
            self._validate([], {Modality.VIDEO: torch.zeros(2, 4)})

    def test_rejects_unexpected_modality(self):
        with self.assertRaisesRegex(RuntimeError, "unexpected embedding modality"):
            self._validate(
                [_item(Modality.VIDEO, 2, [(1, 2)])],
                {Modality.IMAGE: torch.zeros(2, 4)},
            )


if __name__ == "__main__":
    unittest.main()
