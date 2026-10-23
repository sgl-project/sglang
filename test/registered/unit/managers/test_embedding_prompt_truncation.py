"""Unit tests for per-request embedding prompt truncation in TokenizerManager."""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import (  # noqa: E402
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _mm_inputs(offsets, **fields):
    item = MultimodalDataItem(modality=Modality.IMAGE, offsets=offsets)
    return MultimodalProcessorOutput(mm_items=[item], **fields)


class TestEmbeddingPromptTruncation(CustomTestCase):
    def setUp(self):
        self.manager = object.__new__(TokenizerManager)
        self.manager.context_len = 16
        self.manager.num_reserved_tokens = 0

    def test_keeps_the_prompt_head(self):
        self.assertEqual(
            self.manager._truncate_embedding_prompt(list(range(10)), 4, None),
            [0, 1, 2, 3],
        )
        self.assertEqual(
            self.manager._truncate_embedding_prompt([0, 1], 4, None), [0, 1]
        )

    def test_minus_one_means_the_context_length(self):
        self.assertEqual(
            len(self.manager._truncate_embedding_prompt(list(range(40)), -1, None)),
            15,
        )

    def test_cuts_text_after_media_tokens(self):
        mm_inputs = _mm_inputs([(1, 3)], padded_input_ids=list(range(10)))

        truncated = self.manager._truncate_embedding_prompt(
            list(range(10)), 6, mm_inputs
        )

        self.assertEqual(truncated, [0, 1, 2, 3, 4, 5])
        self.assertEqual(mm_inputs.padded_input_ids, [0, 1, 2, 3, 4, 5])

    def test_refuses_to_cut_media_tokens(self):
        with self.assertRaisesRegex(ValueError, "would cut multimodal tokens"):
            self.manager._truncate_embedding_prompt(
                list(range(10)), 6, _mm_inputs([(4, 8)])
            )

    def test_refuses_models_with_prompt_sized_positions(self):
        mm_inputs = _mm_inputs([(1, 2)], token_type_ids=SimpleNamespace())

        with self.assertRaisesRegex(ValueError, "not supported"):
            self.manager._truncate_embedding_prompt(list(range(10)), 6, mm_inputs)


if __name__ == "__main__":
    unittest.main()
