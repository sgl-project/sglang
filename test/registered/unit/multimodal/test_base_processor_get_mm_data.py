import unittest

import torch

from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _StubProcessor(BaseMultimodalProcessor):
    IM_START_TOKEN_ID = 1
    IM_END_TOKEN_ID = 2
    IM_TOKEN_ID = 3

    def __init__(self, num_tokens):
        self.num_tokens = num_tokens

    def build_input_ids(self, prompt, **kwargs):
        return prompt, [(0, self.num_tokens - 1)], [Modality.IMAGE]

    async def process_mm_data_async(self, *args, **kwargs):
        raise NotImplementedError


class TestGetMmData(unittest.TestCase):
    def test_rejects_short_precomputed_embeddings(self):
        processor = _StubProcessor(num_tokens=4010)
        embeddings = {Modality.IMAGE: torch.empty(877, 16)}

        with self.assertRaisesRegex(RuntimeError, "expected 4010.*got 877"):
            processor.get_mm_data([], embeddings)

    def test_accepts_exact_length_precomputed_embeddings(self):
        processor = _StubProcessor(num_tokens=3)
        embeddings = {Modality.IMAGE: torch.empty(3, 16)}

        output = processor.get_mm_data([], embeddings)

        self.assertEqual(output.mm_items[0].precomputed_embeddings.shape, (3, 16))


if __name__ == "__main__":
    unittest.main(verbosity=2)
