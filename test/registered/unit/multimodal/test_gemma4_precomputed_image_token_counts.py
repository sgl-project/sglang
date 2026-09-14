import unittest

import torch

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputFormat,
)
from sglang.srt.multimodal.processors.gemma4 import Gemma4SGLangProcessor
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def make_item(num_placeholders, num_embeddings):
    return MultimodalDataItem(
        modality=Modality.IMAGE,
        offsets=[(0, num_placeholders - 1)],
        feature=torch.zeros((num_embeddings, 4)),
        format=MultimodalInputFormat.PRECOMPUTED_EMBEDDING,
    )


class TestGemma4PrecomputedImageTokenCounts(CustomTestCase):
    def test_accepts_matching_count(self):
        Gemma4SGLangProcessor._validate_precomputed_image_token_counts(
            [make_item(4, 4)]
        )

    def test_rejects_mismatched_count(self):
        with self.assertRaisesRegex(ValueError, r"placeholders=526, embeddings=517"):
            Gemma4SGLangProcessor._validate_precomputed_image_token_counts(
                [make_item(526, 517)]
            )


if __name__ == "__main__":
    unittest.main()
