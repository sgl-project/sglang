"""CPU coverage for mRoPE DP vision-encoder helpers."""

import unittest

import torch

from sglang.srt.multimodal.mm_utils import _pad_mrope_vision_embeddings_for_tp_gather
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMropeVisionEncoderPadding(CustomTestCase):
    def test_padding_preserves_2d_embedding_prefix(self):
        embeddings = torch.arange(12, dtype=torch.float32).reshape(3, 4)

        padded = _pad_mrope_vision_embeddings_for_tp_gather(embeddings, 5)

        self.assertEqual(padded.shape, (5, 4))
        self.assertTrue(torch.equal(padded[:3], embeddings))

    def test_padding_preserves_3d_embedding_prefix(self):
        embeddings = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)

        padded = _pad_mrope_vision_embeddings_for_tp_gather(embeddings, 5)

        self.assertEqual(padded.shape, (5, 3, 4))
        self.assertTrue(torch.equal(padded[:2], embeddings))

    def test_empty_rank_gets_a_gatherable_shape(self):
        embeddings = torch.empty((0, 3, 4), dtype=torch.bfloat16)

        padded = _pad_mrope_vision_embeddings_for_tp_gather(embeddings, 5)

        self.assertEqual(padded.shape, (5, 3, 4))
        self.assertEqual(padded.dtype, torch.bfloat16)

    def test_already_full_embedding_is_not_copied(self):
        embeddings = torch.randn(5, 4)

        padded = _pad_mrope_vision_embeddings_for_tp_gather(embeddings, 5)

        self.assertIs(padded, embeddings)


if __name__ == "__main__":
    unittest.main()
