import unittest
from unittest import mock

import torch

from sglang.srt.layers.attention.dsv4.candidate_indexer import (
    make_candidate_indexer,
)
from sglang.srt.layers.attention.dsv4.candidate_torch import (
    TorchCandidateIndexer,
    two_level_decode_logits,
)
from sglang.srt.runtime_context import override_platform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDsv41CandidateTorch(CustomTestCase):
    def test_source_masks_unreachable_and_publishes_best_blocks(self):
        logits = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0, 100.0, 100.0], [9.0, 8.0, 7.0, 6.0, 5.0, 4.0]]
        )
        scores, mask = two_level_decode_logits(
            logits,
            torch.tensor([4, 6]),
            is_candidate_source=True,
            uses_candidates=False,
            topk_blocks=1,
            block_size=2,
            published=None,
        )

        self.assertTrue(torch.isneginf(scores[0, 4:]).all())
        self.assertEqual(
            mask.tolist(),
            [
                [False, False, True, True, False, False],
                [False, False, False, False, True, True],
            ],
        )

    def test_consumer_applies_published_mask(self):
        logits = torch.arange(6, dtype=torch.float32).reshape(1, 6)
        published = torch.tensor([[False, True, True, False, False, False]])
        scores, mask = two_level_decode_logits(
            logits,
            torch.tensor([5]),
            is_candidate_source=False,
            uses_candidates=True,
            topk_blocks=1,
            block_size=2,
            published=published,
        )

        self.assertIsNone(mask)
        self.assertEqual(scores[0, 1:3].tolist(), [1.0, 2.0])
        self.assertTrue(torch.isneginf(scores[0, [0, 3, 4, 5]]).all())

    def test_blackwell_without_sparse_kernel_uses_dense_fallback(self):
        with (
            override_platform(device_sm=120),
            mock.patch(
                "sglang.srt.layers.deep_gemm_wrapper.configurer.DEEPGEMM_PAGED_SPARSE_MQA_LOGITS",
                False,
            ),
        ):
            self.assertIsInstance(
                make_candidate_indexer(2048, 8), TorchCandidateIndexer
            )

    def test_pre_blackwell_keeps_inline_mask_path(self):
        with override_platform(device_sm=90):
            self.assertIsNone(make_candidate_indexer(2048, 8))


if __name__ == "__main__":
    unittest.main()
