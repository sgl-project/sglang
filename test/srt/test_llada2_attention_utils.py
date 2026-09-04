import unittest

import torch

from sglang.srt.layers.attention.llada2_attention_utils import (
    build_llada_image_custom_mask,
)


class TestLLaDA2AttentionUtils(unittest.TestCase):
    def test_builds_official_text_query_block_mask(self):
        flattened = build_llada_image_custom_mask([2], [4], "cpu")
        actual = flattened.view(4, 4)
        expected = torch.tensor(
            [
                [True, True, False, False],
                [True, True, False, False],
                [True, True, True, True],
                [True, True, True, True],
            ]
        )

        torch.testing.assert_close(actual, expected)

    def test_builds_ragged_batch_without_cross_request_entries(self):
        flattened = build_llada_image_custom_mask([1, 2], [3, 4], "cpu")

        self.assertEqual(flattened.numel(), 3 * 3 + 4 * 4)

    def test_rejects_invalid_conditioning_spans(self):
        with self.assertRaisesRegex(RuntimeError, "metadata batch mismatch"):
            build_llada_image_custom_mask([1], [2, 3], "cpu")
        with self.assertRaisesRegex(RuntimeError, "must leave query tokens"):
            build_llada_image_custom_mask([2], [2], "cpu")


if __name__ == "__main__":
    unittest.main()
