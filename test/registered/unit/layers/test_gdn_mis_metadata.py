import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.linear.gdn_backend import build_gdn_mis_metadata
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestGDNMISMetadata(CustomTestCase):
    def test_allows_trailing_attention_padding(self):
        forward_batch = SimpleNamespace(
            input_ids=torch.empty(8, dtype=torch.int64),
            extend_seq_lens_cpu=[5],
            extend_prefix_lens_cpu=[0],
            multi_item_delimiter_indices=[torch.tensor([1, 4], dtype=torch.int64)],
            is_prefill_only=True,
        )

        metadata = build_gdn_mis_metadata(forward_batch)

        torch.testing.assert_close(
            torch.cat([metadata.query_token_indices, metadata.item_token_indices])
            .sort()
            .values,
            torch.arange(5, dtype=torch.int64),
        )

    def test_mixed_batch_with_empty_query(self):
        forward_batch = SimpleNamespace(
            input_ids=torch.empty(16, dtype=torch.int64),
            extend_seq_lens_cpu=[9, 7],
            extend_prefix_lens_cpu=[0, 0],
            multi_item_delimiter_indices=[
                torch.tensor([0, 3, 8], dtype=torch.int64),
                torch.tensor([4, 6], dtype=torch.int64),
            ],
            is_prefill_only=True,
        )

        metadata = build_gdn_mis_metadata(forward_batch)

        self.assertEqual(metadata.query_seq_lens_cpu, [4])
        self.assertEqual(metadata.item_seq_lens_cpu, [3, 5, 1, 2, 1])
        torch.testing.assert_close(
            metadata.query_token_indices,
            torch.tensor([9, 10, 11, 12], dtype=torch.int64),
        )
        torch.testing.assert_close(
            metadata.query_cu_seqlens,
            torch.tensor([0, 4], dtype=torch.int32),
        )
        torch.testing.assert_close(
            metadata.query_request_indices,
            torch.tensor([1], dtype=torch.int64),
        )
        torch.testing.assert_close(
            metadata.item_token_indices,
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15], dtype=torch.int64),
        )
        torch.testing.assert_close(
            metadata.item_cu_seqlens,
            torch.tensor([0, 3, 8, 9, 11, 12], dtype=torch.int32),
        )
        torch.testing.assert_close(
            metadata.item_request_indices,
            torch.tensor([0, 0, 0, 1, 1], dtype=torch.int64),
        )

    def test_rejects_sequence_lengths_beyond_input(self):
        forward_batch = SimpleNamespace(
            input_ids=torch.empty(4, dtype=torch.int64),
            extend_seq_lens_cpu=[5],
            extend_prefix_lens_cpu=[0],
            multi_item_delimiter_indices=[torch.tensor([1, 4], dtype=torch.int64)],
            is_prefill_only=True,
        )

        with self.assertRaisesRegex(ValueError, "exceed the input tokens"):
            build_gdn_mis_metadata(forward_batch)

    def test_rejects_missing_final_delimiter(self):
        forward_batch = SimpleNamespace(
            input_ids=torch.empty(5, dtype=torch.int64),
            extend_seq_lens_cpu=[5],
            extend_prefix_lens_cpu=[0],
            multi_item_delimiter_indices=[torch.tensor([2, 3], dtype=torch.int64)],
            is_prefill_only=True,
        )

        with self.assertRaisesRegex(ValueError, "final delimiter"):
            build_gdn_mis_metadata(forward_batch)


if __name__ == "__main__":
    unittest.main()
