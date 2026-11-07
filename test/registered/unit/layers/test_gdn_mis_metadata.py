import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.linear.gdn_backend import build_gdn_mis_metadata
from sglang.srt.layers.pooler import pool_at_delimiter_positions
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestGDNMISMetadata(CustomTestCase):
    def test_only_real_items_and_empty_item_readouts_are_processed(self):
        """Skip delimiters except when they are the existing empty-item readout."""
        for query_len, item_lens in (
            (3, [2]),
            (0, [2, 1]),
            (2, [0, 2, 0]),
            (0, [0]),
            (3, [0, 0]),
        ):
            with self.subTest(query_len=query_len, item_lens=item_lens):
                delimiters = []
                expected_segments = []
                offset = query_len
                for item_len in item_lens:
                    delimiters.append(offset)
                    if item_len:
                        expected_segments.append(
                            list(range(offset + 1, offset + 1 + item_len))
                        )
                    else:
                        expected_segments.append([offset])
                    offset += item_len + 1
                delimiters.append(offset)
                forward_batch = SimpleNamespace(
                    input_ids=torch.empty(offset + 1, dtype=torch.int64),
                    extend_seq_lens_cpu=[offset + 1],
                    extend_prefix_lens_cpu=[0],
                    multi_item_delimiter_indices=[
                        torch.tensor(delimiters, dtype=torch.int64)
                    ],
                    is_prefill_only=True,
                )

                metadata = build_gdn_mis_metadata(forward_batch)
                boundaries = metadata.item_cu_seqlens.tolist()
                actual_segments = [
                    metadata.item_token_indices[start:end].tolist()
                    for start, end in zip(boundaries, boundaries[1:])
                ]
                self.assertEqual(actual_segments, expected_segments)

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
            torch.tensor([0, 2, 3], dtype=torch.int64),
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
        self.assertEqual(metadata.item_seq_lens_cpu, [2, 4, 1])
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
            torch.tensor([1, 2, 4, 5, 6, 7, 14], dtype=torch.int64),
        )
        torch.testing.assert_close(
            metadata.item_cu_seqlens,
            torch.tensor([0, 2, 6, 7], dtype=torch.int32),
        )
        torch.testing.assert_close(
            metadata.item_request_indices,
            torch.tensor([0, 0, 1], dtype=torch.int64),
        )

        packed_output = torch.zeros(16, 1)
        indices = metadata.item_token_indices
        packed_output[indices, 0] = indices.float() + 1
        pooled = pool_at_delimiter_positions(
            data=packed_output, forward_batch=forward_batch, device=torch.device("cpu")
        )
        torch.testing.assert_close(pooled[0][1:], torch.tensor([[3.0], [8.0]]))
        torch.testing.assert_close(pooled[1][1:], torch.tensor([[15.0]]))

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
