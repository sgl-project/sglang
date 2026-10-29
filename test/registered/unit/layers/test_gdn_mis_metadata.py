import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.layers.attention.linear.gdn_backend import build_gdn_mis_metadata
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


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


class TestFlashInferMISPositions(CustomTestCase):
    def test_mrope_axes_take_the_item_positions(self):
        """Every mRoPE axis places each item right after its request's query,
        so a Qwen3.5 item does not rotate by the items packed before it, and
        an empty query leaves no position at -1, which RoPE kernels index."""
        positions = torch.cat([torch.arange(11), torch.arange(7), torch.arange(7)])
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            input_ids=torch.empty(25, dtype=torch.int64),
            extend_seq_lens_cpu=[11, 7, 7],
            extend_prefix_lens_cpu=[0, 0, 0],
            multi_item_delimiter_indices=[
                torch.tensor([3, 6, 10], dtype=torch.int64),
                torch.tensor([2, 4, 6], dtype=torch.int64),
                torch.tensor([0, 3, 6], dtype=torch.int64),
            ],
            positions=positions,
            mrope_positions=positions.repeat(3, 1),
        )

        FlashInferAttnBackend._process_multi_item_scoring(
            SimpleNamespace(enable_mis=True), forward_batch
        )

        # Query, then each delimiter at the last query position and its item after.
        # Without a query, items start at 0 and their delimiters take 0 too.
        expected = torch.tensor(
            [0, 1, 2, 2, 3, 4, 2, 3, 4, 5, 2]
            + [0, 1, 1, 2, 1, 2, 1]
            + [0, 0, 1, 0, 0, 1, 0]
        )
        torch.testing.assert_close(forward_batch.positions, expected)
        torch.testing.assert_close(forward_batch.mrope_positions, expected.repeat(3, 1))


if __name__ == "__main__":
    unittest.main()
