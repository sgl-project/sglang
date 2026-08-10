"""CPU tests for linear-attention ragged verify layout helpers."""

import unittest

import torch

from sglang.srt.layers.attention.linear.utils import (
    gather_ragged_verify_from_dense,
    ragged_verify_dense_scatter_indices,
    scatter_ragged_verify_to_dense,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestLinearAttentionRaggedVerify(CustomTestCase):
    def test_maps_packed_tokens_to_dense_request_steps(self):
        query_start_loc = torch.tensor([0, 3, 4, 8], dtype=torch.int32)

        indices = ragged_verify_dense_scatter_indices(
            query_start_loc=query_start_loc,
            seq_len=8,
            draft_token_num=4,
        )

        expected = torch.tensor([0, 1, 2, 4, 8, 9, 10, 11], dtype=torch.int64)
        torch.testing.assert_close(indices, expected)

    def test_maps_uncovered_graph_tier_tokens_to_ghost_row(self):
        query_start_loc = torch.tensor([0, 2, 3], dtype=torch.int32)

        indices = ragged_verify_dense_scatter_indices(
            query_start_loc=query_start_loc,
            seq_len=5,
            draft_token_num=4,
        )

        expected = torch.tensor([0, 1, 4, 8, 8], dtype=torch.int64)
        torch.testing.assert_close(indices, expected)

    def test_scatter_and_gather_preserve_packed_token_order(self):
        query_start_loc = torch.tensor([0, 3, 4, 8], dtype=torch.int32)
        packed = torch.arange(1, 17, dtype=torch.float32).view(8, 2)

        dense, indices = scatter_ragged_verify_to_dense(
            packed,
            query_start_loc=query_start_loc,
            draft_token_num=4,
        )

        expected_dense = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [0.0, 0.0]],
                [[7.0, 8.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [
                    [9.0, 10.0],
                    [11.0, 12.0],
                    [13.0, 14.0],
                    [15.0, 16.0],
                ],
            ]
        )
        torch.testing.assert_close(dense, expected_dense)
        # Production callers pass causal_conv1d_update(...).transpose(1, 2),
        # which is non-contiguous. Preserve reshape (not view) support here.
        conv_layout = (dense + 100).transpose(1, 2).contiguous()
        processed = conv_layout.transpose(1, 2)
        self.assertFalse(processed.is_contiguous())
        gathered = gather_ragged_verify_from_dense(
            processed, dense_token_indices=indices
        )
        torch.testing.assert_close(gathered, packed + 100)

    def test_gather_maps_uncovered_graph_tier_tokens_to_zero_ghost(self):
        query_start_loc = torch.tensor([0, 2, 3], dtype=torch.int32)
        packed = torch.arange(1, 6, dtype=torch.float32).unsqueeze(-1)

        dense, indices = scatter_ragged_verify_to_dense(
            packed,
            query_start_loc=query_start_loc,
            draft_token_num=4,
        )
        gathered = gather_ragged_verify_from_dense(dense, dense_token_indices=indices)

        expected = torch.tensor([[1.0], [2.0], [3.0], [0.0], [0.0]])
        torch.testing.assert_close(gathered, expected)


if __name__ == "__main__":
    unittest.main()
