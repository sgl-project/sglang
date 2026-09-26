"""CPU regression tests for the pooled-indexer breakable-graph bridge."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsa.kpool_prefill_cuda_graph import (
    _kpool_indexer_prefill_with_output,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeIndexer:
    def __init__(self, topk_width: int):
        self.topk_width = topk_width
        self.call = None

    def _forward_cuda_impl(self, **kwargs):
        self.call = kwargs
        rows = sum(kwargs["forward_batch"].extend_seq_lens_cpu)
        return torch.arange(rows * self.topk_width, dtype=torch.int32).reshape(
            rows, self.topk_width
        )


class TestKPoolPrefillCudaGraph(unittest.TestCase):
    def test_dp_padding_uses_real_extend_rows_and_sentinels_tail(self):
        padded_tokens = 8
        real_tokens = 5
        topk_width = 4
        forward_batch = SimpleNamespace(
            # DP MAX_LEN padding expands this scalar and all token-axis inputs.
            extend_num_tokens=padded_tokens,
            # DSA metadata remains request-shaped on this rank.
            extend_seq_lens_cpu=[3, 2],
        )
        context = SimpleNamespace(forward_batch=forward_batch)
        indexer = _FakeIndexer(topk_width)
        x = torch.arange(padded_tokens * 2).reshape(padded_tokens, 2)
        q_lora = x + 100
        positions = torch.tensor([0, 1, 2, 0, 1, 0, 0, 0])
        output = torch.empty((padded_tokens, topk_width), dtype=torch.int32)

        with patch(
            "sglang.srt.layers.attention.dsa.kpool_prefill_cuda_graph."
            "get_tc_piecewise_forward_context",
            return_value=context,
        ):
            _kpool_indexer_prefill_with_output(
                indexer, x, q_lora, positions, output, layer_id=7
            )

        self.assertEqual(indexer.call["x"].shape[0], real_tokens)
        self.assertEqual(indexer.call["q_lora"].shape[0], real_tokens)
        torch.testing.assert_close(indexer.call["positions"], positions[:real_tokens])
        torch.testing.assert_close(
            output[:real_tokens],
            torch.arange(real_tokens * topk_width, dtype=torch.int32).reshape(
                real_tokens, topk_width
            ),
        )
        torch.testing.assert_close(
            output[real_tokens:],
            torch.full(
                (padded_tokens - real_tokens, topk_width), -1, dtype=torch.int32
            ),
        )


if __name__ == "__main__":
    unittest.main()
