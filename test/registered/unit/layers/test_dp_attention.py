"""CPU unit tests for publishing DP-attention buffer sizes from a ForwardBatch."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    _DpGatheredBufferWrapper,
    get_dp_global_num_tokens,
    get_global_dp_buffer_len,
    get_local_dp_buffer_len,
    is_dp_max_padding,
    set_dp_buffer_len,
    set_dp_buffer_len_from_batch,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _batch(**overrides):
    fields = dict(
        global_dp_buffer_len=8,
        global_num_tokens_cpu=[3, 1],
        global_num_tokens_padded_cpu=[4, 4],
        global_num_tokens_gpu=torch.tensor([3, 1]),
        dp_padding_mode=DpPaddingMode.MAX_LEN,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


class TestSetDpBufferLenFromBatch(unittest.TestCase):
    def setUp(self):
        self.addCleanup(set_dp_buffer_len, 0, 0, False)

    def test_publishes_the_padded_list_and_this_ranks_entry(self):
        batch = _batch()
        with get_parallel().override(attn_dp_rank=1):
            set_dp_buffer_len_from_batch(batch)
        self.assertEqual(get_global_dp_buffer_len(), 8)
        self.assertEqual(get_local_dp_buffer_len(), 4)
        self.assertTrue(is_dp_max_padding())
        self.assertEqual(get_dp_global_num_tokens(), [4, 4])
        self.assertIs(
            _DpGatheredBufferWrapper.get_dp_global_num_tokens_gpu(),
            batch.global_num_tokens_gpu,
        )

    def test_falls_back_to_the_raw_list_when_no_padded_list_is_carried(self):
        batch = _batch(
            global_dp_buffer_len=4,
            global_num_tokens_padded_cpu=None,
            dp_padding_mode=DpPaddingMode.SUM_LEN,
        )
        with get_parallel().override(attn_dp_rank=0):
            set_dp_buffer_len_from_batch(batch)
        self.assertEqual(get_local_dp_buffer_len(), 3)
        self.assertFalse(is_dp_max_padding())
        self.assertEqual(get_dp_global_num_tokens(), [3, 1])

    def test_a_single_entry_list_ignores_the_rank(self):
        batch = _batch(
            global_dp_buffer_len=3,
            global_num_tokens_cpu=[3],
            global_num_tokens_padded_cpu=None,
            global_num_tokens_gpu=torch.tensor([3]),
            dp_padding_mode=DpPaddingMode.SUM_LEN,
        )
        with get_parallel().override(attn_dp_rank=1):
            set_dp_buffer_len_from_batch(batch)
        self.assertEqual(get_local_dp_buffer_len(), 3)


if __name__ == "__main__":
    unittest.main()
