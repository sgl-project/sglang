"""Per-rank rows that bound each TBO micro-batch's MORI dispatch.

Every rank must agree on each micro-batch's bound: a receiver caps its MoE
input by the sum, so one rank under-counting a sender's micro-batch drops rows.
An idle or empty-split rank starts BOTH children at token 0, so the child must
be identified by position, not by its start token. CPU-only.
"""

import unittest

import torch

from sglang.srt.batch_overlap.two_batch_overlap import TboForwardBatchPreparer
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_rows = TboForwardBatchPreparer._child_dp_global_num_tokens


class TestTboChildDpRows(CustomTestCase):
    def test_children_partition_parent_rows(self):
        parent, split = [8192, 0, 6254, 3000], [4096, 0, 3127, 1400]
        first = _rows(parent, split, first_child=True)
        second = _rows(parent, split, first_child=False)
        self.assertEqual(first, [4096, 0, 3127, 1400])
        self.assertEqual(second, [4096, 0, 3127, 1600])
        self.assertEqual([a + b for a, b in zip(first, second)], parent)

    def test_split_is_clamped_to_parent_rows(self):
        self.assertEqual(_rows([10, 4], [12, -1], first_child=True), [10, 0])
        self.assertEqual(_rows([10, 4], [12, -1], first_child=False), [0, 4])

    def test_missing_or_mismatched_split_keeps_parent_rows(self):
        self.assertEqual(_rows([8, 8], None, first_child=False), [8, 8])
        self.assertEqual(_rows([8, 8], [4], first_child=True), [8, 8])
        self.assertIsNone(_rows(None, [4, 4], first_child=True))


def _idle_batch(parent_rows, split_rows) -> ForwardBatch:
    empty = torch.zeros(0, dtype=torch.long)
    batch = ForwardBatch(
        forward_mode=ForwardMode.IDLE,
        batch_size=0,
        input_ids=empty,
        positions=empty,
        out_cache_loc=empty,
        req_pool_indices=empty,
        seq_lens=torch.zeros(0, dtype=torch.int32),
        seq_lens_cpu=torch.zeros(0, dtype=torch.int32),
        seq_lens_sum=0,
        spec_info=None,
    )
    batch.full_prefill_cuda_graph_capture = None
    batch.global_num_tokens_cpu = parent_rows
    batch.tbo_global_split_num_tokens = split_rows
    return batch


class TestTboIdleRankChildRows(CustomTestCase):
    def test_idle_rank_bounds_second_micro_batch_by_the_rest(self):
        # An idle rank's split is 0, so both of its children span [0, 0).
        parent, split = [0, 2080, 0, 0], [0, 1000, 0, 0]
        rows = []
        for first_child in (True, False):
            with (
                get_context().override_server_args(
                    attention_backend="fa3", moe_dense_tp_size=None
                ),
                get_parallel().override(attn_tp_size=1),
            ):
                child = TboForwardBatchPreparer.filter_batch(
                    _idle_batch(parent, split),
                    start_token_index=0,
                    end_token_index=0,
                    start_seq_index=0,
                    end_seq_index=0,
                    first_child=first_child,
                    out_num_token_non_padded=torch.tensor(0),
                )
            rows.append(child.tbo_dp_global_num_tokens_bound)
        self.assertEqual(rows, [[0, 1000, 0, 0], [0, 1080, 0, 0]])


if __name__ == "__main__":
    unittest.main()
