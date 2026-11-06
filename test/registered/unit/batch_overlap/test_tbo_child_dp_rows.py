"""Per-rank rows that bound each TBO micro-batch's MORI dispatch.

Every rank must agree on each micro-batch's bound: a receiver caps its MoE
input by the sum, so one rank under-counting a sender's micro-batch drops rows.
An idle or empty-split rank starts BOTH children at token 0, so the child must
be identified by position, not by its start token. CPU-only.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.batch_overlap import two_batch_overlap
from sglang.srt.batch_overlap.two_batch_overlap import (
    TboDPAttentionPreparer,
    TboForwardBatchPreparer,
)
from sglang.srt.layers.moe.utils import DeepEPMode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


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
    batch.global_tbo_split_num_tokens = split_rows
    return batch


class TestTboIdleRankChildRows(CustomTestCase):
    def test_idle_rank_bounds_second_micro_batch_by_the_rest(self):
        """An idle rank's split is 0, so both of its children span [0, 0); the
        bound goes by position, never by where a child starts."""
        parent, split = [0, 2080, 0, 0], [0, 1000, 0, 0]
        batch = _idle_batch(parent, split)
        children = []
        for _ in range(2):
            with (
                get_context().override_server_args(
                    attention_backend="fa3", moe_dense_tp_size=None
                ),
                get_parallel().override(attn_tp_size=1),
            ):
                children.append(
                    TboForwardBatchPreparer.filter_batch(
                        batch,
                        start_token_index=0,
                        end_token_index=0,
                        start_seq_index=0,
                        end_seq_index=0,
                        out_num_token_non_padded=torch.tensor(0),
                    )
                )
        child_a, child_b = children
        self.assertIsNone(child_a.tbo_dp_global_num_tokens_bound)
        TboForwardBatchPreparer._bound_children_dp_rows(
            batch, child_a=child_a, child_b=child_b
        )
        self.assertEqual(child_a.tbo_dp_global_num_tokens_bound, [0, 1000, 0, 0])
        self.assertEqual(child_b.tbo_dp_global_num_tokens_bound, [0, 1080, 0, 0])

        # Without the gathered split, both children keep the parent's rows.
        batch.global_tbo_split_num_tokens = None
        TboForwardBatchPreparer._bound_children_dp_rows(
            batch, child_a=child_a, child_b=child_b
        )
        self.assertEqual(child_a.tbo_dp_global_num_tokens_bound, parent)
        self.assertEqual(child_b.tbo_dp_global_num_tokens_bound, parent)


class TestTboSplitRowsGather(CustomTestCase):
    def test_mori_gathers_each_rank_first_micro_batch_rows(self):
        """Every rank publishes its first micro-batch's rows; one left at 0 makes the
        other ranks bound that rank's first micro-batch at 0 rows and drop it."""
        # Two sequences of 4000 and 5000 tokens split into two 4500-row halves.
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            spec_info=None,
            extend_lens=[4000, 5000],
            extend_num_tokens=9000,
            is_extend_in_batch=True,
        )
        backend = Mock(
            is_none=Mock(return_value=False),
            tbo_uses_micro_batch_rows=Mock(return_value=True),
        )
        preparer = TboDPAttentionPreparer()
        with (
            patch.object(two_batch_overlap, "is_tbo_enabled", return_value=True),
            patch.object(
                two_batch_overlap, "get_moe_a2a_backend", return_value=backend
            ),
            patch.object(
                two_batch_overlap, "get_deepep_mode", return_value=DeepEPMode.NORMAL
            ),
        ):
            can_run, _ = preparer.prepare_all_gather(batch)
        self.assertTrue(can_run)
        self.assertEqual(preparer.local_tbo_split_num_tokens, 4500)
        gathered = torch.tensor([4500, 4096])
        self.assertEqual(preparer.compute_split_num_tokens(gathered), [4500, 4096])


if __name__ == "__main__":
    unittest.main()
