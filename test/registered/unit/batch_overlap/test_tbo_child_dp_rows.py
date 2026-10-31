"""Per-rank rows that bound each TBO micro-batch's MORI dispatch.

Every rank must agree on each micro-batch's bound: a receiver caps its MoE
input by the sum, so one rank under-counting a sender's micro-batch drops rows.
An idle or empty-split rank starts BOTH children at token 0, so the child must
be identified by position, not by its start token. The MORI EPv2 op behind the
second micro-batch holds about half a chunk, so a larger second micro-batch must
keep the whole step unsplit. CPU-only.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.batch_overlap import two_batch_overlap
from sglang.srt.batch_overlap.two_batch_overlap import (
    TboDPAttentionPreparer,
    TboForwardBatchPreparer,
    compute_split_seq_index,
    compute_split_token_index,
)
from sglang.srt.layers.moe.token_dispatcher.moriep import (
    mori_tbo_child_max_dispatch_tokens_per_rank,
)
from sglang.srt.layers.moe.utils import DeepEPMode
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


def _micro_batch_b_rows(extend_lens) -> int:
    split_seq = compute_split_seq_index(
        ForwardMode.EXTEND, sum(extend_lens), extend_lens, None
    )
    split_tokens = compute_split_token_index(
        split_seq, ForwardMode.EXTEND, extend_lens, None
    )
    return sum(extend_lens) - split_tokens


class TestTboChildDispatchCap(CustomTestCase):
    """MORI EPv2 sizes the op behind micro-batch B for about half a chunk; a step
    whose micro-batch B would overflow it must run unsplit on every rank."""

    def test_child_rows_cover_both_split_kinds(self):
        self.assertEqual(mori_tbo_child_max_dispatch_tokens_per_rank(8192), 4352)
        # Small ops keep their full size: 256-row rounding reaches it first.
        self.assertEqual(mori_tbo_child_max_dispatch_tokens_per_rank(256), 256)
        with patch.object(
            two_batch_overlap, "get_tbo_token_distribution_threshold", return_value=0.48
        ):
            for m in (1024, 4096, 8192, 16384):
                cap = mori_tbo_child_max_dispatch_tokens_per_rank(m)
                # One long sequence: two-chunk split, B = ceil(M / 2).
                self.assertLessEqual(_micro_batch_b_rows([m]), cap)
                # Sequence-boundary split just inside the threshold: B ~ 0.52 M.
                first = int(m * 0.48) + 1
                self.assertLessEqual(_micro_batch_b_rows([first, m - first]), cap)

    def _local_can_run_tbo(self, extend_lens, child_cap):
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            spec_info=None,
            extend_lens=extend_lens,
            extend_num_tokens=sum(extend_lens),
            is_extend_in_batch=True,
        )
        backend = Mock(is_none=Mock(return_value=False))
        preparer = TboDPAttentionPreparer()
        with (
            patch.object(two_batch_overlap, "is_tbo_enabled", return_value=True),
            patch.object(
                two_batch_overlap, "get_moe_a2a_backend", return_value=backend
            ),
            patch.object(
                two_batch_overlap, "get_deepep_mode", return_value=DeepEPMode.NORMAL
            ),
            patch.object(
                two_batch_overlap, "mori_tbo_child_dispatch_cap", return_value=child_cap
            ),
        ):
            can_run, mode = preparer.prepare_all_gather(batch)
        return preparer, can_run, mode

    def test_oversized_micro_batch_b_keeps_every_rank_unsplit(self):
        # Rank 0's micro-batch B (4500 rows) overflows a 4352-row op; rank 1's fits.
        lens_by_rank = ([4000, 5000], [4096, 4096])
        results = [self._local_can_run_tbo(lens, 4352) for lens in lens_by_rank]
        self.assertEqual([r[1] for r in results], [False, True])
        self.assertEqual(results[0][0].local_tbo_split_num_tokens, 0)
        gathered = torch.tensor([[int(r[1]), r[2]] for r in results])
        for preparer, _, _ in results:
            self.assertEqual(preparer.compute_output(gathered), (None, None))

    def test_unbounded_backend_keeps_the_split(self):
        _, can_run, _ = self._local_can_run_tbo([4000, 5000], None)
        self.assertTrue(can_run)


if __name__ == "__main__":
    unittest.main()
