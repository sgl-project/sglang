"""Regression: TBO filter_batch resets the attention plan marker on children.

filter_batch's completeness guard raises for any non-None ForwardBatch field
missing from the child dict; the plan marker defaults to False (non-None) and
crashed TBO cuda-graph capture until reset. CPU-only.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.batch_overlap.two_batch_overlap as tbo
from sglang.srt.batch_overlap.two_batch_overlap import TboForwardBatchPreparer
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_target_verify_batch(bs: int) -> ForwardBatch:
    return ForwardBatch(
        forward_mode=ForwardMode.TARGET_VERIFY,
        batch_size=bs,
        input_ids=torch.zeros(bs, dtype=torch.long),
        positions=torch.zeros(bs, dtype=torch.long),
        out_cache_loc=torch.zeros(bs, dtype=torch.long),
        req_pool_indices=torch.zeros(bs, dtype=torch.long),
        seq_lens=torch.ones(bs, dtype=torch.int32),
        seq_lens_cpu=torch.ones(bs, dtype=torch.int32),
        seq_lens_sum=bs,
        spec_info=None,
    )


def _filter(batch: ForwardBatch, *, lo: int, hi: int) -> ForwardBatch:
    fake_args = SimpleNamespace(moe_dense_tp_size=None, attention_backend="fa3")
    with patch.object(tbo, "get_attention_tp_size", lambda: 1), patch.object(
        tbo, "get_global_server_args", lambda: fake_args
    ):
        return TboForwardBatchPreparer.filter_batch(
            batch,
            start_token_index=lo,
            end_token_index=hi,
            start_seq_index=lo,
            end_seq_index=hi,
            out_num_token_non_padded=torch.tensor(hi - lo),
        )


class TestTboFilterBatchMarker(CustomTestCase):
    def test_filter_batch_resets_plan_marker_on_children(self):
        child = _filter(_make_target_verify_batch(8), lo=0, hi=4)
        self.assertEqual(child.batch_size, 4)
        self.assertFalse(child.forward_metadata_ready)
        self.assertIsNone(child.forward_metadata_planned_bs)
        self.assertIsNone(child.forward_metadata_planned_num_tokens)
        self.assertFalse(child.forward_metadata_replan_equivalent)

    def test_pre_planned_parent_does_not_leak_ready_into_children(self):
        parent = _make_target_verify_batch(8)
        parent.mark_forward_metadata_ready(replan_equivalent=True)
        child = _filter(parent, lo=0, hi=4)
        self.assertFalse(child.forward_metadata_ready)
        self.assertFalse(child.forward_metadata_replan_equivalent)


if __name__ == "__main__":
    unittest.main()
