"""Regression: TBO filter_batch must handle ForwardBatch's attention plan
marker fields.

`TboForwardBatchPreparer.filter_batch` ends with a completeness guard that
raises for any ForwardBatch field that is non-None but absent from the child
``output_dict``. The plan marker added by the skip_attn_backend_init refactor
(`forward_metadata_ready` etc.) defaults to ``False`` — non-None — so an
unhandled marker crashes TBO cuda-graph capture (DP + spec + TBO).
The children must be reset to "unplanned" so each sub-batch is planned by the
TBO-aware init flow.

Pure dataclass logic — CPU only.
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
        # Default marker (forward_metadata_ready=False) is non-None and would
        # trip the completeness guard if unhandled.
        child = _filter(_make_target_verify_batch(8), lo=0, hi=4)
        self.assertEqual(child.batch_size, 4)
        self.assertFalse(child.forward_metadata_ready)
        self.assertIsNone(child.forward_metadata_planned_bs)
        self.assertIsNone(child.forward_metadata_planned_num_tokens)
        self.assertFalse(child.forward_metadata_replan_equivalent)

    def test_pre_planned_parent_does_not_leak_ready_into_children(self):
        # Even if the parent was marked, children must start unplanned — a
        # stale "ready" would wrongly skip the child's own planning.
        parent = _make_target_verify_batch(8)
        parent.mark_forward_metadata_ready(replan_equivalent=True)
        child = _filter(parent, lo=0, hi=4)
        self.assertFalse(child.forward_metadata_ready)
        self.assertFalse(child.forward_metadata_replan_equivalent)


if __name__ == "__main__":
    unittest.main()
