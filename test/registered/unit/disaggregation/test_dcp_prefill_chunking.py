"""Cached-prefix sends must obey DCP pack capacity independently of compute."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.utils import build_dcp_token_transfer_plan
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestDcpPrefillChunking(CustomTestCase):
    def make_request(self, total, prefix=0, limit=256):
        sender = SimpleNamespace(
            get_max_transfer_tokens=lambda: limit,
            should_send_kv_chunk=lambda n, last: n > 0 or last,
            send=Mock(),
        )
        req = SimpleNamespace(
            rid="cached-prefix",
            kv=SimpleNamespace(req_pool_idx=0),
            origin_input_ids=[0] * total,
            extend_range=SimpleNamespace(end=total),
            start_send_idx=prefix,
            disagg_decode_prefix_len=prefix,
            disagg_kv_sender=sender,
        )
        sched = SimpleNamespace(
            enable_staging=False,
            token_to_kv_pool_allocator=SimpleNamespace(
                page_size=64,
                translate_kv_indices_for_transfer=lambda x: x,
            ),
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.arange(total).reshape(1, -1),
                req_index_to_mamba_index_mapping=torch.tensor([17]),
                translate_mamba_indices=lambda x: x,
            ),
            disagg_metadata_buffers=SimpleNamespace(set_buf=Mock()),
            disagg_prefill_bootstrap_queue=SimpleNamespace(
                kv_manager=SimpleNamespace(
                    kv_args=SimpleNamespace(state_types=[StateType.MAMBA])
                )
            ),
            disagg_prefill_pending_chunk_rids={req.rid},
        )
        return sched, req

    def test_large_cached_prefix_partial_tail_and_decode_prefix(self):
        # More than DCP4 * capacity: merely quadrupling the allocation would
        # still not bound this send. Reuse the same final-state path as Kimi.
        for prefix in (0, 256):
            sched, req = self.make_request(total=2055, prefix=prefix)
            SchedulerDisaggregationPrefillMixin._send_kv_chunk(
                sched, req, last_chunk=True
            )
            calls = req.disagg_kv_sender.send.call_args_list
            np.testing.assert_array_equal(
                np.concatenate([c.args[0] for c in calls]),
                np.arange(prefix // 64, 33),
            )
            self.assertEqual(
                sum(c.kwargs["num_kv_tokens"] for c in calls), 2055 - prefix
            )
            self.assertTrue(all(c.kwargs["num_kv_tokens"] <= 256 for c in calls))
            self.assertEqual(calls[-1].kwargs["num_kv_tokens"], 7)
            self.assertTrue(all(c.args[1] is None for c in calls[:-1]))
            self.assertEqual(int(calls[-1].args[1][0][0]), 17)
            self.assertEqual(req.start_send_idx, 2055)
            self.assertNotIn(req.rid, sched.disagg_prefill_pending_chunk_rids)

            # Chunk offsets must preserve both sharded target and replicated
            # draft token mappings, including a nonzero decode cache prefix.
            for rank in range(4):
                kwargs = dict(
                    physical_page_size=64,
                    dcp_size=4,
                    dcp_rank=rank,
                    decode_prefix_len=prefix,
                )
                dest = np.arange(10, 19, dtype=np.int32)
                full = build_dcp_token_transfer_plan(
                    np.arange(prefix // 64, 33, dtype=np.int32),
                    dest,
                    num_kv_tokens=2055 - prefix,
                    **kwargs,
                )
                plans = []
                page_offset = 0
                for call in calls:
                    plan = build_dcp_token_transfer_plan(
                        call.args[0],
                        dest,
                        src_page_offset=page_offset,
                        num_kv_tokens=call.kwargs["num_kv_tokens"],
                        **kwargs,
                    )
                    self.assertLessEqual(plan.target_src_token_indices.size, 256 // 4)
                    plans.append(plan)
                    page_offset += len(call.args[0])
                for field in (
                    "target_src_token_indices",
                    "target_dst_token_indices",
                    "draft_src_token_indices",
                    "draft_dst_token_indices",
                ):
                    np.testing.assert_array_equal(
                        np.concatenate([getattr(p, field) for p in plans]),
                        getattr(full, field),
                    )

    def test_partial_nonfinal_page_is_sent_with_final_tail(self):
        sched, req = self.make_request(total=1031)
        SchedulerDisaggregationPrefillMixin._send_kv_chunk(sched, req, end_idx=1000)
        self.assertEqual(req.start_send_idx, 960)
        self.assertIn(req.rid, sched.disagg_prefill_pending_chunk_rids)
        SchedulerDisaggregationPrefillMixin._send_kv_chunk(sched, req, last_chunk=True)
        calls = req.disagg_kv_sender.send.call_args_list
        np.testing.assert_array_equal(
            np.concatenate([c.args[0] for c in calls]), np.arange(17)
        )
        self.assertEqual(sum(c.kwargs["num_kv_tokens"] for c in calls), 1031)
        self.assertTrue(all(c.args[1] is None for c in calls[:-1]))

    def test_empty_final_send_retains_state(self):
        sched, req = self.make_request(total=256, prefix=256)
        SchedulerDisaggregationPrefillMixin._send_kv_chunk(sched, req, last_chunk=True)
        call = req.disagg_kv_sender.send.call_args
        self.assertEqual(len(call.args[0]), 0)
        self.assertEqual(call.kwargs["num_kv_tokens"], 0)
        self.assertIsNotNone(call.args[1])

    def test_direct_transfer_remains_unsplit(self):
        sched, req = self.make_request(total=2055, limit=None)
        SchedulerDisaggregationPrefillMixin._send_kv_chunk(sched, req, last_chunk=True)
        req.disagg_kv_sender.send.assert_called_once()


if __name__ == "__main__":
    unittest.main()
