import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDSATailReqPoolIndex(unittest.TestCase):
    def test_decode_preallocation_uses_kv_owned_req_pool_idx(self):
        req = SimpleNamespace(
            rid="decode-dsa-tail",
            priority=0,
            origin_input_ids=[1, 2, 3],
            output_ids=[],
            finished_reason=None,
            return_logprob=False,
            sampling_params=SimpleNamespace(max_new_tokens=1),
            kv=SimpleNamespace(req_pool_idx=3, cache_protected_len=0),
            time_stats=MagicMock(),
        )
        receiver = MagicMock()
        decode_req = SimpleNamespace(
            req=req,
            waiting_for_input=True,
            kv_receiver=receiver,
            metadata_buffer_index=-1,
            is_rebootstrap=False,
        )

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pp_size = 1
        queue.queue = [decode_req]
        queue.pending_reqs = []
        queue.retracted_queue = []
        queue.num_reserved_decode_tokens = 0
        queue._num_published_destinations = 0
        queue._resolve_pending_reqs = MagicMock()
        queue._update_handshake_waiters = MagicMock()
        queue._uses_swa_tail_prealloc = MagicMock(return_value=False)
        queue._allocatable_token_budgets = MagicMock(return_value=1024)
        queue._hicache_pending_restore_tokens = MagicMock(return_value=0)
        queue._pre_alloc_fill_len = MagicMock(return_value=3)
        queue._pre_alloc = MagicMock(return_value=torch.arange(3))
        queue.req_to_token_pool = SimpleNamespace(
            available_size=lambda: 1,
            req_to_token=torch.arange(64).reshape(8, 8),
        )
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.req_to_metadata_buffer_idx_allocator.available_size.return_value = 1
        queue.req_to_metadata_buffer_idx_allocator.alloc.return_value = 0
        queue.token_to_kv_pool_allocator = MagicMock()
        queue.token_to_kv_pool_allocator.page_size = 1
        queue.token_to_kv_pool_allocator.translate_kv_indices_for_transfer.side_effect = (
            lambda indices: indices
        )
        queue.token_to_kv_pool = MagicMock()
        queue.token_to_kv_pool.page_size = 1
        queue.transfer_queue = SimpleNamespace(enable_staging=False)
        queue.kv_manager = SimpleNamespace(
            kv_args=SimpleNamespace(state_types=[StateType.DSA_TAIL])
        )
        queue.scheduler = SimpleNamespace(
            running_batch=SimpleNamespace(reqs=[]),
            enable_priority_scheduling=False,
            enable_hisparse=False,
            enable_decode_hicache=False,
            server_args=SimpleNamespace(disaggregation_decode_enable_radix_cache=False),
        )

        expected_tail = np.array([31, 32], dtype=np.int32)
        with patch(
            "sglang.srt.disaggregation.decode.get_dsa_tail_state_indices",
            return_value=expected_tail,
        ) as get_tail:
            preallocated, failed = queue.pop_preallocated()

        self.assertEqual(preallocated, [decode_req])
        self.assertEqual(failed, [])
        get_tail.assert_called_once_with(queue.token_to_kv_pool, 3, 3)
        state_indices = receiver.send_metadata.call_args.args[2]
        np.testing.assert_array_equal(state_indices[0], expected_tail)

    def test_prefill_final_send_uses_kv_owned_req_pool_idx(self):
        sender = MagicMock()
        sender.should_send_kv_chunk.return_value = True
        req = SimpleNamespace(
            rid="prefill-dsa-tail",
            origin_input_ids=[1, 2, 3, 4],
            extend_range=SimpleNamespace(end=4),
            start_send_idx=0,
            disagg_decode_prefix_len=0,
            kv=SimpleNamespace(req_pool_idx=5),
            disagg_kv_sender=sender,
        )
        pool = MagicMock()
        allocator = MagicMock()
        allocator.page_size = 1
        allocator.get_kvcache.return_value = pool
        allocator.translate_kv_indices_for_transfer.side_effect = (
            lambda indices: indices
        )
        scheduler = SimpleNamespace(
            token_to_kv_pool_allocator=allocator,
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.arange(64).reshape(8, 8)
            ),
            disagg_metadata_buffers=MagicMock(),
            disagg_prefill_bootstrap_queue=SimpleNamespace(
                kv_manager=SimpleNamespace(
                    kv_args=SimpleNamespace(state_types=[StateType.DSA_TAIL])
                )
            ),
            enable_staging=False,
            disagg_prefill_pending_chunk_rids={req.rid},
        )

        expected_tail = np.array([51], dtype=np.int32)
        with patch(
            "sglang.srt.disaggregation.prefill.get_dsa_tail_state_indices",
            return_value=expected_tail,
        ) as get_tail:
            SchedulerDisaggregationPrefillMixin.send_kv_chunk(
                scheduler, req, last_chunk=True
            )

        get_tail.assert_called_once_with(pool, 5, 4)
        state_indices = sender.send.call_args.args[1]
        np.testing.assert_array_equal(state_indices[0], expected_tail)
        self.assertEqual(req.start_send_idx, 4)
        self.assertNotIn(req.rid, scheduler.disagg_prefill_pending_chunk_rids)


if __name__ == "__main__":
    unittest.main()
