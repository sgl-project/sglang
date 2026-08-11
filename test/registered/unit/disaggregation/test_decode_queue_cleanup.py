import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVReceiver
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodeTransferQueue,
    HiCacheRestoreResult,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class FakeReceiver:
    def __init__(self):
        self.clear_called = False

    def clear(self):
        self.clear_called = True

    def failure_exception(self):
        return None


class TestDecodeQueueCleanup(CustomTestCase):
    def test_paged_swa_retraction_resume_uses_physical_page_budget(self):
        page_size = 128
        fill_len = 574
        physical_tokens_per_req = 5 * page_size
        physical_available = 18 * page_size

        reqs = [
            SimpleNamespace(
                rid=f"req-{i}",
                origin_input_ids=[0] * fill_len,
                output_ids=[],
                is_retracted=True,
                load_kv_cache=MagicMock(),
            )
            for i in range(4)
        ]

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.retracted_queue = reqs.copy()
        queue.num_reserved_decode_tokens = 0
        queue.req_to_token_pool = SimpleNamespace(available_size=lambda: len(reqs))
        queue.token_to_kv_pool_allocator = SimpleNamespace(page_size=page_size)
        queue.scheduler = SimpleNamespace(
            sliding_window_size=2047,
            server_args=SimpleNamespace(disable_radix_cache=True),
        )
        queue._uses_swa_tail_prealloc = MagicMock(return_value=True)
        queue._swa_aware_allocatable_token_budgets = MagicMock(
            return_value=(physical_available, physical_available)
        )

        def pre_alloc(_req):
            nonlocal physical_available
            self.assertGreaterEqual(physical_available, physical_tokens_per_req)
            physical_available -= physical_tokens_per_req

        queue._pre_alloc = MagicMock(side_effect=pre_alloc)

        resumed = queue.resume_retracted_reqs()

        self.assertEqual(resumed, reqs[:3])
        self.assertEqual(queue.retracted_queue, reqs[3:])
        self.assertEqual(physical_available, 3 * page_size)
        self.assertEqual(queue._pre_alloc.call_count, 3)

    def test_prealloc_abort_clears_receiver_before_removing_request(self):
        receiver = FakeReceiver()
        req = SimpleNamespace(
            rid="abort-prealloc",
            finished_reason=FINISH_ABORT("aborted"),
            return_logprob=False,
        )
        decode_req = SimpleNamespace(req=req, kv_receiver=receiver)

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pp_size = 1
        queue.queue = [decode_req]
        queue.pending_reqs = []
        queue.retracted_queue = []
        queue._dp_rank_query_first_attempt = {}
        queue._resolve_pending_reqs = MagicMock()
        queue._update_handshake_waiters = MagicMock()
        queue._uses_swa_tail_prealloc = MagicMock(return_value=False)
        queue._allocatable_token_budgets = MagicMock(return_value=0)
        queue._hicache_pending_restore_tokens = MagicMock(return_value=0)

        scheduler = MagicMock()
        scheduler.running_batch.reqs = []
        scheduler.enable_priority_scheduling = False
        scheduler.enable_hisparse = False
        scheduler.output_streamer = MagicMock()
        queue.scheduler = scheduler

        preallocated, failed = queue.pop_preallocated()

        self.assertEqual(preallocated, [])
        self.assertEqual(failed, [decode_req])
        self.assertEqual(queue.queue, [])
        self.assertTrue(receiver.clear_called)
        self.assertIsNone(decode_req.kv_receiver)
        scheduler.output_streamer.stream_output.assert_called_once_with(
            [req], req.return_logprob
        )

    def test_prealloc_abort_also_drops_from_pending_reqs(self):
        # Same DecodeRequest lives in both queue and pending_reqs (add() slow
        # path). Aborting must drop it from both, and compare by identity since
        # DecodeRequest's dataclass __eq__ would compare the tensor receiver.
        class BadEqReceiver(FakeReceiver):
            def __eq__(self, other):
                raise TypeError("use identity comparison, not value equality")

            __hash__ = object.__hash__

        receiver = BadEqReceiver()
        req = SimpleNamespace(
            rid="abort-shared",
            finished_reason=FINISH_ABORT("aborted"),
            return_logprob=False,
        )
        decode_req = SimpleNamespace(req=req, kv_receiver=receiver)

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pp_size = 1
        queue.queue = [decode_req]
        queue.pending_reqs = [decode_req]  # same object, dual ownership
        queue.retracted_queue = []
        queue._dp_rank_query_first_attempt = {}
        queue._resolve_pending_reqs = MagicMock()
        queue._update_handshake_waiters = MagicMock()
        queue._uses_swa_tail_prealloc = MagicMock(return_value=False)
        queue._allocatable_token_budgets = MagicMock(return_value=0)
        queue._hicache_pending_restore_tokens = MagicMock(return_value=0)

        scheduler = MagicMock()
        scheduler.running_batch.reqs = []
        scheduler.enable_priority_scheduling = False
        scheduler.enable_hisparse = False
        scheduler.output_streamer = MagicMock()
        queue.scheduler = scheduler

        # Must not raise on the receiver __eq__ above.
        preallocated, failed = queue.pop_preallocated()

        self.assertEqual(preallocated, [])
        self.assertEqual(failed, [decode_req])
        self.assertEqual(queue.queue, [])
        self.assertTrue(all(r is not decode_req for r in queue.pending_reqs))
        self.assertIsNone(decode_req.kv_receiver)

    def test_ensure_prefill_info_tolerates_cleared_receiver(self):
        # A req whose kv_receiver was already cleared must not crash on .abort().
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue._max_ensure_retries = 1
        queue._ensure_retry_interval = 0
        queue._ensure_retry_count = {"127.0.0.1:11500": 0}
        queue._ensure_last_attempt_time = {}
        queue.kv_manager = MagicMock()
        queue.kv_manager.try_ensure_parallel_info.return_value = False

        cleared_req = SimpleNamespace(
            req=SimpleNamespace(rid="cleared"), kv_receiver=None
        )
        addr_to_reqs = {"127.0.0.1:11500": [cleared_req]}

        ready, remaining = queue._ensure_prefill_info(addr_to_reqs)

        self.assertEqual(ready, {})
        self.assertEqual(remaining, [])

    def _make_dp_rank_query_queue(self, decode_req, prefill_info):
        addr = "10.0.0.1:8998"
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pending_reqs = [decode_req]
        queue._ensure_retry_count = {}
        queue._ensure_last_attempt_time = {}
        queue._ensure_retry_interval = 0
        queue._max_ensure_retries = 15
        queue.kv_manager = SimpleNamespace(
            try_ensure_parallel_info=lambda addr: True,
            connection_lock=threading.Lock(),
            prefill_info_table={addr: prefill_info},
            handle_instance_replacement=MagicMock(),
            record_failure=MagicMock(),
        )
        queue._dp_rank_query_first_attempt = {}
        queue._dp_rank_query_timeout = 300.0
        queue._dp_rank_query_fresh_retry_grace = 150.0
        queue._dp_rank_query_last_fresh_retry = {}
        return queue

    def test_unresolvable_dp_rank_query_fails_request_after_timeout(self):
        """Regression: a room the current prefill never registered resolved
        to nothing on every cycle and re-entered pending_reqs with no
        timeout armed (init() never ran) -- it cycled forever. The pass must
        arm its own deadline and abort the receiver on expiry."""
        prefill_info = SimpleNamespace(
            instance_id="live", dp_size=2, follow_bootstrap_room=False
        )
        req = SimpleNamespace(
            bootstrap_room=7,
            disagg_prefill_dp_rank=None,
            bootstrap_host="10.0.0.1",
            bootstrap_port=8998,
        )
        decode_req = SimpleNamespace(req=req, kv_receiver=MagicMock())
        queue = self._make_dp_rank_query_queue(decode_req, prefill_info)

        with patch.object(
            CommonKVReceiver, "query_prefill_dp_ranks", return_value={}
        ) as query:
            queue._resolve_pending_reqs()
        query.assert_called_once_with("10.0.0.1:8998", [7], expected_instance_id="live")
        self.assertEqual(queue.pending_reqs, [decode_req])
        self.assertIn(7, queue._dp_rank_query_first_attempt)
        decode_req.kv_receiver.abort.assert_not_called()

        queue._dp_rank_query_first_attempt[7] -= 301.0
        with patch.object(CommonKVReceiver, "query_prefill_dp_ranks", return_value={}):
            queue._resolve_pending_reqs()
        decode_req.kv_receiver.abort.assert_called_once_with()
        # Regression: abort() records a generic AbortReq reason; the deadline
        # path must overwrite it so the client sees why the request failed.
        queue.kv_manager.record_failure.assert_called_once()
        self.assertIn(
            "Could not resolve prefill dp_rank",
            queue.kv_manager.record_failure.call_args.args[1],
        )
        self.assertEqual(queue.pending_reqs, [])
        self.assertEqual(queue._dp_rank_query_first_attempt, {})

        resolved_req = SimpleNamespace(req=req, kv_receiver=MagicMock())
        queue = self._make_dp_rank_query_queue(resolved_req, prefill_info)
        with patch.object(
            CommonKVReceiver, "query_prefill_dp_ranks", return_value={"7": 1}
        ):
            queue._resolve_pending_reqs()
        resolved_req.kv_receiver.init.assert_called_once_with(1)
        self.assertEqual(queue.pending_reqs, [])
        self.assertEqual(queue._dp_rank_query_first_attempt, {})

    def test_persistent_dp_rank_query_mismatch_evicts_stale_topology(self):
        """Regression: on a persistent mismatch (None) the pass re-asked with
        the same stale expectation until the deadline aborted the rooms --
        and the abort exempts the receiver from the failure-path repair. The
        pass must evict the snapshot (generation-gated) and retire the rooms
        in flight against it instead."""
        prefill_info = SimpleNamespace(
            instance_id="stale", dp_size=2, follow_bootstrap_room=False
        )
        req = SimpleNamespace(
            bootstrap_room=7,
            disagg_prefill_dp_rank=None,
            bootstrap_host="10.0.0.1",
            bootstrap_port=8998,
        )
        decode_req = SimpleNamespace(req=req, kv_receiver=MagicMock())
        queue = self._make_dp_rank_query_queue(decode_req, prefill_info)

        with patch.object(
            CommonKVReceiver, "query_prefill_dp_ranks", return_value=None
        ):
            queue._resolve_pending_reqs()

        # Retirement, not cache-only invalidation: in-flight rooms against
        # the evicted snapshot would otherwise wait out the full timeout
        # (the heartbeat can no longer indict them once the replacement's
        # topology is published).
        queue.kv_manager.handle_instance_replacement.assert_called_once_with(
            "10.0.0.1:8998",
            expected_prefill_info=prefill_info,
            detected_by="dp-rank resolution",
        )
        self.assertEqual(queue.pending_reqs, [decode_req])
        decode_req.kv_receiver.abort.assert_not_called()

    def test_externally_removed_rooms_leave_no_deadline_entry(self):
        """Regression: the early return on an emptied queue skipped deadline
        pruning, stranding removed rooms' entries; a reused room id
        inherited the expired deadline and failed immediately."""
        prefill_info = SimpleNamespace(
            instance_id="live", dp_size=2, follow_bootstrap_room=False
        )
        req = SimpleNamespace(
            bootstrap_room=7,
            disagg_prefill_dp_rank=None,
            bootstrap_host="10.0.0.1",
            bootstrap_port=8998,
        )
        decode_req = SimpleNamespace(req=req, kv_receiver=MagicMock())
        queue = self._make_dp_rank_query_queue(decode_req, prefill_info)
        with patch.object(CommonKVReceiver, "query_prefill_dp_ranks", return_value={}):
            queue._resolve_pending_reqs()
        self.assertIn(7, queue._dp_rank_query_first_attempt)

        # pop_preallocated's abort scan drops the request from pending_reqs
        # without running the resolution pass again.
        queue.pending_reqs = []
        queue._resolve_pending_reqs()
        self.assertEqual(queue._dp_rank_query_first_attempt, {})

    def test_drop_failed_pending_reqs_prunes_deadlines_with_the_requests(self):
        """Regression: pop_preallocated left removed requests' deadline
        entries for the next pass to prune; a room reused before then
        inherited the expired deadline via setdefault and aborted on its
        first cycle."""
        prefill_info = SimpleNamespace(
            instance_id="live", dp_size=2, follow_bootstrap_room=False
        )
        failed_req = SimpleNamespace(
            req=SimpleNamespace(bootstrap_room=7), kv_receiver=None
        )
        kept_req = SimpleNamespace(
            req=SimpleNamespace(bootstrap_room=8), kv_receiver=MagicMock()
        )
        queue = self._make_dp_rank_query_queue(failed_req, prefill_info)
        queue.pending_reqs = [failed_req, kept_req]
        queue._dp_rank_query_first_attempt = {7: 100.0, 8: 200.0}

        queue._drop_failed_pending_reqs([failed_req])

        self.assertEqual(queue.pending_reqs, [kept_req])
        self.assertEqual(queue._dp_rank_query_first_attempt, {8: 200.0})

    def test_stale_missing_rooms_force_reconnect_through_address_resolution(self):
        """Regression: a draining replaced instance presents the expected id
        with the replacement's rooms simply absent, so neither the mismatch
        machinery nor the heartbeat ever fired and every request burned its
        deadline. Rooms missing past the grace period drop the pooled
        session (rate-limited); a freshly missing room must not force churn."""
        prefill_info = SimpleNamespace(
            instance_id="live", dp_size=2, follow_bootstrap_room=False
        )
        req = SimpleNamespace(
            bootstrap_room=7,
            disagg_prefill_dp_rank=None,
            bootstrap_host="10.0.0.1",
            bootstrap_port=8998,
        )
        decode_req = SimpleNamespace(req=req, kv_receiver=MagicMock())
        queue = self._make_dp_rank_query_queue(decode_req, prefill_info)

        with patch.object(
            CommonKVReceiver, "query_prefill_dp_ranks", return_value={}
        ), patch(
            "sglang.srt.disaggregation.decode._drop_bootstrap_session"
        ) as drop_session:
            # Freshly missing: within the grace period, no reconnect.
            queue._resolve_pending_reqs()
            drop_session.assert_not_called()

            # Past the grace period: drop the pooled session once.
            queue._dp_rank_query_first_attempt[7] -= 151.0
            queue._resolve_pending_reqs()
            drop_session.assert_called_once_with("10.0.0.1:8998")

            # Immediately after: rate-limited, no second drop.
            queue._resolve_pending_reqs()
            drop_session.assert_called_once_with("10.0.0.1:8998")

        self.assertEqual(queue.pending_reqs, [decode_req])
        decode_req.kv_receiver.abort.assert_not_called()

    @patch("sglang.srt.disaggregation.decode.release_kv_cache")
    @patch("sglang.srt.disaggregation.decode.prepare_abort")
    @patch("sglang.srt.disaggregation.decode.poll_and_all_reduce")
    def test_transfer_failure_clears_receiver_before_removing_request(
        self, mock_poll, mock_prepare_abort, mock_release_kv_cache
    ):
        receiver = FakeReceiver()
        req = SimpleNamespace(
            rid="failed-transfer",
            bootstrap_room=7,
            return_logprob=False,
        )
        decode_req = SimpleNamespace(
            req=req,
            kv_receiver=receiver,
            metadata_buffer_index=3,
            hicache_restore_status=HiCacheRestoreResult.READY,
        )

        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.queue = [decode_req]
        queue.enable_staging = False
        queue.gloo_group = MagicMock()
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.tp_rank = 0
        queue.tree_cache = MagicMock()
        queue.metadata_buffers = SimpleNamespace(bootstrap_room=[None] * 4)
        queue.spec_algorithm = MagicMock()
        queue.spec_algorithm.is_none.return_value = True
        queue._clean_hicache_prefetch_resources = MagicMock()

        scheduler = MagicMock()
        scheduler.enable_decode_hicache = False
        scheduler.enable_hisparse = False
        scheduler.output_streamer = MagicMock()
        scheduler.metrics_reporter.enable_metrics = False
        queue.scheduler = scheduler

        mock_poll.return_value = [KVPoll.Failed]

        transferred = queue.pop_transferred()

        self.assertEqual(transferred, [])
        self.assertEqual(queue.queue, [])
        self.assertTrue(receiver.clear_called)
        self.assertIsNone(decode_req.kv_receiver)
        queue.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(3)
        scheduler.output_streamer.stream_output.assert_called_once_with(
            [req], req.return_logprob
        )
        mock_prepare_abort.assert_called_once()
        mock_release_kv_cache.assert_called_once_with(
            req, queue.tree_cache, is_insert=False
        )

    def test_retracted_decode_requests_keep_scheduler_non_idle(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.running_batch = MagicMock()
        scheduler.running_batch.is_empty.return_value = True
        scheduler.chunked_req = None
        scheduler.dllm_manager = MagicMock()
        scheduler.dllm_manager.any_staging_reqs.return_value = False
        scheduler.last_batch = None
        scheduler.cur_batch_for_debug = None
        scheduler.enable_overlap = False
        scheduler.ps = ParallelState.trivial()
        scheduler.running_mbs = []
        scheduler.waiting_queue = []
        scheduler.grammar_manager = SimpleNamespace(grammar_queue=[])
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            queue=[], retracted_queue=[object()]
        )
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(queue=[])
        scheduler.decode_offload_manager = None
        scheduler.enable_hisparse = False
        scheduler.enable_hierarchical_cache = False

        self.assertFalse(scheduler.is_fully_idle())


if __name__ == "__main__":
    unittest.main()
