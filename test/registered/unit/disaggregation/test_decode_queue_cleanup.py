import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodeTransferQueue,
    HiCacheRestoreResult,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
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
    def test_prealloc_abort_clears_receiver_before_removing_request(self):
        receiver = FakeReceiver()
        req = SimpleNamespace(
            rid="abort-prealloc",
            finished_reason=FINISH_ABORT("aborted"),
            return_logprob=False,
        )
        decode_req = SimpleNamespace(req=req, kv_receiver=receiver)

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.queue = [decode_req]
        queue.pending_reqs = []
        queue.retracted_queue = []
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
        queue.queue = [decode_req]
        queue.pending_reqs = [decode_req]  # same object, dual ownership
        queue.retracted_queue = []
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

    @patch("sglang.srt.disaggregation.decode.release_kv_cache")
    @patch("sglang.srt.disaggregation.decode.prepare_abort")
    @patch("sglang.srt.disaggregation.decode.poll_and_all_reduce")
    def test_pp_failed_consensus_overrides_local_success(
        self, mock_poll, mock_prepare_abort, mock_release_kv_cache
    ):
        receiver = FakeReceiver()
        req = SimpleNamespace(
            rid="global-failed-local-success",
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
        queue._clean_hicache_prefetch_resources = MagicMock()
        queue._process_hicache_local_restores = MagicMock()

        scheduler = MagicMock()
        scheduler.enable_decode_hicache = True
        scheduler.enable_hisparse = False
        scheduler.output_streamer = MagicMock()
        scheduler.metrics_reporter.enable_metrics = False
        queue.scheduler = scheduler

        # A second local poll would report success, but the PP failure consensus
        # must force the request down the failure cleanup path on every stage.
        mock_poll.return_value = [KVPoll.Success]

        transferred = queue.pop_transferred(pp_good_rids=[], pp_bad_rids=[req.rid])

        self.assertEqual(transferred, [])
        self.assertEqual(queue.queue, [])
        self.assertTrue(receiver.clear_called)
        self.assertIsNone(decode_req.kv_receiver)
        mock_poll.assert_not_called()
        queue._process_hicache_local_restores.assert_not_called()
        queue._clean_hicache_prefetch_resources.assert_called_once_with(decode_req)
        mock_prepare_abort.assert_called_once()
        mock_release_kv_cache.assert_called_once_with(
            req, queue.tree_cache, is_insert=False
        )

    @patch("sglang.srt.disaggregation.decode.poll_and_all_reduce")
    def test_pp_success_consensus_skips_second_local_poll(self, mock_poll):
        receiver = FakeReceiver()
        req = SimpleNamespace(
            rid="global-success-local-changed",
            bootstrap_room=8,
            return_logprob=False,
            finished_reason=None,
        )
        decode_req = SimpleNamespace(
            req=req,
            kv_receiver=receiver,
            metadata_buffer_index=2,
            hicache_restore_status=HiCacheRestoreResult.READY,
        )

        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.queue = [decode_req]
        queue.enable_staging = False
        queue.gloo_group = MagicMock()
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.metadata_buffers = SimpleNamespace(bootstrap_room=[None] * 3)
        queue._commit_transfer_to_req = MagicMock()

        scheduler = MagicMock()
        scheduler.enable_decode_hicache = False
        scheduler.metrics_reporter.enable_metrics = False
        queue.scheduler = scheduler
        mock_poll.return_value = [KVPoll.Failed]

        transferred = queue.pop_transferred(pp_good_rids=[req.rid], pp_bad_rids=[])

        self.assertEqual(transferred, [req])
        self.assertEqual(queue.queue, [])
        mock_poll.assert_not_called()
        queue._commit_transfer_to_req.assert_called_once_with(decode_req)
        queue.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(2)

    def test_process_decode_transfer_queue_preserves_consensus_classes(self):
        queue = MagicMock()
        released_req = SimpleNamespace(rid="good")
        queue.pop_transferred.return_value = [released_req]
        scheduler = SimpleNamespace(
            disagg_decode_transfer_queue=queue,
            enable_hisparse=False,
            waiting_queue=[],
        )
        release_payload = [["good"], ["bad"]]

        forwarded = SchedulerPPMixin.process_decode_transfer_queue(
            scheduler, release_payload
        )

        self.assertIs(forwarded, release_payload)
        self.assertEqual(scheduler.waiting_queue, [released_req])
        queue.pop_transferred.assert_called_once_with(
            pp_good_rids=["good"], pp_bad_rids=["bad"]
        )

    def test_decode_transfer_consensus_intersects_success_and_unions_failure(self):
        queue = MagicMock()
        queue.get_finished_rids.return_value = (
            ["good-on-both", "good-only-current"],
            ["bad-current"],
        )
        scheduler = SimpleNamespace(
            pp_group=SimpleNamespace(is_first_rank=False),
            disagg_decode_transfer_queue=queue,
            _pp_recv_pyobj_from_prev_stage=MagicMock(
                return_value=[
                    ["good-on-both", "good-only-previous"],
                    ["bad-previous"],
                ]
            ),
        )

        good_rids, bad_rids = SchedulerPPMixin._pp_pd_get_decode_transferred_ids(
            scheduler
        )

        self.assertEqual(set(good_rids), {"good-on-both"})
        self.assertEqual(set(bad_rids), {"bad-current", "bad-previous"})

    @patch("sglang.srt.disaggregation.decode.prepare_abort")
    def test_pp_prealloc_failed_consensus_overrides_local_success(
        self, mock_prepare_abort
    ):
        receiver = FakeReceiver()
        req = SimpleNamespace(
            rid="global-failed-local-waiting",
            bootstrap_room=9,
            return_logprob=False,
            finished_reason=None,
        )
        decode_req = SimpleNamespace(
            req=req,
            kv_receiver=receiver,
            waiting_for_input=True,
        )

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.queue = [decode_req]
        queue.pending_reqs = []
        queue.tp_rank = 0
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
        scheduler.metrics_reporter.enable_metrics = False
        queue.scheduler = scheduler
        mock_prepare_abort.side_effect = lambda target, *_args, **_kwargs: setattr(
            target, "finished_reason", FINISH_ABORT("PP consensus failure")
        )

        good, failed = queue.pop_preallocated(pp_good_rids=[], pp_bad_rids=[req.rid])

        self.assertEqual(good, [])
        self.assertEqual(failed, [decode_req])
        self.assertEqual(queue.queue, [])
        self.assertTrue(receiver.clear_called)
        self.assertIsNone(decode_req.kv_receiver)
        queue._update_handshake_waiters.assert_not_called()
        mock_prepare_abort.assert_called_once()
        scheduler.output_streamer.stream_output.assert_called_once_with(
            [req], req.return_logprob
        )

    def test_process_prealloc_queue_preserves_consensus_classes(self):
        queue = MagicMock()
        good_req = SimpleNamespace(req=SimpleNamespace(rid="good"))
        failed_req = SimpleNamespace(req=SimpleNamespace(rid="bad"))
        queue.pop_preallocated.return_value = ([good_req], [failed_req])
        transfer_queue = MagicMock()
        scheduler = SimpleNamespace(
            disagg_decode_prealloc_queue=queue,
            disagg_decode_transfer_queue=transfer_queue,
        )
        queue.retracted_queue = []
        payload = [["good"], ["bad"]]

        forwarded = SchedulerPPMixin.process_prealloc_queue(scheduler, payload)

        self.assertIs(forwarded, payload)
        queue.pop_preallocated.assert_called_once_with(
            pp_good_rids=["good"], pp_bad_rids=["bad"]
        )
        transfer_queue.extend.assert_called_once_with([good_req])

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
