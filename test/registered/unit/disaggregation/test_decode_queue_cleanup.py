import unittest
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodeTransferQueue,
    HiCacheRestoreResult,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
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


class FakeWaitingReq:
    def __init__(self, rid: str, wait_queue_entry_time: float):
        self.rid = rid
        self.time_stats = SimpleNamespace(wait_queue_entry_time=wait_queue_entry_time)
        self.mamba_pool_idx = None


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
        scheduler.cur_batch = None
        scheduler.enable_overlap = False
        scheduler.ps = SimpleNamespace(pp_size=1)
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

    @patch("sglang.srt.managers.scheduler.release_kv_cache")
    @patch("sglang.srt.managers.scheduler.time.perf_counter")
    def test_waiting_timeout_releases_disagg_decode_kv_cache(
        self, mock_perf_counter, mock_release_kv_cache
    ):
        timed_out_req = FakeWaitingReq("timed-out", wait_queue_entry_time=98.0)
        fresh_req = FakeWaitingReq("fresh", wait_queue_entry_time=99.5)

        def assert_marked_aborted_before_release(req, _tree_cache):
            self.assertIs(req, timed_out_req)
            self.assertIsInstance(req.finished_reason, FINISH_ABORT)
            self.assertEqual(
                req.finished_reason.message, "Request waiting timeout reached."
            )
            self.assertEqual(
                req.finished_reason.status_code, HTTPStatus.SERVICE_UNAVAILABLE
            )

        mock_release_kv_cache.side_effect = assert_marked_aborted_before_release

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.waiting_queue = [timed_out_req, fresh_req]
        scheduler.enable_hicache_storage = False
        scheduler.enable_hierarchical_cache = False
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.tree_cache = MagicMock()
        scheduler.ipc_channels = SimpleNamespace(
            send_to_tokenizer=SimpleNamespace(send_output=MagicMock())
        )
        mock_perf_counter.return_value = 100.0

        with envs.SGLANG_REQ_WAITING_TIMEOUT.override(1.0):
            scheduler._abort_on_waiting_timeout()

        self.assertEqual(scheduler.waiting_queue, [fresh_req])
        mock_release_kv_cache.assert_called_once_with(
            timed_out_req, scheduler.tree_cache
        )
        scheduler.ipc_channels.send_to_tokenizer.send_output.assert_called_once()
        abort_req = scheduler.ipc_channels.send_to_tokenizer.send_output.call_args[0][0]
        self.assertEqual(abort_req.rid, "timed-out")
        self.assertEqual(abort_req.finished_reason["type"], "abort")

    @patch("sglang.srt.managers.scheduler.time.perf_counter")
    def test_waiting_timeout_terminates_hierarchical_prefetch(self, mock_perf_counter):
        timed_out_req = FakeWaitingReq("timed-out", wait_queue_entry_time=98.0)

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.waiting_queue = [timed_out_req]
        scheduler.enable_hicache_storage = False
        scheduler.enable_hierarchical_cache = True
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.tree_cache = MagicMock()
        scheduler.ipc_channels = SimpleNamespace(
            send_to_tokenizer=SimpleNamespace(send_output=MagicMock())
        )
        mock_perf_counter.return_value = 100.0

        with envs.SGLANG_REQ_WAITING_TIMEOUT.override(1.0):
            scheduler._abort_on_waiting_timeout()

        self.assertEqual(scheduler.waiting_queue, [])
        scheduler.tree_cache.terminate_prefetch.assert_called_once_with("timed-out")

    @patch("sglang.srt.managers.scheduler.maybe_release_metadata_buffer")
    @patch("sglang.srt.managers.scheduler.time.perf_counter")
    def test_waiting_timeout_aborts_finalized_prefill_sender_before_metadata_release(
        self, mock_perf_counter, mock_release_metadata_buffer
    ):
        timed_out_req = FakeWaitingReq("timed-out", wait_queue_entry_time=98.0)
        timed_out_req.pending_bootstrap = False
        timed_out_req.disagg_kv_sender = MagicMock()

        def assert_sender_aborted_before_release(*_args, **_kwargs):
            timed_out_req.disagg_kv_sender.abort.assert_called_once_with()

        mock_release_metadata_buffer.side_effect = assert_sender_aborted_before_release

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.waiting_queue = [timed_out_req]
        scheduler.enable_hicache_storage = False
        scheduler.enable_hierarchical_cache = False
        scheduler.disaggregation_mode = DisaggregationMode.PREFILL
        scheduler.tree_cache = MagicMock()
        scheduler.req_to_metadata_buffer_idx_allocator = MagicMock()
        scheduler.ipc_channels = SimpleNamespace(
            send_to_tokenizer=SimpleNamespace(send_output=MagicMock())
        )
        mock_perf_counter.return_value = 100.0

        with envs.SGLANG_REQ_WAITING_TIMEOUT.override(1.0):
            scheduler._abort_on_waiting_timeout()

        self.assertEqual(scheduler.waiting_queue, [])
        timed_out_req.disagg_kv_sender.abort.assert_called_once_with()
        mock_release_metadata_buffer.assert_called_once_with(
            timed_out_req, scheduler.req_to_metadata_buffer_idx_allocator
        )

    def test_disagg_decode_scheduler_runs_waiting_timeout_cleanup(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler._abort_on_waiting_timeout = MagicMock()
        scheduler.get_new_prebuilt_batch = MagicMock(return_value=None)
        scheduler.running_batch = MagicMock()
        scheduler.running_batch.is_empty.return_value = True
        scheduler.dp_attn_adapter = MagicMock()
        scheduler.dp_attn_adapter.maybe_prepare_mlp_sync_batch.return_value = None

        batch = scheduler.get_next_disagg_decode_batch_to_run()

        self.assertIsNone(batch)
        scheduler._abort_on_waiting_timeout.assert_called_once_with()

    def test_disagg_prefill_scheduler_runs_waiting_timeout_cleanup(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.process_pending_chunked_abort = MagicMock()
        scheduler._abort_on_waiting_timeout = MagicMock()
        scheduler.running_batch = SimpleNamespace(batch_is_full=True)
        scheduler.process_prefill_chunk = MagicMock()
        scheduler.resolve_waiting_queue_bootstrap = MagicMock()
        scheduler.get_new_batch_prefill = MagicMock(return_value=None)
        scheduler.dp_attn_adapter = MagicMock()
        scheduler.dp_attn_adapter.maybe_prepare_mlp_sync_batch.return_value = None

        batch = scheduler.get_next_disagg_prefill_batch_to_run()

        self.assertIsNone(batch)
        scheduler._abort_on_waiting_timeout.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
