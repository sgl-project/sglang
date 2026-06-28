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
    def test_hisparse_request_length_uses_full_token_pool_capacity(self):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.max_total_num_tokens = 10
        queue._uses_swa_tail_prealloc = MagicMock(return_value=False)

        scheduler = MagicMock()
        scheduler.enable_hisparse = True
        scheduler.tp_worker.model_runner.max_token_pool_size = 20
        scheduler.output_streamer = MagicMock()
        queue.scheduler = scheduler

        req = SimpleNamespace(
            rid="hisparse-host-backed",
            origin_input_ids=list(range(15)),
            return_logprob=False,
        )

        self.assertFalse(queue._check_if_req_exceed_kv_capacity(req))
        scheduler.output_streamer.stream_output.assert_not_called()

    @patch("sglang.srt.disaggregation.decode.prepare_abort")
    def test_hisparse_request_length_rejects_above_full_token_pool_capacity(
        self, mock_prepare_abort
    ):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.max_total_num_tokens = 10
        queue._uses_swa_tail_prealloc = MagicMock(return_value=False)

        scheduler = MagicMock()
        scheduler.enable_hisparse = True
        scheduler.tp_worker.model_runner.max_token_pool_size = 20
        scheduler.output_streamer = MagicMock()
        queue.scheduler = scheduler

        req = SimpleNamespace(
            rid="hisparse-too-long",
            origin_input_ids=list(range(21)),
            return_logprob=False,
        )

        self.assertTrue(queue._check_if_req_exceed_kv_capacity(req))
        mock_prepare_abort.assert_called_once()
        message = mock_prepare_abort.call_args.args[1]
        self.assertIn("21 > 20", message)
        scheduler.output_streamer.stream_output.assert_called_once_with(
            [req], req.return_logprob
        )

    def test_non_hisparse_request_length_keeps_max_total_num_tokens_capacity(self):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.max_total_num_tokens = 10
        queue._uses_swa_tail_prealloc = MagicMock(return_value=False)

        scheduler = MagicMock()
        scheduler.enable_hisparse = False
        scheduler.tp_worker.model_runner.max_token_pool_size = 20
        scheduler.output_streamer = MagicMock()
        queue.scheduler = scheduler

        req = SimpleNamespace(
            rid="normal-too-long",
            origin_input_ids=list(range(15)),
            return_logprob=False,
        )

        with patch("sglang.srt.disaggregation.decode.prepare_abort") as mock_abort:
            self.assertTrue(queue._check_if_req_exceed_kv_capacity(req))
            message = mock_abort.call_args.args[1]
            self.assertIn("15 > 10", message)

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


if __name__ == "__main__":
    unittest.main()
