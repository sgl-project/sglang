"""Unit test for the race condition fix in DecodePreallocQueue.

Verifies that:
1. _ensure_prefill_info handles None kv_receiver gracefully (no crash).
2. pop_preallocated synchronizes pending_reqs after abort cleanup.

This test does NOT require GPU or a running server — it uses mock objects
to simulate the exact race condition between _update_handshake_waiters
and _ensure_prefill_info.

Related: https://github.com/sgl-project/sglang/issues/21680
"""

from __future__ import annotations

import unittest
from http import HTTPStatus
from unittest.mock import MagicMock

from sglang.srt.disaggregation.decode import DecodePreallocQueue, DecodeRequest
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="extra-a", runner_config="1-gpu-small")


class FakeKVReceiver:
    """Minimal fake KV receiver for testing."""

    def __init__(self):
        self.conclude_state = None
        self._aborted = False
        self._cleared = False

    def abort(self):
        self._aborted = True

    def clear(self):
        self._cleared = True

    def poll(self):
        return self.conclude_state


def _make_fake_req(rid="test-001"):
    """Create a minimal Req-like mock for testing."""
    req = MagicMock()
    req.rid = rid
    req.bootstrap_room = 12345
    req.bootstrap_host = "10.0.0.1"
    req.bootstrap_port = 30200
    req.finished_reason = None
    req.return_logprob = False
    req.origin_input_ids = [1, 2, 3]
    req.output_ids = []
    req.sampling_params = MagicMock(max_new_tokens=100)
    req.priority = 0
    req.seqlen = 10
    req.time_stats = MagicMock()
    return req


class TestDecodePreallocQueueRaceCondition(CustomTestCase):
    """Test fix for race condition between handshake abort and prefill info timeout."""

    def _create_mock_queue_obj(self):
        """Create a mock DecodePreallocQueue with minimal required attributes."""
        queue_obj = MagicMock(spec=DecodePreallocQueue)
        queue_obj._ensure_retry_count = {}
        queue_obj._ensure_last_attempt_time = {}
        queue_obj._max_ensure_retries = 15
        queue_obj._ensure_retry_interval = 1.0
        queue_obj.pending_reqs = []
        queue_obj.scheduler = MagicMock()
        queue_obj.scheduler.running_batch.reqs = []
        queue_obj.scheduler.enable_priority_scheduling = False
        queue_obj.scheduler.enable_hisparse = False
        queue_obj.scheduler.metrics_reporter.enable_metrics = False
        queue_obj.enable_staging = False
        queue_obj.gloo_group = MagicMock()
        queue_obj._resolve_pending_reqs = MagicMock()
        queue_obj._update_handshake_waiters = MagicMock()
        queue_obj._uses_swa_tail_prealloc = MagicMock(return_value=False)
        queue_obj._allocatable_token_budgets = MagicMock(return_value=10000)
        queue_obj._hicache_pending_restore_tokens = MagicMock(return_value=0)
        queue_obj.retracted_queue = []
        queue_obj.num_reserved_decode_tokens = 256
        return queue_obj

    def test_ensure_prefill_info_handles_none_kv_receiver(self):
        """_ensure_prefill_info should not crash when kv_receiver is already None."""
        queue = self._create_mock_queue_obj()
        queue._ensure_retry_count = {"10.0.0.1:30200": 14}
        queue._ensure_last_attempt_time = {"10.0.0.1:30200": 0}
        queue._ensure_retry_interval = 0
        queue.kv_manager = MagicMock()
        queue.kv_manager.try_ensure_parallel_info.return_value = False

        # Simulate: kv_receiver already cleared by handshake failure path
        req = _make_fake_req()
        req.finished_reason = FINISH_ABORT()
        decode_req = DecodeRequest(req=req, kv_receiver=None)

        addr_to_reqs = {"10.0.0.1:30200": [decode_req]}

        # Should NOT raise AttributeError
        ready, remaining = DecodePreallocQueue._ensure_prefill_info(
            queue, addr_to_reqs
        )

        # Retry state should be cleaned up
        self.assertNotIn("10.0.0.1:30200", queue._ensure_retry_count)
        self.assertNotIn("10.0.0.1:30200", queue._ensure_last_attempt_time)

    def test_ensure_prefill_info_aborts_valid_receiver(self):
        """_ensure_prefill_info should still call abort() when kv_receiver is valid."""
        queue = self._create_mock_queue_obj()
        queue._ensure_retry_count = {"10.0.0.1:30200": 14}
        queue._ensure_last_attempt_time = {"10.0.0.1:30200": 0}
        queue._ensure_retry_interval = 0
        queue.kv_manager = MagicMock()
        queue.kv_manager.try_ensure_parallel_info.return_value = False

        receiver = FakeKVReceiver()
        req = _make_fake_req()
        decode_req = DecodeRequest(req=req, kv_receiver=receiver)

        addr_to_reqs = {"10.0.0.1:30200": [decode_req]}

        DecodePreallocQueue._ensure_prefill_info(queue, addr_to_reqs)

        self.assertTrue(receiver._aborted)

    def test_pop_preallocated_syncs_pending_reqs_on_abort(self):
        """pop_preallocated should remove aborted requests from pending_reqs."""
        queue_obj = self._create_mock_queue_obj()

        # Request in both queue and pending_reqs (the race condition setup)
        req = _make_fake_req(rid="abort-me")
        receiver = FakeKVReceiver()
        decode_req = DecodeRequest(req=req, kv_receiver=receiver)

        queue_obj.queue = [decode_req]
        queue_obj.pending_reqs = [decode_req]

        # Simulate: marked as FINISH_ABORT by _update_handshake_waiters
        req.finished_reason = FINISH_ABORT()

        # Call the real method
        preallocated, failed = DecodePreallocQueue.pop_preallocated(
            queue_obj, rids_to_check=None
        )

        # Verify: pending_reqs must be cleaned
        self.assertEqual(len(queue_obj.pending_reqs), 0)
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0].req.rid, "abort-me")

    def test_pop_preallocated_preserves_valid_pending_reqs(self):
        """pop_preallocated should not remove non-aborted requests from pending_reqs."""
        queue_obj = self._create_mock_queue_obj()

        # One aborted, one healthy
        req_abort = _make_fake_req(rid="abort-me")
        req_abort.finished_reason = FINISH_ABORT()
        receiver_abort = FakeKVReceiver()
        decode_req_abort = DecodeRequest(req=req_abort, kv_receiver=receiver_abort)

        req_healthy = _make_fake_req(rid="keep-me")
        receiver_healthy = FakeKVReceiver()
        decode_req_healthy = DecodeRequest(req=req_healthy, kv_receiver=receiver_healthy)

        queue_obj.queue = [decode_req_abort, decode_req_healthy]
        queue_obj.pending_reqs = [decode_req_abort, decode_req_healthy]

        preallocated, failed = DecodePreallocQueue.pop_preallocated(
            queue_obj, rids_to_check=None
        )

        # Only the aborted one should be removed from pending_reqs
        self.assertEqual(len(queue_obj.pending_reqs), 1)
        self.assertEqual(queue_obj.pending_reqs[0].req.rid, "keep-me")


if __name__ == "__main__":
    unittest.main()
