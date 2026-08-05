"""Unit tests for Mooncake KV-transfer metrics and the Transferring state.

The Mooncake backend inherits transfer_total_bytes accounting from
CommonKVSender, but until now it never measured transfer_latency_s, so
compute_and_observe_kv_transfer_metrics() fell back to
(completion_time - prefill_transfer_queue_entry_time), which only captures the
last chunk of a chunked prefill. These tests pin the NIXL-parity behavior:
the sender records the wall time from the first non-empty send() to the first
poll() that observes KVPoll.Success.

It also never reported KVPoll.Transferring (the prefill room status jumped
straight from WaitingForInput to Success), so the transfer_worker must mark
the room Transferring when it dequeues a chunk, again matching NIXL.
"""

import threading
import time
import unittest
from collections import defaultdict
from types import SimpleNamespace

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll, KVTransferMetric
from sglang.srt.disaggregation.common.utils import FastQueue, TransferKVChunk
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVManager,
    MooncakeKVSender,
    TransferInfo,
)
from sglang.srt.observability.trace import TraceNullContext
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROOM = 41
KV_ITEM_LENS_SUM = 100
STATE_ITEM_LENS_SUM = 7


class _StubManager:
    """Only the attributes MooncakeKVSender touches on the send/poll path."""

    def __init__(self):
        self.enable_all_cp_ranks_for_transfer = False
        self.is_dummy_cp_rank = False
        self.kv_item_lens_sum = KV_ITEM_LENS_SUM
        self.state_item_lens_sum = STATE_ITEM_LENS_SUM
        self.status = KVPoll.WaitingForInput
        self.transfer_requests = []

    def get_kv_replica_factor(self):
        return 1

    def add_transfer_request(self, *args, **kwargs):
        self.transfer_requests.append((args, kwargs))

    def check_status(self, bootstrap_room):
        return self.status


def _make_sender(num_kv_indices):
    """MooncakeKVSender bypassing __init__, wiring only the fields the path reads."""
    sender = MooncakeKVSender.__new__(MooncakeKVSender)
    sender.kv_mgr = _StubManager()
    sender.bootstrap_room = ROOM
    sender.conclude_state = None
    sender.curr_idx = 0
    sender._transfer_metric = KVTransferMetric()
    sender._transfer_num_kv_indices = 0
    sender._transfer_num_state_indices = 0
    sender._transfer_start_time = None
    sender.trace_ctx = TraceNullContext()
    sender.init(num_kv_indices, aux_index=3)
    return sender


class TestMooncakeTransferMetric(CustomTestCase):
    def test_latency_measured_from_first_send_to_success_poll(self):
        # Chunked prefill: latency must cover all chunks, not just the last one.
        sender = _make_sender(num_kv_indices=8)

        sender.send(np.arange(4, dtype=np.int32))
        self.assertIsNotNone(sender._transfer_start_time)
        self.assertEqual(sender.poll(), KVPoll.WaitingForInput)
        self.assertIsNone(sender.get_transfer_metric().transfer_latency_s)

        time.sleep(0.02)
        sender.send(np.arange(4, 8, dtype=np.int32))  # last chunk
        sender.kv_mgr.status = KVPoll.Success
        self.assertEqual(sender.poll(), KVPoll.Success)
        self.assertEqual(sender.conclude_state, KVPoll.Success)

        metric = sender.get_transfer_metric()
        self.assertIsNotNone(metric.transfer_latency_s)
        # Started before the sleep, concluded after it.
        self.assertGreaterEqual(metric.transfer_latency_s, 0.019)

        # Concluded senders must not re-measure on subsequent polls.
        latency = metric.transfer_latency_s
        self.assertEqual(sender.poll(), KVPoll.Success)
        self.assertEqual(sender.get_transfer_metric().transfer_latency_s, latency)

    def test_start_time_set_once_and_only_for_non_empty_chunks(self):
        sender = _make_sender(num_kv_indices=8)

        # A page-aligned empty chunk carries no payload: the clock must not start.
        sender.send(np.array([], dtype=np.int32))
        self.assertIsNone(sender._transfer_start_time)

        sender.send(np.arange(4, dtype=np.int32))
        start = sender._transfer_start_time
        self.assertIsNotNone(start)

        sender.send(np.arange(4, 8, dtype=np.int32))
        self.assertEqual(sender._transfer_start_time, start)

    def test_failed_transfer_reports_no_latency(self):
        sender = _make_sender(num_kv_indices=4)

        sender.send(np.arange(4, dtype=np.int32))
        sender.kv_mgr.status = KVPoll.Failed
        self.assertEqual(sender.poll(), KVPoll.Failed)
        self.assertEqual(sender.conclude_state, KVPoll.Failed)
        self.assertIsNone(sender.get_transfer_metric().transfer_latency_s)

    def test_metric_combines_latency_and_total_bytes(self):
        # End-to-end: prefill.py consumes both fields to derive speed_gb_s.
        sender = _make_sender(num_kv_indices=6)

        sender.send(np.arange(6, dtype=np.int32))
        sender.kv_mgr.status = KVPoll.Success
        self.assertEqual(sender.poll(), KVPoll.Success)

        metric = sender.get_transfer_metric()
        self.assertEqual(metric.transfer_total_bytes, 6 * KV_ITEM_LENS_SUM)
        self.assertGreater(metric.transfer_latency_s, 0.0)


def _make_manager(room, transfer_info):
    """MooncakeKVManager bypassing __init__, wiring only what transfer_worker reads.

    Records every update_status() transition while delegating to the real
    (monotonic max) implementation.
    """
    mgr = MooncakeKVManager.__new__(MooncakeKVManager)
    mgr.enable_trace = False
    mgr.enable_staging = False
    mgr.is_mla_backend = False
    mgr.is_hybrid_mla_backend = False
    mgr.attn_tp_rank = 0
    mgr.attn_tp_size = 1
    mgr.attn_cp_rank = 0
    mgr.attn_cp_size = 1
    mgr.pp_rank = 0
    mgr.pp_size = 1
    mgr.session_lock = threading.Lock()
    mgr.failure_lock = threading.Lock()
    mgr.failed_sessions = set()
    mgr.session_failures = defaultdict(int)
    mgr.failure_records = {}
    mgr.request_status = {room: KVPoll.WaitingForInput}
    mgr.transfer_infos = {room: {transfer_info.mooncake_session_id: transfer_info}}
    mgr.decode_kv_args_table = {
        transfer_info.mooncake_session_id: SimpleNamespace(dst_aux_ptrs=[])
    }
    mgr.req_to_decode_prefix_len = {}

    transitions = []
    real_update_status = MooncakeKVManager.update_status

    def recording_update_status(bootstrap_room, status):
        transitions.append((bootstrap_room, status))
        real_update_status(mgr, bootstrap_room, status)

    mgr.update_status = recording_update_status
    mgr.send_aux = lambda *args, **kwargs: 0
    mgr.sync_status_to_decode_endpoint = lambda *args, **kwargs: None
    return mgr, transitions


class TestMooncakeTransferringState(CustomTestCase):
    def test_transfer_worker_marks_room_transferring_before_success(self):
        # prefill.py polls the inflight queue and expects Transferring while a
        # chunk is being moved; mooncake previously jumped 2 -> 4 directly.
        room = 91
        transfer_info = TransferInfo(
            room=room,
            endpoint="127.0.0.1",
            dst_port=7777,
            mooncake_session_id="sess-a",
            dst_kv_indices=np.array([], dtype=np.int32),
            dst_aux_index=0,
            dst_state_indices=[],
            required_dst_info_num=1,
            is_dummy=False,
        )
        mgr, transitions = _make_manager(room, transfer_info)

        queue = FastQueue()
        worker = threading.Thread(
            target=mgr.transfer_worker,
            args=(queue, None),
            daemon=True,
        )
        worker.start()
        # Empty-payload last chunk: exercises the full status path without a
        # real TransferEngine (zero-length transfers return 0 immediately).
        queue.put(
            TransferKVChunk(
                room=room,
                prefill_kv_indices=np.array([], dtype=np.int32),
                index_slice=slice(0, 0),
                is_last_chunk=True,
                prefill_aux_index=0,
                state_indices=None,
                trace_ctx=TraceNullContext(),
            )
        )

        deadline = time.perf_counter() + 5.0
        while (
            time.perf_counter() < deadline
            and mgr.request_status.get(room) != KVPoll.Success
        ):
            time.sleep(0.005)

        self.assertEqual(mgr.request_status.get(room), KVPoll.Success)
        self.assertEqual(
            transitions,
            [(room, KVPoll.Transferring), (room, KVPoll.Success)],
        )
        # Terminal cleanup still happens after the Transferring hop.
        self.assertNotIn(room, mgr.transfer_infos)

    def test_sender_poll_passes_through_transferring_without_concluding(self):
        sender = _make_sender(num_kv_indices=4)

        sender.send(np.arange(4, dtype=np.int32))
        sender.kv_mgr.status = KVPoll.Transferring
        self.assertEqual(sender.poll(), KVPoll.Transferring)
        self.assertIsNone(sender.conclude_state)
        self.assertIsNone(sender.get_transfer_metric().transfer_latency_s)

    def test_update_status_never_downgrades_success_to_transferring(self):
        # The worker sets Transferring unconditionally after the Failed check;
        # this is only safe because update_status() is monotonic (max).
        room = 92
        mgr = MooncakeKVManager.__new__(MooncakeKVManager)
        mgr.request_status = {room: KVPoll.Success}
        MooncakeKVManager.update_status(mgr, room, KVPoll.Transferring)
        self.assertEqual(mgr.request_status[room], KVPoll.Success)


if __name__ == "__main__":
    unittest.main()
