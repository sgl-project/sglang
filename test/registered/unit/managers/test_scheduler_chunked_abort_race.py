"""Tests for deferred chunked-prefill aborts."""

import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    enter_scope,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

import threading  # noqa: E402

from sglang.srt.disaggregation.base.conn import KVPoll  # noqa: E402
from sglang.srt.disaggregation.common.conn import (  # noqa: E402
    AckTarget,
    CommonKVManager,
    CommonKVSender,
)
from sglang.srt.disaggregation.mode import DisaggregationMode  # noqa: E402
from sglang.srt.managers.req_types import FINISH_ABORT  # noqa: E402
from sglang.srt.managers.schedule_batch import Req  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.scheduler_components import (  # noqa: E402
    request_retirer as retirer_mod,
)
from sglang.test.scheduler_test_utils import FakeReq, empty_request_lists  # noqa: E402

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class _FakeReq(FakeReq):
    """Minimal stand-in for Req: only the fields the abort paths touch."""

    request_finish = Req.request_finish
    clear_requested_finish = Req.clear_requested_finish
    finish_outside_step = Req.finish_outside_step

    def __init__(self, rid: str):
        self.rid = rid
        # Mirrors Req.kv; the abort paths read only these fields.
        self.kv = SimpleNamespace(
            holds_kv=True, holds_mamba=False, retraction_backup=None
        )
        self.to_finish = None
        self._finished = False
        self.discard_output_reason = None
        self.return_logprob = False
        self.output_ids = []
        self.weight_version_events = []
        self.prefill_weight_versions = None
        self.time_stats = Mock()
        self.cache_request_handle = None

    def finished(self):
        return self._finished


def _make_scheduler(pending_req, *, chunked_req, running_reqs) -> Scheduler:
    sched = Scheduler.__new__(Scheduler)
    empty_request_lists(sched)
    sched.enable_continuous_input_polling = False
    sched.result_queue = deque()
    if chunked_req is not None:
        sched.chunked_reqs.join([chunked_req])
    sched._pending_chunked_abort_req = pending_req
    sched.disaggregation_mode = None
    sched.enable_hicache_storage = False
    sched.mm_receiver = None
    sched.rust_server = None
    sched.running_batch = SimpleNamespace(reqs=running_reqs)
    sched.last_batch = None
    sched.beam_coordinator = Mock()
    sched.output_streamer = Mock()
    sched.enable_hisparse = False
    sched.tree_cache = Mock()
    sched.hisparse_coordinator = None
    sched.uocr_runtime = None
    sched.ipc_channels = Mock()
    sched.init_request_retirer()
    return sched


class TestPendingChunkedAbortRace(CustomTestCase):
    def setUp(self):
        enter_scope(self, published_topology())

    def test_req_left_chunked_slot_is_aborted(self):
        req = _FakeReq("zombie_rid")
        sched = _make_scheduler(req, chunked_req=None, running_reqs=[req])

        sched.process_pending_chunked_abort()

        self.assertIsNotNone(req.to_finish, "recorded abort was never applied")
        self.assertIsNone(sched._pending_chunked_abort_req)

    def test_discarded_chunk_keeps_its_abort_reason(self):
        req = _FakeReq("failed_load")
        req.discard_output_reason = FINISH_ABORT("External KV cache load failed", 503)
        sched = _make_scheduler(req, chunked_req=req, running_reqs=[])

        with (
            patch.object(sched.request_retirer, "_detach"),
            patch.object(retirer_mod, "release_kv_cache"),
        ):
            sched.process_pending_chunked_abort()

        self.assertEqual(req.finished_reason.status_code, 503)
        (abort_req, _), _ = sched.ipc_channels.send_to_tokenizer.send_output.call_args
        self.assertEqual(abort_req.finished_reason, req.discard_output_reason.to_json())

    def test_aborted_chunked_beam_leader_retires_its_group(self):
        req = _FakeReq("beam_leader")
        sched = _make_scheduler(req, chunked_req=req, running_reqs=[])

        with (
            patch.object(sched.request_retirer, "_detach"),
            patch.object(retirer_mod, "release_kv_cache"),
        ):
            sched.process_pending_chunked_abort()

        sched.beam_coordinator.retire_group.assert_called_once_with(req)

    def test_aborted_chunked_req_releases_hisparse_before_kv(self):
        # Each chunk's extend allocates HiSparse spec scratch for the row.
        req = _FakeReq("hisparse_chunked")
        sched = _make_scheduler(req, chunked_req=req, running_reqs=[])
        calls = []
        hisparse_coordinator = Mock()
        hisparse_coordinator.release_request.side_effect = lambda req: calls.append(
            ("hisparse", req)
        )
        sched.request_retirer.hisparse_coordinator = hisparse_coordinator

        with (
            patch.object(sched.request_retirer, "_detach"),
            patch.object(
                retirer_mod,
                "release_kv_cache",
                side_effect=lambda req, *a, **kw: calls.append(("kv", req)),
            ),
        ):
            sched.process_pending_chunked_abort()

        self.assertEqual(calls, [("hisparse", req), ("kv", req)])


class _SendBookkeeping(CommonKVManager):
    """A connection's room status and send bookkeeping, without sockets."""

    def __init__(self):
        self.request_status = {}
        self.deferred_bootstrap = None
        self._deferred_ack_targets = {}
        self._deferred_ack_poisoned_rooms = set()
        self._staging_outstanding = {}
        self.failure_lock = threading.Lock()
        self.failure_records = {}
        self.req_to_decode_prefix_len = {}
        self.transfer_infos = {}
        self._staging_ctx = None


class TestPrefillAbortRetiresSendRecord(CustomTestCase):
    """REGRESSION: abort paths that never clear the sender still drop the
    room's send record after its first chunk was sent."""

    ROOM = 42

    def setUp(self):
        enter_scope(self, published_topology())

    def _sender_after_first_chunk(self):
        manager = _SendBookkeeping()
        manager.update_status(self.ROOM, KVPoll.Bootstrapping)
        manager.update_status(self.ROOM, KVPoll.Transferring)
        sender = CommonKVSender.__new__(CommonKVSender)
        sender.kv_mgr = manager
        sender.bootstrap_room = self.ROOM
        sender.conclude_state = None
        sender._owns_bootstrap_room = True
        sender.trace_ctx = Mock()
        # The first chunk, queued for the transfer worker and finished by it.
        record = manager.send_record(self.ROOM)
        manager.note_send_enqueued(self.ROOM)
        manager.note_send_finished(self.ROOM, record)
        self.assertEqual(manager.send_counts(self.ROOM), (1, 1))
        # Decode aborted too and waits for this rank's drain acknowledgement,
        # owed once a chunk still in flight drains: the abort leaves it owed.
        manager._staging_outstanding[self.ROOM] = 1
        manager.register_deferred_ack_target(
            self.ROOM, AckTarget("10.0.0.2", 7000, generation=1)
        )
        return manager, sender

    def _req(self, rid, sender):
        req = _FakeReq(rid)
        req.disagg_kv_sender = sender
        req.metadata_buffer_index = -1
        req.pending_bootstrap = False
        req.cache_request_handle = Mock()
        req.finished_reason = None
        return req

    def _assert_retired(self, manager, sender):
        # The abort concluded the room failed and cleared it; the drain
        # acknowledgement stays owed while a chunk is still in flight.
        self.assertEqual(manager._send_records, {})
        self.assertEqual(sender.conclude_state, KVPoll.Failed)
        self.assertNotIn(self.ROOM, manager.request_status)
        self.assertIn(self.ROOM, manager._deferred_ack_targets)

    def test_pending_chunked_abort(self):
        manager, sender = self._sender_after_first_chunk()
        req = self._req("chunked", sender)
        sched = _make_scheduler(req, chunked_req=req, running_reqs=[])
        sched.disaggregation_mode = DisaggregationMode.PREFILL
        sched.req_to_metadata_buffer_idx_allocator = Mock()

        with (
            patch.object(sched.request_retirer, "_detach"),
            patch.object(retirer_mod, "release_kv_cache"),
        ):
            sched.process_pending_chunked_abort()

        self._assert_retired(manager, sender)

    def test_aborted_prefill_result(self):
        manager, sender = self._sender_after_first_chunk()
        req = self._req("aborted-result", sender)
        sched = _make_scheduler(None, chunked_req=None, running_reqs=[])
        sched.req_to_metadata_buffer_idx_allocator = Mock()

        with (
            patch.object(sched.request_retirer, "_detach"),
            patch.object(retirer_mod, "release_kv_cache"),
        ):
            self.assertTrue(sched._retire_aborted_prefill_result(req))

        self._assert_retired(manager, sender)


if __name__ == "__main__":
    unittest.main(verbosity=2)
