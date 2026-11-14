"""CPU coverage for KVServe transfer completion and buffer ownership."""

import contextlib
import queue
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import msgspec

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.decode import DecodeTransferQueue
from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.disaggregation.mooncake.conn import (
    KVArgsRegisterInfo,
    MooncakeKVManager,
    MooncakeKVSender,
)
from sglang.srt.disaggregation.mooncake.kvserve import ACK, READY, KVServeAdapter
from sglang.srt.disaggregation.mooncake.kvserve_optimized import OptimizedKVServeAdapter
from sglang.srt.disaggregation.mooncake.kvserve_pipeline import (
    EncodedTransferPipeline,
    RestorePipeline,
    SourceRoomGate,
)
from sglang.srt.disaggregation.mooncake.kvserve_transport import ReceiveCredits
from sglang.srt.disaggregation.prefill import (
    SchedulerDisaggregationPrefillMixin,
    _send_done,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    ListKind,
    move,
)
from sglang.srt.managers.scheduler_components.request_retirer import RequestRetirer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class TestAdapterProtocol(unittest.TestCase):
    def setUp(self):
        self.adapter = object.__new__(KVServeAdapter)
        self.adapter.manager = SimpleNamespace(record_failure=Mock())
        self.adapter.lock = threading.Lock()
        self.adapter.submit_lock = threading.Lock()
        self.adapter.page_size = 1
        self.adapter.has_mamba = False
        self.adapter.expected_rooms = {7: {3, 5}}
        self.adapter.restored_rooms = {7: {0: set(), 1: set()}}

    def test_success_requires_both_parts_of_every_registered_page(self):
        for restored, expected in (
            ({0: set(), 1: set()}, KVPoll.Failed),
            ({0: {3, 5}, 1: set()}, KVPoll.Failed),
            ({0: {3, 5}, 1: {3, 6}}, KVPoll.Failed),
            ({0: {3, 5}, 1: {3, 5}}, KVPoll.Success),
        ):
            with self.subTest(restored=restored):
                self.adapter.restored_rooms[7] = restored
                self.assertEqual(
                    self.adapter.check_success(7, KVPoll.Success), expected
                )

    def test_peer_profile_and_capacities_must_match(self):
        self.adapter.runtime = SimpleNamespace(wire_profile="profile")
        slot = {"ptrs": [4096, 8192], "capacities": [4096, 128]}
        self.adapter.registration = lambda: {"slots": [slot, slot]}
        peer = {
            "version": 3,
            "has_mamba": False,
            "page_size": 1,
            "profile": "profile",
            "slots": [dict(slot) for _ in range(8)],
        }
        self.adapter.validate_peer(peer)
        for other in (
            None,
            {**peer, "profile": "other"},
            {**peer, "page_size": 16},
            {**peer, "slots": [{**slot, "capacities": [4096, 127]}]},
        ):
            with self.subTest(peer=other), self.assertRaises(ValueError):
                self.adapter.validate_peer(other)

    def test_sender_selects_matching_advertised_buffers(self):
        def runtime(profile, ptr):
            slot = SimpleNamespace(
                arena_a=SimpleNamespace(data_ptr=lambda: ptr, numel=lambda: 4096),
                aux=SimpleNamespace(data_ptr=lambda: ptr + 4096, numel=lambda: 128),
            )
            return SimpleNamespace(wire_profile=profile, slots=[slot, slot])

        fused, standard = runtime("fused", 4096), runtime("standard", 16384)
        self.adapter.runtime, self.adapter.runtimes = fused, (fused, standard)
        peer = self.adapter.registration()
        self.assertEqual(peer["profile"], "fused")
        self.assertEqual(peer["alternatives"][0]["profile"], "standard")
        self.adapter.runtime, self.adapter.runtimes = standard, (standard,)
        selected = self.adapter.validate_peer(peer)
        self.assertEqual(selected["slots"][0]["ptrs"], [16384, 20480])
        for invalid in (
            {**peer, "alternatives": []},
            {**peer, "alternatives": [selected, selected]},
            {**peer, "alternatives": [{**selected, "page_size": 16}]},
        ):
            with self.subTest(peer=invalid), self.assertRaises(ValueError):
                self.adapter.validate_peer(invalid)

    def test_ready_selects_one_advertised_profile_and_never_switches(self):
        adapter = object.__new__(OptimizedKVServeAdapter)
        fused = SimpleNamespace(wire_profile="fused")
        standard = SimpleNamespace(wire_profile="standard")
        adapter.runtime, adapter.runtimes = fused, (fused, standard)
        adapter._receive_profile = None
        adapter._closing = False
        adapter.submit_lock = threading.Lock()
        adapter._receive_pipeline_v2 = SimpleNamespace(
            accept=Mock(side_effect=lambda metas, before_enqueue: before_enqueue(metas))
        )

        def send(metas):
            return adapter.handle_ready([READY, msgspec.msgpack.encode(metas)])

        for invalid in (
            [],
            [{"profile": []}],
            [{"profile": "unknown"}],
            [{"profile": "fused"}, {"profile": "standard"}],
        ):
            with self.subTest(metas=invalid), self.assertRaises(ValueError):
                send(invalid)
        self.assertIsNone(adapter._receive_profile)
        self.assertTrue(send([{"profile": "standard"}]))
        self.assertIs(adapter.runtime, standard)
        self.assertTrue(send([{"profile": "standard"}]))
        with self.assertRaises(ValueError):
            send([{"profile": "fused"}])
        self.assertEqual(adapter._receive_pipeline_v2.accept.call_count, 7)

    def test_close_releases_all_advertised_runtimes(self):
        fused = SimpleNamespace(shutdown=Mock())
        standard = SimpleNamespace(shutdown=Mock())
        self.adapter.runtime, self.adapter.runtimes = standard, (fused, standard)
        self.adapter.poisoned = False
        self.adapter.pending_lock = threading.Lock()
        self.adapter.pending = {}
        self.adapter.expected_rooms = {}
        self.adapter.registered_ptrs = [4096, 8192]
        self.adapter.notifier = SimpleNamespace(close=Mock())
        self.adapter.manager.engine = SimpleNamespace(batch_deregister=Mock())
        self.adapter.close()
        self.adapter.manager.engine.batch_deregister.assert_called_once_with(
            [4096, 8192]
        )
        fused.shutdown.assert_called_once()
        standard.shutdown.assert_called_once()

    def test_duplicate_and_stale_ack_never_release_another_credit(self):
        self.adapter.epoch = "current"
        self.adapter.pending_lock = threading.Lock()
        event = threading.Event()
        pool = ReceiveCredits(1)
        slot = pool.acquire()[0]
        self.adapter.pending = {"1": (event, {}, pool, slot)}
        for value in (
            {"epoch": "old", "sequence": "1"},
            {"epoch": "current", "sequence": "other"},
        ):
            self.adapter.handle_ack([ACK, msgspec.msgpack.encode([value])])
        self.assertFalse(event.is_set())
        self.assertEqual(pool.available, [])
        ack = [ACK, msgspec.msgpack.encode([{"epoch": "current", "sequence": "1"}])]
        self.adapter.handle_ack(ack)
        self.adapter.handle_ack(ack)
        self.assertTrue(event.is_set())
        self.assertEqual(pool.available, [0])

    def test_unsettled_transfer_cannot_free_registered_memory(self):
        self.adapter.poisoned = True
        self.adapter.runtime = SimpleNamespace(shutdown=Mock())
        with self.assertRaises(RuntimeError):
            self.adapter.close()
        self.adapter.runtime.shutdown.assert_not_called()

    def test_paged_indices_exclude_the_invalid_tail(self):
        self.assertEqual(KVServeAdapter.token_ids([3, 7], 4, 5), [12, 13, 14, 15, 28])
        with self.assertRaises(ValueError):
            KVServeAdapter.token_ids([3, 7], 4, 4)

    def test_failed_sender_waits_until_its_source_reader_drains(self):
        sender = object.__new__(MooncakeKVSender)
        sender.bootstrap_room = 7
        sender.conclude_state = None
        sender._kvserve_source_gate = SourceRoomGate()
        sender.kv_mgr = SimpleNamespace(
            kvserve_enabled=True, kvserve=None, request_status={7: KVPoll.Transferring}
        )
        sender.trace_ctx = SimpleNamespace(trace_req_finish=Mock())

        def fail_after_enqueue(_):
            self.assertTrue(sender._kvserve_source_gate.acquire())
            return KVPoll.Failed

        self.assertEqual(
            sender._poll_with_status(fail_after_enqueue), KVPoll.Transferring
        )
        self.assertFalse(sender._kvserve_source_gate.acquire())
        sender._kvserve_source_gate.release()
        self.assertEqual(sender._poll_with_status(fail_after_enqueue), KVPoll.Failed)

    def test_registration_preserves_native_dcp_and_optional_codec_fields(self):
        manager = object.__new__(MooncakeKVManager)
        manager.get_session_id = lambda: "session"
        manager.local_ip, manager.rank_port = "127.0.0.1", 1234
        manager.attn_tp_size, manager.dcp_size, manager.dcp_rank = 1, 1, 0
        manager.enable_staging = False
        manager.exchange_aux_item_lens = False
        manager.kv_args = SimpleNamespace(
            kv_data_ptrs=[4096],
            aux_data_ptrs=[8192],
            state_data_ptrs=[],
            state_item_lens=[],
            state_dim_per_tensor=[],
            state_layer_ids=[],
            kv_layer_ids=[0],
            engine_rank=0,
            kv_item_lens=[64],
            draft_rows_dcp_collapsed=True,
        )
        for profile in (None, {"version": 3, "profile": "default"}):
            with self.subTest(profile=profile):
                manager.kvserve = (
                    None
                    if profile is None
                    else SimpleNamespace(registration=lambda: profile)
                )
                msg = manager.build_receiver_registration_message()
                self.assertEqual(msg[21], b"1")
                parsed = KVArgsRegisterInfo.from_zmq(msg)
                self.assertTrue(parsed.dst_draft_rows_dcp_collapsed)
                self.assertEqual(parsed.kvserve, profile)
                legacy = KVArgsRegisterInfo.from_zmq(msg[:21])
                self.assertFalse(legacy.dst_draft_rows_dcp_collapsed)
                self.assertIsNone(legacy.kvserve)

    def test_enqueue_records_native_send_and_owns_source_until_done(self):
        manager = object.__new__(MooncakeKVManager)
        manager.disaggregation_mode = DisaggregationMode.PREFILL
        manager.request_status = {7: KVPoll.Transferring}
        manager.transfer_infos = {7: {"127.0.0.1:1234": object()}}
        manager.transfer_queues = [queue.Queue()]
        manager.kvserve_enabled = True
        manager._send_records = None
        gate = SourceRoomGate()
        manager.add_transfer_request(7, [3], slice(0, 1), False, source_gate=gate)
        chunk = manager.transfer_queues[0].get_nowait()
        self.assertEqual(manager.send_counts(7), (1, 0))
        self.assertIs(chunk.send_record, manager.send_record(7))
        self.assertFalse(gate.close_and_is_idle())
        chunk._kvserve_source_gate.release()
        self.assertTrue(gate.close_and_is_idle())
        manager.note_send_finished(7, chunk.send_record)
        self.assertEqual(manager.send_counts(7), (1, 1))


class TestRequestOwnership(unittest.TestCase):
    def test_failed_send_does_not_allow_page_move_before_codec_drains(self):
        gate = SourceRoomGate()
        self.assertTrue(gate.acquire())
        sender = SimpleNamespace(
            bootstrap_room=7,
            kv_mgr=SimpleNamespace(
                kvserve_enabled=True, request_status={7: KVPoll.Failed}
            ),
            is_source_release_safe=gate.close_and_is_idle,
        )
        self.assertFalse(_send_done((sender, 0)))
        gate.release()
        self.assertTrue(_send_done((sender, 0)))

    def test_waiting_abort_sends_output_once_and_releases_after_drain(self):
        class Request:
            rid = "pending"
            owner_list = None
            pending_bootstrap = False
            finished_reason = None
            metadata_buffer_index = -1
            return_logprob = False
            time_stats = SimpleNamespace(set_completion_time=Mock())

            def finish_outside_step(self, reason):
                self.finished_reason = reason

        req = Request()
        gate = SourceRoomGate()
        self.assertTrue(gate.acquire())
        req.disagg_kv_sender = SimpleNamespace(
            kv_mgr=SimpleNamespace(kvserve_enabled=True),
            abort=Mock(),
            clear=Mock(),
            is_source_release_safe=gate.close_and_is_idle,
        )
        scheduler = SchedulerDisaggregationPrefillMixin()
        scheduler.disaggregation_mode = DisaggregationMode.PREFILL
        scheduler.waiting_queue = IndexedReqList(ListKind.WAITING)
        scheduler.disagg_prefill_transfer_queue = IndexedReqList(ListKind.PD_SENDING)
        scheduler.req_to_metadata_buffer_idx_allocator = Mock()
        scheduler.scheduler_stage_metrics = None
        scheduler.attn_cp_cpu_group = scheduler.attn_tp_cpu_group = None
        scheduler.output_streamer = SimpleNamespace(stream_output=Mock())
        scheduler.request_retirer = RequestRetirer(
            tree_cache=Mock(),
            beam_coordinator=Mock(),
            hisparse_coordinator=None,
            uocr_runtime=None,
            send_abort_output=Mock(),
            stream_output=Mock(),
        )
        scheduler.request_retirer.release = Mock(
            side_effect=lambda item, **_: item.owner_list._leave([item])
        )
        scheduler.handle_inflight_transfer_failure = Mock(
            side_effect=lambda item: scheduler.request_retirer.release(
                item, checkpoint=False
            )
        )
        move([req], scheduler.waiting_queue)
        abort = SimpleNamespace(
            matches=lambda rid: rid == req.rid, finished_reason=None
        )
        with patch(
            "sglang.srt.managers.scheduler_components.request_retirer.make_abort_req",
            return_value=object(),
        ):
            Scheduler._abort_waiting(scheduler, abort)
        scheduler.request_retirer.send_abort_output.assert_called_once()
        scheduler.request_retirer.release.assert_not_called()
        req.disagg_kv_sender.clear.assert_not_called()
        self.assertFalse(scheduler.waiting_queue)
        self.assertTrue(req._kvserve_abort_output_sent)
        self.assertIs(req.owner_list, scheduler.disagg_prefill_transfer_queue)

        gate.release()
        self.assertTrue(req.disagg_kv_sender.is_source_release_safe())
        with patch(
            "sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group",
            return_value=[KVPoll.Failed],
        ):
            scheduler.process_disagg_prefill_transfer_queue()
        scheduler.request_retirer.release.assert_called_once_with(req, checkpoint=False)
        scheduler.output_streamer.stream_output.assert_called_once_with([], False, None)
        self.assertIsNone(req.owner_list)

    def test_page_move_and_release_wait_for_local_restore(self):
        transfer = object.__new__(DecodeTransferQueue)
        transfer.scheduler = SimpleNamespace(request_retirer=Mock())
        receiver = SimpleNamespace(
            is_local_restore_release_safe=Mock(return_value=False),
            poll=Mock(return_value=KVPoll.Success),
        )
        req = SimpleNamespace(kv_receiver=receiver)
        self.assertFalse(transfer._receive_done(req))
        with self.assertRaisesRegex(RuntimeError, "unsettled local restore"):
            transfer._do_release(req, 0)
        transfer.scheduler.request_retirer.release.assert_not_called()
        receiver.is_local_restore_release_safe.return_value = True
        transfer.enable_staging = False
        self.assertTrue(transfer._receive_done(req))


class TestTransferBuffers(unittest.TestCase):
    def test_output_lease_is_reused_only_after_transfer_finishes(self):
        started, finish = threading.Event(), threading.Event()

        def write(_lease, _job):
            started.set()
            if not finish.wait(3):
                raise TimeoutError("transfer did not finish")
            return 42

        pipeline = EncodedTransferPipeline(
            [(object(), object())], write, Mock(), workers=1
        )
        self.addCleanup(pipeline.close)
        self.addCleanup(finish.set)
        lease = pipeline.acquire()
        future = pipeline.submit(lease, {})
        self.assertTrue(started.wait(2))
        self.assertEqual(pipeline.available, [])
        finish.set()
        self.assertEqual(future.result(2), 42)
        self.assertIs(pipeline.acquire(), lease)
        pipeline.discard_unsubmitted(lease)

    def test_failed_transfer_quarantines_its_output_lease(self):
        def fail(_lease, _job):
            raise RuntimeError("transfer failed")

        failed = Mock()
        pipeline = EncodedTransferPipeline(
            [(object(), object())], fail, failed, workers=1
        )
        self.addCleanup(pipeline.close)
        lease = pipeline.acquire()
        with self.assertRaisesRegex(RuntimeError, "transfer failed"):
            pipeline.submit(lease, {}).result(2)
        self.assertEqual(lease.state, "quarantined")
        self.assertEqual(pipeline.available, [])
        failed.assert_called_once()


class FakeReceiveAdapter(KVServeAdapter):
    def __init__(self, failed_completion=False):
        self.lock = threading.Lock()
        self.slot_locks = [threading.Lock(), threading.Lock()]
        self.device, self.poisoned, self.verify = "fake", False, False
        self.chunk_size, self.cache_blocks, self.page_size = 1024, 100, 1
        self.expected_rooms = {7: {10}}
        self.restored_rooms = {7: {0: set(), 1: set()}}
        self.completions = [threading.Event(), threading.Event()]
        self.submitted, self.acks = queue.Queue(), queue.Queue()
        self.failed_completion = failed_completion
        self.runtime = SimpleNamespace(
            slots=[
                SimpleNamespace(arena_a=SimpleNamespace(numel=lambda: 1024))
                for _ in range(2)
            ],
            wire_profile="profile",
            aux_bytes=lambda *_: 32,
            decode=self.decode,
        )
        self.manager = SimpleNamespace(
            request_status={7: KVPoll.Transferring},
            check_status=lambda _: KVPoll.Transferring,
            record_failure=Mock(),
            update_status=Mock(),
        )

    def decode(self, slot, *_args, **_kwargs):
        self.submitted.put(slot)

        def synchronize():
            if self.failed_completion:
                raise RuntimeError("restore completion failed")
            if not self.completions[slot].wait(3):
                raise TimeoutError("restore did not finish")

        return SimpleNamespace(synchronize=synchronize)

    def reply(self, meta, result):
        self.acks.put((meta, result.copy()))

    def record(self, _value):
        pass

    def control_failed(self, _error):
        self.poisoned = True


def ready(slot, part):
    return {
        "slot_id": slot,
        "room": 7,
        "epoch": "epoch",
        "sequence": str(slot),
        "version": 3,
        "profile": "profile",
        "indices": [10],
        "tokens": 1,
        "part": part,
        "aux_bytes": 32,
        "codec_bytes": 128,
        "u8_bytes": 256,
    }


class TestRestoreCompletion(unittest.TestCase):
    def make_pipeline(self, adapter):
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(device=lambda _: contextlib.nullcontext())
        )
        return RestorePipeline(adapter, fake_torch, KVPoll.Failed)

    def test_profile_selection_runs_only_after_ready_validation(self):
        adapter = FakeReceiveAdapter()
        pipeline = self.make_pipeline(adapter)
        self.addCleanup(pipeline.close)
        self.addCleanup(lambda: [event.set() for event in adapter.completions])
        select = Mock()
        with self.assertRaises(ValueError):
            pipeline.accept([{**ready(0, 0), "slot_id": 10}], before_enqueue=select)
        select.assert_not_called()
        frames = [ready(0, 0)]
        pipeline.accept(frames, before_enqueue=select)
        select.assert_called_once_with(frames)

    def test_ack_and_page_accounting_wait_for_each_restore(self):
        adapter = FakeReceiveAdapter()
        pipeline = self.make_pipeline(adapter)
        self.addCleanup(pipeline.close)
        self.addCleanup(lambda: [event.set() for event in adapter.completions])
        pipeline.accept([ready(0, 0), ready(1, 1)])
        self.assertEqual(
            {adapter.submitted.get(timeout=2), adapter.submitted.get(timeout=2)}, {0, 1}
        )
        self.assertTrue(adapter.acks.empty())
        with self.assertRaises(ValueError):
            pipeline.accept([ready(0, 0)])
        adapter.completions[1].set()
        meta, result = adapter.acks.get(timeout=2)
        self.assertEqual(meta["part"], 1)
        self.assertNotIn("error", result)
        self.assertEqual(adapter.restored_rooms[7], {0: set(), 1: {10}})
        adapter.completions[0].set()
        self.assertEqual(adapter.acks.get(timeout=2)[0]["part"], 0)
        self.assertEqual(adapter.restored_rooms[7], {0: {10}, 1: {10}})

    def test_failed_restore_wait_keeps_destination_quarantined(self):
        adapter = FakeReceiveAdapter(failed_completion=True)
        pipeline = self.make_pipeline(adapter)
        pipeline.accept([ready(0, 0)])
        _, result = adapter.acks.get(timeout=2)
        self.assertIn("restore completion failed", result["error"])
        self.assertFalse(pipeline.room_is_idle(7))
        self.assertTrue(adapter.poisoned)
        with self.assertRaisesRegex(RuntimeError, "Unsettled"):
            pipeline.close()


if __name__ == "__main__":
    unittest.main()
