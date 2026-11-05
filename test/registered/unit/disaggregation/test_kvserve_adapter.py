"""CPU coverage for KVServe transfer completion and buffer ownership."""

import contextlib
import queue
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import msgspec

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVSender
from sglang.srt.disaggregation.mooncake.kvserve import ACK, KVServeAdapter
from sglang.srt.disaggregation.mooncake.kvserve_pipeline import (
    EncodedTransferPipeline,
    RestorePipeline,
    SourceRoomGate,
)
from sglang.srt.disaggregation.mooncake.kvserve_transport import ReceiveCredits
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
