import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.managers.overlap_utils import (
    CONFIDENCE_RELAY_RING_DEPTH,
    ConfidenceRelay,
    FutureMap,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeEvent:
    def __init__(self, name=None, log=None):
        self.name = name
        self.log = log
        self.wait_calls = 0
        self.synchronize_calls = 0
        self.record_calls = 0

    def wait(self):
        self.wait_calls += 1

    def synchronize(self):
        self.synchronize_calls += 1

    def record(self):
        self.record_calls += 1
        if self.log is not None:
            self.log.append((self.name, "record"))

    def query(self):
        return True


class _FakeStream:
    def __init__(self, name=None, log=None):
        self.name = name
        self.log = log
        self.waited_events = []

    def wait_event(self, event):
        self.waited_events.append(event)
        if self.log is not None:
            self.log.append((self.name, "wait", event.name))


class _FakeDeviceModule:
    def __init__(self, stream):
        self._current_stream = stream

    def current_stream(self):
        return self._current_stream

    def stream(self, stream):
        return contextlib.nullcontext()

    def Event(self):
        return _FakeEvent()


class _RecordingConfidenceRelay:
    def __init__(self, log):
        self.log = log

    def scatter(self, indices, confidence):
        self.log.append(("relay", "scatter"))

    def wait_for_previous_snapshot(self, stream):
        self.log.append(("relay", "wait_snapshot"))

    def issue_ring_copy(self, *, stream, publish_stream):
        self.log.append(("relay", "snapshot"))


def _future_map(*, needs_cpu_seq_lens, publish_stream):
    future_map = object.__new__(FutureMap)
    future_map.device = torch.device("cpu")
    future_map.needs_cpu_seq_lens = needs_cpu_seq_lens
    future_map.publish_ready = _FakeEvent()
    future_map._publish_stream = publish_stream
    future_map._publish_fresh = True
    future_map.new_seq_lens_buf = torch.tensor([0, 11, 22], dtype=torch.int64)
    future_map.new_seq_lens_cpu_pinned = None
    future_map.fwd_prepare_d2h_stream = None
    return future_map


def _batch():
    return SimpleNamespace(
        spec_info=SimpleNamespace(future_indices=torch.tensor([1, 2])),
        seq_lens=torch.tensor([-1, -1], dtype=torch.int64),
        seq_lens_cpu=torch.tensor([-1, -1], dtype=torch.int64),
        seq_lens_sum=-2,
        req_pool_indices_cpu=torch.tensor([1, 2], dtype=torch.int64),
    )


class TestFutureMapHostSync(unittest.TestCase):
    def test_gpu_only_scheduler_path_does_not_wait_or_gather(self):
        publish_stream = _FakeStream()
        future_map = _future_map(
            needs_cpu_seq_lens=False, publish_stream=publish_stream
        )
        batch = _batch()

        future_map.resolve_seq_lens_cpu(batch)

        self.assertEqual(future_map.publish_ready.wait_calls, 0)
        self.assertEqual(future_map.publish_ready.synchronize_calls, 0)
        self.assertTrue(torch.equal(batch.seq_lens, torch.tensor([-1, -1])))
        self.assertIsNone(batch.seq_lens_cpu)
        self.assertIsNone(batch.seq_lens_sum)

    def test_same_forward_stream_gather_needs_no_event_wait(self):
        forward_stream = _FakeStream()
        future_map = _future_map(
            needs_cpu_seq_lens=False, publish_stream=forward_stream
        )
        batch = _batch()

        with mock.patch(
            "sglang.srt.managers.overlap_utils.torch.get_device_module",
            return_value=_FakeDeviceModule(forward_stream),
        ):
            future_map.resolve_seq_lens_device(batch)

        self.assertEqual(forward_stream.waited_events, [])
        self.assertTrue(torch.equal(batch.seq_lens, torch.tensor([11, 22])))

    def test_off_stream_seed_uses_device_wait_not_host_wait(self):
        publish_stream = _FakeStream()
        forward_stream = _FakeStream()
        future_map = _future_map(
            needs_cpu_seq_lens=False, publish_stream=publish_stream
        )
        batch = _batch()

        with mock.patch(
            "sglang.srt.managers.overlap_utils.torch.get_device_module",
            return_value=_FakeDeviceModule(forward_stream),
        ):
            future_map.resolve_seq_lens_device(batch)

        self.assertEqual(forward_stream.waited_events, [future_map.publish_ready])
        self.assertEqual(future_map.publish_ready.wait_calls, 0)
        self.assertEqual(future_map.publish_ready.synchronize_calls, 0)

    def test_legacy_cpu_consumer_keeps_existing_wait_and_copy(self):
        publish_stream = _FakeStream()
        future_map = _future_map(needs_cpu_seq_lens=True, publish_stream=publish_stream)
        batch = _batch()

        future_map.resolve_seq_lens_cpu(batch)

        total_waits = (
            future_map.publish_ready.wait_calls
            + future_map.publish_ready.synchronize_calls
        )
        self.assertEqual(total_waits, 1)
        self.assertTrue(torch.equal(batch.seq_lens_cpu, torch.tensor([11, 22])))
        self.assertEqual(batch.seq_lens_sum, 33)

    def test_flush_reset_invalidates_publication_and_confidence_ring(self):
        relay = ConfidenceRelay(
            device=torch.device("cpu"),
            req_pool_size=3,
            pool=SimpleNamespace(),
        )
        relay.ring_pos = 7
        relay.gen_ring = torch.tensor([[1, 2, 3]], dtype=torch.int64)
        future_map = _future_map(needs_cpu_seq_lens=False, publish_stream=_FakeStream())
        future_map.confidence_relay = relay

        future_map.reset()

        self.assertIsNone(future_map.publish_ready)
        self.assertIsNone(future_map._publish_stream)
        self.assertFalse(future_map._publish_fresh)
        self.assertEqual(relay.ring_pos, 0)
        self.assertTrue(torch.equal(relay.gen_ring, torch.full((1, 3), -1)))

    def test_publish_waits_before_scatter_and_records_after_snapshot(self):
        log = []
        previous_stream = _FakeStream("previous", log)
        publish_stream = _FakeStream("publish", log)
        publish_ready = _FakeEvent("publish_ready", log)
        future_map = _future_map(
            needs_cpu_seq_lens=False, publish_stream=previous_stream
        )
        future_map.publish_ready = publish_ready
        future_map.spec_algo = SimpleNamespace(is_some=lambda: True)
        future_map.needs_confidence_relay = True
        future_map.confidence_relay = _RecordingConfidenceRelay(log)
        future_map.fwd_prepare_d2h_stream = _FakeStream("d2h", log)

        with mock.patch(
            "sglang.srt.managers.overlap_utils.torch.get_device_module",
            return_value=_FakeDeviceModule(publish_stream),
        ):
            future_map.publish(
                torch.tensor([1, 2]),
                torch.tensor([13, 24]),
                torch.ones((2, 2)),
            )

        self.assertEqual(
            log,
            [
                ("publish", "wait", "publish_ready"),
                ("relay", "wait_snapshot"),
                ("relay", "scatter"),
                ("relay", "snapshot"),
                ("publish_ready", "record"),
            ],
        )
        self.assertEqual(publish_ready.wait_calls, 0)
        self.assertEqual(publish_ready.synchronize_calls, 0)

    def test_confidence_snapshot_and_slot_reuse_use_device_events(self):
        depth = CONFIDENCE_RELAY_RING_DEPTH
        log = []
        publish_stream = _FakeStream("publish", log)
        d2h_stream = _FakeStream("d2h", log)
        snapshot_ready = [_FakeEvent(f"snapshot-{slot}", log) for slot in range(depth)]
        copy_done = [_FakeEvent(f"copy-{slot}", log) for slot in range(depth)]
        relay = ConfidenceRelay(
            device=torch.device("cpu"),
            req_pool_size=2,
            pool=SimpleNamespace(req_generation=torch.tensor([4, 5])),
            confidence_buf=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            confidence_snapshot_ring=torch.empty((depth, 2, 2)),
            conf_ring=torch.empty((depth, 2, 2)),
            gen_ring=torch.zeros((depth, 2), dtype=torch.int64),
            snapshot_ready=snapshot_ready,
            copy_done=copy_done,
            copy_issued=[False] * depth,
            initialized=True,
        )

        with mock.patch(
            "sglang.srt.managers.overlap_utils.torch.get_device_module",
            return_value=_FakeDeviceModule(publish_stream),
        ):
            relay.issue_ring_copy(stream=d2h_stream, publish_stream=publish_stream)
            first_snapshot = relay.confidence_snapshot_ring[0].clone()
            first_host_copy = relay.conf_ring[0].clone()

            # Flush rewinds the logical ring but must preserve its physical
            # in-flight-copy fence before slot zero is overwritten.
            relay.reset()
            self.assertEqual(relay.last_snapshot_slot, 0)
            relay.wait_for_previous_snapshot(publish_stream)
            relay.confidence_buf.fill_(9)
            relay.issue_ring_copy(stream=d2h_stream, publish_stream=publish_stream)

        self.assertTrue(
            torch.equal(first_snapshot, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        )
        self.assertTrue(
            torch.equal(first_host_copy, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        )
        self.assertTrue(torch.equal(relay.conf_ring[0], torch.full((2, 2), 9.0)))
        self.assertEqual(
            log,
            [
                ("snapshot-0", "record"),
                ("d2h", "wait", "snapshot-0"),
                ("copy-0", "record"),
                ("publish", "wait", "snapshot-0"),
                ("publish", "wait", "copy-0"),
                ("snapshot-0", "record"),
                ("d2h", "wait", "snapshot-0"),
                ("copy-0", "record"),
            ],
        )
        for event in snapshot_ready + copy_done:
            self.assertEqual(event.wait_calls, 0)
            self.assertEqual(event.synchronize_calls, 0)



if __name__ == "__main__":
    unittest.main()
