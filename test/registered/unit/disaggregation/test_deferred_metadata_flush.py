"""Unit tests for the deferred transfer-metadata readback on the decode side.

``pop_preallocated`` used to read the destination page/state indices back with
``.cpu()`` and send the transfer metadata inline. That readback sits on the
scheduler stream, which the WAR barrier fences behind the in-flight forward, so
it stalled the very path that has to launch the next forward. The indices are now
registered as device tensors and read back by ``flush_pending_metadata()`` after
the launch, on a private stream gated on an event.

Covered here:
  - ``_page_indices_device`` produces what ``kv_to_page_indices`` produced, on
    the device and in the int32 wire dtype;
  - ``flush_pending_metadata`` sends one metadata message per pending entry, with
    the right buffer index / prefix length / state indices, drains the queue, and
    is a no-op when empty; rebootstrap requests also get their recompute submit;
  - ``_ensure_metadata_stream`` never hands back a stream that aliases the
    forward or schedule stream (that would put the readback back behind the WAR
    barrier), and stays inert without a device module;
  - the pinned staging helpers keep their CPU fast path.

    python -m pytest test/registered/unit/disaggregation/test_deferred_metadata_flush.py -v
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    _page_indices_device,
    _PendingMetadataSend,
    _pinned_int64_pair,
    _pinned_to_device_int64,
)
from sglang.srt.mem_cache.allocator.base import pinned_int64_pair
from sglang.srt.mem_cache.common import kv_to_page_indices
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=40, suite="base-a-test-cpu")


class _FakeReceiver:
    def __init__(self):
        self.sent = []

    def send_metadata(self, page_indices, buffer_index, state_indices, **kwargs):
        self.sent.append((page_indices, buffer_index, state_indices, kwargs))


def _make_decode_req(*, buffer_index, is_rebootstrap=False, payload="payload"):
    return SimpleNamespace(
        kv_receiver=_FakeReceiver(),
        metadata_buffer_index=buffer_index,
        is_rebootstrap=is_rebootstrap,
        req=SimpleNamespace(build_rebootstrap_payload=lambda: payload),
    )


def _make_queue(pending):
    """A prealloc queue carrying only what flush_pending_metadata reads."""
    queue = object.__new__(DecodePreallocQueue)
    queue._pending_metadata = list(pending)
    queue._metadata_stream = None
    queue._prealloc_done = None
    queue.scheduler = SimpleNamespace(device_module=None)
    queue.kv_manager = SimpleNamespace(
        submitted=[],
        submit_prefill_recompute=lambda receiver, payload: queue.kv_manager.submitted.append(
            (receiver, payload)
        ),
    )
    return queue


class _FakeStream:
    """Enough of a device stream for the aliasing check: an identity handle."""

    _next = 100

    def __init__(self, priority=0):
        _FakeStream._next += 1
        self.cuda_stream = _FakeStream._next
        self.priority = priority


class TestPageIndicesDevice(CustomTestCase):
    def test_matches_kv_to_page_indices(self):
        for page_size in (1, 8, 64):
            kv_indices = torch.arange(page_size * 5, dtype=torch.int64)

            device_pages = _page_indices_device(kv_indices, page_size)

            self.assertEqual(torch.int32, device_pages.dtype)
            self.assertEqual(
                kv_to_page_indices(kv_indices, page_size).tolist(),
                device_pages.tolist(),
            )

    def test_empty_input(self):
        self.assertEqual(
            0, _page_indices_device(torch.empty(0, dtype=torch.int64), 64).numel()
        )


class TestPinnedStagingHelpers(CustomTestCase):
    def test_pair_returns_the_same_tensor_on_cpu(self):
        for fn in (pinned_int64_pair, _pinned_int64_pair):
            host, device = fn([3, 7], "cpu")

            self.assertIs(host, device)
            self.assertEqual(torch.int64, host.dtype)
            self.assertEqual([3, 7], host.tolist())

    def test_to_device_keeps_cpu_tensor_and_casts_to_int64(self):
        src = torch.tensor([1, 2, 3], dtype=torch.int32)

        out = _pinned_to_device_int64(src, "cpu")

        self.assertEqual(torch.int64, out.dtype)
        self.assertEqual([1, 2, 3], out.tolist())

    @unittest.skipUnless(torch.cuda.is_available(), "needs a device for pinned staging")
    def test_pair_stages_through_pinned_memory_on_device(self):
        host, device = pinned_int64_pair([5, 9], "cuda")

        self.assertTrue(host.is_pinned())
        self.assertEqual("cuda", device.device.type)
        torch.cuda.synchronize()
        self.assertEqual([5, 9], device.cpu().tolist())


class TestFlushPendingMetadata(CustomTestCase):
    def test_empty_queue_is_a_noop(self):
        queue = _make_queue([])

        queue.flush_pending_metadata()

        self.assertEqual([], queue._pending_metadata)

    def test_sends_one_message_per_pending_entry_and_drains(self):
        reqs = [_make_decode_req(buffer_index=i) for i in range(3)]
        pending = [
            _PendingMetadataSend(
                req,
                torch.tensor([i, i + 1], dtype=torch.int32),
                None,
                decode_prefix_len=10 * i,
            )
            for i, req in enumerate(reqs)
        ]
        queue = _make_queue(pending)

        queue.flush_pending_metadata()

        self.assertEqual([], queue._pending_metadata, "queue must be drained")
        for i, req in enumerate(reqs):
            self.assertEqual(1, len(req.kv_receiver.sent))
            page_indices, buffer_index, state_indices, kwargs = req.kv_receiver.sent[0]
            self.assertEqual([i, i + 1], page_indices.tolist())
            self.assertEqual(i, buffer_index)
            self.assertIsNone(state_indices)
            self.assertEqual({"decode_prefix_len": 10 * i}, kwargs)

    def test_state_indices_tensors_are_converted_and_non_tensors_kept(self):
        req = _make_decode_req(buffer_index=0)
        pending = [
            _PendingMetadataSend(
                req,
                torch.tensor([1], dtype=torch.int32),
                [torch.tensor([4, 5], dtype=torch.int32), 7],
                decode_prefix_len=0,
            )
        ]

        _make_queue(pending).flush_pending_metadata()

        _, _, state_indices, _ = req.kv_receiver.sent[0]
        self.assertEqual([4, 5], state_indices[0].tolist())
        self.assertEqual(7, state_indices[1])

    def test_device_kv_indices_are_passed_through(self):
        req = _make_decode_req(buffer_index=0)
        pending = [
            _PendingMetadataSend(
                req,
                torch.tensor([1], dtype=torch.int32),
                None,
                decode_prefix_len=0,
                device_kv_indices=torch.tensor([8, 9], dtype=torch.int32),
            )
        ]

        _make_queue(pending).flush_pending_metadata()

        _, _, _, kwargs = req.kv_receiver.sent[0]
        self.assertEqual([8, 9], kwargs["device_kv_indices"].tolist())

    def test_rebootstrap_request_also_submits_prefill_recompute(self):
        plain = _make_decode_req(buffer_index=0)
        reboot = _make_decode_req(buffer_index=1, is_rebootstrap=True, payload="again")
        pending = [
            _PendingMetadataSend(
                r, torch.tensor([1], dtype=torch.int32), None, decode_prefix_len=0
            )
            for r in (plain, reboot)
        ]
        queue = _make_queue(pending)

        queue.flush_pending_metadata()

        self.assertEqual(1, len(queue.kv_manager.submitted))
        receiver, payload = queue.kv_manager.submitted[0]
        self.assertIs(reboot.kv_receiver, receiver)
        self.assertEqual("again", payload)

    def test_flush_runs_on_the_private_stream_and_syncs_it(self):
        """The readback is gated on the preallocation event and joined before send."""
        events = []

        class _Stream:
            def wait_event(self, event):
                events.append(("wait", event))

            def synchronize(self):
                events.append(("sync",))

        class _Ctx:
            def __enter__(self_inner):
                events.append(("enter",))

            def __exit__(self_inner, *exc):
                events.append(("exit",))
                return False

        req = _make_decode_req(buffer_index=0)
        pending = [
            _PendingMetadataSend(
                req, torch.tensor([1], dtype=torch.int32), None, decode_prefix_len=0
            )
        ]
        queue = _make_queue(pending)
        stream = _Stream()
        queue._metadata_stream = stream
        queue._prealloc_done = "prealloc_done"
        queue.scheduler = SimpleNamespace(
            device_module=SimpleNamespace(stream=lambda s: _Ctx())
        )

        queue.flush_pending_metadata()

        self.assertEqual(
            [("wait", "prealloc_done"), ("enter",), ("exit",), ("sync",)], events
        )
        self.assertEqual(1, len(req.kv_receiver.sent))

    def test_to_host_passes_cpu_tensors_through(self):
        t = torch.tensor([1, 2], dtype=torch.int32)

        self.assertIs(t, DecodePreallocQueue._to_host(t, None))


class TestEnsureMetadataStream(CustomTestCase):
    def _queue(self, scheduler):
        queue = object.__new__(DecodePreallocQueue)
        queue._metadata_stream = None
        queue._prealloc_done = None
        queue._metadata_stream_ready = False
        queue.scheduler = scheduler
        return queue

    def test_no_device_module_leaves_the_stream_unset(self):
        queue = self._queue(SimpleNamespace(device_module=None, device="cuda"))

        queue._ensure_metadata_stream()

        self.assertIsNone(queue._metadata_stream)
        self.assertTrue(queue._metadata_stream_ready, "must not retry every call")

    def test_non_cuda_device_leaves_the_stream_unset(self):
        module = SimpleNamespace(Stream=_FakeStream, Event=lambda: "event")
        queue = self._queue(SimpleNamespace(device_module=module, device="cpu"))

        queue._ensure_metadata_stream()

        self.assertIsNone(queue._metadata_stream)

    def test_creates_a_stream_and_event(self):
        module = SimpleNamespace(Stream=_FakeStream, Event=lambda: "event")
        queue = self._queue(
            SimpleNamespace(
                device_module=module,
                device="cuda",
                forward_stream=_FakeStream(),
                schedule_stream=_FakeStream(),
            )
        )

        queue._ensure_metadata_stream()

        self.assertIsNotNone(queue._metadata_stream)
        self.assertEqual("event", queue._prealloc_done)

    def test_never_returns_a_stream_aliasing_forward_or_schedule(self):
        """Aliasing either stream would put the readback behind the WAR barrier."""
        forward, schedule = _FakeStream(), _FakeStream()
        handles = iter([forward.cuda_stream, schedule.cuda_stream, 999])

        def stream_factory(priority=0):
            s = _FakeStream(priority)
            s.cuda_stream = next(handles)
            return s

        module = SimpleNamespace(Stream=stream_factory, Event=lambda: "event")
        queue = self._queue(
            SimpleNamespace(
                device_module=module,
                device="cuda",
                forward_stream=forward,
                schedule_stream=schedule,
            )
        )

        queue._ensure_metadata_stream()

        self.assertEqual(999, queue._metadata_stream.cuda_stream)

    def test_gives_up_when_every_redraw_aliases(self):
        forward = _FakeStream()

        def stream_factory(priority=0):
            s = _FakeStream(priority)
            s.cuda_stream = forward.cuda_stream
            return s

        module = SimpleNamespace(Stream=stream_factory, Event=lambda: "event")
        queue = self._queue(
            SimpleNamespace(
                device_module=module,
                device="cuda",
                forward_stream=forward,
                schedule_stream=None,
            )
        )

        queue._ensure_metadata_stream()

        self.assertIsNone(queue._metadata_stream, "must not use an aliased stream")


if __name__ == "__main__":
    unittest.main()
