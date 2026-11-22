"""A PD decode row's DeepSeek-V4 request-scoped state (C128 ring, pending pairs)
is reset before the peer writes the transferred part of it. Under overlap the
row's previous owner can still have a redundant forward in flight that writes
this state, so a send held behind that forward must reset after it; a send not
held must not go out while the reset is still queued on the device."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.decode import DecodePreallocQueue, DecodeRequest
from sglang.srt.disaggregation.utils import TransferBackend
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

ROW = 3


class _Streams:
    """Stand-in device module. Work issued on a stream stays queued until an
    event recorded after it is synchronized; an event recorded on a stream is
    what a held send waits for."""

    def __init__(self):
        self.schedule = "schedule"
        self.forward = "forward"
        self.current = self.schedule
        self.queued = []  # (stream, work) not yet known complete on the host
        self.log = []  # (stream, work) in issue order

    def issue(self, work):
        self.queued.append((self.current, work))
        self.log.append((self.current, work))

    @contextmanager
    def stream(self, stream):
        previous, self.current = self.current, stream
        try:
            yield
        finally:
            self.current = previous

    def Event(self):
        streams = self

        class _Event:
            def record(self, stream=None):
                self.stream = stream or streams.current
                self.upto = [w for w in streams.queued if w[0] == self.stream]
                streams.log.append((self.stream, "event"))

            def synchronize(self):
                for work in self.upto:
                    if work in streams.queued:
                        streams.queued.remove(work)

        return _Event()


def _queue(streams, *, enable_overlap, defer_publish, hisparse=False):
    queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
    queue.scheduler = SimpleNamespace(
        enable_overlap=enable_overlap,
        enable_hisparse=hisparse,
        forward_stream=streams.forward,
        device_module=streams,
    )
    queue._defer_publish = defer_publish
    queue.transfer_backend = TransferBackend.MOONCAKE
    queue.draft_token_to_kv_pool = None
    queue.token_to_kv_pool_allocator = SimpleNamespace(
        page_size=1, translate_kv_indices_for_transfer=lambda ids: ids
    )
    queue.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.arange(64, dtype=torch.int32).reshape(4, 16)
    )
    queue.kv_manager = SimpleNamespace(
        kv_args=SimpleNamespace(
            state_types=[StateType.DSV4_REQUEST_STATE], state_sub_pools=[None]
        )
    )
    queue.transfer_queue = SimpleNamespace(enable_staging=False)
    pool = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
    pool.request_state_transfer_indices = Mock(return_value=[0, 1])
    pool.clear_request_scoped_state = Mock(
        side_effect=lambda row: streams.issue("clear")
    )
    queue.token_to_kv_pool = pool
    queue.req_to_metadata_buffer_idx_allocator = SimpleNamespace(alloc=lambda: 0)
    queue._storage = Mock(return_value=Mock(move_epoch=Mock(return_value=0)))
    queue._record_receive = Mock()
    queue._num_published_destinations = 0
    return queue


def _publish(queue, streams, *, fill_len=8, dst_kv_indices=None):
    req = SimpleNamespace(
        kv=SimpleNamespace(req_pool_idx=ROW),
        time_stats=Mock(),
        bootstrap_host="prefill",
    )
    receiver = Mock()
    # What a send not held finds queued on the device when it goes out.
    receiver.send_metadata.side_effect = lambda *a, **k: streams.log.append(
        ("host", "send", tuple(streams.queued))
    )
    entry = DecodeRequest(req=req, kv_receiver=receiver)
    queue._publish_destination(
        entry,
        dst_kv_indices=(
            torch.arange(fill_len) if dst_kv_indices is None else dst_kv_indices
        ),
        prefix_len=0,
        total_prefix_len=0,
        fill_len=fill_len,
    )
    return entry


class TestRequestStateClearOrder(CustomTestCase):
    def test_a_held_send_clears_after_the_forward_in_flight(self):
        """The send waits for the forward stream's event: the clear runs on that
        stream before it, so the previous owner's late writes precede it."""
        streams = _Streams()
        queue = _queue(streams, enable_overlap=True, defer_publish=False)

        entry = _publish(queue, streams)

        self.assertIsNotNone(entry.pending_metadata)
        forward_work = [w for s, w, *_ in streams.log if s == "forward"]
        self.assertEqual(forward_work, ["clear", "event"])
        queue.token_to_kv_pool.clear_request_scoped_state.assert_called_once_with(ROW)

    def test_a_send_not_held_goes_out_after_the_clear_completed(self):
        """A deferred publish (one iteration later) and a scheduler without
        overlap send at once: the clear stays on the current stream, and the
        send waits until it has run, whatever readbacks precede the send."""
        for enable_overlap, defer_publish in ((True, True), (False, False)):
            with self.subTest(overlap=enable_overlap, deferred=defer_publish):
                streams = _Streams()
                queue = _queue(
                    streams, enable_overlap=enable_overlap, defer_publish=defer_publish
                )

                entry = _publish(queue, streams)

                self.assertIsNone(entry.pending_metadata)
                self.assertIn(("schedule", "clear"), streams.log)
                self.assertEqual(streams.log[-1], ("host", "send", ()))

    def test_a_short_hisparse_request_without_overlap_sends_after_the_clear(self):
        """A HiSparse request of 1-3 tokens holds no compressed host entry, so
        the destination's page readback is empty and does not wait on the
        device; its C128 state still transfers, so the clear must be done."""
        for fill_len in (1, 2, 3):
            with self.subTest(fill_len=fill_len):
                streams = _Streams()
                queue = _queue(
                    streams, enable_overlap=False, defer_publish=False, hisparse=True
                )

                entry = _publish(
                    queue,
                    streams,
                    fill_len=fill_len,
                    dst_kv_indices=torch.empty(0, dtype=torch.int64),
                )

                self.assertIsNone(entry.pending_metadata)
                self.assertEqual(streams.log[-1], ("host", "send", ()))


if __name__ == "__main__":
    unittest.main()
