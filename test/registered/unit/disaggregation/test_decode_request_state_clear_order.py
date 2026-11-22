"""A PD decode row's DeepSeek-V4 request-scoped state (C128 ring, pending pairs)
is reset before the peer writes the transferred part of it. Under overlap the
row's previous owner can still have a redundant forward in flight that writes
this state, so a send held behind that forward must reset after it."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.decode import DecodePreallocQueue, DecodeRequest
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

ROW = 3


class _Streams:
    """Stand-in device module: logs each stream's work in issue order."""

    def __init__(self):
        self.schedule = "schedule"
        self.forward = "forward"
        self.current = self.schedule
        self.log = []

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
                streams.log.append((stream or streams.current, "event"))

        return _Event()


def _queue(streams, *, enable_overlap, defer_publish):
    queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
    queue.scheduler = SimpleNamespace(
        enable_overlap=enable_overlap,
        enable_hisparse=False,
        forward_stream=streams.forward,
        device_module=streams,
    )
    queue._defer_publish = defer_publish
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
        side_effect=lambda row: streams.log.append((streams.current, "clear"))
    )
    queue.token_to_kv_pool = pool
    queue.req_to_metadata_buffer_idx_allocator = SimpleNamespace(alloc=lambda: 0)
    queue._transfer_page_indices = Mock(return_value=[0])
    queue._storage = Mock(return_value=Mock(move_epoch=Mock(return_value=0)))
    queue._record_receive = Mock()
    queue._num_published_destinations = 0
    return queue


def _publish(queue):
    req = SimpleNamespace(kv=SimpleNamespace(req_pool_idx=ROW), time_stats=Mock())
    entry = DecodeRequest(req=req, kv_receiver=Mock())
    queue._publish_destination(
        entry,
        dst_kv_indices=torch.arange(8),
        prefix_len=0,
        total_prefix_len=0,
        fill_len=8,
    )
    return entry


class TestRequestStateClearOrder(CustomTestCase):
    def test_a_held_send_clears_after_the_forward_in_flight(self):
        """The send waits for the forward stream's event: the clear runs on that
        stream before it, so the previous owner's late writes precede it."""
        streams = _Streams()
        queue = _queue(streams, enable_overlap=True, defer_publish=False)

        entry = _publish(queue)

        self.assertIsNotNone(entry.pending_metadata)
        forward_work = [event for stream, event in streams.log if stream == "forward"]
        self.assertEqual(forward_work, ["clear", "event"])
        queue.token_to_kv_pool.clear_request_scoped_state.assert_called_once_with(ROW)

    def test_a_send_not_held_clears_on_the_publishing_stream(self):
        """A deferred publish (one iteration later) and a scheduler without
        overlap send once the current stream is done: the clear stays on it."""
        for enable_overlap, defer_publish in ((True, True), (False, False)):
            with self.subTest(overlap=enable_overlap, deferred=defer_publish):
                streams = _Streams()
                queue = _queue(
                    streams, enable_overlap=enable_overlap, defer_publish=defer_publish
                )

                entry = _publish(queue)

                self.assertIsNone(entry.pending_metadata)
                self.assertEqual(streams.log, [("schedule", "clear")])
                entry.kv_receiver.send_metadata.assert_called_once()


if __name__ == "__main__":
    unittest.main()
