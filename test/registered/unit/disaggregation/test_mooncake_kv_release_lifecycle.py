"""CPU integration coverage for writes, cancellation, drain ACKs and page reuse.

Run the production Mooncake worker, descriptor builder, control-message handlers
and decode release queue. Only the SDK, sockets and GPU page allocator are
replaced. This checks protocol ordering, not real RDMA completion semantics.
"""

import concurrent.futures
import threading
import unittest
from collections import defaultdict
from contextlib import ExitStack
from queue import SimpleQueue
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from sglang.srt.disaggregation import decode as decode_mod
from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import ABORT_ACK_TAG, CommonKVSender
from sglang.srt.disaggregation.common.utils import TransferKVChunk
from sglang.srt.disaggregation.decode import DecodeTransferQueue
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVManager,
    MooncakeKVReceiver,
    TransferInfo,
)
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

ROOM = 106
PAGE_BYTES = 16
SRC_PAGES = np.array([1, 3], dtype=np.int32)
DST_PAGES = np.array([2, 4], dtype=np.int32)
GUARD = 0xCD
WAIT_SECONDS = 5


class _PagePool:
    def __init__(self, ranks):
        # Each rank writes separate K/V buffers. Unselected pages are guards.
        self.data = np.full((ranks, 2, 6, PAGE_BYTES), GUARD, dtype=np.uint8)
        self.owner = None
        self.releases = []

    def allocate(self, owner):
        if self.owner is not None:
            return None
        self.owner = owner
        return DST_PAGES.copy()

    def release(self, req, tree_cache, *, checkpoint):
        assert not checkpoint
        assert self.owner == req.rid, (self.owner, req.rid)
        self.releases.append(req.rid)
        self.owner = None


class _ControlledEngine:
    """Pause the first SDK call before touching its destination bytes."""

    def __init__(self, source, destination):
        self.source = source
        self.destination = destination
        self.entered = threading.Event()
        self.resume = threading.Event()
        self.calls = 0

    @staticmethod
    def _view(buffer, address, length):
        offset = address - buffer.ctypes.data
        assert 0 <= offset and offset + length <= buffer.nbytes, (
            "descriptor outside registered buffer",
            offset,
            length,
            buffer.nbytes,
        )
        return buffer.reshape(-1)[offset : offset + length]

    def batch_transfer_sync(self, session, sources, destinations, lengths):
        self.calls += 1
        if self.calls == 1:
            self.entered.set()
            assert self.resume.wait(WAIT_SECONDS), "test did not release the write"
        for src, dst, length in zip(sources, destinations, lengths, strict=True):
            self._view(self.destination, dst, length)[:] = self._view(
                self.source, src, length
            )
        return 0


def _manager(rank=0):
    # Avoid hardware registration and background network services. Methods
    # below remain the real production implementations, including ACK tracking.
    mgr = MooncakeKVManager.__new__(MooncakeKVManager)
    mgr.enable_deferred_decode_kv_release = True
    mgr._deferred_abort_ack_tracker = {}
    mgr._deferred_abort_generation = 0
    mgr._deferred_ack_targets = {}
    mgr._deferred_ack_poisoned_rooms = set()
    mgr._staging_outstanding = defaultdict(int)
    mgr.request_status = {}
    mgr.req_to_decode_prefix_len = {}
    mgr.transfer_infos = {}
    mgr.failure_records = {}
    mgr.failure_lock = threading.Lock()
    mgr.deferred_bootstrap = None
    mgr.defer_decode_allocation = False
    mgr.enable_trace = mgr.enable_staging = False
    mgr.is_mla_backend = mgr.is_hybrid_mla_backend = False
    mgr.enable_custom_mem_pool = False
    mgr.max_transfer_batch_indices = 0
    mgr.attn_tp_rank = rank
    mgr.attn_tp_size = mgr.pp_size = mgr.attn_cp_size = 1
    mgr.pp_rank = mgr.attn_cp_rank = 0
    mgr.bootstrap_port = 8998
    mgr.session_lock = threading.Lock()
    mgr.session_failures = defaultdict(int)
    mgr.failed_sessions = set()
    mgr.state_layout_rejections = {}
    mgr.prefill_response_tracker = defaultdict(set)
    mgr.required_prefill_response_num_table = {}
    mgr.addr_to_rooms_tracker = defaultdict(set)
    return mgr


class _Lifecycle:
    def __init__(self, testcase, ranks=1):
        self.test = testcase
        self.ranks = ranks
        self.pool = _PagePool(ranks)
        self.source = np.arange(self.pool.data.size, dtype=np.uint8).reshape(
            self.pool.data.shape
        )
        self.decode = _manager()
        self.decode.local_ip, self.decode.rank_port = "127.0.0.1", 6000
        self.prefills = [_manager(rank) for rank in range(ranks)]
        self.messages = SimpleQueue()
        for mgr in self.prefills:
            mgr.attn_tp_size = ranks
            mgr._send_multipart_locked = lambda address, parts, **kwargs: (
                self.messages.put(list(parts))
            )
        self.clock = 100.0
        self.resources = ExitStack()
        testcase.addCleanup(self.resources.close)
        self.resources.enter_context(
            patch.object(
                decode_mod, "time", SimpleNamespace(monotonic=lambda: self.clock)
            )
        )
        self.resources.enter_context(
            patch.object(decode_mod, "release_kv_cache", self.pool.release)
        )
        self.queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        self.queue._deferred_releases = []
        self.queue.deferred_kv_release_timeout = 30.0
        self.queue.enable_host_receive = self.queue.enable_staging = False
        self.queue.tree_cache = object()
        self.queue.metadata_buffers = SimpleNamespace(bootstrap_room={})
        self.freed_metadata = []
        self.queue.req_to_metadata_buffer_idx_allocator = SimpleNamespace(
            free=self.freed_metadata.append
        )
        self.queue.scheduler = SimpleNamespace(
            metrics_reporter=SimpleNamespace(enable_metrics=False)
        )
        self.workers = concurrent.futures.ThreadPoolExecutor(
            max_workers=ranks, thread_name_prefix="pd-lifecycle"
        )
        self.resources.callback(self.workers.shutdown, wait=True)
        self.engines = []
        self.futures = []
        # Always unblock SDK calls before joining workers, even after a failure.
        self.resources.callback(self._unblock)

    def _unblock(self):
        for engine in self.engines:
            engine.resume.set()

    def start(self, owner):
        self.test.assertIsNotNone(self.pool.allocate(owner))
        self.decode.update_status(ROOM, KVPoll.WaitingForInput)
        self.decode.required_prefill_response_num_table[ROOM] = self.ranks
        receiver = MooncakeKVReceiver.__new__(MooncakeKVReceiver)
        receiver.kv_mgr = self.decode
        receiver.bootstrap_room = ROOM
        receiver.bootstrap_addr = "prefill"
        receiver.init_time = 1.0  # Destination metadata has been published.
        receiver.abort_notified = False
        receiver._abort_generation = None
        receiver.bootstrap_infos = [{"rank": rank} for rank in range(self.ranks)]
        self.aborts = [SimpleQueue() for _ in self.prefills]
        receiver._connect_to_bootstrap_server = lambda info: (
            SimpleNamespace(send_multipart=self.aborts[info["rank"]].put),
            threading.Lock(),
        )
        self.receiver = receiver
        self.req = SimpleNamespace(
            req=SimpleNamespace(bootstrap_room=ROOM, rid=owner),
            kv_receiver=receiver,
            metadata_buffer_index=0,
        )
        self.queue.metadata_buffers.bootstrap_room[0] = ROOM
        self.engines = []
        self.futures = []
        for rank, mgr in enumerate(self.prefills):
            src, dst = self.source[rank], self.pool.data[rank]
            engine = _ControlledEngine(src, dst)
            self.engines.append(engine)
            mgr.engine = engine
            mgr.update_status(ROOM, KVPoll.WaitingForInput)
            mgr.kv_args = SimpleNamespace(
                kv_data_ptrs=[buf.ctypes.data for buf in src],
                kv_item_lens=[PAGE_BYTES, PAGE_BYTES],
                kv_layer_ids=[],
                prefill_start_layer=0,
                aux_data_ptrs=[],
                aux_item_lens=[],
            )
            mgr.decode_kv_args_table = {
                "decode": SimpleNamespace(
                    requires_dcp_relayout=False,
                    dst_kv_ptrs=[buf.ctypes.data for buf in dst],
                    dst_kv_layer_ids=[],
                    dst_kv_item_len=PAGE_BYTES,
                    dst_attn_tp_size=self.ranks,
                    dst_aux_ptrs=[],
                )
            }
            mgr.transfer_infos[ROOM] = {
                "decode": TransferInfo(
                    ROOM, "127.0.0.1", 6000, "decode", DST_PAGES.copy(), 0, [], 1, False
                )
            }
            chunks = SimpleQueue()
            chunks.put(
                TransferKVChunk(ROOM, SRC_PAGES.copy(), slice(0, 2), True, 0, None)
            )
            chunks.put(None)
            self.futures.append(
                self.workers.submit(mgr.transfer_worker, chunks, self.workers)
            )
        for rank, engine in enumerate(self.engines):
            ready = engine.entered.wait(WAIT_SECONDS)
            if not ready and self.futures[rank].done():
                self.futures[rank].result()
            self.test.assertTrue(ready, f"rank {rank} did not submit its write")

    def abort(self):
        self.receiver.abort()
        self.generation = self.receiver._abort_generation
        for rank, mgr in enumerate(self.prefills):
            self.test.assertTrue(
                mgr._handle_abort_notification(
                    self.aborts[rank].get(timeout=WAIT_SECONDS)
                )
            )
        self.queue._defer_release(self.req)

    def drain(self, rank):
        self.engines[rank].resume.set()
        self.futures[rank].result(timeout=WAIT_SECONDS)

    def deliver(self, parts):
        if not self.decode.handle_abort_ack_message(parts):
            room, status, rank, reason = self.decode.parse_kv_status_message(parts)
            self.decode.apply_prefill_status(
                bootstrap_room=room,
                status=status,
                prefill_rank=rank,
                failure_reason=reason,
            )

    def deliver_all(self):
        acks = []
        while not self.messages.empty():
            parts = self.messages.get_nowait()
            self.deliver(parts)
            if parts[0] == ABORT_ACK_TAG:
                acks.append(parts)
        return acks

    def resolve(self):
        self.queue.resolve_deferred_releases()

    def assert_held(self, owner):
        self.resolve()
        outstanding = [m._staging_outstanding.get(ROOM, 0) for m in self.prefills]
        context = (
            f"room={ROOM}, generation={self.generation}, "
            f"owner={self.pool.owner}, outstanding={outstanding}"
        )
        self.test.assertEqual(self.pool.owner, owner, context)
        self.test.assertIsNone(self.pool.allocate("unexpected reuse"), context)
        self.test.assertEqual(self.queue.num_pending_deferred_releases(), 1, context)

    def assert_bytes(self, value=None):
        expected = np.full_like(self.pool.data, GUARD)
        if value is None:
            expected[:, :, DST_PAGES, :] = self.source[:, :, SRC_PAGES, :]
        else:
            expected[:, :, DST_PAGES, :] = value
        np.testing.assert_array_equal(self.pool.data, expected)

    def assert_released(self, owner):
        self.resolve()
        self.test.assertIsNone(self.pool.owner)
        self.test.assertEqual(self.pool.releases.count(owner), 1)
        self.test.assertEqual(len(self.freed_metadata), len(self.pool.releases))
        self.test.assertEqual(self.queue.num_pending_deferred_releases(), 0)
        self.test.assertNotIn(ROOM, self.decode._deferred_abort_ack_tracker)
        self.test.assertIsNone(self.req.kv_receiver)
        self.test.assertEqual(self.queue.metadata_buffers.bootstrap_room[0], 0)
        for mgr in self.prefills:
            self.test.assertEqual(mgr._staging_outstanding.get(ROOM, 0), 0)
            self.test.assertNotIn(ROOM, mgr._deferred_ack_targets)
            sender = SimpleNamespace(kv_mgr=mgr, bootstrap_room=ROOM)
            CommonKVSender.clear(sender)

    def reuse_and_check(self):
        self.test.assertIsNotNone(self.pool.allocate("B"))
        self.pool.data[:, :, DST_PAGES, :] = 0xA5
        self.deliver_all()
        self.resolve()
        self.assert_bytes(0xA5)
        self.test.assertEqual(self.pool.owner, "B")


class TestMooncakeKVReleaseLifecycle(unittest.TestCase):
    def setUp(self):
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy", device="cpu"), role="scheduler")

    def test_completed_transfer_can_be_released_and_reused(self):
        h = _Lifecycle(self)
        h.start("A")
        h.drain(0)
        h.deliver_all()
        self.assertEqual(h.decode.check_status(ROOM), KVPoll.Success)
        h.assert_bytes()
        # Cancelling an already-completed request must ACK immediately.
        h.abort()
        h.deliver_all()
        h.assert_released("A")
        h.reuse_and_check()

    def test_cancel_during_write_holds_pages_until_drain(self):
        h = _Lifecycle(self)
        h.start("A")
        h.abort()
        self.assertEqual(h.deliver_all(), [], "ACK arrived before the write drained")
        h.assert_held("A")
        np.testing.assert_array_equal(h.pool.data, GUARD)
        h.drain(0)
        acks = h.deliver_all()
        self.assertEqual([m[0] for m in acks], [ABORT_ACK_TAG])
        h.assert_bytes()
        h.assert_released("A")
        h.reuse_and_check()

    def test_all_ranks_must_drain_and_duplicate_ack_does_not_count_twice(self):
        h = _Lifecycle(self, ranks=2)
        h.start("A")
        h.abort()
        h.assert_held("A")
        h.drain(0)
        acks = h.deliver_all()
        self.assertEqual(len(acks), 1)
        h.deliver(acks[0])
        h.assert_held("A")
        h.drain(1)
        h.deliver_all()
        h.assert_bytes()
        h.assert_released("A")
        h.deliver(acks[0])
        h.resolve()
        self.assertEqual(h.pool.releases, ["A"])
        h.reuse_and_check()

    def test_old_generation_ack_cannot_release_reused_room(self):
        h = _Lifecycle(self)
        h.start("A")
        h.abort()
        h.drain(0)
        old_acks = h.deliver_all()
        old_generation = h.generation
        h.assert_released("A")
        h.start("C")
        h.abort()
        self.assertGreater(h.generation, old_generation)
        for parts in old_acks:
            h.deliver(parts)
        h.assert_held("C")
        h.drain(0)
        h.deliver_all()
        h.assert_released("C")
        h.reuse_and_check()

    def test_device_timeout_fallback_is_distinct_from_drain_ack(self):
        h = _Lifecycle(self)
        h.start("A")
        h.abort()
        h.drain(0)
        # Drop the ACK. This models a control-channel loss after the write;
        # device pages retain the existing timeout fallback, unlike host pages.
        dropped_acks = []
        while not h.messages.empty():
            parts = h.messages.get_nowait()
            if parts[0] == ABORT_ACK_TAG:
                dropped_acks.append(parts)
            else:
                h.deliver(parts)
        self.assertEqual(len(dropped_acks), 1)
        h.assert_held("A")
        h.clock += h.queue.deferred_kv_release_timeout + 1
        h.assert_released("A")
        h.reuse_and_check()


if __name__ == "__main__":
    unittest.main()
