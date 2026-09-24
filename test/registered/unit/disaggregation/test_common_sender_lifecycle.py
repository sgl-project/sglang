import importlib.util
import queue
import sys
import threading
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVSender, KVTransferError
from sglang.srt.disaggregation.common.staging_handler import PrefillStagingContext
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager, MooncakeKVSender
from sglang.srt.disaggregation.nixl.conn import NixlKVManager, NixlKVSender
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    KVClassType,
    TransferBackend,
    get_kv_class,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def make_sender(manager_type, room=42, *, dummy_cp=False):
    manager = object.__new__(manager_type)
    manager.disaggregation_mode = DisaggregationMode.PREFILL
    manager.request_status = {}
    manager.failure_records = {}
    manager.failure_lock = threading.Lock()
    manager.req_to_decode_prefix_len = {room: 0}
    manager.transfer_infos = {room: {"127.0.0.1:1234": SimpleNamespace(dst_port=1234)}}
    manager.is_dummy_cp_rank = dummy_cp
    manager.enable_all_cp_ranks_for_transfer = False
    manager.attn_cp_size = 1
    manager.attn_cp_rank = 0
    manager.bootstrap_timeout = 5
    manager.enable_staging = False
    manager._staging_outstanding = {}
    manager._staging_ctx = None
    manager._num_shards = 1
    manager.transfer_queues = [queue.Queue()]
    manager._transfer_queues = manager.transfer_queues
    with get_context().override_server_args(dp_size=1, enable_trace=False):
        sender = CommonKVSender(manager, "unused", room, [0], 0)
    return sender, manager, manager.transfer_queues[0]


class TestCommonSenderLifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # These tests exercise Mori's Python queue adapter, not its RDMA engine.
        if importlib.util.find_spec("mori") is None:
            package, cpp, io = (
                ModuleType("mori"),
                ModuleType("mori.cpp"),
                ModuleType("mori.io"),
            )
            cpp.TransferStatus = type("TransferStatus", (), {})
            for name in (
                "BackendType",
                "EngineDesc",
                "IOEngine",
                "IOEngineConfig",
                "MemoryDesc",
                "MemoryLocationType",
                "PollCqMode",
                "RdmaBackendConfig",
            ):
                setattr(io, name, type(name, (), {}))
            io.StatusCode = SimpleNamespace(SUCCESS=0, IN_PROGRESS=1)
            package.cpp, package.io = cpp, io
            stub = patch.dict(
                sys.modules, {"mori": package, "mori.cpp": cpp, "mori.io": io}
            )
            stub.start()
            cls.addClassCleanup(stub.stop)
        from sglang.srt.disaggregation.mori.conn import MoriKVManager, MoriKVSender

        cls.mori_manager, cls.mori_sender = MoriKVManager, MoriKVSender

    def test_factory_preserves_backend_classes_with_common_sender_implementation(self):
        from sglang.srt.disaggregation.ascend.conn import AscendKVSender

        for backend, expected in (
            (TransferBackend.MOONCAKE, MooncakeKVSender),
            (TransferBackend.MORI, self.mori_sender),
            (TransferBackend.NIXL, NixlKVSender),
            (TransferBackend.ASCEND, AscendKVSender),
        ):
            with self.subTest(backend=backend):
                sender_type = get_kv_class(backend, KVClassType.SENDER)
                self.assertIs(sender_type, expected)
                self.assertIsNot(sender_type, CommonKVSender)
                self.assertTrue(issubclass(sender_type, CommonKVSender))
                for method in (
                    "__init__",
                    "send",
                    "poll",
                    "clear",
                    "abort",
                    "failure_exception",
                ):
                    self.assertIs(
                        getattr(sender_type, method), getattr(CommonKVSender, method)
                    )

    def test_two_chunks_keep_backend_queue_contracts(self):
        for manager_type in (MooncakeKVManager, self.mori_manager, NixlKVManager):
            with self.subTest(backend=manager_type.__name__):
                sender, manager, chunks = make_sender(manager_type)
                sender.init(4, aux_index=9)
                sender.send(np.array([10, 11], dtype=np.int32), num_kv_tokens=128)
                sender.send(
                    np.array([12, 13], dtype=np.int32),
                    state_indices=[[3, 4], None],
                    num_kv_tokens=100,
                )
                first, last = chunks.get_nowait(), chunks.get_nowait()
                np.testing.assert_array_equal(first.prefill_kv_indices, [10, 11])
                self.assertEqual(first.index_slice, slice(0, 2))
                self.assertFalse(first.is_last_chunk)
                self.assertIsNone(first.prefill_aux_index)
                self.assertIsNone(first.state_indices)
                np.testing.assert_array_equal(last.prefill_kv_indices, [12, 13])
                self.assertEqual(last.index_slice, slice(2, 4))
                self.assertTrue(last.is_last_chunk)
                self.assertEqual(last.prefill_aux_index, 9)
                np.testing.assert_array_equal(last.state_indices[0], [3, 4])
                self.assertIsNone(last.state_indices[1])
                self.assertEqual(last.num_kv_tokens, 100)
                self.assertEqual(sender._transfer_num_kv_indices, 4)
                self.assertEqual(sender._transfer_num_state_indices, 2)
                if manager_type is NixlKVManager:
                    self.assertEqual((first.chunk_id, last.chunk_id), (0, 1))

    def test_mori_consumes_early_event_once_and_skips_cp_replicated_state(self):
        sender, manager, chunks = make_sender(self.mori_manager)
        manager.attn_cp_size = 2
        manager.attn_cp_rank = 1
        sender.init(2, aux_index=9)
        event = object()
        sender._early_send_wait_event = event
        with get_context().override_server_args(enable_dsa_cache_layer_split=False):
            sender.send(np.array([10], dtype=np.int32))
            sender.send(np.array([11], dtype=np.int32), state_indices=[[3]])
        first, last = chunks.get_nowait(), chunks.get_nowait()
        self.assertIs(first.wait_event, event)
        self.assertIsNone(last.wait_event)
        self.assertIsNone(last.state_indices)
        self.assertEqual(sender._transfer_num_state_indices, 0)

    def test_zero_kv_final_chunk_still_sends_aux_and_state(self):
        for manager_type in (MooncakeKVManager, self.mori_manager, NixlKVManager):
            with self.subTest(backend=manager_type.__name__):
                sender, _, chunks = make_sender(manager_type)
                sender.init(0, aux_index=9)
                sender.send(np.array([], dtype=np.int32), state_indices=[[3]])
                chunk = chunks.get_nowait()
                self.assertTrue(chunk.is_last_chunk)
                self.assertEqual(chunk.prefill_aux_index, 9)
                np.testing.assert_array_equal(chunk.state_indices[0], [3])

    def test_dummy_cp_rank_finishes_only_on_last_chunk_without_enqueuing(self):
        for manager_type in (MooncakeKVManager, self.mori_manager, NixlKVManager):
            with self.subTest(backend=manager_type.__name__):
                sender, _, chunks = make_sender(manager_type, dummy_cp=True)
                sender.init(2, aux_index=9)
                sender.send(np.array([10], dtype=np.int32))
                self.assertEqual(sender.poll(), KVPoll.WaitingForInput)
                sender.send(np.array([11], dtype=np.int32))
                self.assertEqual(sender.poll(), KVPoll.Success)
                self.assertTrue(chunks.empty())

    def test_success_waits_for_deferred_chunk_and_survives_clear(self):
        for manager_type in (MooncakeKVManager, self.mori_manager, NixlKVManager):
            with self.subTest(backend=manager_type.__name__):
                sender, manager, _ = make_sender(manager_type)
                sender._transfer_start_time = 5.0
                manager.request_status[42] = KVPoll.Success
                manager._staging_outstanding[42] = 1
                self.assertEqual(sender.poll(), KVPoll.Transferring)
                self.assertIsNone(sender.conclude_state)
                manager._staging_outstanding[42] = 0
                with patch(
                    "sglang.srt.disaggregation.common.conn.time.perf_counter",
                    return_value=7.0,
                ):
                    self.assertEqual(sender.poll(), KVPoll.Success)
                self.assertEqual(sender._transfer_metric.transfer_latency_s, 2.0)
                sender.clear()
                self.assertEqual(sender.poll(), KVPoll.Success)
                self.assertNotIn(42, manager.request_status)

    def test_timeout_latches_failed_and_cleans_up(self):
        sender, manager, _ = make_sender(self.mori_manager)
        sender.init_time = 1.0
        with patch(
            "sglang.srt.disaggregation.common.conn.time.time", return_value=10.0
        ):
            self.assertEqual(sender.poll(), KVPoll.Failed)
        self.assertEqual(sender.conclude_state, KVPoll.Failed)
        self.assertIn("timed out", manager.failure_records[42])
        with self.assertRaises(KVTransferError) as raised:
            sender.failure_exception()
        self.assertFalse(raised.exception.is_from_another_rank)
        self.assertNotIn(42, manager.failure_records)
        self.assertNotIn(42, manager.transfer_infos)
        self.assertEqual(sender.poll(), KVPoll.Failed)

    def test_abort_stops_enqueue_and_failure_cleanup_keeps_other_rooms(self):
        sender, manager, chunks = make_sender(NixlKVManager)
        sender.init(1, aux_index=9)
        manager._staging_ctx = PrefillStagingContext()
        manager._staging_ctx.prefetched_rooms = {42, 43}
        manager._staging_ctx.prefetch_requested = {(42, 0, "peer"), (43, 0, "peer")}
        sender.abort()
        sender.send(np.array([1], dtype=np.int32))
        self.assertTrue(chunks.empty())
        with self.assertRaisesRegex(KVTransferError, "Aborted"):
            sender.failure_exception()
        self.assertEqual(manager._staging_ctx.prefetched_rooms, {43})
        self.assertEqual(manager._staging_ctx.prefetch_requested, {(43, 0, "peer")})

    def test_missing_room_is_failed_without_resurrecting_it(self):
        sender, manager, _ = make_sender(self.mori_manager)
        del manager.request_status[42]
        self.assertEqual(sender.poll(), KVPoll.Failed)
        self.assertNotIn(42, manager.request_status)
        with self.assertRaises(KVTransferError) as raised:
            sender.failure_exception()
        self.assertTrue(raised.exception.is_from_another_rank)

    def test_mooncake_trace_context_reaches_worker_and_finishes(self):
        trace = Mock(tracing_enable=True)
        with (
            patch(
                "sglang.srt.disaggregation.common.conn.get_observability",
                return_value=SimpleNamespace(enable_trace=True),
            ),
            patch(
                "sglang.srt.disaggregation.common.conn.TraceReqContext",
                return_value=trace,
            ),
        ):
            sender, manager, chunks = make_sender(MooncakeKVManager)
        sender.init(1, aux_index=9)
        sender.send(np.array([1], dtype=np.int32))
        self.assertIs(chunks.get_nowait().trace_ctx, trace.copy_for_thread.return_value)
        trace.trace_req_start.assert_called_once()
        trace.trace_slice_start.assert_called_once_with("mooncake_send", 1)
        trace.trace_slice_end.assert_called_once_with("mooncake_send", 1)
        manager.request_status[42] = KVPoll.Success
        self.assertEqual(sender.poll(), KVPoll.Success)
        self.assertEqual(sender.poll(), KVPoll.Success)
        trace.trace_req_finish.assert_called_once()


if __name__ == "__main__":
    unittest.main()
