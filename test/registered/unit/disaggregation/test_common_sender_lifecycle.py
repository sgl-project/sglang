import importlib.util
import queue
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import ANY, Mock, patch

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVSender
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager, MooncakeKVSender
from sglang.srt.disaggregation.nixl.conn import NixlKVManager
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def make_sender(manager_type, room=42, *, sender_type=CommonKVSender):
    manager = object.__new__(manager_type)
    manager.disaggregation_mode = DisaggregationMode.PREFILL
    manager.request_status = {}
    manager.transfer_infos = {room: {"127.0.0.1:1234": SimpleNamespace(dst_port=1234)}}
    manager.is_dummy_cp_rank = False
    manager.enable_all_cp_ranks_for_transfer = False
    manager.attn_cp_size = 1
    manager.attn_cp_rank = 0
    manager.enable_staging = False
    manager._staging_outstanding = {}
    manager._deferred_ack_targets = {}
    manager._deferred_ack_poisoned_rooms = set()
    manager._num_shards = 1
    manager.transfer_queues = [queue.Queue()]
    manager._transfer_queues = manager.transfer_queues
    with get_context().override_server_args(dp_size=1, enable_trace=False):
        sender = sender_type(manager, "unused", room, [0], 0)
    return sender, manager, manager.transfer_queues[0]


class TestCommonSenderLifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # These tests exercise Mori's Python queue adapter, not its RDMA engine.
        if "mori" not in sys.modules and importlib.util.find_spec("mori") is None:
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
        from sglang.srt.disaggregation.mori.conn import MoriKVManager

        cls.mori_manager = MoriKVManager

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

    def test_success_waits_for_deferred_staging_chunk(self):
        sender, manager, _ = make_sender(MooncakeKVManager)
        manager.request_status[42] = KVPoll.Success
        manager._staging_outstanding[42] = 1
        self.assertEqual(sender.poll(), KVPoll.Transferring)
        manager._staging_outstanding[42] = 0
        self.assertEqual(sender.poll(), KVPoll.Success)

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
            sender, manager, chunks = make_sender(
                MooncakeKVManager, sender_type=MooncakeKVSender
            )
        sender.init(1, aux_index=9)
        sender.send(np.array([1], dtype=np.int32))
        self.assertIs(chunks.get_nowait().trace_ctx, trace.copy_for_thread.return_value)
        trace.trace_req_start.assert_called_once()
        trace.trace_slice_start.assert_called_once_with("mooncake_send", 1, ANY)
        trace.trace_slice_end.assert_called_once_with("mooncake_send", 1)
        manager.request_status[42] = KVPoll.Success
        self.assertEqual(sender.poll(), KVPoll.Success)
        self.assertEqual(sender.poll(), KVPoll.Success)
        trace.trace_req_finish.assert_called_once()


if __name__ == "__main__":
    unittest.main()
