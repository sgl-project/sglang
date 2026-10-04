"""Unit tests for srt/disaggregation/common/conn — receiver connection_pool invalidation."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


import importlib.util
import struct
import sys
import threading
import unittest
from collections import defaultdict
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import msgspec
import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVManager, CommonKVReceiver
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.disaggregation.nixl.conn import NixlKVManager
from sglang.test.test_utils import CustomTestCase


class _ConcreteReceiver(CommonKVReceiver):
    def poll(self) -> KVPoll:
        raise NotImplementedError

    def failure_exception(self):
        raise NotImplementedError


def _manager(manager_class=CommonKVManager):
    """Construct control-plane state without creating an engine or worker threads."""
    manager = manager_class.__new__(manager_class)
    manager.request_status = {}
    manager.prefill_response_tracker = {}
    manager.required_prefill_response_num_table = {}
    manager.addr_to_rooms_tracker = defaultdict(set)
    manager._staging_outstanding = {}
    manager._deferred_ack_targets = {}
    manager._deferred_ack_poisoned_rooms = set()
    manager.enable_staging = False
    return manager


def _receiver(connection_pool, entries):
    receiver = object.__new__(_ConcreteReceiver)
    receiver.kv_mgr = SimpleNamespace(
        connection_pool=connection_pool,
        connection_lock=threading.Lock(),
    )
    receiver._connection_pool_entries = entries
    return receiver


class _FetchingReceiver(_ConcreteReceiver):
    def _get_bootstrap_info_from_server(
        self, prefill_dp_rank, prefill_cp_rank, target_tp_rank, target_pp_rank
    ):
        self.fetch_count += 1
        return {"rank_ip": "10.0.0.1", "rank_port": 2001, "pp_rank": target_pp_rank}

    def _register_kv_args(self):
        return True


def _fetching_receiver(connection_pool):
    receiver = object.__new__(_FetchingReceiver)
    receiver.kv_mgr = SimpleNamespace(
        connection_pool=connection_pool,
        connection_lock=threading.Lock(),
        is_mla_backend=False,
    )
    receiver.bootstrap_addr = "prefill:8998"
    receiver.bootstrap_room = 1
    receiver.prefill_dp_rank = 0
    receiver.target_cp_ranks = [0]
    receiver.target_tp_rank = 0
    receiver.target_tp_ranks = [0]
    receiver.target_pp_ranks = [0]
    receiver._connection_pool_entries = {}
    receiver.fetch_count = 0
    return receiver


class TestReceiverConnectionPool(CustomTestCase):
    def test_invalidate_removes_matching_generation(self):
        stale = [
            {"rank_ip": "10.0.0.1", "rank_port": 1001},
            {"rank_ip": "10.0.0.1", "rank_port": 1002},
        ]
        retained = [{"rank_ip": "10.0.0.2", "rank_port": 2001}]
        receiver = _receiver(
            {"stale": stale, "retained": retained},
            {"stale": stale},
        )

        receiver.invalidate_cached_bootstrap_infos()

        self.assertEqual(receiver.kv_mgr.connection_pool, {"retained": retained})
        self.assertEqual(receiver._connection_pool_entries, {})

    def test_invalidate_preserves_concurrent_replacement_generation(self):
        stale = [{"rank_ip": "10.0.0.1", "rank_port": 1001}]
        replacement = [{"rank_ip": "10.0.0.1", "rank_port": 2001}]
        receiver = _receiver(
            {"key": replacement},
            {"key": stale},
        )

        receiver.invalidate_cached_bootstrap_infos()

        self.assertEqual(receiver.kv_mgr.connection_pool, {"key": replacement})

    def test_invalidate_removes_all_matching_cp_entries(self):
        stale_cp0 = [{"rank_ip": "10.0.0.1", "rank_port": 1001}]
        stale_cp1 = [{"rank_ip": "10.0.0.1", "rank_port": 1002}]
        receiver = _receiver(
            {"cp0": stale_cp0, "cp1": stale_cp1},
            {"cp0": stale_cp0, "cp1": stale_cp1},
        )

        receiver.invalidate_cached_bootstrap_infos()

        self.assertEqual(receiver.kv_mgr.connection_pool, {})

    def test_next_receiver_refetches_after_invalidation(self):
        stale = [{"rank_ip": "10.0.0.1", "rank_port": 1001}]
        connection_pool = {"prefill:8998_0_0_0": stale}
        stale_receiver = _receiver(
            connection_pool,
            {"prefill:8998_0_0_0": stale},
        )
        stale_receiver.invalidate_cached_bootstrap_infos()

        receiver = _fetching_receiver(connection_pool)
        receiver._setup_bootstrap_infos()

        self.assertEqual(receiver.fetch_count, 1)
        self.assertEqual(receiver.bootstrap_infos[0]["rank_port"], 2001)
        self.assertIs(
            connection_pool["prefill:8998_0_0_0"],
            receiver._connection_pool_entries["prefill:8998_0_0_0"],
        )

    @patch("sglang.srt.disaggregation.common.conn.time.time", return_value=3.0)
    def test_waiting_timeout_invalidates_cached_generation(self, _mock_time):
        stale = [{"rank_ip": "10.0.0.1", "rank_port": 1001}]
        receiver = _receiver({"key": stale}, {"key": stale})
        receiver.bootstrap_room = 1
        receiver.bootstrap_infos = stale
        receiver.init_time = 1.0
        receiver.abort_notified = True
        receiver.kv_mgr.waiting_timeout = 1.0
        receiver.kv_mgr.record_failure = Mock()
        receiver.kv_mgr.update_status = Mock()

        self.assertEqual(receiver._check_waiting_timeout(), KVPoll.Failed)
        self.assertEqual(receiver.kv_mgr.connection_pool, {})


class TestReceiverControlMessages(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if "mori" not in sys.modules and importlib.util.find_spec("mori") is None:
            package = ModuleType("mori")
            cpp = ModuleType("mori.cpp")
            cpp.TransferStatus = type("TransferStatus", (), {})
            io = ModuleType("mori.io")
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

        cls.backends = {
            "mooncake": MooncakeKVManager,
            "nixl": NixlKVManager,
            "mori": MoriKVManager,
        }

    def _make_receiver(self, backend):
        manager = _manager(self.backends[backend])
        manager.local_ip = "127.0.0.1"
        manager.rank_port = 5000
        manager.attn_tp_size = 2
        manager.dcp_size, manager.dcp_rank = 1, 0
        manager.kv_args = SimpleNamespace(
            kv_data_ptrs=[0x1000],
            kv_data_lens=[256],
            kv_item_lens=[16],
            kv_data_mem_kinds=["VRAM"],
            kv_layer_ids=[7],
            aux_data_ptrs=[0x2000],
            state_data_ptrs=[[0x3000]],
            state_item_lens=[[32]],
            state_dim_per_tensor=[[4]],
            state_layer_ids=[[7]],
            engine_rank=0,
            gpu_id=0,
        )
        manager.agent = SimpleNamespace(
            name="nixl-peer", get_agent_metadata=lambda: b"agent-desc"
        )
        manager.engine_desc = SimpleNamespace(
            key="mori-peer", pack=lambda: b"engine-desc"
        )
        manager.kv_mem_descs = [SimpleNamespace(pack=lambda: b"kv-desc")]
        manager.aux_mem_descs = [SimpleNamespace(pack=lambda: b"aux-desc")]
        manager.state_mem_descs = [[SimpleNamespace(pack=lambda: b"state-desc")]]
        manager.engine = SimpleNamespace(get_session_id=lambda: "mooncake-peer")
        receiver = CommonKVReceiver(manager, "prefill:8998", 17)
        receiver.required_dst_info_num = 3
        receiver.bootstrap_infos = [
            {"rank_ip": "::1", "rank_port": port, "is_dummy": port == 6001}
            for port in (6000, 6001, 6002)
        ]
        sent = []

        def connect(info):
            def send(frames):
                sent.append((info["rank_port"], frames))

            return SimpleNamespace(send_multipart=send), threading.Lock()

        receiver._connect_to_bootstrap_server = connect
        return receiver, sent

    def test_registration_wire_is_preserved_for_all_peers(self):
        """Registration refactors must preserve each backend's legacy frame layout."""
        ptr = struct.pack("Q", 0x1000)
        aux = struct.pack("Q", 0x2000)
        state = struct.pack("<IIQ", 1, 8, 0x3000)
        item = struct.pack("<III", 1, 4, 32)
        dim = struct.pack("<III", 1, 4, 4)
        layer = struct.pack("I", 7)
        state_layer = struct.pack("<III", 1, 4, 7)
        expected = {
            "mooncake": [
                b"None",
                b"127.0.0.1",
                b"5000",
                b"mooncake-peer",
                ptr,
                aux,
                state,
                b"0",
                b"2",
                b"16",
                item,
                dim,
                layer,
                state_layer,
                b"",
                b"",
                b"1",
                b"0",
                b"",
                struct.pack("Q", 16),
            ],
            "nixl": [
                b"NixlMsgGuard",
                b"None",
                b"127.0.0.1",
                b"5000",
                b"nixl-peer",
                b"agent-desc",
                ptr,
                aux,
                state,
                b"0",
                b"2",
                b"0",
                b"16",
                item,
                dim,
                b"",
                b"",
                b"16",
                b"VRAM",
                struct.pack("Q", 16),
                state_layer,
                layer,
                b"1",
                b"0",
            ],
            "mori": [
                b"MoriMsgGuard",
                b"None",
                b"127.0.0.1",
                b"5000",
                b"engine-desc",
                msgspec.msgpack.encode([b"kv-desc"]),
                msgspec.msgpack.encode([b"aux-desc"]),
                msgspec.msgpack.encode([[b"state-desc"]]),
                b"0",
                b"2",
                b"0",
                b"16",
                item,
                dim,
            ],
        }
        for backend in self.backends:
            with self.subTest(backend=backend):
                receiver, sent = self._make_receiver(backend)
                self.assertTrue(receiver._register_kv_args())
                self.assertEqual(
                    sent, [(port, expected[backend]) for port in (6000, 6001, 6002)]
                )

    def test_metadata_wire_is_preserved_for_all_peers(self):
        for backend in self.backends:
            for states in (None, [], [[3, 5]]):
                with self.subTest(backend=backend, states=states):
                    receiver, sent = self._make_receiver(backend)
                    receiver.send_metadata(
                        np.array([2, 4], dtype=np.int32),
                        aux_index=9,
                        state_indices=states,
                        decode_prefix_len=2,
                    )
                    self.assertEqual([port for port, _ in sent], [6000, 6001, 6002])
                    for port, frames in sent:
                        dummy = port == 6001
                        indices = b"" if dummy else struct.pack("<ii", 2, 4)
                        state = (
                            struct.pack("<IIii", 1, 8, 3, 5)
                            if states and not dummy
                            else b""
                        )
                        common = [b"17", b"127.0.0.1", b"5000"]
                        if backend == "mooncake":
                            expected = common + [
                                b"mooncake-peer",
                                indices,
                                b"" if dummy else b"9",
                                state,
                                b"3",
                                b"2",
                                b"",
                            ]
                        elif backend == "nixl":
                            expected = (
                                [b"NixlMsgGuard"]
                                + common
                                + [
                                    b"nixl-peer",
                                    indices,
                                    b"9",
                                    b"3",
                                    state,
                                    b"2",
                                    b"1" if dummy else b"0",
                                ]
                            )
                        else:
                            expected = (
                                [b"MoriMsgGuard"]
                                + common
                                + [
                                    b"mori-peer",
                                    indices,
                                    b"" if dummy else b"9",
                                    state,
                                    b"3",
                                    b"2",
                                ]
                            )
                        self.assertEqual(frames, expected)

    def test_mooncake_device_indices_are_forwarded(self):
        receiver, sent = self._make_receiver("mooncake")
        receiver.send_metadata(
            np.array([2, 4], dtype=np.int32),
            aux_index=9,
            device_kv_indices=np.array([11, 13], dtype=np.int32),
        )
        self.assertEqual(
            [frames[-1] for _, frames in sent],
            [struct.pack("<ii", 11, 13), b"", struct.pack("<ii", 11, 13)],
        )


class TestCommonReceiverLifecycle(CustomTestCase):
    def _make_receiver(self):
        manager = _manager()
        receiver = CommonKVReceiver(manager, "prefill:8998", 17)
        manager.update_status(17, KVPoll.WaitingForInput)
        return receiver, manager

    def test_success_waits_for_all_distinct_prefill_ranks(self):
        receiver, manager = self._make_receiver()
        manager.required_prefill_response_num_table[17] = 2
        for rank in (3, 3):
            manager.apply_prefill_status(
                bootstrap_room=17, status=KVPoll.Success, prefill_rank=rank
            )
            self.assertEqual(receiver.poll(), KVPoll.WaitingForInput)
        manager.apply_prefill_status(
            bootstrap_room=17, status=KVPoll.Success, prefill_rank=4
        )
        self.assertEqual(receiver.poll(), KVPoll.Success)
        self.assertEqual(manager.prefill_response_tracker[17], {3, 4})

    def test_chunk_ready_round_trip(self):
        _, manager = self._make_receiver()
        manager._prefill_unique_rank = lambda: 3
        manager._send_multipart_locked = Mock()
        manager._staging_handler = Mock()
        manager.send_chunk_ready(
            room=17,
            chunk_idx=2,
            page_start=8,
            num_pages=4,
            writer_id="peer_name_with_underscores",
            targets=[("127.0.0.1", 5000)],
        )
        frames = manager._send_multipart_locked.call_args.args[1]
        self.assertEqual(
            frames,
            [
                b"CHUNK_READY",
                b"17",
                b"2",
                b"8",
                b"4",
                b"peer_name_with_underscores",
                b"3",
            ],
        )
        self.assertTrue(manager.handle_chunk_ready(frames))
        manager._staging_handler.handle_chunk_arrived.assert_called_once_with(
            17, 2, 8, 4, "peer_name_with_underscores"
        )

    def test_completion_racing_clear_cannot_recreate_rank_tracker(self):
        """A notification that observed a live room must not reinsert it after clear."""
        receiver, manager = self._make_receiver()
        observed = threading.Event()
        resume = threading.Event()

        class PausedLookup(dict):
            def __contains__(states, room):
                live = super().__contains__(room)
                observed.set()
                if not resume.wait(timeout=5):
                    raise RuntimeError("clear did not release notification lookup")
                return live

        manager.request_status = PausedLookup(manager.request_status)
        result = []
        worker = threading.Thread(
            target=lambda: result.append(
                manager.apply_prefill_status(
                    bootstrap_room=17, status=KVPoll.Success, prefill_rank=0
                )
            )
        )
        worker.start()
        try:
            self.assertTrue(observed.wait(timeout=5))
            receiver.clear()
        finally:
            resume.set()
            worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(result, [None])
        self.assertEqual(manager.prefill_response_tracker, {})


if __name__ == "__main__":
    unittest.main()
