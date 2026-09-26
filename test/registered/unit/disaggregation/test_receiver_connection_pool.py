"""Receiver control-message publication and connection-pool invalidation."""

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
import zmq

from sglang.srt.disaggregation.base.conn import KVPoll, KVTransferDestination
from sglang.srt.disaggregation.common.conn import (
    CommonKVManager,
    CommonKVReceiver,
    KVTransferError,
)
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.disaggregation.nixl.conn import NixlKVManager
from sglang.srt.disaggregation.utils import KVClassType, TransferBackend, get_kv_class
from sglang.test.test_utils import CustomTestCase


def _manager(manager_class=CommonKVManager):
    """Construct control-plane state without creating an engine or worker threads."""
    manager = manager_class.__new__(manager_class)
    manager.request_status = {}
    manager.prefill_response_tracker = {}
    manager.required_prefill_response_num_table = {}
    manager.addr_to_rooms_tracker = defaultdict(set)
    manager.failure_records = {}
    manager.failure_lock = threading.Lock()
    manager.connection_pool = {}
    manager.connection_lock = threading.Lock()
    manager.waiting_timeout = 5.0
    manager.enable_staging = False
    return manager


def _receiver(connection_pool, entries):
    receiver = object.__new__(CommonKVReceiver)
    receiver.kv_mgr = SimpleNamespace(
        connection_pool=connection_pool,
        connection_lock=threading.Lock(),
    )
    receiver._connection_pool_entries = entries
    return receiver


class _FetchingReceiver(CommonKVReceiver):
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
        if importlib.util.find_spec("mori") is None:
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

    def test_factory_preserves_backend_classes_with_common_receiver_implementation(
        self,
    ):
        from sglang.srt.disaggregation.ascend.conn import AscendKVReceiver
        from sglang.srt.disaggregation.mooncake.conn import MooncakeKVReceiver
        from sglang.srt.disaggregation.mori.conn import MoriKVReceiver
        from sglang.srt.disaggregation.nixl.conn import NixlKVReceiver

        for backend, expected in (
            (TransferBackend.MOONCAKE, MooncakeKVReceiver),
            (TransferBackend.NIXL, NixlKVReceiver),
            (TransferBackend.MORI, MoriKVReceiver),
            (TransferBackend.ASCEND, AscendKVReceiver),
        ):
            with self.subTest(backend=backend):
                receiver_type = get_kv_class(backend, KVClassType.RECEIVER)
                self.assertIs(receiver_type, expected)
                self.assertIsNot(receiver_type, CommonKVReceiver)
                self.assertTrue(issubclass(receiver_type, CommonKVReceiver))
                for method in (
                    "__init__",
                    "send_metadata",
                    "poll",
                    "clear",
                    "abort",
                    "failure_exception",
                ):
                    self.assertIs(
                        getattr(receiver_type, method),
                        getattr(CommonKVReceiver, method),
                    )

    def _make_receiver(self, backend, *, fault=None, error=None):
        manager = _manager(self.backends[backend])
        manager.local_ip = "127.0.0.1"
        manager.rank_port = 5000
        manager.attn_tp_size = 2
        manager.dcp_size, manager.dcp_rank = 1, 0
        manager.enable_staging = False
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
        manager.update_status(17, KVPoll.WaitingForInput)
        receiver.required_dst_info_num = 3
        receiver.bootstrap_infos = [
            {"rank_ip": "::1", "rank_port": port, "is_dummy": port == 6001}
            for port in (6000, 6001, 6002)
        ]
        stale, replaced, replacement = ["stale"], ["old"], ["new"]
        manager.connection_pool = {"stale": stale, "replaced": replacement}
        receiver._connection_pool_entries = {"stale": stale, "replaced": replaced}
        lock = threading.Lock()
        connected, sent = [], []

        class Address(str):
            def encode(address, *args, **kwargs):
                self.assertTrue(
                    lock.locked(), "frame encoding moved outside the socket lock"
                )
                if fault == "encode" and connected[-1] == 6001:
                    raise error
                return super().encode(*args, **kwargs)

        manager.local_ip = Address(manager.local_ip)

        def connect(info):
            port = info["rank_port"]
            connected.append(port)
            if fault == "connect" and port == 6001:
                raise error

            def send(frames):
                self.assertTrue(lock.locked())
                if fault == "send" and port == 6001:
                    raise error
                sent.append((port, frames))

            return SimpleNamespace(send_multipart=send), lock

        receiver._connect_to_bootstrap_server = connect
        return receiver, connected, sent, lock

    @staticmethod
    def _publish(receiver, operation, *, states=None):
        if operation == "_register_kv_args":
            return receiver._register_kv_args()
        return receiver.send_metadata(
            np.array([2, 4], dtype=np.int32),
            aux_index=9,
            state_indices=states,
            decode_prefix_len=2,
        )

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
                receiver, connected, sent, lock = self._make_receiver(backend)
                self.assertTrue(self._publish(receiver, "_register_kv_args"))
                self.assertEqual(connected, [6000, 6001, 6002])
                self.assertEqual(
                    sent, [(port, expected[backend]) for port in connected]
                )
                self.assertFalse(lock.locked())
                self.assertIsNone(receiver.init_time)

    def test_metadata_wire_and_success_bookkeeping(self):
        """Legacy bytes are unchanged while publication bookkeeping is shared."""
        for backend in self.backends:
            for states in (None, [], [[3, 5]]):
                with self.subTest(backend=backend, states=states):
                    receiver, connected, sent, lock = self._make_receiver(backend)
                    self._publish(receiver, "send_metadata", states=states)
                    self.assertEqual(connected, [6000, 6001, 6002])
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
                    self.assertIsNotNone(receiver.init_time)
                    self.assertFalse(lock.locked())
                    self.assertTrue(receiver.metadata_published)
                    self.assertEqual(
                        receiver.kv_mgr.prefill_response_tracker[17], set()
                    )
                    self.assertEqual(
                        set(receiver.kv_mgr.connection_pool), {"stale", "replaced"}
                    )

    def test_destination_interface_preserves_device_wire_and_rejects_host(self):
        for backend in self.backends:
            with self.subTest(backend=backend):
                args = dict(kv_indices=np.array([2, 4], dtype=np.int32), aux_index=9)
                default, _, default_sent, _ = self._make_receiver(backend)
                default.send_metadata(**args)
                device, _, device_sent, _ = self._make_receiver(backend)
                device.send_metadata(**args, destination=KVTransferDestination.DEVICE)
                self.assertEqual(device_sent, default_sent)

                host, connected, sent, _ = self._make_receiver(backend)
                self.assertFalse(host.supports_host_destination)
                with self.assertRaisesRegex(NotImplementedError, "Host KV"):
                    host.send_metadata(**args, destination=KVTransferDestination.HOST)
                self.assertEqual(connected, [])
                self.assertEqual(sent, [])
                self.assertFalse(host.metadata_published)

    def test_missing_bootstrap_fails_all_backends_without_publication(self):
        for backend in self.backends:
            with self.subTest(backend=backend):
                receiver, connected, sent, _ = self._make_receiver(backend)
                receiver.bootstrap_infos = None
                self._publish(receiver, "send_metadata")
                self.assertEqual(receiver.kv_mgr.check_status(17), KVPoll.Failed)
                self.assertEqual(receiver.conclude_state, KVPoll.Failed)
                self.assertIn("bootstrap", receiver.kv_mgr.failure_records[17])
                self.assertEqual(connected, [])
                self.assertEqual(sent, [])
                self.assertIsNone(receiver.init_time)
                self.assertFalse(receiver.metadata_published)

    def test_no_room_does_not_publish(self):
        for backend in self.backends:
            with self.subTest(backend=backend):
                receiver, connected, sent, _ = self._make_receiver(backend)
                receiver.bootstrap_room = None
                self._publish(receiver, "send_metadata")
                self.assertEqual(connected, [])
                self.assertEqual(sent, [])
                self.assertFalse(receiver.metadata_published)

    def test_staging_is_registered_before_any_metadata_is_published(self):
        """Prefill may request staging as soon as the first metadata message is sent."""
        managers = {"mooncake": MooncakeKVManager, "nixl": NixlKVManager}
        for backend, manager_class in managers.items():
            for fault in (None, "send"):
                with self.subTest(backend=backend, fault=fault):
                    receiver, connected, _, _ = self._make_receiver(
                        backend, fault=fault, error=zmq.ZMQError(zmq.EHOSTUNREACH)
                    )
                    manager = receiver.kv_mgr
                    manager.enable_staging = True
                    manager._staging_ctx = SimpleNamespace(
                        allocator=object(), room_bootstrap={}, room_receivers={}
                    )
                    manager.register_staging_room_bootstrap = (
                        manager_class.register_staging_room_bootstrap.__get__(manager)
                    )
                    receiver.chunk_staging_infos = ["old chunk"]
                    connect = receiver._connect_to_bootstrap_server

                    def connect_after_staging(info):
                        self.assertIs(manager._staging_ctx.room_receivers[17], receiver)
                        self.assertIs(
                            manager._staging_ctx.room_bootstrap[17],
                            receiver.bootstrap_infos,
                        )
                        self.assertEqual(receiver.chunk_staging_infos, [])
                        return connect(info)

                    receiver._connect_to_bootstrap_server = connect_after_staging
                    self._publish(receiver, "send_metadata", states=[[3, 5]])
                    self.assertEqual(
                        connected, [6000, 6001] if fault else [6000, 6001, 6002]
                    )
                    self.assertEqual(receiver.init_time is None, fault is not None)
                    self.assertEqual(receiver.metadata_published, fault is None)

    def test_device_indices_remain_a_mooncake_wire_extension(self):
        """A shared entry point must neither drop Mooncake device indices nor accept them elsewhere."""
        for backend in self.backends:
            with self.subTest(backend=backend):
                receiver, connected, sent, _ = self._make_receiver(backend)
                kwargs = dict(
                    kv_indices=np.array([2, 4], dtype=np.int32),
                    aux_index=9,
                    device_kv_indices=np.array([11, 13], dtype=np.int32),
                )
                if backend == "mooncake":
                    receiver.send_metadata(**kwargs)
                    self.assertEqual(
                        [frames[-1] for _, frames in sent],
                        [struct.pack("<ii", 11, 13), b"", struct.pack("<ii", 11, 13)],
                    )
                else:
                    with self.assertRaises(TypeError):
                        receiver.send_metadata(**kwargs)
                    self.assertEqual(connected, [])
                    self.assertFalse(receiver.metadata_published)

    def test_zmq_failure_stops_publication_and_preserves_cache_policy(self):
        """A partially published request must fail without touching replacement caches."""
        for backend in self.backends:
            for operation in ("_register_kv_args", "send_metadata"):
                for fault in ("connect", "encode", "send"):
                    with self.subTest(
                        backend=backend, operation=operation, fault=fault
                    ):
                        receiver, connected, sent, lock = self._make_receiver(
                            backend,
                            fault=fault,
                            error=zmq.ZMQError(zmq.EHOSTUNREACH),
                        )
                        result = self._publish(receiver, operation, states=[[3, 5]])
                        self.assertEqual(
                            result, False if operation == "_register_kv_args" else None
                        )
                        self.assertEqual(connected, [6000, 6001])
                        self.assertEqual([port for port, _ in sent], [6000])
                        self.assertEqual(receiver.conclude_state, KVPoll.Failed)
                        self.assertEqual(
                            receiver.kv_mgr.check_status(17), KVPoll.Failed
                        )
                        self.assertEqual(
                            receiver.kv_mgr.failure_records[17],
                            f"{operation} to prefill ::1:6001 failed",
                        )
                        expected = (
                            {"replaced"}
                            if operation == "send_metadata"
                            else {"stale", "replaced"}
                        )
                        self.assertEqual(set(receiver.kv_mgr.connection_pool), expected)
                        self.assertEqual(
                            receiver.kv_mgr.connection_pool["replaced"], ["new"]
                        )
                        self.assertIsNone(receiver.init_time)
                        self.assertFalse(receiver.metadata_published)
                        self.assertEqual(
                            receiver.kv_mgr.prefill_response_tracker[17], set()
                        )
                        self.assertFalse(lock.locked())

    def test_non_zmq_errors_are_not_converted_to_transfer_failures(self):
        """Encoding/programming errors keep their exception and leave failure state alone."""
        for backend in self.backends:
            for operation in ("_register_kv_args", "send_metadata"):
                for fault in ("encode", "send"):
                    with self.subTest(
                        backend=backend, operation=operation, fault=fault
                    ):
                        error = ValueError("invalid frame")
                        receiver, connected, sent, lock = self._make_receiver(
                            backend, fault=fault, error=error
                        )
                        with self.assertRaises(ValueError) as caught:
                            self._publish(receiver, operation)
                        self.assertIs(caught.exception, error)
                        self.assertEqual(connected, [6000, 6001])
                        self.assertEqual([port for port, _ in sent], [6000])
                        self.assertEqual(
                            receiver.kv_mgr.check_status(17), KVPoll.WaitingForInput
                        )
                        self.assertEqual(receiver.kv_mgr.failure_records, {})
                        self.assertIsNone(receiver.conclude_state)
                        self.assertFalse(lock.locked())
                        self.assertEqual(
                            set(receiver.kv_mgr.connection_pool), {"stale", "replaced"}
                        )


class TestCommonReceiverLifecycle(CustomTestCase):
    def _make_receiver(self, status=KVPoll.WaitingForInput):
        manager = _manager()
        receiver = CommonKVReceiver(manager, "prefill:8998", 17)
        manager.update_status(17, status)
        return receiver, manager

    def test_constructor_initializes_shared_request_state(self):
        manager = _manager()
        receiver = CommonKVReceiver(manager, "prefill:8998", 17)

        self.assertEqual(manager.request_status[17], KVPoll.Bootstrapping)
        self.assertEqual(manager.addr_to_rooms_tracker["prefill:8998"], {17})
        self.assertFalse(receiver.metadata_published)
        self.assertFalse(receiver.abort_notified)
        self.assertFalse(receiver.require_staging)
        self.assertIsNone(receiver.init_time)
        self.assertIsNone(receiver.conclude_state)

    @patch("sglang.srt.disaggregation.common.conn.time.time", return_value=20.0)
    def test_unpublished_metadata_does_not_start_timeout(self, _mock_time):
        receiver, manager = self._make_receiver()
        receiver.init_time = 1.0

        self.assertEqual(receiver.poll(), KVPoll.WaitingForInput)
        self.assertEqual(manager.failure_records, {})

    @patch("sglang.srt.disaggregation.common.conn.time.time", return_value=20.0)
    def test_transferring_state_obeys_the_published_deadline(self, _mock_time):
        receiver, manager = self._make_receiver(KVPoll.Transferring)
        receiver.metadata_published = True
        receiver.init_time = 1.0

        self.assertEqual(receiver.poll(), KVPoll.Failed)
        self.assertIn("timed out", manager.failure_records[17])
        self.assertEqual(receiver.conclude_state, KVPoll.Failed)

    def test_missing_room_fails_instead_of_waiting_forever(self):
        receiver, manager = self._make_receiver()
        manager.request_status.pop(17)

        self.assertEqual(receiver.poll(), KVPoll.Failed)
        self.assertEqual(receiver.poll(), KVPoll.Failed)
        self.assertNotIn(17, manager.request_status)

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

    def test_success_is_published_only_after_staging_is_notified(self):
        receiver, manager = self._make_receiver()
        receiver.metadata_published = True
        manager.required_prefill_response_num_table[17] = 1
        manager.enable_staging = True
        manager._staging_handler = Mock()
        manager._staging_handler.is_staging_room.return_value = True

        def notify_staging(room):
            self.assertEqual(room, 17)
            self.assertEqual(receiver.poll(), KVPoll.WaitingForInput)

        manager._staging_handler.submit_last_scatter_async.side_effect = notify_staging
        manager.apply_prefill_status(
            bootstrap_room=17, status=KVPoll.Success, prefill_rank=0
        )
        self.assertEqual(receiver.poll(), KVPoll.Success)
        manager._staging_handler.submit_last_scatter_async.assert_called_once_with(17)

    def test_success_cannot_override_failure_or_recreate_cleared_room(self):
        receiver, manager = self._make_receiver()
        manager.required_prefill_response_num_table[17] = 1
        manager.apply_prefill_status(
            bootstrap_room=17,
            status=KVPoll.Failed,
            prefill_rank=0,
            failure_reason="remote write failed",
        )
        manager.apply_prefill_status(
            bootstrap_room=17, status=KVPoll.Success, prefill_rank=0
        )
        self.assertEqual(receiver.poll(), KVPoll.Failed)
        self.assertEqual(manager.failure_records[17], "remote write failed")
        receiver.clear()
        manager.apply_prefill_status(
            bootstrap_room=17, status=KVPoll.Success, prefill_rank=0
        )
        self.assertNotIn(17, manager.request_status)
        self.assertNotIn(17, manager.prefill_response_tracker)

    def test_shared_chunk_ready_wire_and_late_message(self):
        receiver, manager = self._make_receiver()
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
        manager._staging_handler.reset_mock()
        receiver.clear()
        self.assertTrue(manager.handle_chunk_ready(frames))
        manager._staging_handler.handle_chunk_arrived.assert_not_called()
        self.assertFalse(manager.handle_chunk_ready([b"KV_STATUS"]))
        self.assertTrue(manager.handle_chunk_ready([b"CHUNK_READY", b"bad"]))

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

    def test_propagated_failure_overrides_local_success(self):
        """A TP-wide failure retires even a receiver whose local transfer succeeded."""
        receiver, _ = self._make_receiver(KVPoll.Success)
        self.assertEqual(receiver.poll(), KVPoll.Success)
        with self.assertRaises(KVTransferError) as caught:
            receiver.failure_exception()
        self.assertTrue(caught.exception.is_from_another_rank)
        self.assertEqual(receiver.poll(), KVPoll.Failed)

    def test_clear_preserves_terminal_result_and_retires_unfinished_receiver(self):
        for initial in (KVPoll.Success, KVPoll.Failed, KVPoll.WaitingForInput):
            with self.subTest(initial=initial):
                receiver, manager = self._make_receiver(initial)
                manager.prefill_response_tracker[17] = set()
                manager.required_prefill_response_num_table[17] = 1
                if initial in (KVPoll.Success, KVPoll.Failed):
                    self.assertEqual(receiver.poll(), initial)
                receiver.clear()
                receiver.clear()

                expected = initial if initial == KVPoll.Success else KVPoll.Failed
                self.assertEqual(receiver.poll(), expected)
                self.assertNotIn(17, manager.request_status)
                self.assertNotIn(17, manager.prefill_response_tracker)
                self.assertNotIn(17, manager.required_prefill_response_num_table)
                self.assertNotIn(17, manager.addr_to_rooms_tracker["prefill:8998"])

    def test_abort_retains_receive_and_drain_state_until_explicit_clear(self):
        receiver, manager = self._make_receiver()
        receiver.bootstrap_infos = [{"rank_ip": "127.0.0.1", "rank_port": 5000}]
        receiver._send_abort_notification = Mock()
        status = manager.prefill_response_tracker[17]
        manager._deferred_abort_ack_tracker = {17: {0}}

        receiver.abort()
        receiver.abort()

        self.assertEqual(receiver.poll(), KVPoll.Failed)
        self.assertEqual(manager.request_status[17], KVPoll.Failed)
        self.assertIs(manager.prefill_response_tracker[17], status)
        self.assertEqual(manager._deferred_abort_ack_tracker, {17: {0}})
        self.assertIn("Abort", manager.failure_records[17])
        receiver._send_abort_notification.assert_called_once_with()

        receiver.clear()
        self.assertNotIn(17, manager.prefill_response_tracker)
        self.assertEqual(manager._deferred_abort_ack_tracker, {17: {0}})

    def test_failure_exception_preserves_local_reason_and_propagation_flag(self):
        for reason in ("peer disconnected during transfer", None):
            with self.subTest(reason=reason):
                receiver, manager = self._make_receiver()
                manager.prefill_response_tracker[17] = set()
                if reason is not None:
                    manager.record_failure(17, reason)
                with self.assertRaises(KVTransferError) as caught:
                    receiver.failure_exception()

                self.assertEqual(caught.exception.bootstrap_room, 17)
                self.assertEqual(caught.exception.is_from_another_rank, reason is None)
                if reason is not None:
                    self.assertEqual(caught.exception.failure_reason, reason)
                self.assertNotIn(17, manager.failure_records)
                self.assertNotIn(17, manager.prefill_response_tracker)
                self.assertEqual(receiver.poll(), KVPoll.Failed)


if __name__ == "__main__":
    unittest.main()
