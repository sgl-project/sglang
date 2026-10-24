"""Basic CPU unit tests for NIXL disaggregation control paths."""

import json
import os
import sys
import threading
import types
import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import requests

from sglang.srt.disaggregation.base.conn import KVPoll, StateType
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.common.staging_handler import PrefillStagingContext
from sglang.srt.disaggregation.nixl.conn import (
    KVArgsRegisterInfo,
    NixlKVManager,
    NixlKVReceiver,
    TransferInfo,
    TransferKVChunk,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class StagingFakeAgent:
    def __init__(self, register_result=None):
        self.register_result = (
            register_result if register_result is not None else ["desc"]
        )
        self.register_memory_calls = []
        self.get_xfer_descs_calls = []
        self.initialize_xfer_calls = []
        self.transfer_calls = []

    def register_memory(self, addrs, mem_type):
        self.register_memory_calls.append((addrs, mem_type))
        return self.register_result

    def get_xfer_descs(self, reqs, mem_type):
        self.get_xfer_descs_calls.append((reqs, mem_type))
        return f"{mem_type}_{len(self.get_xfer_descs_calls)}"

    def initialize_xfer(self, *args):
        self.initialize_xfer_calls.append(args)
        return "handle"

    def transfer(self, handle):
        self.transfer_calls.append(handle)
        return "DONE"


class FakeQueue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


class FakeStagingAllocator:
    ALLOC_OVERSIZED = -2


def _fake_staging_buffer_module(mock_gather=None):
    module = types.ModuleType("sglang.srt.disaggregation.common.staging_buffer")
    module.StagingAllocator = FakeStagingAllocator
    module.compute_head_slice_params = lambda *args: (0, 1, 0, 1)
    module.compute_staging_layout = lambda *args: (2, [256, 256], 512)
    module.resolve_total_kv_heads = lambda kv_args, attn_tp_size: 2
    module.gather_all_layers_to_staging = mock_gather or MagicMock()
    return module


class TestNixlBackendInitialization(CustomTestCase):
    def _initialize_manager(
        self,
        agent,
        device_module,
        mode,
        gpu_id,
        *,
        backend="UCX",
        ib_device=None,
        backend_params=None,
        ucx_net_devices="",
    ):
        args = SimpleNamespace(
            pp_rank=0,
            engine_rank=0,
            gpu_id=gpu_id,
            ib_device=ib_device,
            kv_data_ptrs=[],
            kv_item_lens=[],
            state_types=[],
        )
        mgr = object.__new__(NixlKVManager)
        mgr.kv_args = args
        mgr.disaggregation_mode = mode
        mgr.enable_deferred_decode_kv_release = False
        mgr.pp_size = mgr.attn_cp_size = mgr.dcp_size = 1

        api = types.ModuleType("nixl._api")
        api.nixl_agent = MagicMock(return_value=agent)
        api.nixl_agent_config = MagicMock()
        api.nixl_thread_sync_t = SimpleNamespace(NIXL_THREAD_SYNC_STRICT="strict")
        nixl = types.ModuleType("nixl")
        nixl._api = api
        agent.get_plugin_list.return_value = [backend]

        with (
            patch.dict(sys.modules, {"nixl": nixl, "nixl._api": api}),
            patch.dict(
                "os.environ",
                {
                    "SGLANG_DISAGGREGATION_NIXL_BACKEND": backend,
                    "SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS": json.dumps(
                        backend_params or {}
                    ),
                    "UCX_NET_DEVICES": ucx_net_devices,
                    "SGLANG_DISAGGREGATION_ENGINE_INIT_TIMEOUT": "5",
                    "SGLANG_DISAGGREGATION_QUEUE_SIZE": "0",
                    "SGLANG_DISAGG_STAGING_BUFFER": "false",
                },
            ),
            get_context().override_server_args(device="cuda") as server_args,
            patch.object(CommonKVManager, "__init__", return_value=None),
            patch(
                "sglang.srt.disaggregation.nixl.conn.torch.get_device_module",
                return_value=device_module,
            ) as get_device_module,
            patch.object(NixlKVManager, "register_buffer_to_engine"),
            patch.object(NixlKVManager, "_start_bootstrap_thread"),
            patch.object(NixlKVManager, "_start_decode_listener_thread"),
            patch.object(NixlKVManager, "_start_heartbeat_checker_thread"),
        ):
            self.assertIsNone(server_args.device)
            NixlKVManager.__init__(mgr, args, mode, server_args)
            get_device_module.assert_called_once_with("cuda")
            self.assertEqual(os.environ["UCX_NET_DEVICES"], ucx_net_devices)

    def test_backend_initialization_selects_device_in_deadline_thread(self):
        caller_thread = threading.get_ident()
        for mode, gpu_id in (
            (DisaggregationMode.PREFILL, 3),
            (DisaggregationMode.DECODE, 1),
        ):
            with self.subTest(mode=mode, gpu_id=gpu_id):
                calls = []
                device_module = MagicMock()
                device_module.set_device.side_effect = lambda device: calls.append(
                    ("set_device", device, threading.get_ident())
                )
                agent = MagicMock()
                agent.create_backend.side_effect = lambda backend, params: calls.append(
                    ("create_backend", backend, threading.get_ident())
                )

                self._initialize_manager(agent, device_module, mode, gpu_id)

                self.assertEqual(len(calls), 2)
                backend_thread = calls[1][2]
                self.assertNotEqual(backend_thread, caller_thread)
                self.assertEqual(
                    calls,
                    [
                        ("set_device", gpu_id, backend_thread),
                        ("create_backend", "UCX", backend_thread),
                    ],
                )
                agent.create_backend.assert_called_once_with(
                    "UCX",
                    {"num_threads": "8"} if mode == DisaggregationMode.PREFILL else {},
                )

    def test_backend_initialization_respects_ucx_device_selection(self):
        mapping = '{"1":"rdma0","3":"rdma4"}'
        net_devices = "rdma0:1,rdma4:1,lo"
        cases = (
            ("UCX", 1, mapping, "", {}, {"device_list": "rdma0"}),
            ("UCX", 1, "rdma0,rdma4", "", {}, {"device_list": "rdma0, rdma4"}),
            ("UCX", 3, mapping, "", {}, {"device_list": "rdma4"}),
            ("UCX", 1, mapping, net_devices, {}, {}),
            ("UCX", 3, mapping, net_devices, {}, {}),
            ("UCX", 1, None, net_devices, {}, {}),
            (
                "UCX",
                1,
                mapping,
                net_devices,
                {"device_list": "rdma4"},
                {"device_list": "rdma4"},
            ),
            ("LIBFABRIC", 1, mapping, net_devices, {}, {}),
        )
        for backend, gpu_id, ib_device, net, params, expected in cases:
            with (
                self.subTest(backend=backend, gpu_id=gpu_id, net=net, params=params),
                patch("sglang.srt.disaggregation.nixl.conn.logger.warning") as warning,
            ):
                agent = MagicMock()
                self._initialize_manager(
                    agent,
                    MagicMock(),
                    DisaggregationMode.DECODE,
                    gpu_id,
                    backend=backend,
                    ib_device=ib_device,
                    backend_params=params,
                    ucx_net_devices=net,
                )
                agent.create_backend.assert_called_once_with(backend, expected)
                if backend == "UCX" and ib_device and net:
                    warning.assert_called_once()
                    self.assertIn(
                        "Per-GPU CLI device mappings", warning.call_args[0][0]
                    )
                else:
                    warning.assert_not_called()

        with self.assertRaisesRegex(ValueError, "Conflicting NIXL UCX device_list"):
            self._initialize_manager(
                MagicMock(),
                MagicMock(),
                DisaggregationMode.DECODE,
                1,
                ib_device=mapping,
                backend_params={"device_list": "rdma4"},
            )

    def test_backend_initialization_propagates_error(self):
        error = RuntimeError("backend initialization failed")
        agent = MagicMock()
        agent.create_backend.side_effect = error

        with self.assertRaises(RuntimeError) as raised:
            self._initialize_manager(
                agent, MagicMock(), DisaggregationMode.DECODE, gpu_id=3
            )

        self.assertIs(raised.exception, error)


class TestNixlEmptyStateTransfer(CustomTestCase):
    def test_prepared_mamba_rejects_stride_mismatch_before_fallback(self):
        """Different SSM dtypes must not write into the wrong decode request slot."""
        for src_stride, dst_stride in ((32, 16), (16, 32)):
            with self.subTest(src_stride=src_stride, dst_stride=dst_stride):
                mgr = object.__new__(NixlKVManager)
                mgr.agent = MagicMock()
                mgr.pp_size = mgr.state_tp_size = 1
                mgr.is_mla_backend = False
                mgr._use_strided_descs = True
                mgr._prepared_state_components = frozenset({0})
                mgr._prepared_state_plans = {}
                mgr._state_layout_rejections = {}
                mgr.kv_args = SimpleNamespace(
                    state_types=[StateType.MAMBA],
                    state_data_ptrs=[[0x10000, 0x11000]],
                    state_data_lens=[[16 * 16, src_stride * 16]],
                    state_item_lens=[[16, src_stride]],
                    state_layer_ids=[[0, 0]],
                    mla_compression_ratios=None,
                )
                peer = SimpleNamespace(
                    agent_name="decode",
                    decode_tp_size=1,
                    dst_dcp_size=1,
                    requires_dcp_relayout=False,
                    dst_state_data_ptrs=[[0x20000, 0x21000]],
                    dst_state_data_lens=[[16 * 16, dst_stride * 16]],
                    dst_state_item_lens=[[16, dst_stride]],
                    dst_state_layer_ids=[[0, 0]],
                )

                # Registration records the rejection; each transfer to the peer fails.
                mgr._prepare_state_xfer(peer)

                self.assertIn("stride mismatch", mgr._state_layout_rejections["decode"])
                self.assertEqual(mgr._prepared_state_plans, {})
                mgr.agent.prep_xfer_dlist.assert_not_called()
                with self.assertRaisesRegex(RuntimeError, "stride mismatch"):
                    list(
                        mgr.maybe_send_extra(
                            "decode",
                            [[1]],
                            peer.dst_state_data_ptrs,
                            [[2]],
                            0,
                            1,
                        )
                    )
                mgr.agent.transfer.assert_not_called()

    def test_paired_state_entries_reject_item_length_mismatch(self):
        mgr = object.__new__(NixlKVManager)
        mgr.agent = StagingFakeAgent()
        mgr.is_mla_backend = False
        mgr.pp_size = 1
        mgr.kv_args = SimpleNamespace(prefill_start_layer=0, kv_data_ptrs=[1])

        with self.assertRaisesRegex(RuntimeError, "item length mismatch"):
            mgr._send_kvcache_generic(
                peer_name="decode",
                src_data_ptrs=[10],
                dst_data_ptrs=[20],
                item_lens=[32],
                prefill_data_indices=np.array([3], dtype=np.int32),
                dst_data_indices=np.array([5], dtype=np.int32),
                dst_gpu_id=0,
                state_type=StateType.QSA_PENDING,
                force_flat=True,
                src_layer_ids=[24],
                dst_layer_ids=[24],
                dst_item_lens=[48],
            )
        self.assertEqual(mgr.agent.initialize_xfer_calls, [])


class TestNixlTransferWorker(CustomTestCase):
    def _make_manager(self, room):
        mgr = object.__new__(NixlKVManager)
        mgr.request_status = {room: KVPoll.WaitingForInput}
        mgr.transfer_infos = {
            room: {
                "agent": TransferInfo(
                    room=room,
                    endpoint="127.0.0.1",
                    dst_port=5555,
                    agent_name="agent",
                    dst_kv_indices=np.array([2], dtype=np.int32),
                    dst_aux_index=0,
                    required_dst_info_num=1,
                    dst_state_indices=[],
                )
            }
        }
        mgr.decode_kv_args_table = {
            "agent": SimpleNamespace(
                decode_tp_size=1,
                dst_kv_ptrs=[0],
                dst_aux_ptrs=[0],
                gpu_id=0,
                staging_base_ptr=0,
                staging_total_size=0,
                kv_xfer_segments=None,
                dst_homogeneous_mem_kind="VRAM",
                # Non-DCP peer. Without this the worker raises AttributeError
                # and lands in the same Failed status the assertions expect,
                # so the transfer path would go unexercised.
                requires_dcp_relayout=False,
                dcp_dst_region_indices=None,
                dcp_token_item_lens=None,
            )
        }
        mgr.req_to_decode_prefix_len = {room: 4}
        mgr.enable_staging = False
        mgr.enable_deferred_decode_kv_release = False
        mgr._staging_ctx = None
        mgr._staging_outstanding = defaultdict(int)
        mgr.is_mla_backend = False
        mgr.is_hybrid_mla_backend = False
        mgr.attn_tp_size = 1
        mgr.kv_args = SimpleNamespace(
            engine_rank=0, kv_data_ptrs=[0], num_draft_entries=0
        )
        mgr.failure_lock = threading.Lock()
        mgr.failure_records = {}
        mgr.send_kv_status_message = MagicMock()

        def check_xfer_state(_handle):
            mgr.update_status(room, KVPoll.Failed)
            return "DONE"

        mgr.agent = SimpleNamespace(check_xfer_state=check_xfer_state)
        return mgr

    def _make_chunk(self, room, prefill_kv_indices, is_last_chunk):
        return TransferKVChunk(
            room=room,
            prefill_kv_indices=np.array(prefill_kv_indices, dtype=np.int32),
            index_slice=slice(0, len(prefill_kv_indices)),
            is_last_chunk=is_last_chunk,
            prefill_aux_index=0 if is_last_chunk else None,
            state_indices=None,
        )

    def _run_worker_once(self, mgr, chunk):
        queue = SimpleNamespace(get=MagicMock(side_effect=[chunk, SystemExit()]))
        with self.assertRaises(SystemExit):
            mgr.transfer_worker(queue)

    def test_given_last_chunk_aborts_mid_transfer_when_worker_finishes_then_failed_status_is_preserved(
        self,
    ):
        room = 21
        mgr = self._make_manager(room)
        mgr.send_aux = MagicMock(return_value="aux_handle")
        chunk = self._make_chunk(room, [], is_last_chunk=True)

        self._run_worker_once(mgr, chunk)

        self.assertEqual(mgr.request_status[room], KVPoll.Failed)
        self.assertNotIn(room, mgr.transfer_infos)
        self.assertNotIn(room, mgr.req_to_decode_prefix_len)
        mgr.send_aux.assert_called_once()

    def test_given_non_last_chunk_aborts_mid_transfer_when_worker_finishes_then_failed_status_is_preserved(
        self,
    ):
        room = 22
        mgr = self._make_manager(room)
        mgr.send_kvcache = MagicMock(return_value="kv_handle")
        chunk = self._make_chunk(room, [1], is_last_chunk=False)

        self._run_worker_once(mgr, chunk)

        self.assertEqual(mgr.request_status[room], KVPoll.Failed)
        self.assertIn(room, mgr.transfer_infos)
        self.assertIn(room, mgr.req_to_decode_prefix_len)
        mgr.send_kvcache.assert_called_once()

    def test_dcp_destinations_use_disjoint_pack_regions_before_chunk_barrier(self):
        room = 23
        mgr = self._make_manager(room)
        agents = ("agent0a", "agent0b", "agent1")
        dcp_ranks = (0, 0, 1)
        mgr.transfer_infos[room] = {
            agent: TransferInfo(
                room=room,
                endpoint="127.0.0.1",
                dst_port=5555 + i,
                agent_name=agent,
                dst_kv_indices=np.array([2 + i], dtype=np.int32),
                dst_aux_index=0,
                required_dst_info_num=len(agents),
                dst_state_indices=[],
            )
            for i, agent in enumerate(agents)
        }
        mgr.decode_kv_args_table = {
            agent: SimpleNamespace(
                decode_tp_size=len(agents),
                dst_kv_ptrs=[0x3000 + i * 0x100],
                dst_aux_ptrs=[0],
                gpu_id=0,
                staging_base_ptr=0,
                staging_total_size=0,
                kv_xfer_segments=None,
                dst_homogeneous_mem_kind="VRAM",
                requires_dcp_relayout=True,
                dst_dcp_size=2,
                dst_dcp_rank=dcp_rank,
                dcp_dst_region_indices=[0],
                dcp_token_item_lens=[4],
            )
            for i, (agent, dcp_rank) in enumerate(zip(agents, dcp_ranks))
        }
        mgr.kv_args = SimpleNamespace(
            engine_rank=0,
            kv_data_ptrs=[0x1000],
            page_size=4,
            num_draft_entries=0,
        )
        mgr._dcp_pack_buffers = [SimpleNamespace(get_size=lambda: 16)]

        packed_rank0 = ([0x9000], np.arange(2, dtype=np.int64))
        packed_rank1 = ([0x9008], np.arange(2, dtype=np.int64))
        try_pack = MagicMock(side_effect=[packed_rank0, packed_rank1])
        dcp_pack_module = types.ModuleType("sglang.srt.disaggregation.common.dcp_pack")
        dcp_pack_module.try_pack_dcp_src = try_pack
        submitted = []

        def send_kvcache_dcp(*args, **kwargs):
            submitted.append((args[0], args[-1]))
            # One handle per transfer part; the worker extends its handle list.
            return [f"handle-{args[0]}"]

        mgr.send_kvcache_dcp = MagicMock(side_effect=send_kvcache_dcp)
        submitted_counts_at_poll = []

        def check_xfer_state(_handle):
            submitted_counts_at_poll.append(len(submitted))
            return "DONE"

        mgr.agent = SimpleNamespace(check_xfer_state=check_xfer_state)
        chunk = self._make_chunk(room, [1], is_last_chunk=False)
        chunk.num_kv_tokens = 4

        with patch.dict(
            sys.modules,
            {"sglang.srt.disaggregation.common.dcp_pack": dcp_pack_module},
        ):
            self._run_worker_once(mgr, chunk)

        self.assertEqual(try_pack.call_count, 2)
        self.assertEqual(
            [call.kwargs["pack_offset_bytes"] for call in try_pack.call_args_list],
            [0, 8],
        )
        self.assertEqual(
            submitted,
            [
                ("agent0a", packed_rank0),
                ("agent0b", packed_rank0),
                ("agent1", packed_rank1),
            ],
        )
        self.assertEqual(submitted_counts_at_poll, [3, 3, 3])

    def test_success_is_sent_after_kv_aux_and_state_handles_complete(self):
        room = 24
        mgr = self._make_manager(room)
        peer = mgr.decode_kv_args_table["agent"]
        peer.dst_state_data_ptrs = [[0]]
        peer.decode_tp_rank = 0
        peer.dst_state_item_lens = [[4]]
        peer.dst_state_dim_per_tensor = [[1]]
        peer.dst_state_layer_ids = [[0]]
        events = []
        mgr.send_kvcache = MagicMock(return_value="kv")
        mgr.maybe_send_extra = MagicMock(return_value=["state0", "state1"])
        mgr.send_aux = MagicMock(return_value="aux")
        mgr.agent.check_xfer_state = lambda handle: events.append(handle) or "DONE"
        mgr.send_kv_status_message.side_effect = lambda **kwargs: events.append(
            kwargs["status"]
        )
        chunk = self._make_chunk(room, [1], is_last_chunk=True)
        chunk.state_indices = [[0]]

        self._run_worker_once(mgr, chunk)

        self.assertEqual(events, ["kv", "state0", "state1", "aux", KVPoll.Success])
        self.assertEqual(
            mgr.send_kv_status_message.call_args.kwargs["targets"],
            [("127.0.0.1", 5555)],
        )
        self.assertNotIn(room, mgr.transfer_infos)

    def test_failed_handle_never_emits_success(self):
        room = 25
        mgr = self._make_manager(room)
        mgr.send_kvcache = MagicMock(return_value="kv")
        mgr.send_aux = MagicMock(return_value="aux")
        mgr.agent.check_xfer_state = lambda handle: "ERR" if handle == "kv" else "DONE"
        self._run_worker_once(mgr, self._make_chunk(room, [1], is_last_chunk=True))
        self.assertEqual(mgr.request_status[room], KVPoll.Failed)
        mgr.send_kv_status_message.assert_called_once()
        self.assertEqual(
            mgr.send_kv_status_message.call_args.kwargs["status"], KVPoll.Failed
        )

    def _make_staging_manager(self, room):
        mgr = self._make_manager(room)
        req = TransferInfo(**vars(mgr.transfer_infos[room]["agent"]))
        req.dst_kv_indices = np.array([100, 101], dtype=np.int32)
        mgr.transfer_infos[room] = {"agent": req}
        peer = SimpleNamespace(**vars(mgr.decode_kv_args_table["agent"]))
        peer.decode_tp_size = 2
        peer.staging_base_ptr = 0x8000
        mgr.decode_kv_args_table = {"agent": peer}
        mgr.enable_staging = True
        mgr._staging_ctx = PrefillStagingContext()
        mgr._try_create_staging_strategy = MagicMock(return_value=object())
        mgr.agent.check_xfer_state = MagicMock(return_value="DONE")
        mgr.send_aux = MagicMock(side_effect=lambda peer, *args: f"aux-{peer}")
        mgr.send_chunk_ready = MagicMock()
        return mgr

    def _run_staging_worker(self, mgr, chunks):
        pending = list(chunks)

        def get():
            if not pending:
                raise SystemExit()
            return pending.pop(0)

        queue = SimpleNamespace(get=get, put=pending.append)
        with self.assertRaises(SystemExit):
            mgr.transfer_worker(queue, staging_buffer=object())

    def test_deferred_staging_chunk_still_transfers_after_last_chunk_success(self):
        room = 27
        mgr = self._make_staging_manager(room)
        first = self._make_chunk(room, [1], is_last_chunk=False)
        last = self._make_chunk(room, [2], is_last_chunk=True)
        last.index_slice = slice(1, 2)
        events = []
        attempts = []

        def stage(_strategy, chunk, _indices, _req, _info, queue):
            page_start = chunk.index_slice.start
            attempts.append(page_start)
            if len(attempts) == 1:
                queue.put(chunk)
                return None, True, 0
            return f"kv-{page_start}", False, page_start

        mgr._do_staging_transfer = stage
        mgr.send_chunk_ready.side_effect = lambda **kw: events.append(
            ("chunk", kw["chunk_idx"])
        )
        mgr.send_kv_status_message.side_effect = lambda **kw: events.append(
            ("status", kw["status"])
        )
        self._run_staging_worker(mgr, [first, last])
        self.assertEqual(attempts, [0, 1, 0])
        self.assertEqual(
            events, [("chunk", 1), ("status", KVPoll.Success), ("chunk", 0)]
        )
        self.assertNotIn(room, mgr.transfer_infos)
        self.assertNotIn(room, mgr._staging_outstanding)


class TestNixlDecodeListener(CustomTestCase):
    def test_decode_listener_receives_zmq_chunk_and_success(self):
        mgr = object.__new__(NixlKVManager)
        mgr.request_status = {}
        mgr.failure_records = {}
        mgr.failure_lock = threading.Lock()
        mgr.prefill_response_tracker = {}
        mgr.required_prefill_response_num_table = {11: 1}
        mgr.addr_to_rooms_tracker = defaultdict(set)
        mgr._deferred_ack_targets = {}
        mgr._deferred_ack_poisoned_rooms = set()
        mgr._staging_outstanding = {}
        mgr.enable_staging = False
        mgr._staging_handler = MagicMock()
        receiver = NixlKVReceiver(mgr, "prefill:8998", 11)
        mgr.update_status(11, KVPoll.WaitingForInput)
        mgr.server_socket = MagicMock()
        mgr.server_socket.recv_multipart.side_effect = [
            [b"CHUNK_READY", b"11", b"0", b"0", b"2", b"decode_agent", b"0"],
            [b"KV_STATUS", b"11", str(int(KVPoll.Success)).encode(), b"0", b""],
            SystemExit(),
        ]
        with patch("sglang.srt.disaggregation.nixl.conn.threading.Thread") as thread:
            mgr._start_decode_listener_thread()
            with self.assertRaises(SystemExit):
                thread.call_args.kwargs["target"]()
        mgr._staging_handler.handle_chunk_arrived.assert_called_once_with(
            11, 0, 0, 2, "decode_agent"
        )
        self.assertEqual(receiver.poll(), KVPoll.Success)


class TestNixlNodeFailure(CustomTestCase):
    def _make_manager(self):
        mgr = object.__new__(NixlKVManager)
        mgr.connection_lock = threading.Lock()
        # Connection keys are "{addr}_{dp_rank}_{cp_rank}_{tp_rank}".
        mgr.connection_pool = {
            "10.0.0.1:8998_0_0_0": [{"rank_ip": "10.0.0.1"}],
            "10.0.0.1:8998_0_0_1": [{"rank_ip": "10.0.0.1"}],
            "10.0.0.2:8998_0_0_0": [{"rank_ip": "10.0.0.2"}],
        }
        mgr.prefill_info_table = {
            "10.0.0.1:8998": object(),
            "10.0.0.2:8998": object(),
        }
        mgr.addr_to_rooms_tracker = defaultdict(set)
        mgr.addr_to_rooms_tracker["10.0.0.1:8998"] = {3, 4, 5}
        mgr.heartbeat_failures = {"10.0.0.1:8998": 2, "10.0.0.2:8998": 1}
        mgr.request_status = {
            3: KVPoll.WaitingForInput,
            4: KVPoll.Transferring,
            5: KVPoll.Success,
        }
        mgr.failure_records = {}
        mgr.failure_lock = threading.Lock()
        mgr.update_status = CommonKVManager.update_status.__get__(mgr, CommonKVManager)
        mgr.check_status = CommonKVManager.check_status.__get__(mgr, CommonKVManager)
        mgr.record_failure = CommonKVManager.record_failure.__get__(
            mgr, CommonKVManager
        )
        return mgr

    def test_handle_node_failure_removes_connections_and_marks_pending_rooms(self):
        mgr = self._make_manager()

        mgr._handle_node_failure("10.0.0.1:8998")

        self.assertNotIn("10.0.0.1:8998_0_0_0", mgr.connection_pool)
        self.assertNotIn("10.0.0.1:8998_0_0_1", mgr.connection_pool)
        self.assertIn("10.0.0.2:8998_0_0_0", mgr.connection_pool)
        self.assertNotIn("10.0.0.1:8998", mgr.prefill_info_table)
        self.assertNotIn("10.0.0.1:8998", mgr.addr_to_rooms_tracker)
        self.assertEqual(mgr.request_status[3], KVPoll.Failed)
        self.assertEqual(mgr.request_status[4], KVPoll.Failed)
        self.assertEqual(mgr.request_status[5], KVPoll.Success)
        self.assertIn(3, mgr.failure_records)
        self.assertIn(4, mgr.failure_records)
        self.assertNotIn(5, mgr.failure_records)
        self.assertEqual(mgr.heartbeat_failures, {"10.0.0.2:8998": 1})

    def test_a_prefill_reached_again_after_a_loss_needs_its_misses_again(self):
        """A prefill dropped after max_failures missed heartbeats, then reached
        again by a new request, is dropped again only after as many: one
        missed heartbeat of a prefill busy for a few seconds is no loss."""
        mgr = self._make_manager()
        addr = "10.0.0.1:8998"
        mgr.prefill_info_table = {addr: object()}
        mgr.heartbeat_interval = 0
        mgr.max_failures = 2
        mgr.heartbeat_failures = {}
        mgr.session_pool_lock = threading.Lock()
        probes = ["miss", "miss", "miss", 200, "miss", 200]
        ok = SimpleNamespace(status_code=200)

        def get(*_args, **_kwargs):
            if probes.pop(0) == "miss":
                raise requests.Timeout("prefill busy")
            return ok

        mgr.session_pool = defaultdict(lambda: SimpleNamespace(get=get))
        rounds = iter(range(len(probes)))
        mgr._heartbeat_shutdown = SimpleNamespace(
            is_set=lambda: False, wait=lambda _s: next(rounds, None) is None
        )
        losses = []
        handle = mgr._handle_node_failure

        def lost(failed):
            losses.append(failed)
            handle(failed)
            # The next request to it registers it again.
            mgr.prefill_info_table[failed] = object()

        mgr._handle_node_failure = lost
        mgr._start_heartbeat_checker_thread().join(10)
        self.assertEqual(probes, [])
        self.assertEqual(losses, [addr])


class TestNixlStaging(CustomTestCase):
    def _make_manager(self, agent=None):
        mgr = object.__new__(NixlKVManager)
        mgr.agent = agent or StagingFakeAgent()
        mgr.attn_tp_size = 2
        mgr.is_mla_backend = False
        mgr.kv_args = SimpleNamespace(
            gpu_id=1,
            engine_rank=1,
            page_size=2,
            total_kv_head_num=2,
            kv_head_num=1,
        )
        mgr.server_args = SimpleNamespace(chunked_prefill_size=4)
        return mgr

    def test_do_staging_transfer_requeues_when_allocation_not_ready(self):
        mgr = self._make_manager()
        mgr._staging_ctx = PrefillStagingContext()
        strategy = MagicMock()
        strategy.check_ready.return_value = (False, 0, -1, 0, -1)
        kv_chunk = TransferKVChunk(
            room=3,
            prefill_kv_indices=np.array([10, 11], dtype=np.int32),
            index_slice=slice(0, 2),
            is_last_chunk=False,
            prefill_aux_index=None,
            state_indices=None,
        )
        req = SimpleNamespace(room=3, agent_name="decode_agent")
        queue = FakeQueue()

        with patch.dict(
            sys.modules,
            {
                "sglang.srt.disaggregation.common.staging_buffer": (
                    _fake_staging_buffer_module()
                )
            },
        ):
            handle, deferred, chunk_idx = mgr._do_staging_transfer(
                strategy,
                kv_chunk,
                kv_chunk.prefill_kv_indices,
                req,
                SimpleNamespace(),
                queue,
            )

        self.assertIsNone(handle)
        self.assertTrue(deferred)
        self.assertEqual(queue.items, [kv_chunk])

    def test_do_staging_transfer_raises_for_oversized_allocation(self):
        mgr = self._make_manager()
        strategy = MagicMock()
        strategy.check_ready.return_value = (
            False,
            0,
            FakeStagingAllocator.ALLOC_OVERSIZED,
            0,
            -1,
        )
        kv_chunk = TransferKVChunk(
            room=3,
            prefill_kv_indices=np.array([10], dtype=np.int32),
            index_slice=slice(0, 1),
            is_last_chunk=False,
            prefill_aux_index=None,
            state_indices=None,
        )

        with self.assertRaisesRegex(RuntimeError, "ring buffer total size"):
            with patch.dict(
                sys.modules,
                {
                    "sglang.srt.disaggregation.common.staging_buffer": (
                        _fake_staging_buffer_module()
                    )
                },
            ):
                mgr._do_staging_transfer(
                    strategy,
                    kv_chunk,
                    kv_chunk.prefill_kv_indices,
                    SimpleNamespace(room=3, agent_name="decode_agent"),
                    SimpleNamespace(),
                    FakeQueue(),
                )


class DlistCaptureAgent:
    """Records prep_xfer_dlist descriptor arrays so tests can inspect them."""

    def __init__(self):
        self.calls = []  # (peer_name, np.ndarray, mem_kind)

    def prep_xfer_dlist(self, peer_name, array, mem_kind):
        self.calls.append((peer_name, np.asarray(array), mem_kind))
        return f"handle_{len(self.calls)}"


class TestNixlHeteroTpReplicatedKV(CustomTestCase):
    """Regression guard for #31295.

    Prefill attention-TP1 -> decode TP4 on a model with only 2 KV heads forces
    GQA replication: decode ranks 0,1 share KV head 0 and ranks 2,3 share KV
    head 1. The shared source dlist must interleave one group per *unique*
    source head-slice (2), and each peer's head_group_idx must map replicated
    decode ranks via integer division (0,0,1,1). The pre-fix code used
    ``num_groups = decode_tp // prefill_tp`` (=4) -- addressing 2x past the
    registered source region, which NIXL rejects with NIXL_ERR_NOT_FOUND -- and
    a modulo head map (0,1,0,1).
    """

    TOTAL_KV_HEADS = 2
    DECODE_TP = 4
    PAGE_SIZE = 1
    BYTES_PER_HEAD = 128  # per token, per head slice
    SRC_KV_ITEM_LEN = TOTAL_KV_HEADS * BYTES_PER_HEAD  # both heads on one prefill rank
    DST_KV_ITEM_LEN = BYTES_PER_HEAD  # one replicated head per decode rank
    NUM_SLOTS = 4
    SRC_PTRS = [0x10000, 0x20000]  # K, V for the single local layer
    REGION_LEN = NUM_SLOTS * SRC_KV_ITEM_LEN

    def _make_manager(self):
        mgr = object.__new__(NixlKVManager)
        mgr.agent = DlistCaptureAgent()
        mgr.attn_tp_size = 1  # prefill attention TP = 1 (DP attention)
        mgr.prep_handle_slice_src = None
        mgr.prep_handles_slice_dst = {}
        mgr.kv_args = SimpleNamespace(
            gpu_id=0,
            engine_rank=0,
            page_size=self.PAGE_SIZE,
            prefill_start_layer=0,
            total_kv_head_num=self.TOTAL_KV_HEADS,
            kv_head_num=self.TOTAL_KV_HEADS,
            kv_item_lens=[self.SRC_KV_ITEM_LEN, self.SRC_KV_ITEM_LEN],
            kv_data_ptrs=list(self.SRC_PTRS),
            kv_data_lens=[self.REGION_LEN, self.REGION_LEN],
        )
        return mgr

    def _decode_args(self, decode_tp_rank):
        return SimpleNamespace(
            agent_name=f"decode_{decode_tp_rank}",
            decode_tp_size=self.DECODE_TP,
            decode_tp_rank=decode_tp_rank,
            dst_kv_item_len=self.DST_KV_ITEM_LEN,
            dst_kv_ptrs=[0x30000, 0x40000],
            dst_num_slots=self.NUM_SLOTS,
            gpu_id=0,
        )

    def test_src_dlist_stays_within_registered_region_and_num_groups(self):
        # Src dlist is built once (shared across peers) on the first call.
        mgr = self._make_manager()
        mgr._init_hetero_tp_prep_handle(
            peer_name="decode_0", decode_kv_args=self._decode_args(0)
        )

        # num_groups must be 2 (one per unique KV head), not decode_tp//prefill_tp=4.
        src_handle, num_groups, _num_ptr_pairs, _num_slots = mgr.prep_handle_slice_src
        self.assertEqual(num_groups, 2)

        # Every source run must lie inside a registered base region
        # [ptr, ptr+REGION_LEN), gaps between its blocks included. Pre-fix,
        # num_groups=4 pushed the top group past the region -> NIXL_ERR_NOT_FOUND.
        src_call = next(c for c in mgr.agent.calls if c[0] == "")
        src_array = src_call[1]
        regions = [(p, p + self.REGION_LEN) for p in self.SRC_PTRS]
        for addr, length, _dev, stride, count in src_array:
            addr = int(addr)
            end = addr + (int(count) - 1) * int(stride) + int(length)
            self.assertTrue(
                any(lo <= addr and end <= hi for lo, hi in regions),
                f"run [{addr:#x}, {end:#x}) escapes all registered source "
                f"regions {[(hex(lo), hex(hi)) for lo, hi in regions]}",
            )

    def test_head_group_idx_maps_replicated_ranks_by_integer_division(self):
        # Each decode rank's per-peer dst handle records its head_group_idx.
        # Expected replicated-KV mapping: ranks 0,1 -> group 0; ranks 2,3 -> group 1.
        expected = {0: 0, 1: 0, 2: 1, 3: 1}
        for rank in range(self.DECODE_TP):
            mgr = self._make_manager()
            mgr._init_hetero_tp_prep_handle(
                peer_name=f"decode_{rank}", decode_kv_args=self._decode_args(rank)
            )
            _dst_handle, _num_slots_dst, head_group_idx = mgr.prep_handles_slice_dst[
                f"decode_{rank}"
            ]
            self.assertEqual(
                head_group_idx,
                expected[rank],
                f"decode rank {rank} mapped to group {head_group_idx}, "
                f"expected {expected[rank]} (modulo bug gives 0,1,0,1)",
            )


class ReloadFakeAgent:
    """NIXL remote metadata as on the NIXL 1.3.0 CI floor: each load starts a
    new generation, and an old handle's failure unloads the peer by name."""

    def __init__(self):
        self.sections = {}
        self.generations = 0
        self.fail_prep_for = None

    def add_remote_agent(self, metadata):
        name = metadata.decode("ascii")
        self.generations += 1
        self.sections[name] = self.generations
        return name

    def remove_remote_agent(self, name):
        del self.sections[name]

    def check_remote_metadata(self, name):
        return name in self.sections

    def prep_xfer_dlist(self, name, descs, mem_kind):
        if name and (name not in self.sections or name == self.fail_prep_for):
            raise RuntimeError("NIXL_ERR_NOT_FOUND")
        return name, self.sections.get(name)

    def make_prepped_xfer(self, op, src, src_indices, dst, dst_indices, notif=b""):
        name, generation = dst
        if generation != self.sections.get(name):
            raise RuntimeError("NIXL_ERR_NOT_FOUND")
        return name, generation

    def transfer(self, handle):
        return "PROC"

    def check_xfer_state(self, handle):
        name, generation = handle
        if name not in self.sections:
            raise RuntimeError("NIXL_ERR_NOT_FOUND")
        if generation != self.sections[name]:
            del self.sections[name]
            raise RuntimeError("NIXL_ERR_REMOTE_DISCONNECT")
        return "DONE"


class TestNixlInvalidatedPeerReload(CustomTestCase):
    def _make_manager(self, peers):
        mgr = object.__new__(NixlKVManager)
        mgr.agent = ReloadFakeAgent()
        mgr.disaggregation_mode = DisaggregationMode.PREFILL
        mgr.is_mla_backend = False
        mgr.is_hybrid_mla_backend = False
        mgr.attn_tp_size = 1
        mgr.pp_size = 1
        mgr.src_mem_kind = "VRAM"
        mgr._num_slots_src = 4
        mgr.prep_handles = {}
        mgr.prep_handles_slice_dst = {}
        mgr._prepared_state_plans = {}
        mgr._prepared_state_components = frozenset()
        mgr._state_layout_rejections = {}
        mgr._peer_reload_lock = threading.Lock()
        mgr._peer_reload_times = {}
        mgr.kv_args = SimpleNamespace(
            gpu_id=0,
            engine_rank=0,
            num_draft_entries=0,
            kv_layer_ids=[],
            kv_data_ptrs=[0x1000, 0x2000],
            kv_data_lens=[64, 64],
            kv_item_lens=[16, 16],
        )
        mgr.decode_kv_args_table = {}
        for name in peers:
            info = KVArgsRegisterInfo(
                room="None",
                endpoint="127.0.0.1",
                dst_port=5555,
                agent_name=name,
                agent_metadata=name.encode("ascii"),
                dst_kv_ptrs=[0x3000, 0x4000],
                dst_kv_mem_kinds=["VRAM", "VRAM"],
                dst_aux_ptrs=[0],
                dst_state_data_ptrs=[],
                gpu_id=0,
                decode_tp_size=1,
                decode_tp_rank=0,
                dst_kv_item_len=16,
                dst_kv_item_lens=[16, 16],
                dst_num_slots=4,
            )
            mgr.decode_kv_args_table[name] = info
            mgr.agent.add_remote_agent(info.agent_metadata)
            mgr._prepare_payload_xfer(info)
        return mgr

    def test_failed_room_reloads_the_invalidated_peer_for_later_rooms(self):
        """A peer invalidated mid-room must serve later rooms again, even after
        the failed room's posted handles fail and the room is cleared."""
        peers = ["healthy", "dropped"]
        mgr = self._make_manager(peers)
        healthy_dlist = mgr.prep_handles["healthy"]
        mgr.request_status = {}
        mgr.transfer_infos = {}
        mgr.req_to_decode_prefix_len = {}
        for room in (31, 32):
            mgr.request_status[room] = KVPoll.WaitingForInput
            mgr.req_to_decode_prefix_len[room] = 0
            mgr.transfer_infos[room] = {
                name: TransferInfo(
                    room=room,
                    endpoint="127.0.0.1",
                    dst_port=5555 + i,
                    agent_name=name,
                    dst_kv_indices=np.array([2], dtype=np.int32),
                    dst_aux_index=0,
                    required_dst_info_num=len(peers),
                    dst_state_indices=[],
                )
                for i, name in enumerate(peers)
            }
        mgr.enable_staging = False
        mgr.enable_deferred_decode_kv_release = False
        mgr._staging_ctx = None
        mgr._staging_outstanding = defaultdict(int)
        mgr._deferred_ack_targets = {}
        mgr._deferred_ack_poisoned_rooms = set()
        mgr.failure_lock = threading.Lock()
        mgr.failure_records = {}
        mgr.send_kv_status_message = MagicMock()
        # A remote disconnect while posting dropped's aux unloads the peer and
        # leaves its KV handle posted.
        disconnect = {"dropped"}

        def send_aux(peer, *args):
            if peer in disconnect:
                disconnect.discard(peer)
                mgr.agent.remove_remote_agent(peer)
                raise RuntimeError("NIXL_ERR_REMOTE_DISCONNECT")
            return peer, mgr.agent.sections[peer]

        mgr.send_aux = send_aux
        # The sender clears a room as soon as it sees it failed.
        update_status = mgr.update_status

        def update_status_and_clear(room, status):
            update_status(room, status)
            if status == KVPoll.Failed:
                mgr.transfer_infos.pop(room, None)

        mgr.update_status = update_status_and_clear
        chunks = [
            TransferKVChunk(
                room=room,
                prefill_kv_indices=np.array([1], dtype=np.int32),
                index_slice=slice(0, 1),
                is_last_chunk=True,
                prefill_aux_index=0,
                state_indices=None,
            )
            for room in (31, 32)
        ]
        queue = SimpleNamespace(get=MagicMock(side_effect=[*chunks, SystemExit()]))

        with self.assertRaises(SystemExit):
            mgr.transfer_worker(queue)

        self.assertEqual(mgr.request_status[31], KVPoll.Failed)
        self.assertIn(31, mgr.failure_records)
        self.assertEqual(mgr.request_status[32], KVPoll.Success)
        self.assertIs(mgr.prep_handles["healthy"], healthy_dlist)

    def test_failed_rebuild_leaves_the_peer_to_a_later_reload(self):
        """A reload whose dlists cannot be rebuilt must leave the peer invalid,
        so a later failure past the interval reloads it."""
        mgr = self._make_manager(["dropped"])
        mgr.agent.remove_remote_agent("dropped")
        for now, prep_fails, reloaded in (
            (100.0, True, False),
            (100.5, False, False),
            (101.5, False, True),
        ):
            mgr.agent.fail_prep_for = "dropped" if prep_fails else None
            with patch(
                "sglang.srt.disaggregation.nixl.conn.time.monotonic",
                return_value=now,
            ):
                mgr._reload_invalidated_peers(dict.fromkeys(["dropped"]))
            self.assertEqual(mgr.agent.check_remote_metadata("dropped"), reloaded, now)

        handle = mgr.send_kvcache(
            "dropped",
            np.array([1], dtype=np.int32),
            mgr.decode_kv_args_table["dropped"].dst_kv_ptrs,
            np.array([2], dtype=np.int32),
            0,
        )
        self.assertEqual(handle, ("dropped", mgr.agent.sections["dropped"]))


if __name__ == "__main__":
    unittest.main()
