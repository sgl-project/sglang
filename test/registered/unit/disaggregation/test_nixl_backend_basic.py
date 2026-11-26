"""Basic CPU unit tests for NIXL disaggregation control paths."""

import json
import os
import sys
import threading
import types
import unittest
from collections import defaultdict
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import requests

from sglang.srt.disaggregation.base.conn import KVPoll, StateType
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.common.staging_handler import PrefillStagingContext
from sglang.srt.disaggregation.common.utils import TransferKVChunk
from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.disaggregation.nixl.conn import (
    KVArgsRegisterInfo,
    NixlKVManager,
    NixlKVReceiver,
    NixlKVSender,
    TransferInfo,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


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
        mgr.customized_info_enabled = False
        mgr.exchange_aux_item_lens = False
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
                unified_layout_error=None,
                decode_tp_size=1,
                dst_kv_ptrs=[0],
                dst_aux_ptrs=[0],
                dst_aux_item_lens=None,
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
        mgr.enable_kv_packing = False
        mgr._kv_packers = []
        mgr.enable_deferred_decode_kv_release = False
        mgr._staging_ctx = None
        mgr._staging_outstanding = defaultdict(int)
        mgr.is_mla_backend = False
        mgr.is_hybrid_mla_backend = False
        mgr.attn_tp_size = 1
        mgr.kv_args = SimpleNamespace(
            engine_rank=0,
            kv_data_ptrs=[0],
            num_draft_entries=0,
            gpu_id=3,
            aux_data_ptrs=[0],
            aux_item_lens=[64],
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

    @staticmethod
    def _record_device_bind(order):
        """Patch the worker's device lookup so set_device records into `order`."""
        device_module = MagicMock()
        device_module.set_device.side_effect = lambda gpu_id: order.append(
            f"bind:{gpu_id}"
        )
        return (
            patch(
                "sglang.srt.disaggregation.nixl.conn.get_device",
                return_value=SimpleNamespace(device="cuda"),
            ),
            patch(
                "sglang.srt.disaggregation.nixl.conn.torch.get_device_module",
                return_value=device_module,
            ),
        )

    def test_cached_layout_rejection_blocks_aux_only_transfer(self):
        mgr = self._make_manager(34)
        mgr.decode_kv_args_table["agent"].unified_layout_error = "layout mismatch"
        mgr.agent.check_remote_metadata = MagicMock(return_value=False)
        mgr.agent.add_remote_agent = MagicMock()
        mgr.send_kvcache = MagicMock()
        mgr.maybe_send_extra = MagicMock()
        mgr.send_aux = MagicMock()
        chunk = self._make_chunk(34, [], is_last_chunk=True)

        self._run_worker_once(mgr, chunk)

        self.assertEqual(mgr.check_status(34), KVPoll.Failed)
        mgr.send_kvcache.assert_not_called()
        mgr.maybe_send_extra.assert_not_called()
        mgr.send_aux.assert_not_called()
        self.assertIn("layout mismatch", mgr.failure_records[34])
        mgr.agent.check_remote_metadata.assert_not_called()
        mgr.agent.add_remote_agent.assert_not_called()

    def test_discarded_chunks_complete_without_transfers(self):
        for status in (KVPoll.Failed, KVPoll.Success):
            with self.subTest(status=status):
                mgr = self._make_manager(35)
                mgr.request_status[35] = status
                if status == KVPoll.Success:
                    mgr.transfer_infos.clear()
                chunk = self._make_chunk(35, [1], is_last_chunk=False)
                chunk.send_record = mgr.send_record(35)
                mgr.note_send_enqueued(35)
                mgr.send_kvcache = MagicMock()
                if status == KVPoll.Failed:
                    wait_event = chunk.wait_event = MagicMock()
                    indices_event = chunk.indices_event = MagicMock()
                self._run_worker_once(mgr, chunk)
                self.assertEqual(mgr.send_counts(35), (1, 1))
                mgr.send_kvcache.assert_not_called()
                if status == KVPoll.Failed:
                    wait_event.synchronize.assert_not_called()
                    indices_event.synchronize.assert_not_called()
                    mgr.disaggregation_mode = DisaggregationMode.PREFILL
                    mgr.transfer_queues = [FakeQueue()]
                    mgr.add_transfer_request(35, np.array([1]), slice(0, 1), False)
                    self.assertEqual(mgr.transfer_queues[0].items, [])
                    self.assertEqual(mgr.send_counts(35), (1, 1))

    def test_abort_during_forward_wait_skips_indices_wait(self):
        mgr = self._make_manager(36)
        chunk = self._make_chunk(36, [1], is_last_chunk=False)
        chunk.send_record = mgr.send_record(36)
        mgr.note_send_enqueued(36)
        chunk.wait_event = MagicMock(
            synchronize=lambda: mgr.update_status(36, KVPoll.Failed)
        )
        event = chunk.indices_event = MagicMock()
        mgr.send_kvcache = MagicMock()
        device, device_module = self._record_device_bind([])
        with device, device_module:
            self._run_worker_once(mgr, chunk)
        event.synchronize.assert_not_called()
        mgr.send_kvcache.assert_not_called()
        self.assertEqual(mgr.send_counts(36), (1, 1))

    def test_incompatible_aux_layout_fails_before_transfer_and_worker_continues(self):
        context = get_context().override_server_args(
            disaggregation_enable_kv_checksum=False
        )
        context.install()
        self.addCleanup(context.restore)
        mgr = self._make_manager(21)
        mgr.customized_info_enabled = True
        mgr.exchange_aux_item_lens = True
        healthy = self._make_manager(22)
        mgr.kv_args.aux_data_ptrs = [0] * 8
        mgr.kv_args.aux_item_lens = [64] * 6 + [4096, 64]
        for manager, capacity in ((mgr, 2048), (healthy, 4096)):
            peer = manager.decode_kv_args_table["agent"]
            peer.dst_aux_ptrs = [0] * 8
            peer.dst_aux_item_lens = [64] * 6 + [capacity, 64]
        healthy_req = healthy.transfer_infos[22]["agent"]
        healthy_req.agent_name = "healthy"
        mgr.transfer_infos[22] = {"healthy": healthy_req}
        mgr.request_status[22] = KVPoll.WaitingForInput
        mgr.decode_kv_args_table["healthy"] = healthy.decode_kv_args_table["agent"]
        mgr.agent.check_xfer_state = MagicMock(return_value="DONE")
        mgr.agent.check_remote_metadata = MagicMock(return_value=True)
        mgr.send_kv_status_message = MagicMock()
        mgr.send_kvcache = MagicMock(return_value="kv_handle")
        mgr.send_aux = MagicMock(return_value="aux_handle")
        chunks = [self._make_chunk(room, [1], True) for room in (21, 22)]
        queue = SimpleNamespace(get=MagicMock(side_effect=[*chunks, SystemExit()]))

        with self.assertRaises(SystemExit):
            mgr.transfer_worker(queue)

        self.assertEqual(mgr.request_status[21], KVPoll.Failed)
        self.assertIn("metadata buffer layout mismatch", mgr.failure_records[21])
        self.assertEqual(mgr.request_status[22], KVPoll.Success)
        self.assertNotIn(22, mgr.failure_records)
        mgr.send_kvcache.assert_called_once()
        self.assertEqual(mgr.send_kvcache.call_args.args[0], "healthy")
        mgr.send_aux.assert_called_once()
        self.assertEqual(mgr.send_aux.call_args.args[0], "healthy")
        self.assertEqual(
            [
                (
                    call.kwargs["bootstrap_room"],
                    call.kwargs["status"],
                    call.kwargs["failure_reason"],
                )
                for call in mgr.send_kv_status_message.call_args_list
            ],
            [
                (21, KVPoll.Failed, mgr.failure_records[21]),
                (22, KVPoll.Success, None),
            ],
        )
        self.assertEqual(mgr._staging_outstanding.get(21, 0), 0)

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

    @staticmethod
    def _make_cp_tp_prefill(mgr, room, *, peer_only, n=4):
        """Prefill TP rank 1 under CP-TP group sharing: attention TP 1, and
        the CP, KV-shard and GDN widths all ``n``."""
        mgr.kv_shard_state_peer_only = peer_only
        mgr.attn_tp_size = 1
        mgr.attn_cp_size = mgr.kv_shard_size = mgr.state_tp_size = n
        mgr.kv_args.engine_rank = 1
        mgr.transfer_infos[room]["agent"].required_dst_info_num = n
        mgr.decode_kv_args_table["agent"].decode_tp_size = n

    def test_kv_sharded_completion_waits_for_handles(self):
        for peer_only, decode_tp_rank, is_peer in (
            (True, 1, True),
            (True, 2, False),
            (True, 5, True),
            (True, 6, False),
            (False, 2, True),
        ):
            # A non-peer sends no state or aux, so it has no handle to fail.
            for final_state in ("DONE", "ERR") if is_peer else ("DONE",):
                with self.subTest(
                    peer_only=peer_only,
                    decode_tp_rank=decode_tp_rank,
                    final_state=final_state,
                ):
                    room = 24
                    mgr = self._make_manager(room)
                    self._make_cp_tp_prefill(mgr, room, peer_only=peer_only)
                    mgr.transfer_infos[room]["agent"].dst_state_indices = [[2]]
                    dst_info = mgr.decode_kv_args_table["agent"]
                    dst_info.decode_tp_rank = decode_tp_rank
                    dst_info.dst_state_data_ptrs = [[0]]
                    dst_info.dst_state_item_lens = [[4]]
                    dst_info.dst_state_dim_per_tensor = [[1]]
                    dst_info.dst_state_layer_ids = [0]
                    mgr.maybe_send_extra = MagicMock(return_value=["state_handle"])
                    mgr.send_aux = MagicMock(return_value="aux_handle")
                    mgr.conclude_transfer = MagicMock(wraps=mgr.conclude_transfer)
                    mgr.conclude_failure = MagicMock(wraps=mgr.conclude_failure)
                    mgr._reload_invalidated_peers = MagicMock()
                    mgr.agent = MagicMock()
                    polls = defaultdict(int)
                    notification_counts_at_poll = []

                    def check_xfer_state(handle):
                        notification_counts_at_poll.append(
                            mgr.send_kv_status_message.call_count
                        )
                        mgr.conclude_transfer.assert_not_called()
                        polls[handle] += 1
                        # State may still be writing after the aux handle fails.
                        if polls[handle] < (3 if handle == "state_handle" else 2):
                            return "PROC"
                        return final_state if handle == "aux_handle" else "DONE"

                    mgr.agent.check_xfer_state.side_effect = check_xfer_state
                    chunk = self._make_chunk(room, [], is_last_chunk=True)
                    chunk.state_indices = [[1]]

                    self._run_worker_once(mgr, chunk)

                    # Only the rank-matched peer sends state and aux.
                    self.assertEqual(mgr.maybe_send_extra.call_count, int(is_peer))
                    self.assertEqual(mgr.send_aux.call_count, int(is_peer))
                    self.assertEqual(
                        set(polls),
                        {"aux_handle", "state_handle"} if is_peer else set(),
                    )
                    self.assertTrue(all(count >= 2 for count in polls.values()))
                    self.assertEqual(
                        notification_counts_at_poll, [0] * sum(polls.values())
                    )
                    expected_status = (
                        KVPoll.Success if final_state == "DONE" else KVPoll.Failed
                    )
                    mgr.conclude_transfer.assert_called_once()
                    self.assertEqual(
                        mgr.conclude_transfer.call_args.kwargs["status"],
                        expected_status,
                    )
                    mgr.send_kv_status_message.assert_called_once_with(
                        targets=[("127.0.0.1", 5555)],
                        bootstrap_room=room,
                        status=expected_status,
                        failure_reason=None
                        if final_state == "DONE"
                        else f"NIXL transfer encountered ERR room={room}",
                    )
                    self.assertEqual(mgr.request_status[room], expected_status)
                    if final_state == "DONE":
                        mgr.conclude_failure.assert_not_called()
                    else:
                        mgr.conclude_failure.assert_called_once()

    def test_kv_sharded_pairing_mismatch_fails_the_room_on_every_rank(self):
        # Prefill rank 1 is not decode rank 0's peer, so it would send nothing
        # but KV pages; it must still fail the room.
        for decode_tp_size, required_dst_info_num, message in (
            (2, 2, "prefill=4 decode=2"),
            (4, 1, "1 of 4 decode ranks"),
        ):
            with self.subTest(
                decode_tp_size=decode_tp_size,
                required_dst_info_num=required_dst_info_num,
            ):
                room = 25
                mgr = self._make_manager(room)
                self._make_cp_tp_prefill(mgr, room, peer_only=True)
                info = mgr.transfer_infos[room]["agent"]
                info.required_dst_info_num = required_dst_info_num
                dst_info = mgr.decode_kv_args_table["agent"]
                dst_info.decode_tp_size = decode_tp_size
                dst_info.decode_tp_rank = 0
                mgr.send_kvcache = MagicMock(return_value="kv_handle")
                mgr.send_aux = MagicMock(return_value="aux_handle")
                mgr.maybe_send_extra = MagicMock(return_value=["state_handle"])
                mgr._reload_invalidated_peers = MagicMock()

                self._run_worker_once(mgr, self._make_chunk(room, [1], True))

                self.assertEqual(mgr.request_status[room], KVPoll.Failed)
                self.assertIn(message, mgr.failure_records[room])
                mgr.send_kv_status_message.assert_called_once()
                self.assertEqual(
                    mgr.send_kv_status_message.call_args.kwargs["status"],
                    KVPoll.Failed,
                )
                mgr.send_kvcache.assert_not_called()
                mgr.send_aux.assert_not_called()
                mgr.maybe_send_extra.assert_not_called()

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
                unified_layout_error=None,
                decode_tp_size=len(agents),
                dst_kv_ptrs=[0x3000 + i * 0x100],
                dst_aux_ptrs=[0],
                dst_aux_item_lens=None,
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
                dst_draft_rows_dcp_collapsed=False,
            )
            for i, (agent, dcp_rank) in enumerate(zip(agents, dcp_ranks))
        }
        mgr.kv_args = SimpleNamespace(
            engine_rank=0,
            kv_data_ptrs=[0x1000],
            page_size=4,
            num_draft_entries=0,
            aux_data_ptrs=[0],
            aux_item_lens=[64],
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

    def test_packed_kv_from_this_worker_settles_before_success(self):
        """Prefill TP < decode TP with packing on: the worker posts through its own packer, then settles."""
        room = 26
        mgr = self._make_manager(room)
        mgr.decode_kv_args_table["agent"].decode_tp_size = 2
        mgr.enable_kv_packing = True
        mgr._kv_packers = ["packer0", "packer1"]
        events = []

        def send_packed(*, packer, peer_name, src_pages, dst_pages, handles):
            events.append(packer)
            handles.append("packed")

        mgr._send_packed_kvcache = MagicMock(side_effect=send_packed)
        mgr.send_kvcache_slice = MagicMock(side_effect=AssertionError("slice path"))
        mgr.send_aux = MagicMock(return_value="aux")
        mgr.agent.check_xfer_state = lambda handle: events.append(handle) or "DONE"
        mgr.send_kv_status_message.side_effect = lambda **kwargs: events.append(
            kwargs["status"]
        )
        queue = SimpleNamespace(
            get=MagicMock(
                side_effect=[
                    self._make_chunk(room, [1], is_last_chunk=True),
                    SystemExit(),
                ]
            )
        )
        with self.assertRaises(SystemExit):
            mgr.transfer_worker(queue, worker_index=1)

        self.assertEqual(events, ["packer1", "packed", "aux", KVPoll.Success])

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

    def _run_staging_worker(self, mgr, chunks, staging_buffer=None):
        pending = list(chunks)

        def get():
            if not pending:
                raise SystemExit()
            return pending.pop(0)

        queue = SimpleNamespace(get=get, put=pending.append)
        with self.assertRaises(SystemExit):
            mgr.transfer_worker(
                queue,
                staging_buffer=(
                    staging_buffer if staging_buffer is not None else object()
                ),
            )

    def _two_staged_peers(self, room):
        mgr = self._make_staging_manager(room)
        requests = mgr.transfer_infos[room]
        requests["other"] = TransferInfo(
            **{**vars(requests["agent"]), "agent_name": "other", "dst_port": 5556}
        )
        peer = mgr.decode_kv_args_table["agent"]
        peer.decode_tp_rank = 0
        peer.staging_total_size = 4096
        peer.dst_kv_item_len = 4
        mgr.decode_kv_args_table["other"] = SimpleNamespace(
            **{**vars(peer), "decode_tp_rank": 1, "staging_base_ptr": 0xA000}
        )
        return mgr

    def test_staged_fanout_reads_each_head_from_its_own_source_until_done(self):
        import torch

        from sglang.srt.disaggregation.common import staging_buffer as staging

        # Two pages of two tokens each. Distinct layer, token, head, K/V
        # values expose an incorrect source offset or a premature slot reuse.
        keys = [
            torch.arange(24, dtype=torch.uint8).reshape(6, 2, 2) + i * 32
            for i in range(2)
        ]
        values = [k + 96 for k in keys]
        pages = [2, 0]
        payload_size = 32
        for buf_size in (4096, payload_size):
            overlap = buf_size > payload_size
            with self.subTest(buf_size=buf_size):
                mgr = self._two_staged_peers(27)
                mgr.kv_args.total_kv_head_num = 2
                mgr.kv_args.page_size = 2
                mgr.kv_buffer_tensors = dict(
                    k_buffers=keys, v_buffers=values, page_size=2
                )
                # Only device allocation and stream operations are mocked. Use
                # the torch gather, byte views and NIXL descriptors from production.
                buf = object.__new__(staging.StagingBuffer)
                buf.buffer = torch.empty(buf_size, dtype=torch.uint8)
                buf.device, buf.device_module = buf.buffer.device, torch.cpu
                buf.gpu_id, buf.data_ptr, buf.size_bytes = (
                    0,
                    buf.buffer.data_ptr(),
                    buf_size,
                )
                buf._gather_stream = MagicMock()
                strategy = MagicMock(staging_buffer=buf)

                strategy.check_ready.return_value = (
                    True,
                    0,
                    128,
                    0,
                    128 + payload_size,
                )
                mgr._try_create_staging_strategy.return_value = strategy
                writes, completed, posted, received = {}, [], [], {}
                notifications = []
                pending_polls = []

                def init_xfer(op, src, dst, peer):
                    self.assertEqual(op, "WRITE")
                    writes[peer] = (src, dst)
                    return peer

                def finish(peer):
                    if peer in received:
                        return
                    src, _ = writes[peer]
                    offset = int(src[0, 0]) - buf.get_ptr()
                    count = int(src[0, 1])
                    received[peer] = bytes(buf.buffer[offset : offset + count].tolist())
                    completed.append(peer)

                def state(peer):
                    if peer not in writes:  # auxiliary last-chunk metadata
                        return "DONE"
                    if peer == "agent" and "other" not in completed:
                        # Force the first poll to observe an active WRITE. In the
                        # one-slot case it completes on the next poll. Bound the
                        # delay to make a serial regression fail without hanging.
                        pending_polls.append(peer)
                        if len(pending_polls) <= (3 if overlap else 1):
                            return "PROC"
                    finish(peer)
                    return "DONE"

                mgr.agent = SimpleNamespace(
                    get_xfer_descs=lambda desc, kind: desc,
                    initialize_xfer=init_xfer,
                    transfer=lambda peer: posted.append(peer) or "PROC",
                    check_xfer_state=state,
                )
                mgr.send_chunk_ready.side_effect = lambda **kw: notifications.append(
                    (kw["writer_id"], kw["writer_id"] in received, kw["targets"])
                )
                chunk = self._make_chunk(27, pages, is_last_chunk=True)
                with (
                    patch.object(staging, "_USE_TRITON_STAGING", False),
                    patch.object(
                        staging, "device_stream_context", return_value=nullcontext()
                    ),
                ):
                    self._run_staging_worker(mgr, [chunk], staging_buffer=buf)

                self.assertEqual(posted, ["agent", "other"])
                self.assertEqual(completed, ["other", "agent"] if overlap else posted)
                for rank, peer in enumerate(posted):
                    expected = torch.cat(
                        [t[[4, 5, 0, 1], rank].flatten() for t in keys + values]
                    )
                    self.assertEqual(received[peer], bytes(expected.tolist()))
                    self.assertEqual(
                        int(writes[peer][1][0, 0]),
                        mgr.decode_kv_args_table[peer].staging_base_ptr + 128,
                    )
                self.assertEqual([n[0] for n in notifications], completed)
                self.assertCountEqual(
                    notifications,
                    [
                        ("agent", True, [("127.0.0.1", 5555)]),
                        ("other", True, [("127.0.0.1", 5556)]),
                    ],
                )
                self.assertEqual(
                    [c.args[0] for c in mgr.send_aux.call_args_list], posted
                )
                self.assertEqual(mgr.request_status[27], KVPoll.Success)

    def test_staged_writers_stay_within_the_decode_reservation(self):
        import torch

        from sglang.srt.disaggregation.common import staging_buffer as staging

        mgr = self._two_staged_peers(38)
        mgr.agent = StagingFakeAgent()
        mgr.attn_tp_size = mgr.kv_args.total_kv_head_num = 2
        kv = torch.empty((2, 1, 4), dtype=torch.uint8)
        mgr.kv_buffer_tensors = dict(k_buffers=[kv], v_buffers=[kv], page_size=2)
        req = mgr.transfer_infos[38]["agent"]
        dst = mgr.decode_kv_args_table["agent"]
        dst.decode_tp_size = 1
        chunk = self._make_chunk(38, [0], False)
        source = MagicMock()
        source.fits.return_value = True
        source.get_ptr.return_value = 0x2000
        strategy = MagicMock()

        def transfer(offset, end):
            strategy.check_ready.return_value = (True, 0, offset, 1, end)
            return mgr._do_staging_transfer(
                strategy, chunk, chunk.prefill_kv_indices, req, dst, source
            )

        with patch.object(staging, "gather_all_layers_to_staging") as gather:
            # The pool has 4096 bytes. Even writer 0 must reject a slot
            # containing just its own 16 bytes when the two writers need 32.
            with self.assertRaisesRegex(RuntimeError, "cannot fit chunk"):
                transfer(128, 144)
            for offset, end in ((128, 128), (144, 128), (128, 4097), (-1, 31)):
                with (
                    self.subTest(offset=offset, end=end),
                    self.assertRaises(ValueError),
                ):
                    transfer(offset, end)
            gather.assert_not_called()
            self.assertEqual(mgr.agent.initialize_xfer_calls, [])
            for rank in range(2):
                mgr.kv_args.engine_rank = rank
                handle, deferred, chunk_idx = transfer(128, 160)
                self.assertEqual((handle, deferred, chunk_idx), ("handle", False, 0))
                dst_desc, _ = mgr.agent.get_xfer_descs_calls[-1]
                self.assertEqual(
                    dst_desc.tolist(),
                    [[dst.staging_base_ptr + 128 + rank * 16, 16, dst.gpu_id]],
                )

    def _two_room_fanout(self, room):
        mgr = self._two_staged_peers(room)
        next_room = room + 1
        mgr.transfer_infos[next_room] = {
            name: TransferInfo(**{**vars(req), "room": next_room})
            for name, req in mgr.transfer_infos[room].items()
        }
        mgr.request_status[next_room] = KVPoll.WaitingForInput
        mgr.req_to_decode_prefix_len[next_room] = 0
        mgr._reload_invalidated_peers = MagicMock()
        mgr._staging_source_slice = MagicMock(return_value=(0, 1, 32))
        buffer = MagicMock()
        buffer.get_size.return_value = 4096
        strategy = MagicMock(staging_buffer=buffer)
        strategy.check_ready.return_value = (True, 0, 0, 0, 32)
        mgr._try_create_staging_strategy.return_value = strategy
        return mgr, buffer, strategy

    def test_staged_fanout_notifies_fast_peer_before_gathering_the_next(self):
        room = 37
        mgr, buffer, _ = self._two_room_fanout(room)
        events = []

        def send(peer, *_args, staging_buffer):
            self.assertIs(staging_buffer, buffer)
            events.append(("gather", peer))
            return peer

        mgr.send_kvcache_staged = send
        mgr.send_chunk_ready.side_effect = lambda **kw: events.append(
            ("ready", kw["writer_id"])
        )
        self._run_staging_worker(
            mgr, [self._make_chunk(room, [0], True)], staging_buffer=buffer
        )
        self.assertEqual(
            events,
            [
                ("gather", "agent"),
                ("ready", "agent"),
                ("gather", "other"),
                ("ready", "other"),
            ],
        )
        buffer.view.assert_not_called()
        self.assertEqual(mgr.request_status[room], KVPoll.Success)

    def test_staged_fanout_progresses_across_opposite_per_peer_watermarks(self):
        # Each decode rank has one slot. The second reservation becomes
        # writable only after that rank scatters the first from CHUNK_READY.
        rooms = (30, 31)
        mgr, buffer, strategy = self._two_room_fanout(rooms[0])
        order = {"agent": list(rooms), "other": list(reversed(rooms))}
        delivered = []
        writes = []
        completed_aux = []

        def ready(req, _start, _count, *, session_id):
            if strategy.check_ready.call_count > 16:
                raise AssertionError("staging worker made no progress")
            active = order[session_id]
            return bool(active and active[0] == req.room), 0, 0, 0, 32

        def chunk_ready(*, room, writer_id, **kwargs):
            self.assertEqual(order[writer_id].pop(0), room)
            delivered.append((room, writer_id))

        def send(peer, indices, *args, **kwargs):
            transfer = int(indices[0]), peer
            writes.append(transfer)
            return ("kv", *transfer)

        def state(handle):
            kind, room, peer = handle
            if kind == "aux":
                completed_aux.append((room, peer))
            return "DONE"

        def status(*, bootstrap_room, status, **kwargs):
            self.assertEqual(status, KVPoll.Success)
            for peer in order:
                self.assertIn((bootstrap_room, peer), delivered)
                self.assertIn((bootstrap_room, peer), completed_aux)

        strategy.check_ready.side_effect = ready
        mgr.send_kvcache_staged = send
        mgr.send_chunk_ready.side_effect = chunk_ready
        mgr.send_aux.side_effect = lambda peer, room, *_: ("aux", room, peer)
        mgr.send_kv_status_message.side_effect = status
        mgr.agent.check_xfer_state.side_effect = state
        chunks = [self._make_chunk(room, [room], True) for room in rooms]
        for chunk in chunks:
            chunk.prefill_aux_index = chunk.room
        self._run_staging_worker(mgr, chunks, staging_buffer=buffer)
        expected = [(room, peer) for room in rooms for peer in order]
        self.assertCountEqual(writes, expected)
        self.assertCountEqual(delivered, expected)
        self.assertCountEqual(
            [(c.args[1], c.args[0]) for c in mgr.send_aux.call_args_list], expected
        )
        self.assertEqual(mgr.send_kv_status_message.call_count, len(rooms))
        self.assertTrue(all(mgr.request_status[r] == KVPoll.Success for r in rooms))
        self.assertFalse(mgr._staging_outstanding)

    def test_staged_fanout_sends_last_chunk_aux_when_one_peer_has_no_pages(self):
        room = 33
        mgr, buffer, strategy = self._two_room_fanout(room)
        mgr.transfer_infos[room]["agent"].dst_kv_indices = np.array([], dtype=np.int32)
        mgr.send_kvcache_staged = MagicMock(return_value="kv-other")
        self._run_staging_worker(
            mgr, [self._make_chunk(room, [0], True)], staging_buffer=buffer
        )
        self.assertEqual(
            [c.kwargs["session_id"] for c in strategy.check_ready.call_args_list],
            ["other"],
        )
        mgr.send_kvcache_staged.assert_called_once()
        self.assertEqual(mgr.send_kvcache_staged.call_args.args[0], "other")
        self.assertEqual(mgr.send_chunk_ready.call_args.kwargs["writer_id"], "other")
        self.assertCountEqual(
            [c.args[0] for c in mgr.send_aux.call_args_list], ["agent", "other"]
        )
        self.assertEqual(mgr.request_status[room], KVPoll.Success)

    def test_staged_fanout_rejects_invalid_peer_with_pending_reservation(self):
        mgr, buffer, strategy = self._two_room_fanout(32)
        mgr.decode_kv_args_table["agent"].unified_layout_error = "rejected peer layout"

        def ready(req, *args, **kwargs):
            if strategy.check_ready.call_count > 4:
                raise AssertionError("worker kept waiting for incompatible peer")
            return req.agent_name != "agent", 0, -1, 0, 32

        strategy.check_ready.side_effect = ready
        mgr.send_kvcache_staged = MagicMock()
        self._run_staging_worker(
            mgr, [self._make_chunk(32, [0], True)], staging_buffer=buffer
        )
        self.assertEqual(mgr.failure_records[32], "rejected peer layout")
        mgr.send_kvcache_staged.assert_not_called()
        mgr.send_aux.assert_not_called()

    @staticmethod
    def _cpu_staging_source():
        # Backing memory is shared by views of this source, independent across
        # allocations. The POST fake may inspect these bytes much later.
        raw = memoryview(bytearray(512))
        return SimpleNamespace(
            raw=raw,
            get_size=lambda: len(raw),
            view=lambda offset, size: SimpleNamespace(raw=raw[offset : offset + size]),
        )

    def test_staged_fanout_keeps_source_after_ambiguous_post(self):
        from sglang.srt.disaggregation.nixl.conn import _NixlPostError

        room, next_room = 34, 35
        for deferred_release in (False, True):
            with self.subTest(deferred_release=deferred_release):
                mgr, _, _ = self._two_room_fanout(room)
                mgr.enable_deferred_decode_kv_release = deferred_release
                mgr.poison_deferred_ack_room = MagicMock()
                mgr._maybe_ack_drained_abort = MagicMock()
                source = self._cpu_staging_source()
                spare = self._cpu_staging_source()
                mgr._staging_ctx.buffers = [source]
                mgr._allocate_staging_buffers = MagicMock(return_value=[spare])
                posted = {}
                completed = []

                def send(peer, indices, *_args, staging_buffer):
                    r = int(indices[0])
                    payload = bytes([r, int(peer == "other")]) * 16
                    window = staging_buffer.raw[: len(payload)]
                    window[:] = payload
                    posted[r, peer] = (window, payload)
                    if r == room and peer == "other":
                        raise _NixlPostError("partially posted", (r, peer))
                    return r, peer

                def state(handle):
                    if handle == (room, "other"):
                        return "ERR"
                    if handle == (room, "agent") and (room, "other") not in posted:
                        return "PROC"
                    if handle in posted:
                        completed.append(handle)
                    return "DONE"

                mgr.send_kvcache_staged = send
                mgr.agent.check_xfer_state.side_effect = state
                chunks = [self._make_chunk(r, [r], True) for r in (room, next_room)]
                self._run_staging_worker(mgr, chunks, staging_buffer=source)
                self.assertEqual(mgr.request_status[room], KVPoll.Failed)
                self.assertEqual(mgr.request_status[next_room], KVPoll.Success)
                self.assertIn((room, "agent"), completed)
                self.assertEqual(mgr._staging_outstanding[room], 1)
                mgr._allocate_staging_buffers.assert_called_once_with(1)
                self.assertEqual(len(mgr._staging_ctx.buffers), 2)
                self.assertIs(mgr._staging_ctx.buffers[0], source)
                self.assertIs(mgr._staging_ctx.buffers[1], spare)
                mgr.poison_deferred_ack_room.assert_called_once_with(room)
                self.assertNotIn(
                    (room,),
                    [c.args for c in mgr._maybe_ack_drained_abort.call_args_list],
                )
                self.assertCountEqual(
                    [
                        (c.kwargs["room"], c.kwargs["writer_id"])
                        for c in mgr.send_chunk_ready.call_args_list
                    ],
                    [(next_room, "agent"), (next_room, "other")],
                )
                # Re-read the ambiguous WRITE's source after a complete healthy
                # room. A GPU/NIC could still be reading it at this time.
                ambiguous_window, expected = posted[room, "other"]
                self.assertIs(ambiguous_window.obj, source.raw.obj)
                self.assertEqual(bytes(ambiguous_window), expected)
                for peer in ("agent", "other"):
                    self.assertIs(posted[next_room, peer][0].obj, spare.raw.obj)
                mgr.send_kv_status_message.assert_called_once()
                self.assertEqual(
                    mgr.send_kv_status_message.call_args.kwargs["bootstrap_room"],
                    next_room,
                )
                self.assertEqual(
                    mgr.send_kv_status_message.call_args.kwargs["status"],
                    KVPoll.Success,
                )

    def test_fanout_retains_unsettled_staging_across_rooms(self):
        from sglang.srt.disaggregation.nixl.conn import _NixlPostError

        first, second, blocked, recovered = 28, 29, 30, 31
        mgr, _, _ = self._two_room_fanout(first)
        for room in (blocked, recovered):
            mgr.transfer_infos[room] = {
                name: TransferInfo(**{**vars(req), "room": room})
                for name, req in mgr.transfer_infos[first].items()
            }
            mgr.request_status[room] = KVPoll.WaitingForInput
            mgr.req_to_decode_prefix_len[room] = 0
        mgr.poison_deferred_ack_room = MagicMock()
        source, spare = self._cpu_staging_source(), self._cpu_staging_source()
        mgr._allocate_staging_buffers = MagicMock(return_value=[spare])
        posted = {}

        def send(peer, indices, *_args, staging_buffer):
            room = int(indices[0])
            window = staging_buffer.raw[:8]
            window[:] = bytes([room]) * 8
            posted[room, peer] = window
            if room == second:
                raise _NixlPostError("rejected by backend", (room, peer))
            return room, peer

        def state(handle):
            if (
                handle == (first, "agent")
                and mgr.request_status[blocked] != KVPoll.Failed
            ):
                raise RuntimeError("NIXL_ERR_NOT_FOUND for old registration")
            if handle == (second, "agent"):
                # A terminal/error or stale handle on the rejected POST never
                # permits reuse. It can remain unrecoverable after reconnect.
                raise RuntimeError("NIXL_ERR_BACKEND for rejected POST")
            return "DONE"

        mgr.send_kvcache_staged = send
        mgr.agent.check_xfer_state.side_effect = state
        rooms = (first, second, blocked, recovered)
        chunks = [self._make_chunk(r, [r], True) for r in rooms]
        self._run_staging_worker(mgr, chunks, staging_buffer=source)
        self.assertEqual(
            [mgr.request_status[r] for r in rooms],
            [KVPoll.Failed] * 3 + [KVPoll.Success],
        )
        mgr._allocate_staging_buffers.assert_called_once_with(1)
        self.assertEqual(
            list(posted),
            [
                (first, "agent"),
                (second, "agent"),
                (recovered, "agent"),
                (recovered, "other"),
            ],
        )
        self.assertIs(posted[first, "agent"].obj, source.raw.obj)
        self.assertIs(posted[second, "agent"].obj, spare.raw.obj)
        self.assertEqual(bytes(posted[second, "agent"]), bytes([second]) * 8)
        self.assertIs(posted[recovered, "agent"].obj, source.raw.obj)
        self.assertCountEqual(
            [c.args for c in mgr.poison_deferred_ack_room.call_args_list],
            [(first,), (second,)],
        )
        self.assertEqual(
            [
                (c.kwargs["bootstrap_room"], c.kwargs["status"])
                for c in mgr.send_kv_status_message.call_args_list
            ],
            [(blocked, KVPoll.Failed), (recovered, KVPoll.Success)],
        )

    def test_fanout_drains_old_source_after_spare_allocation_fails(self):
        room, next_room, recovered = 28, 29, 30
        mgr, buffer, _ = self._two_room_fanout(room)
        mgr.transfer_infos[recovered] = {
            name: TransferInfo(**{**vars(req), "room": recovered})
            for name, req in mgr.transfer_infos[room].items()
        }
        mgr.request_status[recovered] = KVPoll.WaitingForInput
        mgr.req_to_decode_prefix_len[recovered] = 0
        mgr._allocate_staging_buffers = MagicMock(
            side_effect=RuntimeError("out of memory")
        )
        mgr.poison_deferred_ack_room = MagicMock()
        mgr.send_kvcache_staged = MagicMock(
            side_effect=lambda peer, indices, *a, **kw: (int(indices[0]), peer)
        )

        def state(handle):
            if handle == (room, "agent"):
                # The old WRITE finishes only after a later room tries the
                # same worker while its source is still owned by the NIC.
                return (
                    "DONE" if mgr.request_status[next_room] == KVPoll.Failed else "PROC"
                )
            return "ERR" if handle == (room, "other") else "DONE"

        mgr.agent.check_xfer_state.side_effect = state
        chunks = [self._make_chunk(r, [r], True) for r in (room, next_room, recovered)]
        with (
            patch("sglang.srt.disaggregation.nixl.conn.NIXL_ERR_SETTLE_TIMEOUT_S", 0),
            patch("sglang.srt.disaggregation.nixl.conn.NIXL_ERR_SETTLE_POLL_S", 0),
        ):
            self._run_staging_worker(mgr, chunks, staging_buffer=buffer)
        posted = [
            (int(c.args[1][0]), c.args[0])
            for c in mgr.send_kvcache_staged.call_args_list
        ]
        self.assertEqual(
            posted, [(r, p) for r in (room, recovered) for p in ("agent", "other")]
        )
        mgr.poison_deferred_ack_room.assert_called_once_with(room)
        self.assertEqual(mgr._staging_outstanding[room], 1)
        self.assertCountEqual(
            [
                (c.kwargs["room"], c.kwargs["writer_id"])
                for c in mgr.send_chunk_ready.call_args_list
            ],
            [(recovered, "agent"), (recovered, "other")],
        )
        statuses = [
            (c.kwargs["bootstrap_room"], c.kwargs["status"])
            for c in mgr.send_kv_status_message.call_args_list
        ]
        self.assertEqual(
            statuses, [(next_room, KVPoll.Failed), (recovered, KVPoll.Success)]
        )
        self.assertEqual(mgr.request_status[room], KVPoll.Failed)
        mgr._allocate_staging_buffers.assert_called_once_with(1)

    def test_deferred_staging_chunk_still_transfers_after_last_chunk_success(self):
        room = 27
        mgr = self._make_staging_manager(room)
        first = self._make_chunk(room, [1], is_last_chunk=False)
        last = self._make_chunk(room, [2], is_last_chunk=True)
        last.index_slice = slice(1, 2)
        events = []
        attempts = []

        def stage(_strategy, chunk, _indices, _req, _info, _buffer):
            page_start = chunk.index_slice.start
            attempts.append(page_start)
            if len(attempts) == 1:
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

    def test_worker_waits_for_events_before_reading_and_counts_completed_handles(self):
        import torch

        for event_field in ("wait_event", "indices_event"):
            with self.subTest(event=event_field):
                room = 24
                mgr = self._make_manager(room)
                order = []

                def record_read(*args, **kwargs):
                    order.append("read")
                    return "kv_handle"

                def poll(handle):
                    self.assertEqual(mgr.send_counts(room), (1, 0))
                    order.append("completed")
                    return "DONE"

                mgr.send_kvcache = MagicMock(side_effect=record_read)
                mgr.agent.check_xfer_state = poll
                chunk = self._make_chunk(room, [1], is_last_chunk=False)
                if event_field == "indices_event":
                    chunk.prefill_kv_indices = torch.tensor([1], dtype=torch.int32)
                setattr(
                    chunk,
                    event_field,
                    SimpleNamespace(synchronize=lambda: order.append("wait")),
                )
                chunk.send_record = mgr.send_record(room)
                mgr.note_send_enqueued(room)
                get_device, get_device_module = self._record_device_bind(order)
                with get_device, get_device_module:
                    self._run_worker_once(mgr, chunk)

                self.assertEqual(order, ["bind:3", "wait", "read", "completed"])
                self.assertEqual(mgr.send_counts(room), (1, 1))
                self.assertIsInstance(chunk.prefill_kv_indices, np.ndarray)

    def test_early_send_wait_event_is_synchronized_before_the_staging_gather(self):
        """With staging on, the wait precedes the staging strategy build.

        The staged path gathers the pages into the staging buffer before posting
        the RDMA, so a wait that only covers the post leaves the gather racing the
        prior step's writes. gather_all_layers_to_staging() syncs its own gather
        stream, which does not order those writes against the gather.
        """
        room = 25
        mgr = self._make_manager(room)
        mgr.enable_staging = True
        order = []

        def stop_at_strategy(_staging_buffer):
            order.append("gather")
            raise SystemExit

        def record_read(*args, **kwargs):
            # SystemExit, not AssertionError: the worker's except Exception would
            # swallow the latter and re-dequeue this chunk forever.
            order.append("read")
            raise SystemExit

        mgr._try_create_staging_strategy = stop_at_strategy
        mgr.send_kvcache = MagicMock(side_effect=record_read)
        chunk = self._make_chunk(room, [1], is_last_chunk=False)
        chunk.wait_event = SimpleNamespace(synchronize=lambda: order.append("wait"))
        # Bounded: a body that reaches neither probe must end the worker, not spin.
        queue = SimpleNamespace(get=MagicMock(side_effect=[chunk, SystemExit()]))
        get_device, get_device_module = self._record_device_bind(order)

        with get_device, get_device_module, self.assertRaises(SystemExit):
            mgr.transfer_worker(queue, staging_buffer=object())

        self.assertEqual(order, ["bind:3", "wait", "gather"])

    def test_early_send_wait_is_skipped_for_a_failed_room(self):
        """A failed room's chunk is dropped without blocking the worker on its event."""
        room = 26
        mgr = self._make_manager(room)
        mgr.request_status[room] = KVPoll.Failed
        order = []
        mgr.send_kvcache = MagicMock()
        chunk = self._make_chunk(room, [1], is_last_chunk=False)
        chunk.wait_event = SimpleNamespace(synchronize=lambda: order.append("wait"))
        # Patched so a missing skip reaches the wait; unpatched, get_device() raises
        # first and the worker's except Exception hides it.
        get_device, get_device_module = self._record_device_bind(order)

        with get_device, get_device_module:
            self._run_worker_once(mgr, chunk)

        self.assertEqual(order, [])
        mgr.send_kvcache.assert_not_called()


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
        mgr.enable_kv_packing = False
        mgr._kv_packers = []
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


class TestNixlEarlySendWaitEvent(CustomTestCase):
    def test_sender_queues_the_early_send_event_on_the_next_chunk_only(self):
        """send() hands the recorded event to the next queued chunk, then drops it.

        Before this fix NIXL's add_transfer_request accepted the event but never
        put it on the queued chunk, so the worker read prefix pages the prior
        step's forward was still writing, and the transfer still reported
        success. The event is one-shot: a later chunk that reused it would wait on
        writes that already landed and leave its own unordered.
        """
        room = 32
        mgr = object.__new__(NixlKVManager)
        mgr.disaggregation_mode = DisaggregationMode.PREFILL
        mgr.enable_staging = False
        mgr.enable_kv_packing = False
        mgr._kv_packers = []
        mgr.transfer_infos = {room: {"agent": SimpleNamespace(dst_port=5555)}}
        mgr.request_status = {room: KVPoll.WaitingForInput}
        observed = []

        def consume(chunk):
            observed.append(mgr.send_counts(room))
            mgr.note_send_finished(room, chunk.send_record)

        queue = SimpleNamespace(put=MagicMock(side_effect=consume))
        mgr.transfer_queues = [queue]
        mgr.prepare_send_state_indices = lambda state_indices: state_indices
        sender = object.__new__(NixlKVSender)
        sender.kv_mgr = mgr
        sender.bootstrap_room = room
        sender.aux_index = None
        sender.conclude_state = None
        sender._transfer_start_time = None
        sender.trace_ctx = MagicMock()
        sender._prepare_send_indices = lambda kv_indices, state_indices: (
            kv_indices,
            slice(0, len(kv_indices)),
            False,
            False,
        )
        sender._record_transfer_indices = MagicMock()
        event = object()
        sender._early_send_wait_event = event

        indices_event = object()
        sender.send(np.array([1], dtype=np.int32), indices_event=indices_event)
        sender.send(np.array([2], dtype=np.int32))

        first, second = (c.args[0] for c in queue.put.call_args_list)
        self.assertIs(first.wait_event, event)
        self.assertIsNone(second.wait_event)

        self.assertIs(first.indices_event, indices_event)
        self.assertIsNone(second.indices_event)
        self.assertEqual(observed, [(1, 0), (2, 1)])
        self.assertEqual(mgr.send_counts(room), (2, 2))


class TestNixlNodeFailure(CustomTestCase):
    def _make_manager(self):
        mgr = object.__new__(NixlKVManager)
        mgr.connection_lock = threading.Lock()
        # Connection keys are "{addr}_{dp_rank}_{cp_rank}_{tp_rank}".
        mgr.connection_pool = {
            "10.0.0.1:8998_0_0_0": [{"rank_ip": "10.0.0.1"}],
            "10.0.0.1:8998_0_0_1": [{"rank_ip": "10.0.0.1"}],
            "10.0.0.2:8998_0_0_0": [{"rank_ip": "10.0.0.2"}],
            "10.0.0.1:89981_0_0_0": [{"rank_ip": "10.0.0.1"}],
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
        self.assertIn("10.0.0.1:89981_0_0_0", mgr.connection_pool)
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
        mgr.customized_info_enabled = False
        mgr.exchange_aux_item_lens = False
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

    def test_send_aux_uses_destination_row_stride(self):
        context = get_context().override_server_args(
            disaggregation_enable_kv_checksum=False
        )
        context.install()
        self.addCleanup(context.restore)
        for enabled in (False, True):
            with self.subTest(customized_info=enabled):
                agent = StagingFakeAgent()
                mgr = self._make_manager(agent)
                mgr.customized_info_enabled = enabled
                mgr.exchange_aux_item_lens = enabled
                src_ptrs = [0x1000 + i * 0x10000 for i in range(8)]
                dst_ptrs = [0x2000 + i * 0x10000 for i in range(8)]
                mgr.kv_args.aux_data_ptrs = src_ptrs
                mgr.kv_args.aux_item_lens = [64] * 8
                dst_lens = [128] * 6 + [64, 64] if enabled else None

                mgr.send_aux("peer", 2, dst_ptrs, dst_lens, 1)

                self.assertEqual(
                    agent.get_xfer_descs_calls,
                    [
                        ([(ptr + 128, 64, 0) for ptr in src_ptrs], "DRAM"),
                        (
                            [
                                (ptr + (128 if enabled and i < 6 else 64), 64, 0)
                                for i, ptr in enumerate(dst_ptrs)
                            ],
                            "DRAM",
                        ),
                    ],
                )

    def test_send_aux_rejects_metadata_buffer_count_mismatch(self):
        context = get_context().override_server_args(
            disaggregation_enable_kv_checksum=False
        )
        context.install()
        self.addCleanup(context.restore)
        agent = StagingFakeAgent()
        mgr = self._make_manager(agent)
        mgr.customized_info_enabled = True
        mgr.exchange_aux_item_lens = True
        mgr.kv_args.aux_data_ptrs = [0x1000] * 8
        mgr.kv_args.aux_item_lens = [64] * 8

        with self.assertRaisesRegex(ValueError, "metadata buffer layout mismatch"):
            mgr.send_aux("peer", 0, [0x2000] * 9, [64] * 8, 0)

        self.assertEqual(agent.get_xfer_descs_calls, [])

    def test_do_staging_transfer_defers_when_allocation_not_ready(self):
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
                staging_buffer=None,
            )

        self.assertIsNone(handle)
        self.assertTrue(deferred)
        self.assertEqual(chunk_idx, 0)

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
                    staging_buffer=None,
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
        mgr.customized_info_enabled = False
        mgr.exchange_aux_item_lens = False
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
            aux_data_ptrs=[0],
            aux_item_lens=[16],
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
                dst_aux_item_lens=None,
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
        mgr.enable_kv_packing = False
        mgr._kv_packers = []
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


class TestNixlUnifiedMemory(CustomTestCase):
    def _manager(self):
        mgr = object.__new__(NixlKVManager)
        mgr.enable_unified_memory = True
        mgr.agent = StagingFakeAgent()
        mgr.attn_tp_size = mgr.state_tp_size = mgr.pp_size = 1
        mgr.is_mla_backend = False
        mgr.kv_args = SimpleNamespace(
            gpu_id=0,
            page_size=16,
            prefill_start_layer=0,
            kv_data_ptrs=[0x10000],
            kv_data_lens=[65536],
            kv_item_lens=[256],
            kv_data_mem_kinds=["VRAM"],
            kv_layer_ids=[],
            state_types=[StateType.SWA, StateType.MAMBA],
            state_wire_keys=["window", "state"],
            state_data_ptrs=[[0x10000], [0x10000]],
            state_data_lens=[[65536], [65536]],
            state_item_lens=[[128], [512]],
            aux_data_ptrs=[0x30000],
            aux_data_lens=[64],
        )
        return mgr

    def _peer(self, mgr):
        return SimpleNamespace(
            agent_name="peer",
            dst_dcp_size=1,
            dst_dcp_rank=0,
            unified_layout_error=None,
            dst_unified_layout=mgr._unified_wire_layout(),
            decode_tp_size=1,
            dst_kv_ptrs=[0x40000],
            dst_kv_item_lens=[256],
            dst_kv_mem_kinds=["VRAM"],
            dst_state_item_lens=[[128], [512]],
            dst_state_data_ptrs=[[0x40000], [0x40000]],
        )

    def test_registration_roundtrip_and_legacy_marker(self):
        mgr = self._manager()
        mgr.enable_staging = mgr.exchange_aux_item_lens = False
        mgr.local_ip, mgr.rank_port = "127.0.0.1", 5555
        mgr.dcp_size, mgr.dcp_rank = 1, 0
        mgr.kv_args.engine_rank = 0
        mgr.kv_args.state_layer_ids = [[], []]
        mgr.kv_args.draft_rows_dcp_collapsed = False
        mgr.agent.name = "decode"
        mgr.agent.get_agent_metadata = lambda: b"metadata"
        message = mgr.build_receiver_registration_message()[1:]
        peer = KVArgsRegisterInfo.from_zmq(message)
        self.assertEqual(peer.dst_unified_layout, mgr._unified_wire_layout())
        mgr._validate_unified_peer(peer)
        legacy = KVArgsRegisterInfo.from_zmq(message[:-1])
        self.assertIsNone(legacy.dst_unified_layout)
        with self.assertRaisesRegex(ValueError, "layout mismatch"):
            mgr._validate_unified_peer(legacy)
        mgr.enable_unified_memory = False
        self.assertEqual(mgr.build_receiver_registration_message()[1:], message[:-1])
        mgr._validate_unified_peer(legacy)
        with self.assertRaisesRegex(ValueError, "layout mismatch"):
            mgr._validate_unified_peer(peer)

    def test_shared_arena_is_registered_once(self):
        mgr = self._manager()
        mgr.register_buffer_to_engine()
        self.assertEqual(
            mgr.agent.register_memory_calls,
            [
                ([(0x10000, 65536, 0, "")], "VRAM"),
                ([(0x30000, 64, 0, "")], "DRAM"),
            ],
        )

    def test_peer_checks_all_entry_geometry_before_preparing(self):
        mgr = self._manager()
        mgr.requires_dcp_relayout = MagicMock(return_value=False)
        mgr.agent.add_remote_agent = MagicMock()
        mgr.disaggregation_mode = DisaggregationMode.PREFILL
        mgr.check_peer_aux_buffer_count = MagicMock()
        mgr._prepare_payload_xfer = MagicMock()
        mgr._prepare_state_xfer = MagicMock()
        for field, value in (
            (None, None),
            ("dst_unified_layout", None),
            ("dst_unified_layout", {**mgr._unified_wire_layout(), "page_size": 32}),
            (
                "dst_unified_layout",
                {**mgr._unified_wire_layout(), "state_wire_keys": ["other", "state"]},
            ),
            ("decode_tp_size", 2),
            ("dst_kv_item_lens", [512]),
            ("dst_kv_ptrs", [0x40000, 0x50000]),
            ("dst_state_item_lens", [[128], [1024]]),
            ("dst_state_data_ptrs", [[0x40000]]),
        ):
            with self.subTest(field=field, value=value):
                peer = self._peer(mgr)
                peer.agent_metadata, peer.dst_aux_ptrs = b"metadata", [0x30000]
                if field is not None:
                    setattr(peer, field, value)
                mgr.decode_kv_args_table = {}
                mgr.agent.add_remote_agent.reset_mock()
                mgr._prepare_payload_xfer.reset_mock()
                mgr._prepare_state_xfer.reset_mock()
                mgr._add_remote_peer(peer)
                self.assertIs(mgr.decode_kv_args_table["peer"], peer)
                if field is None:
                    self.assertIsNone(peer.unified_layout_error)
                    mgr.agent.add_remote_agent.assert_called_once_with(b"metadata")
                    mgr._prepare_payload_xfer.assert_called_once_with(peer)
                    mgr._prepare_state_xfer.assert_called_once_with(peer)
                else:
                    self.assertIn("NIXL unified-memory PD", peer.unified_layout_error)
                    mgr.agent.add_remote_agent.assert_not_called()
                    mgr._prepare_payload_xfer.assert_not_called()
                    mgr._prepare_state_xfer.assert_not_called()

    def test_swa_fallback_copies_one_whole_entry(self):
        mgr = self._manager()
        handle = mgr._send_kvcache_generic(
            peer_name="peer",
            src_data_ptrs=[0x10000],
            dst_data_ptrs=[0x40000],
            item_lens=[128],
            prefill_data_indices=np.array([2, 3], dtype=np.int32),
            dst_data_indices=np.array([5, 6], dtype=np.int32),
            dst_gpu_id=0,
            state_type=StateType.SWA,
            dst_item_lens=[128],
        )
        self.assertEqual(handle, "handle")
        src, dst = mgr.agent.get_xfer_descs_calls
        np.testing.assert_array_equal(src[0], [[0x10100, 256, 0]])
        np.testing.assert_array_equal(dst[0], [[0x40280, 256, 0]])
        self.assertEqual(mgr.agent.transfer_calls, ["handle"])


if __name__ == "__main__":
    unittest.main()
