"""Basic CPU unit tests for NIXL disaggregation control paths."""

import struct
import sys
import threading
import types
import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll, StateType
from sglang.srt.disaggregation.common.conn import CommonKVManager, CommonKVReceiver
from sglang.srt.disaggregation.common.staging_handler import PrefillStagingContext
from sglang.srt.disaggregation.common.utils import pack_int_lists
from sglang.srt.disaggregation.nixl.conn import (
    KVArgsRegisterInfo,
    NixlKVManager,
    NixlKVSender,
    TransferInfo,
    TransferKVChunk,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


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


class FakeTensor:
    shape = (1, 1, 8)

    def element_size(self):
        return 2


class FakeStagingBuffer:
    def __init__(self, ptr=0x9000, size=1 << 20):
        self.ptr = ptr
        self.size = size

    def fits(self, required_bytes):
        return required_bytes <= self.size

    def get_ptr(self):
        return self.ptr


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
    def _initialize_manager(self, agent, device_module, mode, gpu_id):
        args = SimpleNamespace(
            pp_rank=0,
            engine_rank=0,
            gpu_id=gpu_id,
            kv_data_ptrs=[],
            kv_item_lens=[],
        )
        mgr = object.__new__(NixlKVManager)
        mgr.kv_args = args
        mgr.disaggregation_mode = mode
        mgr.enable_deferred_decode_kv_release = False

        api = types.ModuleType("nixl._api")
        api.nixl_agent = MagicMock(return_value=agent)
        api.nixl_agent_config = MagicMock()
        api.nixl_thread_sync_t = SimpleNamespace(NIXL_THREAD_SYNC_STRICT="strict")
        nixl = types.ModuleType("nixl")
        nixl._api = api
        agent.get_plugin_list.return_value = ["UCX"]

        with (
            patch.dict(sys.modules, {"nixl": nixl, "nixl._api": api}),
            patch.dict(
                "os.environ",
                {
                    "SGLANG_DISAGGREGATION_NIXL_BACKEND": "UCX",
                    "SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS": "{}",
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
            api.nixl_agent_config.assert_called_once_with(
                backends=[],
                num_threads=8 if mode == DisaggregationMode.PREFILL else 0,
                enable_prog_thread=True,
                sync_mode="strict",
            )

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

    def test_backend_initialization_propagates_error(self):
        error = RuntimeError("backend initialization failed")
        agent = MagicMock()
        agent.create_backend.side_effect = error

        with self.assertRaises(RuntimeError) as raised:
            self._initialize_manager(
                agent, MagicMock(), DisaggregationMode.DECODE, gpu_id=3
            )

        self.assertIs(raised.exception, error)


class TestNixlTransferInfo(CustomTestCase):
    def test_from_zmq_parses_required_fields(self):
        kv_indices = np.array([3, 5, 8], dtype=np.int32)
        state_indices = [[1, 2], [], [9]]
        msg = [
            b"7",
            b"127.0.0.1",
            b"12345",
            b"decode_agent",
            kv_indices.tobytes(),
            b"4",
            b"2",
            pack_int_lists(state_indices, "i"),
            b"11",
        ]

        info = TransferInfo.from_zmq(msg)

        self.assertEqual(info.room, 7)
        self.assertEqual(info.endpoint, "127.0.0.1")
        self.assertEqual(info.dst_port, 12345)
        self.assertEqual(info.agent_name, "decode_agent")
        np.testing.assert_array_equal(info.dst_kv_indices, kv_indices)
        self.assertEqual(info.dst_aux_index, 4)
        self.assertEqual(info.required_dst_info_num, 2)
        self.assertEqual(info.dst_state_indices, state_indices)
        self.assertEqual(info.decode_prefix_len, 11)

    def test_from_zmq_defaults_optional_fields(self):
        info = TransferInfo.from_zmq(
            [
                b"8",
                b"127.0.0.1",
                b"12346",
                b"agent",
                np.array([1], dtype=np.int32).tobytes(),
                b"0",
                b"1",
            ]
        )

        self.assertEqual(info.dst_state_indices, [])
        self.assertIsNone(info.decode_prefix_len)

    def test_decode_radix_full_hit_is_not_dummy(self):
        info = TransferInfo.from_zmq(
            [
                b"9",
                b"127.0.0.1",
                b"12347",
                b"agent",
                np.array([], dtype=np.int32).tobytes(),
                b"2",
                b"1",
                b"",
                b"128",
            ]
        )

        self.assertFalse(info.is_dummy)

    def test_empty_indices_without_decode_prefix_is_dummy(self):
        info = TransferInfo.from_zmq(
            [
                b"10",
                b"127.0.0.1",
                b"12348",
                b"agent",
                np.array([], dtype=np.int32).tobytes(),
                b"2",
                b"1",
                b"",
                b"0",
            ]
        )

        self.assertTrue(info.is_dummy)

    def test_explicit_dummy_frame_true_is_dummy(self):
        # msg[9] is the explicit is_dummy frame the sender writes
        # (str(int(is_dummy))); it wins over payload inference.
        info = TransferInfo.from_zmq(
            [
                b"11",
                b"127.0.0.1",
                b"12349",
                b"agent",
                np.array([], dtype=np.int32).tobytes(),
                b"2",
                b"1",
                b"",
                b"0",
                b"1",
            ]
        )

        self.assertTrue(info.is_dummy)

    def test_explicit_dummy_frame_true_with_prefix_hit_stays_dummy(self):
        # A dummy rank whose request also has a decode-side prefix hit: the
        # sender sends decode_prefix_len unconditionally, so only the explicit
        # frame distinguishes this from a real full-prefix-hit transfer.
        info = TransferInfo.from_zmq(
            [
                b"12",
                b"127.0.0.1",
                b"12350",
                b"agent",
                np.array([], dtype=np.int32).tobytes(),
                b"2",
                b"1",
                b"",
                b"128",
                b"1",
            ]
        )

        self.assertTrue(info.is_dummy)

    def test_explicit_dummy_frame_false_with_empty_indices_is_real(self):
        # Full prefix hit as the sender encodes it: empty kv indices,
        # decode_prefix_len > 0, explicit is_dummy 0.
        info = TransferInfo.from_zmq(
            [
                b"13",
                b"127.0.0.1",
                b"12351",
                b"agent",
                np.array([], dtype=np.int32).tobytes(),
                b"2",
                b"1",
                b"",
                b"128",
                b"0",
            ]
        )

        self.assertFalse(info.is_dummy)

    def test_explicit_dummy_frame_false_for_real_transfer(self):
        info = TransferInfo.from_zmq(
            [
                b"14",
                b"127.0.0.1",
                b"12352",
                b"agent",
                np.array([3, 5], dtype=np.int32).tobytes(),
                b"2",
                b"1",
                b"",
                b"0",
                b"0",
            ]
        )

        self.assertFalse(info.is_dummy)

    def test_fallback_without_dummy_frame_reads_prefix_hit_dummy_as_real(self):
        # Old-peer fallback: without msg[9], a dummy rank with a decode-side
        # prefix hit is indistinguishable from a real full-prefix-hit transfer
        # and parses as real. The explicit frame above exists for this case.
        info = TransferInfo.from_zmq(
            [
                b"15",
                b"127.0.0.1",
                b"12353",
                b"agent",
                np.array([], dtype=np.int32).tobytes(),
                b"2",
                b"1",
                b"",
                b"128",
            ]
        )

        self.assertFalse(info.is_dummy)


class TestNixlKVArgsRegisterInfo(CustomTestCase):
    def test_from_zmq_preserves_unsigned_pointers_and_optional_fields(self):
        high_ptr = 0xFFFF_81AB_54E0_1000
        kv_ptrs = [high_ptr, high_ptr + 0x1000]
        aux_ptrs = [0x1000, 0x2000]
        state_ptrs = [[high_ptr + 0x2000], [high_ptr + 0x3000, high_ptr + 0x4000]]
        state_item_lens = [[64], [128, 256]]
        state_dims = [[16], [32, 64]]
        staging_ptr = high_ptr + 0x5000

        msg = [
            b"None",
            b"10.0.0.2",
            b"23456",
            b"agent_with_large_ptr",
            b"metadata",
            b"".join(struct.pack("Q", ptr) for ptr in kv_ptrs),
            b"".join(struct.pack("Q", ptr) for ptr in aux_ptrs),
            pack_int_lists(state_ptrs, "Q"),
            b"3",
            b"4",
            b"1",
            b"1024",
            pack_int_lists(state_item_lens, "I"),
            pack_int_lists(state_dims, "I"),
            struct.pack("Q", staging_ptr),
            b"1048576",
            b"64",
            b"DRAM,DRAM",
            b"".join(struct.pack("Q", item_len) for item_len in [1024, 2048]),
            pack_int_lists([[4], [4, 5]], "I"),
            b"".join(struct.pack("I", layer_id) for layer_id in [2, 7]),
            b"4",
            b"3",
        ]

        info = KVArgsRegisterInfo.from_zmq(msg)

        self.assertEqual(info.room, "None")
        self.assertEqual(info.endpoint, "10.0.0.2")
        self.assertEqual(info.dst_port, 23456)
        self.assertEqual(info.agent_name, "agent_with_large_ptr")
        self.assertEqual(info.agent_metadata, b"metadata")
        self.assertEqual(info.dst_kv_ptrs, kv_ptrs)
        self.assertEqual(info.dst_aux_ptrs, aux_ptrs)
        self.assertEqual(info.dst_state_data_ptrs, state_ptrs)
        self.assertEqual(info.gpu_id, 3)
        self.assertEqual(info.decode_tp_size, 4)
        self.assertEqual(info.decode_tp_rank, 1)
        self.assertEqual(info.dst_kv_item_len, 1024)
        self.assertEqual(info.dst_kv_item_lens, [1024, 2048])
        self.assertEqual(info.dst_num_slots, 64)
        self.assertEqual(info.dst_kv_mem_kinds, ["DRAM", "DRAM"])
        self.assertEqual(info.dst_state_item_lens, state_item_lens)
        self.assertEqual(info.dst_state_dim_per_tensor, state_dims)
        self.assertEqual(info.dst_dcp_size, 4)
        self.assertEqual(info.dst_dcp_rank, 3)
        self.assertEqual(info.dst_state_layer_ids, [[4], [4, 5]])
        self.assertEqual(info.dst_kv_layer_ids, [2, 7])
        self.assertEqual(info.staging_base_ptr, staging_ptr)
        self.assertEqual(info.staging_total_size, 1048576)

    def test_from_zmq_allows_missing_state_and_staging_fields(self):
        msg = [
            b"None",
            b"10.0.0.3",
            b"23457",
            b"agent",
            b"metadata",
            struct.pack("Q", 0x1000),
            struct.pack("Q", 0x2000),
            b"",
            b"0",
            b"1",
            b"0",
            b"256",
        ]

        info = KVArgsRegisterInfo.from_zmq(msg)

        self.assertEqual(info.dst_state_data_ptrs, [])
        self.assertEqual(info.dst_state_item_lens, [])
        self.assertEqual(info.dst_state_dim_per_tensor, [])
        self.assertEqual(info.dst_kv_item_lens, [256])
        self.assertEqual(info.dst_dcp_size, 1)
        self.assertEqual(info.dst_dcp_rank, 0)
        self.assertEqual(info.staging_base_ptr, 0)
        self.assertEqual(info.staging_total_size, 0)


class TestNixlKVSenderChunkPolicy(CustomTestCase):
    def test_last_zero_page_chunk_is_sent_for_aux_only_completion(self):
        sender = object.__new__(NixlKVSender)

        self.assertTrue(sender.should_send_kv_chunk(0, last_chunk=True))
        self.assertFalse(sender.should_send_kv_chunk(0, last_chunk=False))
        self.assertTrue(sender.should_send_kv_chunk(3, last_chunk=False))


class TestNixlEmptyStateTransfer(CustomTestCase):
    def test_empty_pp_state_component_is_a_noop(self):
        mgr = object.__new__(NixlKVManager)
        mgr.agent = StagingFakeAgent()
        mgr.is_mla_backend = False
        mgr.pp_size = 2
        mgr.kv_args = SimpleNamespace(prefill_start_layer=0, kv_data_ptrs=[1])

        handle = mgr._send_kvcache_generic(
            peer_name="decode",
            src_data_ptrs=[],
            dst_data_ptrs=[],
            item_lens=[],
            prefill_data_indices=np.array([3], dtype=np.int32),
            dst_data_indices=np.array([5], dtype=np.int32),
            dst_gpu_id=0,
            notif="qsa-empty",
            state_type=StateType.QSA_PENDING,
            force_flat=True,
            src_layer_ids=[],
            dst_layer_ids=[],
        )

        self.assertIsNone(handle)
        self.assertEqual(mgr.agent.get_xfer_descs_calls, [])
        self.assertEqual(mgr.agent.initialize_xfer_calls, [])

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
                notif="qsa-mismatch",
                state_type=StateType.QSA_PENDING,
                force_flat=True,
                src_layer_ids=[24],
                dst_layer_ids=[24],
                dst_item_lens=[48],
            )
        self.assertEqual(mgr.agent.initialize_xfer_calls, [])


class TestNixlAbortHandling(CustomTestCase):
    def _make_manager(self, request_status=None):
        mgr = object.__new__(NixlKVManager)
        mgr.request_status = dict(request_status or {})
        mgr._connect = MagicMock()
        mgr.failure_lock = threading.Lock()
        mgr.failure_records = {}
        # These cases cover the legacy no-ack behavior; the deferred-release ack
        # path is exercised in test_nixl_deferred_kv_release.py.
        mgr.enable_deferred_decode_kv_release = False
        return mgr

    def test_given_known_incomplete_room_when_abort_arrives_then_room_fails_without_ack(
        self,
    ):
        mgr = self._make_manager({11: KVPoll.WaitingForInput})

        handled = mgr._handle_abort_notification(
            [b"ABORT", b"11", b"127.0.0.1", b"5555"]
        )

        self.assertTrue(handled)
        self.assertEqual(mgr.request_status[11], KVPoll.Failed)
        self.assertEqual(
            mgr.failure_records[11],
            "Aborted by decode-side abort notification.",
        )
        mgr._connect.assert_not_called()

    def test_given_successful_room_when_abort_arrives_then_status_is_preserved(self):
        mgr = self._make_manager({12: KVPoll.Success})

        handled = mgr._handle_abort_notification(
            [b"ABORT", b"12", b"127.0.0.1", b"5556"]
        )

        self.assertTrue(handled)
        self.assertEqual(mgr.request_status[12], KVPoll.Success)
        self.assertEqual(mgr.failure_records, {})
        mgr._connect.assert_not_called()

    def test_given_unknown_room_when_abort_arrives_then_status_remains_absent(self):
        mgr = self._make_manager()

        handled = mgr._handle_abort_notification(
            [b"ABORT", b"14", b"127.0.0.1", b"5557"]
        )

        self.assertTrue(handled)
        self.assertNotIn(14, mgr.request_status)
        self.assertEqual(mgr.failure_records, {})
        mgr._connect.assert_not_called()

    def test_given_malformed_abort_when_handled_then_no_exception_or_ack(self):
        mgr = self._make_manager({13: KVPoll.WaitingForInput})

        handled = mgr._handle_abort_notification(
            [b"ABORT", b"invalid-room", b"127.0.0.1", b"5558"]
        )

        self.assertTrue(handled)
        self.assertEqual(mgr.request_status[13], KVPoll.WaitingForInput)
        self.assertEqual(mgr.failure_records, {})
        mgr._connect.assert_not_called()


class TestNixlUpdateStatus(CustomTestCase):
    def _make_manager(self, request_status):
        mgr = object.__new__(NixlKVManager)
        mgr.request_status = dict(request_status)
        return mgr

    def test_given_failed_room_when_status_is_promoted_then_failed_is_preserved(self):
        for status in (KVPoll.Transferring, KVPoll.Success):
            with self.subTest(status=status):
                mgr = self._make_manager({17: KVPoll.Failed})

                mgr.update_status(17, status)

                self.assertEqual(mgr.request_status[17], KVPoll.Failed)

    def test_given_missing_room_when_failed_update_arrives_then_room_is_not_resurrected(
        self,
    ):
        mgr = self._make_manager({})

        mgr.update_status(18, KVPoll.Failed)

        self.assertNotIn(18, mgr.request_status)


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
        mgr.transfer_source_rank = 0
        mgr.kv_args = SimpleNamespace(
            engine_rank=0, kv_data_ptrs=[0], num_draft_entries=0
        )
        mgr.exceptions = {}
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
            chunk_id=0,
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
        self.assertEqual(mgr.send_aux.call_args.args[-1], "")

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
        self.assertEqual(mgr.send_kvcache.call_args.args[5], "")
        self.assertEqual(mgr.maybe_send_extra.call_args.args[5], "")
        self.assertEqual(mgr.send_aux.call_args.args[4], "")
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

    def test_running_sibling_after_failure_cannot_notify_decode_to_release_pages(self):
        room = 26
        mgr = self._make_manager(room)
        mgr.send_kvcache = MagicMock(return_value="kv")
        mgr.send_aux = MagicMock(return_value="aux")
        mgr.agent.check_xfer_state = lambda handle: "ERR" if handle == "kv" else "PROC"
        with patch("sglang.srt.disaggregation.nixl.conn.NIXL_ERR_SETTLE_TIMEOUT_S", 0):
            self._run_worker_once(mgr, self._make_chunk(room, [1], is_last_chunk=True))
        self.assertEqual(mgr.request_status[room], KVPoll.Failed)
        mgr.send_kv_status_message.assert_not_called()
        self.assertGreater(mgr._staging_outstanding[room], 0)

    def _make_staging_manager(self, room, peers=("agent",)):
        mgr = self._make_manager(room)
        original = mgr.transfer_infos[room]["agent"]
        original_peer = mgr.decode_kv_args_table["agent"]
        mgr.transfer_infos[room] = {}
        mgr.decode_kv_args_table = {}
        for index, name in enumerate(peers):
            req = TransferInfo(**vars(original))
            req.agent_name = name
            req.dst_port += index
            req.dst_kv_indices = np.array([100, 101], dtype=np.int32)
            mgr.transfer_infos[room][name] = req
            peer = SimpleNamespace(**vars(original_peer))
            peer.decode_tp_size = 2
            peer.staging_base_ptr = 0x8000
            mgr.decode_kv_args_table[name] = peer
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

    def test_last_chunk_success_waits_for_earlier_deferred_staging_chunk(self):
        room = 27
        mgr = self._make_staging_manager(room)
        first = self._make_chunk(room, [1], is_last_chunk=False)
        last = self._make_chunk(room, [2], is_last_chunk=True)
        last.chunk_id = 1
        last.index_slice = slice(1, 2)
        events = []
        attempts = []

        def stage(_strategy, chunk, _indices, _req, _info, queue):
            attempts.append(chunk.chunk_id)
            if len(attempts) == 1:
                queue.put(chunk)
                return None, True, 0
            return f"kv-{chunk.chunk_id}", False, chunk.chunk_id

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
            events, [("chunk", 1), ("chunk", 0), ("status", KVPoll.Success)]
        )
        self.assertNotIn(room, mgr.transfer_infos)
        self.assertNotIn(room, mgr._staging_outstanding)

    def test_deferred_second_peer_does_not_rewrite_first_peers_staging(self):
        room = 28
        mgr = self._make_staging_manager(room, ("agent0", "agent1"))
        chunk = self._make_chunk(room, [1], is_last_chunk=True)
        attempts = []
        events = []

        def stage(_strategy, chunk, _indices, req, _info, queue):
            attempts.append(req.agent_name)
            if attempts == ["agent0", "agent1"]:
                queue.put(chunk)
                return None, True, 0
            return f"kv-{req.agent_name}", False, 0

        mgr._do_staging_transfer = stage
        mgr.agent.check_xfer_state.side_effect = lambda handle: (
            events.append(("done", handle)) or "DONE"
        )
        mgr.send_chunk_ready.side_effect = lambda **kw: events.append(
            ("chunk", kw["writer_id"])
        )
        self._run_staging_worker(mgr, [chunk])
        self.assertEqual(attempts, ["agent0", "agent1", "agent1"])
        self.assertLess(
            events.index(("done", "kv-agent0")), events.index(("chunk", "agent0"))
        )
        self.assertLess(
            events.index(("done", "kv-agent1")), events.index(("chunk", "agent1"))
        )
        self.assertEqual(mgr.send_chunk_ready.call_count, 2)
        self.assertEqual(mgr.send_aux.call_count, 2)
        mgr.send_kv_status_message.assert_called_once()
        self.assertEqual(
            mgr.send_kv_status_message.call_args.kwargs["status"], KVPoll.Success
        )


class TestNixlReceiverPoll(CustomTestCase):
    def _make_receiver(self, status=KVPoll.WaitingForInput):
        mgr = object.__new__(NixlKVManager)
        mgr.waiting_timeout = 5
        mgr.request_status = {}
        mgr.failure_records = {}
        mgr.failure_lock = threading.Lock()
        mgr.prefill_response_tracker = {}
        mgr.required_prefill_response_num_table = {11: 1}
        mgr.addr_to_rooms_tracker = defaultdict(set)
        mgr.connection_pool = {}
        mgr.connection_lock = threading.Lock()
        mgr.enable_staging = False
        mgr._staging_handler = None
        mgr.agent = MagicMock()
        receiver = CommonKVReceiver(mgr, "prefill:8998", 11)
        mgr.update_status(11, status)
        return receiver, mgr

    def test_returns_existing_conclude_state_without_progressing_manager(self):
        receiver, mgr = self._make_receiver()
        receiver.conclude_state = KVPoll.Success
        self.assertEqual(receiver.poll(), KVPoll.Success)
        mgr.agent.get_new_notifs.assert_not_called()

    def test_returns_bootstrap_status_before_metadata_publication(self):
        receiver, mgr = self._make_receiver(status=KVPoll.Bootstrapping)
        self.assertEqual(receiver.poll(), KVPoll.Bootstrapping)
        mgr.agent.get_new_notifs.assert_not_called()

    @patch("sglang.srt.disaggregation.common.conn.time.time", return_value=20.0)
    def test_waiting_timeout_records_failure(self, _mock_time):
        receiver, mgr = self._make_receiver()
        receiver.metadata_published = True
        receiver.init_time = 10.0
        self.assertEqual(receiver.poll(), KVPoll.Failed)
        mgr.agent.get_new_notifs.assert_not_called()
        self.assertIn("timed out", mgr.failure_records[11])

    @patch("sglang.srt.disaggregation.common.conn.time.time", return_value=20.0)
    def test_zmq_completion_wins_over_waiting_timeout(self, _mock_time):
        receiver, mgr = self._make_receiver()
        receiver.metadata_published = True
        receiver.init_time = 10.0
        mgr.apply_prefill_status(
            bootstrap_room=11, status=KVPoll.Success, prefill_rank=0
        )
        self.assertEqual(receiver.poll(), KVPoll.Success)
        mgr.agent.get_new_notifs.assert_not_called()
        self.assertEqual(mgr.failure_records, {})
        self.assertFalse(receiver.abort_notified)

    def test_multi_rank_success_is_deduplicated_and_clear_drops_room(self):
        receiver, mgr = self._make_receiver()
        receiver.metadata_published = True
        mgr.required_prefill_response_num_table[11] = 2
        for rank in (0, 0):
            mgr.apply_prefill_status(
                bootstrap_room=11, status=KVPoll.Success, prefill_rank=rank
            )
        self.assertEqual(receiver.poll(), KVPoll.WaitingForInput)
        mgr.apply_prefill_status(
            bootstrap_room=11, status=KVPoll.Success, prefill_rank=1
        )
        self.assertEqual(receiver.poll(), KVPoll.Success)
        receiver.clear()
        self.assertNotIn(11, mgr.prefill_response_tracker)
        self.assertNotIn(11, mgr.request_status)
        self.assertEqual(receiver.poll(), KVPoll.Success)
        # A late ZMQ completion cannot resurrect a cleared room.
        mgr.apply_prefill_status(
            bootstrap_room=11, status=KVPoll.Success, prefill_rank=0
        )
        self.assertNotIn(11, mgr.prefill_response_tracker)
        self.assertNotIn(11, mgr.request_status)
        mgr.agent.get_new_notifs.assert_not_called()

    def test_zmq_failure_preserves_reason(self):
        receiver, mgr = self._make_receiver()
        mgr.apply_prefill_status(
            bootstrap_room=11,
            status=KVPoll.Failed,
            prefill_rank=0,
            failure_reason="NIXL WRITE failed",
        )
        self.assertEqual(receiver.poll(), KVPoll.Failed)
        self.assertEqual(mgr.failure_records[11], "NIXL WRITE failed")

    def _run_decode_listener(self, mgr, messages):
        mgr.kv_args = SimpleNamespace(gpu_id=0)
        mgr.server_socket = MagicMock()
        mgr.server_socket.recv_multipart.side_effect = [*messages, SystemExit()]
        with (
            patch("sglang.srt.disaggregation.nixl.conn.threading.Thread") as thread,
            patch(
                "sglang.srt.disaggregation.nixl.conn.get_device",
                return_value=SimpleNamespace(device="cuda"),
            ),
            patch("sglang.srt.disaggregation.nixl.conn.torch.get_device_module"),
        ):
            mgr._start_decode_listener_thread()
            with self.assertRaises(SystemExit):
                thread.call_args.kwargs["target"]()
        mgr.server_socket.poll.assert_not_called()
        mgr.agent.get_new_notifs.assert_not_called()
        self.assertEqual(mgr.server_socket.recv_multipart.call_count, len(messages) + 1)

    def test_decode_listener_receives_zmq_chunk_and_success_without_agent_polling(self):
        receiver, mgr = self._make_receiver()
        mgr._staging_handler = MagicMock()
        self._run_decode_listener(
            mgr,
            [
                [b"CHUNK_READY", b"11", b"0", b"0", b"2", b"decode_agent", b"0"],
                [b"KV_STATUS", b"11", str(int(KVPoll.Success)).encode(), b"0", b""],
            ],
        )
        mgr._staging_handler.handle_chunk_arrived.assert_called_once_with(
            11, 0, 0, 2, "decode_agent"
        )
        self.assertEqual(mgr.request_status[11], KVPoll.Success)
        self.assertEqual(receiver.poll(), KVPoll.Success)
        mgr.agent.get_new_notifs.assert_not_called()

    def test_decode_listener_dispatches_zmq_failure_and_abort_ack(self):
        receiver, mgr = self._make_receiver()
        mgr._deferred_abort_ack_tracker = {11: set()}
        self._run_decode_listener(
            mgr,
            [
                [
                    b"KV_STATUS",
                    b"11",
                    str(int(KVPoll.Failed)).encode(),
                    b"0",
                    b"remote write failed",
                ],
                [b"ABORT_ACK", b"11", b"0"],
            ],
        )
        self.assertEqual(receiver.poll(), KVPoll.Failed)
        self.assertEqual(mgr.failure_records[11], "remote write failed")
        self.assertEqual(mgr._deferred_abort_ack_tracker[11], {0})
        mgr.agent.get_new_notifs.assert_not_called()

    def test_decode_listener_dispatches_zmq_staging_request(self):
        _, mgr = self._make_receiver()
        mgr.enable_staging = True
        mgr._handle_staging_req = MagicMock()
        message = [b"STAGING_REQ", b"11"]
        self._run_decode_listener(mgr, [message])
        mgr._handle_staging_req.assert_called_once_with(message)
        self.assertEqual(mgr.request_status[11], KVPoll.WaitingForInput)


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

    def test_late_failed_update_does_not_resurrect_cleared_room(self):
        mgr = object.__new__(CommonKVManager)
        mgr.request_status = {}

        CommonKVManager.update_status(mgr, 9, KVPoll.Failed)

        self.assertNotIn(9, mgr.request_status)


class TestNixlStaging(CustomTestCase):
    def _make_manager(self, agent=None):
        mgr = object.__new__(NixlKVManager)
        mgr.agent = agent or StagingFakeAgent()
        mgr.attn_tp_size = 2
        mgr.is_mla_backend = False
        mgr.transfer_source_rank = 1
        mgr.kv_args = SimpleNamespace(
            gpu_id=1,
            engine_rank=1,
            page_size=2,
            total_kv_head_num=2,
            kv_head_num=1,
        )
        mgr.server_args = SimpleNamespace(chunked_prefill_size=4)
        return mgr

    def test_register_buffer_to_engine_groups_kv_memory_kinds_in_one_pass(self):
        agent = StagingFakeAgent(register_result=["desc"])
        mgr = self._make_manager(agent)
        mgr.kv_args.kv_data_ptrs = [0x1000, 0x2000, 0x3000]
        mgr.kv_args.kv_data_lens = [64, 128, 256]
        mgr.kv_args.kv_data_mem_kinds = ["VRAM", "DRAM", "VRAM"]
        mgr.kv_args.aux_data_ptrs = [0x4000]
        mgr.kv_args.aux_data_lens = [32]
        mgr.kv_args.state_data_ptrs = []
        mgr.kv_args.state_data_lens = []

        mgr.register_buffer_to_engine()

        self.assertEqual(
            agent.register_memory_calls,
            [
                (
                    [(0x1000, 64, 1, ""), (0x3000, 256, 1, "")],
                    "VRAM",
                ),
                ([(0x2000, 128, 0, "")], "DRAM"),
                ([(0x4000, 32, 0, "")], "DRAM"),
            ],
        )
        self.assertEqual(mgr.kv_descs, [["desc"], ["desc"]])
        self.assertEqual(mgr.aux_descs, ["desc"])

    def test_register_staging_memory_uses_vram_and_fails_on_empty_descs(self):
        agent = StagingFakeAgent(register_result=["staging"])
        mgr = self._make_manager(agent)

        mgr._register_staging_memory(0x1000, 4096)

        self.assertEqual(
            agent.register_memory_calls,
            [([(0x1000, 4096, 1, "")], "VRAM")],
        )

        mgr = self._make_manager(StagingFakeAgent(register_result=[]))
        with self.assertRaisesRegex(RuntimeError, "staging buffer"):
            mgr._register_staging_memory(0x1000, 4096)

    def test_prefetch_staging_reqs_noops_when_disabled_or_missing_kv_buffers(self):
        mgr = self._make_manager()
        mgr.enable_staging = False
        mgr.kv_buffer_tensors = {"k_buffers": [], "v_buffers": [], "page_size": 1}

        mgr._prefetch_staging_reqs(3)

        mgr.enable_staging = True
        mgr.kv_buffer_tensors = None
        mgr._prefetch_staging_reqs(3)

    def test_prefetch_staging_reqs_marks_room_when_no_peer_needs_staging(self):
        mgr = self._make_manager()
        mgr.enable_staging = True
        mgr.kv_buffer_tensors = {"k_buffers": [], "v_buffers": [], "page_size": 1}
        mgr._staging_ctx = PrefillStagingContext()
        mgr.transfer_infos = {
            3: {
                "agent": TransferInfo(
                    room=3,
                    endpoint="127.0.0.1",
                    dst_port=1000,
                    agent_name="agent",
                    dst_kv_indices=np.array([1], dtype=np.int32),
                    dst_aux_index=0,
                    required_dst_info_num=1,
                    dst_state_indices=[],
                )
            }
        }
        mgr.decode_kv_args_table = {
            "agent": SimpleNamespace(decode_tp_size=2),
        }

        mgr._prefetch_staging_reqs(3)

        self.assertIn(3, mgr._staging_ctx.prefetched_rooms)

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
            chunk_id=0,
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
            chunk_id=0,
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

    def test_do_staging_transfer_returns_chunk_without_native_notification(self):
        mgr = self._make_manager()
        strategy = MagicMock()
        strategy.check_ready.return_value = (True, 2, 128, 0, 512)
        strategy.staging_buffer = FakeStagingBuffer()
        kv_chunk = TransferKVChunk(
            room=3,
            prefill_kv_indices=np.array([10, 11], dtype=np.int32),
            index_slice=slice(4, 6),
            is_last_chunk=True,
            chunk_id=7,
            prefill_aux_index=0,
            state_indices=None,
        )
        dst_info = KVArgsRegisterInfo(
            room="None",
            endpoint="127.0.0.1",
            dst_port=1000,
            agent_name="decode_agent",
            agent_metadata=b"",
            dst_kv_ptrs=[],
            dst_kv_mem_kinds=[],
            dst_aux_ptrs=[],
            dst_state_data_ptrs=[],
            gpu_id=5,
            decode_tp_size=1,
            decode_tp_rank=0,
            dst_kv_item_len=128,
            dst_kv_item_lens=[],
            staging_base_ptr=0x8000,
            staging_total_size=4096,
        )
        calls = []
        mgr.send_kvcache_staged = lambda *args, **kwargs: (
            calls.append((args, kwargs)) or "handle"
        )

        handle, deferred, chunk_idx = mgr._do_staging_transfer(
            strategy,
            kv_chunk,
            kv_chunk.prefill_kv_indices,
            SimpleNamespace(room=3, agent_name="decode_agent"),
            dst_info,
            FakeQueue(),
        )

        self.assertEqual(handle, "handle")
        self.assertFalse(deferred)
        self.assertEqual(chunk_idx, 2)
        self.assertEqual(calls[0][0][8], "")

    def test_send_kvcache_staged_uses_one_bulk_vram_write(self):
        mock_gather = MagicMock()
        agent = StagingFakeAgent()
        mgr = self._make_manager(agent)
        mgr.kv_buffer_tensors = {
            "k_buffers": [FakeTensor(), FakeTensor()],
            "v_buffers": [FakeTensor(), FakeTensor()],
            "page_size": 2,
        }

        with patch.dict(
            sys.modules,
            {
                "sglang.srt.disaggregation.common.staging_buffer": (
                    _fake_staging_buffer_module(mock_gather)
                )
            },
        ):
            handle = mgr.send_kvcache_staged(
                "peer",
                np.array([1, 2], dtype=np.int32),
                dst_staging_ptr=0x100000,
                dst_staging_size=1 << 20,
                dst_gpu_id=4,
                dst_tp_rank=0,
                dst_attn_tp_size=1,
                dst_kv_item_len=128,
                notif="3_stg_0_1_1_0_0_2_decode_agent",
                staging_buffer=FakeStagingBuffer(ptr=0x9000, size=1 << 20),
            )

        self.assertEqual(handle, "handle")
        mock_gather.assert_called_once()
        src_reqs, src_mem = agent.get_xfer_descs_calls[0]
        dst_reqs, dst_mem = agent.get_xfer_descs_calls[1]
        self.assertEqual(src_mem, "VRAM")
        self.assertEqual(dst_mem, "VRAM")
        self.assertEqual(src_reqs.shape, (1, 3))
        self.assertEqual(dst_reqs.shape, (1, 3))
        self.assertTrue(np.issubdtype(src_reqs.dtype, np.integer))
        self.assertTrue(np.issubdtype(dst_reqs.dtype, np.integer))
        self.assertEqual(int(src_reqs[0, 0]), 0x9000)
        self.assertGreaterEqual(int(dst_reqs[0, 0]), 0x100000)
        self.assertEqual(agent.initialize_xfer_calls[0][0], "WRITE")
        self.assertEqual(
            agent.initialize_xfer_calls[0][-1],
            b"3_stg_0_1_1_0_0_2_decode_agent",
        )

    def test_send_kvcache_staged_falls_back_when_prefill_buffer_too_small(self):
        mgr = self._make_manager()
        mgr.kv_buffer_tensors = {
            "k_buffers": [FakeTensor(), FakeTensor()],
            "v_buffers": [FakeTensor(), FakeTensor()],
            "page_size": 2,
        }

        with patch.dict(
            sys.modules,
            {
                "sglang.srt.disaggregation.common.staging_buffer": (
                    _fake_staging_buffer_module()
                )
            },
        ):
            handle = mgr.send_kvcache_staged(
                "peer",
                np.array([1, 2], dtype=np.int32),
                dst_staging_ptr=0xA000,
                dst_staging_size=1 << 20,
                dst_gpu_id=4,
                dst_tp_rank=0,
                dst_attn_tp_size=1,
                dst_kv_item_len=128,
                notif="notif",
                staging_buffer=FakeStagingBuffer(size=1),
            )

        self.assertIsNone(handle)


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

        # Every source descriptor [addr, addr+len) must lie inside a registered
        # base region [ptr, ptr+REGION_LEN). Pre-fix, num_groups=4 pushed the
        # top group's addresses past the region -> NIXL_ERR_NOT_FOUND.
        src_call = next(c for c in mgr.agent.calls if c[0] == "")
        src_array = src_call[1]
        regions = [(p, p + self.REGION_LEN) for p in self.SRC_PTRS]
        for addr, length, _dev in src_array:
            addr = int(addr)
            length = int(length)
            self.assertTrue(
                any(lo <= addr and addr + length <= hi for lo, hi in regions),
                f"descriptor [{addr:#x}, {addr + length:#x}) escapes all "
                f"registered source regions {[(hex(lo), hex(hi)) for lo, hi in regions]}",
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


if __name__ == "__main__":
    unittest.main()
