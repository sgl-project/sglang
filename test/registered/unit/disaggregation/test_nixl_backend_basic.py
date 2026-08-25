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

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.common.staging_handler import PrefillStagingContext
from sglang.srt.disaggregation.common.utils import pack_int_lists
from sglang.srt.disaggregation.nixl.conn import (
    KVArgsRegisterInfo,
    NixlKVManager,
    NixlKVReceiver,
    NixlKVSender,
    TransferInfo,
    TransferKVChunk,
    TransferStatus,
    _set_rank_local_gpunetio_oob_port,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=23, suite="base-a-test-cpu")


class NotificationFakeAgent:
    def __init__(self, messages):
        self.messages = messages

    def get_new_notifs(self):
        return {"peer": [msg.encode("ascii") for msg in self.messages]}


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


class TestDcpGpunetioPeerRows(CustomTestCase):
    def test_default_keeps_legacy_dcp_pack(self):
        mgr = object.__new__(CommonKVManager)
        with envs.SGLANG_DISAGG_DCP_GPUNETIO_PEER_ROWS.override(False):
            with envs.SGLANG_DISAGG_DCP_PACK.override(True):
                mgr._configure_dcp_pack_mode()

        self.assertFalse(mgr.enable_dcp_peer_rows)
        self.assertFalse(mgr.enable_dcp_gpunetio_batch_post)
        self.assertTrue(mgr.enable_dcp_pack)

    def test_peer_rows_rejects_non_gpunetio_backend(self):
        mgr = object.__new__(CommonKVManager)
        with envs.SGLANG_DISAGG_DCP_GPUNETIO_PEER_ROWS.override(True):
            with envs.SGLANG_DISAGGREGATION_NIXL_BACKEND.override("UCX"):
                with self.assertRaisesRegex(ValueError, "requires .*GPUNETIO"):
                    mgr._configure_dcp_pack_mode()

    def test_batch_post_requires_peer_rows(self):
        mgr = object.__new__(CommonKVManager)
        with envs.SGLANG_DISAGG_DCP_GPUNETIO_PEER_ROWS.override(False):
            with envs.SGLANG_DISAGG_DCP_GPUNETIO_BATCH_POST.override(True):
                with self.assertRaisesRegex(ValueError, "requires .*PEER_ROWS"):
                    mgr._configure_dcp_pack_mode()

    def test_peer_rows_skips_pack_and_keeps_cyclic_indices(self):
        mgr = object.__new__(NixlKVManager)
        mgr.enable_dcp_peer_rows = True
        mgr.enable_dcp_pack = False
        mgr._dcp_pack_buffers = [object()]
        mgr.kv_args = SimpleNamespace(
            page_size=64,
            kv_data_ptrs=[0x1000],
            gpu_id=0,
        )
        mgr.src_mem_kind = "VRAM"
        mgr._publish_ready_epoch = MagicMock(return_value=0x3000)
        mgr._send_kvcache_generic = MagicMock(return_value="transfer")
        dst_info = SimpleNamespace(
            dst_homogeneous_mem_kind="VRAM",
            dst_dcp_size=4,
            dst_dcp_rank=0,
            dcp_token_item_lens=[16],
            dst_kv_ptrs=[0x2000],
            dcp_dst_region_indices=[0],
            gpu_id=0,
            ready_enabled=True,
            ready_slot_count=8,
            ready_base_ptr=0x4000,
        )

        with patch(
            "sglang.srt.disaggregation.common.dcp_pack.try_pack_dcp_src"
        ) as pack:
            result = mgr.send_kvcache_dcp(
                "decode",
                np.arange(4, dtype=np.int32),
                dst_info,
                np.array([7], dtype=np.int32),
                src_page_offset=0,
                decode_prefix_len=0,
                num_kv_tokens=256,
                notif="ready",
                pack_buffer=mgr._dcp_pack_buffer_for_worker(0),
                ready_slot=1,
                ready_epoch=7,
                ready_src_ptr=0x3000,
                post=False,
            )

        self.assertEqual(result, "transfer")
        pack.assert_not_called()
        mgr._publish_ready_epoch.assert_not_called()
        args = mgr._send_kvcache_generic.call_args.kwargs
        np.testing.assert_array_equal(
            args["prefill_data_indices"], np.arange(0, 256, 4, dtype=np.int32)
        )
        np.testing.assert_array_equal(
            args["dst_data_indices"], np.arange(448, 512, dtype=np.int32)
        )
        self.assertEqual(args["ready_tail"], (0x3000, 0x4008))
        self.assertFalse(args["post"])


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

        self.assertFalse(info.is_dummy())

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

        self.assertTrue(info.is_dummy())

    def test_trailing_ready_fields_are_backward_compatible(self):
        base = [
            b"11",
            b"127.0.0.1",
            b"12349",
            b"agent",
            np.array([1], dtype=np.int32).tobytes(),
            b"0",
            b"1",
            b"",
            b"0",
            b"0",
        ]
        legacy = TransferInfo.from_zmq(base)
        self.assertIsNone(legacy.ready_slot)
        self.assertIsNone(legacy.ready_epoch)

        ready = TransferInfo.from_zmq(base + [b"3", b"7"])
        self.assertEqual((ready.ready_slot, ready.ready_epoch), (3, 7))


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

    def test_trailing_ready_capability_is_parsed(self):
        msg = [
            b"None",
            b"10.0.0.4",
            b"23458",
            b"agent",
            b"metadata",
            struct.pack("Q", 0x1000),
            struct.pack("Q", 0x2000),
            b"",
            b"0",
            b"1",
            b"0",
            b"256",
            b"",
            b"",
            b"",
            b"",
            b"64",
            b"VRAM",
            b"",
            b"",
            b"",
            b"1",
            b"0",
        ]
        legacy = KVArgsRegisterInfo.from_zmq(msg)
        self.assertFalse(legacy.ready_enabled)

        ready = KVArgsRegisterInfo.from_zmq(
            msg + [b"1", struct.pack("Q", 0xABC0), b"64", b"8"]
        )
        self.assertTrue(ready.ready_enabled)
        self.assertEqual((ready.ready_base_ptr, ready.ready_slot_count), (0xABC0, 64))


class TestNixlTransferStatus(CustomTestCase):
    def test_not_done_until_aux_and_expected_count_arrive(self):
        status = TransferStatus()

        self.assertFalse(status.is_done())

        status.received_aux = True
        self.assertFalse(status.is_done())

        status.num_pp_ranks_expected = 1
        self.assertFalse(status.is_done())

        status.expected_kvs_per_pp[0] = 1
        self.assertFalse(status.is_done())

        status.received_kvs_per_pp[0].add(0)
        self.assertTrue(status.is_done())

    def test_zero_kv_aux_only_completion(self):
        status = TransferStatus()
        status.received_aux = True
        status.num_pp_ranks_expected = 1
        status.expected_kvs_per_pp[0] = 0

        self.assertTrue(status.is_done())

    def test_multi_pp_requires_each_rank_expected_chunks(self):
        status = TransferStatus()
        status.received_aux = True
        status.num_pp_ranks_expected = 2
        status.expected_kvs_per_pp[0] = 1
        status.received_kvs_per_pp[0].add(0)

        self.assertFalse(status.is_done())

        status.expected_kvs_per_pp[1] = 2
        status.received_kvs_per_pp[1].update({0, 1})
        self.assertTrue(status.is_done())

    def test_state_required_completion_waits_for_all_pp_ranks(self):
        status = TransferStatus()
        status.received_aux = True
        status.num_pp_ranks_expected = 2
        status.expected_kvs_per_pp[0] = 0
        status.expected_kvs_per_pp[1] = 0
        status.expects_state = True

        self.assertFalse(status.is_done())

        status.received_state_per_pp.add(0)
        self.assertFalse(status.is_done())

        status.received_state_per_pp.add(1)
        self.assertTrue(status.is_done())


class TestDcpPeerRowReadyProtocol(CustomTestCase):
    def _make_ready_manager(self):
        mgr = object.__new__(NixlKVManager)
        mgr._ready_pool = object()
        mgr._ready_slot_count = 2
        mgr._ready_next_epoch = [0, 0]
        mgr._ready_leases = {}
        mgr._ready_poisoned_slots = set()
        return mgr

    def test_ready_slot_bounds_and_epoch_monotonicity(self):
        mgr = self._make_ready_manager()
        with self.assertRaisesRegex(ValueError, "outside aux-slot bounds"):
            mgr.reserve_ready_lease(1, 2)

        self.assertEqual(mgr.reserve_ready_lease(1, 0), (0, 1))
        with self.assertRaisesRegex(RuntimeError, "is leased"):
            mgr.reserve_ready_lease(2, 0)
        mgr._ready_leases.pop((1, 0, -1))
        self.assertEqual(mgr.reserve_ready_lease(2, 0), (0, 2))

    def test_peer_rows_requires_rank_local_oob_port(self):
        params = {"oob_port": "6544"}
        _set_rank_local_gpunetio_oob_port(params, 3)
        self.assertEqual(params["oob_port"], "6547")
        with self.assertRaisesRegex(ValueError, "explicit base"):
            _set_rank_local_gpunetio_oob_port({}, 0)

    def test_peer_rows_require_ready_capability(self):
        mgr = object.__new__(NixlKVManager)
        mgr.enable_dcp_peer_rows = True
        mgr.decode_kv_args_table = {}
        peer = SimpleNamespace(agent_name="peer", ready_enabled=False)
        with self.assertRaisesRegex(RuntimeError, "ready-pool capable"):
            mgr._add_remote_peer(peer)

    def test_notification_waits_before_arrival_and_releases_lease(self):
        mgr = object.__new__(NixlKVManager)
        mgr.agent = NotificationFakeAgent(["9_kv_0_1_3"])
        mgr.transfer_statuses = defaultdict(TransferStatus)
        mgr.required_prefill_response_num_table = {9: 1}
        mgr.enable_staging = False
        mgr._staging_handler = None
        mgr._chunk_writer_counts = defaultdict(lambda: defaultdict(list))
        mgr.enable_dcp_peer_rows = True
        mgr._ready_leases = {(9, 0, 3): (1, 4)}
        calls = []
        mgr._wait_for_peer_row_ready = lambda *args: calls.append(args)

        mgr.update_transfer_status()

        self.assertEqual(calls, [(9, 0, True, 3)])
        self.assertEqual(mgr.transfer_statuses[9].received_kvs_per_pp[3], {0})
        self.assertNotIn((9, 0, 3), mgr._ready_leases)

    def test_notification_gate_failure_fails_before_arrival(self):
        mgr = object.__new__(NixlKVManager)
        mgr.agent = NotificationFakeAgent(["10_kv_0_1_0"])
        mgr.transfer_statuses = defaultdict(TransferStatus)
        mgr.required_prefill_response_num_table = {10: 1}
        mgr.enable_staging = False
        mgr._staging_handler = None
        mgr._chunk_writer_counts = defaultdict(lambda: defaultdict(list))
        mgr.enable_dcp_peer_rows = True
        mgr.request_status = {10: KVPoll.WaitingForInput}
        mgr.failure_records = {}
        mgr.failure_lock = threading.Lock()
        mgr._wait_for_peer_row_ready = MagicMock(side_effect=RuntimeError("stale"))

        mgr.update_transfer_status()

        self.assertEqual(mgr.request_status[10], KVPoll.Failed)
        self.assertEqual(mgr.transfer_statuses[10].received_kvs_per_pp[0], set())
        self.assertIn("stale", mgr.failure_records[10])

    def test_sender_rejects_multi_chunk_or_state(self):
        sender = object.__new__(NixlKVSender)
        sender.kv_mgr = SimpleNamespace(enable_dcp_peer_rows=True)
        sender._send_failed = False
        sender._prepare_send_indices = MagicMock(
            return_value=(np.array([1], dtype=np.int32), slice(0, 1), False, False)
        )
        sender.chunk_id = 0
        with self.assertRaisesRegex(ValueError, "exactly one KV chunk"):
            sender.send(np.array([1], dtype=np.int32))

    def test_ready_tail_is_last_and_uses_fixed_gpunetio_qp(self):
        class TailAgent:
            def __init__(self):
                self.descs = []
                self.kwargs = None

            def get_xfer_descs(self, reqs, _mem_kind):
                self.descs.append(np.asarray(reqs))
                return reqs

            def initialize_xfer(self, *args, **kwargs):
                self.kwargs = kwargs
                return "handle"

            def transfer(self, _handle):
                return "DONE"

        mgr = object.__new__(NixlKVManager)
        mgr.agent = TailAgent()
        mgr.is_mla_backend = True
        mgr.kv_args = SimpleNamespace(gpu_id=0)
        mgr.get_mla_kv_ptrs_with_pp = lambda src, dst, state: (src, dst, 1)
        handle = mgr._send_kvcache_generic(
            peer_name="peer",
            src_data_ptrs=[0x1000],
            dst_data_ptrs=[0x2000],
            item_lens=[16],
            prefill_data_indices=np.array([1], dtype=np.int32),
            dst_data_indices=np.array([3], dtype=np.int32),
            dst_gpu_id=1,
            notif="9_kv_0_1_0",
            force_flat=True,
            bypass_prepped=True,
            ready_tail=(0x3000, 0x4000),
        )

        self.assertEqual(handle, "handle")
        self.assertEqual(tuple(mgr.agent.descs[0][-1]), (0x3000, 8, 0))
        self.assertEqual(tuple(mgr.agent.descs[1][-1]), (0x4000, 8, 1))
        self.assertEqual(
            mgr.agent.kwargs,
            {"backends": ["GPUNETIO"], "custom_param": b"gpunetio_qp=0"},
        )

    def test_generic_can_create_without_posting(self):
        class NoPostAgent:
            def get_xfer_descs(self, reqs, _mem_kind):
                return reqs

            def initialize_xfer(self, *args, **kwargs):
                return "handle"

            def transfer(self, _handle):
                raise AssertionError("deferred transfer was posted")

        mgr = object.__new__(NixlKVManager)
        mgr.agent = NoPostAgent()
        mgr.is_mla_backend = True
        mgr.kv_args = SimpleNamespace(gpu_id=0)
        mgr.get_mla_kv_ptrs_with_pp = lambda src, dst, state: (src, dst, 1)

        handle = mgr._send_kvcache_generic(
            peer_name="peer",
            src_data_ptrs=[0x1000],
            dst_data_ptrs=[0x2000],
            item_lens=[16],
            prefill_data_indices=np.array([1], dtype=np.int32),
            dst_data_indices=np.array([3], dtype=np.int32),
            dst_gpu_id=1,
            notif="9_kv_0_1_0",
            force_flat=True,
            bypass_prepped=True,
            ready_tail=(0x3000, 0x4000),
            post=False,
        )

        self.assertEqual(handle, "handle")


class TestNixlKVSenderChunkPolicy(CustomTestCase):
    def test_last_zero_page_chunk_is_sent_for_aux_only_completion(self):
        sender = object.__new__(NixlKVSender)

        self.assertTrue(sender.should_send_kv_chunk(0, last_chunk=True))
        self.assertFalse(sender.should_send_kv_chunk(0, last_chunk=False))
        self.assertTrue(sender.should_send_kv_chunk(3, last_chunk=False))


class TestDcpGpunetioBatchPost(CustomTestCase):
    def test_worker_rejects_non_four_peer_topology(self):
        room = 32
        mgr = object.__new__(NixlKVManager)
        mgr.enable_dcp_gpunetio_batch_post = True
        mgr.enable_staging = False
        mgr.request_status = {room: KVPoll.WaitingForInput}
        mgr.transfer_infos = {
            room: {
                "peer0": TransferInfo(
                    room=room,
                    endpoint="127.0.0.1",
                    dst_port=1000,
                    agent_name="peer0",
                    dst_kv_indices=np.array([0], dtype=np.int32),
                    dst_aux_index=0,
                    required_dst_info_num=1,
                    dst_state_indices=[],
                    is_dummy_rank=False,
                )
            }
        }
        mgr.exceptions = {}
        mgr.failure_lock = threading.Lock()
        mgr.failure_records = {}
        mgr._staging_outstanding = defaultdict(int)
        mgr.enable_deferred_decode_kv_release = False
        mgr.check_status = MagicMock(return_value=KVPoll.WaitingForInput)
        mgr.update_status = MagicMock()
        mgr.record_failure = MagicMock()

        chunk = TransferKVChunk(
            room=room,
            prefill_kv_indices=np.array([0], dtype=np.int32),
            index_slice=slice(0, 1),
            is_last_chunk=True,
            chunk_id=0,
            prefill_aux_index=0,
            state_indices=[],
            num_kv_tokens=1,
        )
        queue = SimpleNamespace(get=MagicMock(side_effect=[chunk, SystemExit()]))

        with self.assertRaises(SystemExit):
            mgr.transfer_worker(queue)

        self.assertIn("exactly four DCP peers", str(mgr.exceptions[room]))

    def test_worker_batches_four_dcp_handles_before_aux(self):
        room = 31
        mgr = object.__new__(NixlKVManager)
        mgr.enable_dcp_peer_rows = True
        mgr.enable_dcp_gpunetio_batch_post = True
        mgr.enable_staging = False
        mgr.enable_deferred_decode_kv_release = False
        mgr.is_mla_backend = True
        mgr.is_hybrid_mla_backend = False
        mgr.attn_tp_size = 1
        mgr.transfer_source_rank = 0
        mgr.kv_args = SimpleNamespace(engine_rank=0, kv_data_ptrs=[0x1000])
        mgr.request_status = {room: KVPoll.WaitingForInput}
        mgr.transfer_infos = {}
        mgr.decode_kv_args_table = {}
        mgr.req_to_decode_prefix_len = {room: 0}
        mgr._staging_ctx = None
        mgr._staging_outstanding = defaultdict(int)
        mgr.exceptions = {}
        mgr.failure_lock = threading.Lock()
        mgr.failure_records = {}
        mgr._publish_ready_epoch = MagicMock(
            side_effect=lambda slot, _epoch: 0x3000 + slot * 8
        )

        reqs = {}
        dst_infos = {}
        for peer_idx in range(4):
            peer_name = f"peer{peer_idx}"
            reqs[peer_name] = TransferInfo(
                room=room,
                endpoint="127.0.0.1",
                dst_port=1000 + peer_idx,
                agent_name=peer_name,
                dst_kv_indices=np.array([peer_idx], dtype=np.int32),
                dst_aux_index=peer_idx,
                required_dst_info_num=1,
                dst_state_indices=[],
                decode_prefix_len=0,
                is_dummy_rank=False,
                ready_slot=peer_idx,
                ready_epoch=7,
            )
            dst_infos[peer_name] = SimpleNamespace(
                decode_tp_size=1,
                staging_base_ptr=0,
                staging_total_size=0,
                requires_dcp_relayout=True,
                ready_enabled=True,
                ready_slot_count=8,
                ready_base_ptr=0x4000,
                dst_aux_ptrs=[0x5000],
            )
        mgr.transfer_infos[room] = reqs
        mgr.decode_kv_args_table = dst_infos

        events = []
        dcp_handles = iter(["dcp0", "dcp1", "dcp2", "dcp3"])
        aux_handles = iter(["aux0", "aux1", "aux2", "aux3"])

        def send_dcp(*_args, **kwargs):
            events.append(("dcp", kwargs["post"]))
            return next(dcp_handles)

        def send_aux(*_args):
            events.append("aux")
            return next(aux_handles)

        def batch_post(handles):
            events.append(("batch", list(handles)))
            return "PROC"

        def check_state(handle):
            events.append(("check", handle))
            return "DONE"

        mgr.send_kvcache_dcp = MagicMock(side_effect=send_dcp)
        mgr.send_aux = MagicMock(side_effect=send_aux)
        mgr.agent = SimpleNamespace(
            transfer_batch4_gpunetio_experimental=batch_post,
            check_xfer_state=check_state,
        )

        chunk = TransferKVChunk(
            room=room,
            prefill_kv_indices=np.array([2], dtype=np.int32),
            index_slice=slice(0, 1),
            is_last_chunk=True,
            chunk_id=0,
            prefill_aux_index=0,
            state_indices=[],
            num_kv_tokens=1,
        )
        queue = SimpleNamespace(get=MagicMock(side_effect=[chunk, SystemExit()]))

        with self.assertRaises(SystemExit):
            mgr.transfer_worker(queue)

        self.assertEqual(
            [
                event
                for event in events
                if isinstance(event, tuple) and event[0] == "dcp"
            ],
            [("dcp", False)] * 4,
        )
        batch_events = [event for event in events if event[0] == "batch"]
        self.assertEqual(batch_events, [("batch", ["dcp0", "dcp1", "dcp2", "dcp3"])])
        self.assertEqual(events[4], ("batch", ["dcp0", "dcp1", "dcp2", "dcp3"]))
        self.assertEqual(events[5:9], ["aux"] * 4)
        self.assertEqual(mgr.send_aux.call_count, 4)


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
        mgr.kv_args = SimpleNamespace(engine_rank=0, kv_data_ptrs=[0])
        mgr.exceptions = {}
        mgr.failure_lock = threading.Lock()
        mgr.failure_records = {}

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
        self.assertEqual(mgr.send_aux.call_args.args[-1], "21_aux_nokv_0_0")

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


class TestNixlNotifications(CustomTestCase):
    def _make_manager(self, messages, required=None):
        mgr = object.__new__(NixlKVManager)
        mgr.agent = NotificationFakeAgent(messages)
        mgr.transfer_statuses = defaultdict(TransferStatus)
        mgr.required_prefill_response_num_table = required or {}
        mgr.enable_staging = False
        mgr._staging_handler = None
        mgr._chunk_writer_counts = defaultdict(lambda: defaultdict(list))
        return mgr

    def test_kv_last_notification_sets_expected_count(self):
        mgr = self._make_manager(["5_kv_2_1_0"])

        mgr.update_transfer_status()

        status = mgr.transfer_statuses[5]
        self.assertEqual(status.received_kvs_per_pp[0], {2})
        self.assertEqual(status.expected_kvs_per_pp[0], 3)
        self.assertEqual(status.num_pp_ranks_expected, 1)

    def test_staging_notification_preserves_agent_name_with_underscores(self):
        mgr = self._make_manager(["5_stg_0_1_0_2_4_8_agent_with_underscores"])
        calls = []
        mgr._handle_staging_chunk_arrived = lambda *args: calls.append(args)

        mgr.update_transfer_status()

        self.assertEqual(calls, [(5, 2, 4, 8, "agent_with_underscores")])
        status = mgr.transfer_statuses[5]
        self.assertEqual(status.received_kvs_per_pp[0], {0})
        self.assertEqual(status.expected_kvs_per_pp[0], 1)

    def test_aux_nokv_marks_zero_expected_chunks_for_pp_rank(self):
        mgr = self._make_manager(["6_aux_nokv_3"], required={6: 4})

        mgr.update_transfer_status()

        status = mgr.transfer_statuses[6]
        self.assertTrue(status.received_aux)
        self.assertEqual(status.expected_kvs_per_pp[3], 0)
        self.assertEqual(status.num_pp_ranks_expected, 4)

    def test_state_notification_marks_pp_rank(self):
        mgr = self._make_manager(["7_state_2"])

        mgr.update_transfer_status()

        self.assertEqual(mgr.transfer_statuses[7].received_state_per_pp, {2})

    def test_aux_nokv_allows_full_hit_completion(self):
        mgr = self._make_manager(["8_aux_nokv_0"], required={8: 1})

        mgr.update_transfer_status()

        self.assertTrue(mgr.transfer_statuses[8].is_done())


class TestNixlReceiverPoll(CustomTestCase):
    def _make_receiver(self, status=KVPoll.WaitingForInput):
        mgr = MagicMock()
        mgr.waiting_timeout = 5
        mgr.check_status.return_value = status
        mgr.check_transfer_done.return_value = False
        mgr.transfer_statuses = {}
        mgr.addr_to_rooms_tracker = defaultdict(set)
        mgr.addr_to_rooms_tracker["prefill:8998"].add(11)

        receiver = object.__new__(NixlKVReceiver)
        receiver.kv_mgr = mgr
        receiver.bootstrap_room = 11
        receiver.bootstrap_addr = "prefill:8998"
        receiver.started_transfer = False
        receiver.init_time = None
        receiver.conclude_state = None
        receiver.abort_notified = False
        receiver._connection_pool_entries = {}
        return receiver, mgr

    def test_returns_existing_conclude_state_without_polling_manager(self):
        receiver, mgr = self._make_receiver()
        receiver.conclude_state = KVPoll.Success

        self.assertEqual(receiver.poll(), KVPoll.Success)
        mgr.check_status.assert_not_called()

    def test_returns_bootstrap_status_before_transfer_starts(self):
        receiver, mgr = self._make_receiver(status=KVPoll.Bootstrapping)

        self.assertEqual(receiver.poll(), KVPoll.Bootstrapping)
        mgr.update_transfer_status.assert_not_called()

    def test_manager_success_or_failed_status_is_terminal(self):
        for terminal_status in (KVPoll.Success, KVPoll.Failed):
            receiver, _ = self._make_receiver(status=terminal_status)

            self.assertEqual(receiver.poll(), terminal_status)
            self.assertEqual(receiver.conclude_state, terminal_status)

    @patch("sglang.srt.disaggregation.nixl.conn.time.time")
    def test_waiting_timeout_records_failure(self, mock_time):
        mock_time.return_value = 20.0
        receiver, mgr = self._make_receiver(status=KVPoll.WaitingForInput)
        receiver.started_transfer = True
        receiver.init_time = 10.0

        self.assertEqual(receiver.poll(), KVPoll.Failed)
        mgr.record_failure.assert_called_once()
        self.assertIn("timed out", mgr.record_failure.call_args[0][1])
        mgr.update_status.assert_called_once_with(11, KVPoll.Failed)

    @patch("sglang.srt.disaggregation.nixl.conn.time.time")
    def test_queued_completion_wins_over_waiting_timeout(self, mock_time):
        # Past the deadline, but the completion is already queued/observed:
        # draining before the timeout check must yield Success, not a false
        # timeout, and must not send an abort.
        mock_time.return_value = 20.0
        receiver, mgr = self._make_receiver(status=KVPoll.WaitingForInput)
        receiver.started_transfer = True
        receiver.init_time = 10.0
        mgr.transfer_statuses = {11: TransferStatus()}
        mgr.check_transfer_done.return_value = True

        self.assertEqual(receiver.poll(), KVPoll.Success)
        mgr.update_transfer_status.assert_called_once_with()
        mgr.record_failure.assert_not_called()
        mgr.update_status.assert_not_called()
        self.assertNotIn(11, mgr.transfer_statuses)

    @patch("sglang.srt.disaggregation.nixl.conn.time.time")
    def test_transfer_done_returns_success_and_cleans_room_state(self, mock_time):
        mock_time.return_value = 12.0
        receiver, mgr = self._make_receiver(status=KVPoll.WaitingForInput)
        receiver.started_transfer = True
        receiver.init_time = 10.0
        status = TransferStatus()
        status.received_aux = True
        status.num_pp_ranks_expected = 1
        status.expected_kvs_per_pp[0] = 0
        mgr.transfer_statuses = {11: status}
        mgr.check_transfer_done.return_value = True

        self.assertEqual(receiver.poll(), KVPoll.Success)
        self.assertNotIn(11, mgr.transfer_statuses)
        self.assertNotIn(11, mgr.addr_to_rooms_tracker["prefill:8998"])
        self.assertEqual(receiver.conclude_state, KVPoll.Success)


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

        mgr._register_staging_memory(0x1000, 4096, 3)

        self.assertEqual(
            agent.register_memory_calls,
            [([(0x1000, 4096, 3, "")], "VRAM")],
        )

        mgr = self._make_manager(StagingFakeAgent(register_result=[]))
        with self.assertRaisesRegex(RuntimeError, "staging buffer"):
            mgr._register_staging_memory(0x1000, 4096, 3)

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
            handle, deferred = mgr._do_staging_transfer(
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

    def test_do_staging_transfer_builds_staging_notification(self):
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
        mgr.send_kvcache_staged = (
            lambda *args, **kwargs: calls.append((args, kwargs)) or "handle"
        )

        handle, deferred = mgr._do_staging_transfer(
            strategy,
            kv_chunk,
            kv_chunk.prefill_kv_indices,
            SimpleNamespace(room=3, agent_name="decode_agent"),
            dst_info,
            FakeQueue(),
        )

        self.assertEqual(handle, "handle")
        self.assertFalse(deferred)
        self.assertEqual(calls[0][0][8], "3_stg_7_1_1_2_4_2_decode_agent")

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
