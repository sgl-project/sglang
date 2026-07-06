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
)
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
        self.assertIsNotNone(info.staging)
        self.assertEqual(info.staging.base_ptr, staging_ptr)
        self.assertEqual(info.staging.total_size, 1048576)

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
        self.assertIsNone(info.staging)


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


class TestNixlKVSenderChunkPolicy(CustomTestCase):
    def test_last_zero_page_chunk_is_sent_for_aux_only_completion(self):
        sender = object.__new__(NixlKVSender)

        self.assertTrue(sender.should_send_kv_chunk(0, last_chunk=True))
        self.assertFalse(sender.should_send_kv_chunk(0, last_chunk=False))
        self.assertTrue(sender.should_send_kv_chunk(3, last_chunk=False))


class TestNixlBootstrapMessages(CustomTestCase):
    def _make_manager(self, request_status):
        mgr = object.__new__(NixlKVManager)
        mgr.request_status = request_status
        mgr.record_failure = MagicMock()
        mgr.update_status = MagicMock()
        return mgr

    def _run_bootstrap_message(self, mgr, message):
        mgr.server_socket = MagicMock()
        mgr.server_socket.recv_multipart.side_effect = [message, StopIteration]

        with patch(
            "sglang.srt.disaggregation.nixl.conn.threading.Thread"
        ) as mock_thread:
            mgr._start_bootstrap_thread()

        bootstrap_target = mock_thread.call_args.kwargs["target"]
        with self.assertRaises(StopIteration):
            bootstrap_target()

    def test_abort_marks_active_room_failed(self):
        for status in (
            KVPoll.Bootstrapping,
            KVPoll.WaitingForInput,
            KVPoll.Transferring,
        ):
            with self.subTest(status=status):
                mgr = self._make_manager({7: status})

                self._run_bootstrap_message(
                    mgr, [b"ABORT", b"7", b"127.0.0.1", b"12345"]
                )

                mgr.record_failure.assert_called_once_with(
                    7, "Request 7 was aborted by decode."
                )
                mgr.update_status.assert_called_once_with(7, KVPoll.Failed)

    def test_abort_ignores_terminal_or_unknown_room(self):
        for request_status in ({7: KVPoll.Failed}, {7: KVPoll.Success}, {}):
            with self.subTest(request_status=request_status):
                mgr = self._make_manager(request_status)

                self._run_bootstrap_message(
                    mgr, [b"ABORT", b"7", b"127.0.0.1", b"12345"]
                )

                mgr.record_failure.assert_not_called()
                mgr.update_status.assert_not_called()

    def test_malformed_abort_is_ignored(self):
        for message in (
            [b"ABORT"],
            [b"ABORT", b"not-a-room", b"127.0.0.1", b"12345"],
            [b"ABORT", b"7", b"127.0.0.1", b"not-a-port"],
        ):
            with self.subTest(message=message):
                mgr = self._make_manager({7: KVPoll.WaitingForInput})

                self._run_bootstrap_message(mgr, message)

                mgr.record_failure.assert_not_called()
                mgr.update_status.assert_not_called()


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
            staging=SimpleNamespace(base_ptr=0x8000, total_size=4096),
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


if __name__ == "__main__":
    unittest.main()
