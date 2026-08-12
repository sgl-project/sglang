"""Basic CPU unit tests for NIXL disaggregation control paths."""

import asyncio
import json
import struct
import sys
import threading
import types
import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import zmq

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import (
    BOOTSTRAP_INSTANCE_ID_HEADER,
    CommonKVBootstrapServer,
    CommonKVManager,
    CommonKVReceiver,
    InstanceProbeVerdict,
    PrefillRankInfo,
    PrefillServerInfo,
)
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
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=23, suite="base-a-test-cpu")


class TestNixlBootstrapGeneration(CustomTestCase):
    def test_all_bootstrap_responses_report_same_instance(self):
        """Completeness contract: the decode validates /health, the topology
        fetch, the per-rank fetch, and the dp-rank query against one instance
        id; a handler that stops stamping silently disables that path."""
        server = object.__new__(CommonKVBootstrapServer)
        server.instance_id = "prefill-generation"
        server.attn_tp_size = 1
        server.attn_cp_size = 1
        server.dp_size = 1
        server.pp_size = 1
        server.page_size = 1
        server.kv_cache_dtype = "auto"
        server.follow_bootstrap_room = True
        server.enable_dsa_cache_layer_split = False
        server.prefill_http_port = 30000
        server._registered_count = 1

        health = asyncio.run(server._handle_health_check(None))
        topology = asyncio.run(
            server._handle_route_get(
                SimpleNamespace(
                    query={
                        "prefill_dp_rank": "-1",
                        "prefill_cp_rank": "-1",
                        "target_tp_rank": "-1",
                        "target_pp_rank": "-1",
                    }
                )
            )
        )
        server.lock = asyncio.Lock()
        server.prefill_port_table = {
            0: {0: {0: {0: PrefillRankInfo(rank_ip="10.0.0.1", rank_port=7000)}}}
        }
        rank = asyncio.run(
            server._handle_route_get(
                SimpleNamespace(
                    query={
                        "prefill_dp_rank": "0",
                        "prefill_cp_rank": "0",
                        "target_tp_rank": "0",
                        "target_pp_rank": "0",
                    }
                )
            )
        )

        self.assertEqual(
            health.headers[BOOTSTRAP_INSTANCE_ID_HEADER], "prefill-generation"
        )
        self.assertEqual(
            topology.headers[BOOTSTRAP_INSTANCE_ID_HEADER], "prefill-generation"
        )
        # asyncio.Lock binds to the loop that first acquires it, and each
        # asyncio.run above used its own; give the dp-rank handler a fresh one.
        server.lock = asyncio.Lock()
        server.room_to_dp_rank = {1: {"dp_rank": 0, "timestamp": 0}}

        async def _query_body():
            return {"bootstrap_rooms": ["1"]}

        dp_ranks = asyncio.run(
            server._handle_query_dp_ranks(SimpleNamespace(json=_query_body))
        )

        self.assertEqual(
            rank.headers[BOOTSTRAP_INSTANCE_ID_HEADER], "prefill-generation"
        )
        self.assertEqual(
            dp_ranks.headers[BOOTSTRAP_INSTANCE_ID_HEADER], "prefill-generation"
        )
        # The id must stay out of the JSON body: decodes predating the field
        # construct PrefillServerInfo(**json) and crash on unknown keys, so a
        # body key would break old-decode/new-prefill rolling upgrades.
        self.assertNotIn("instance_id", json.loads(topology.text))


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


class TestNixlAbortHandling(CustomTestCase):
    def _make_manager(self, request_status=None):
        mgr = object.__new__(NixlKVManager)
        mgr.request_status = dict(request_status or {})
        mgr.status_lock = threading.Lock()
        mgr._connect = MagicMock()
        mgr.failure_lock = threading.Lock()
        mgr.failure_records = {}
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
        mgr.status_lock = threading.Lock()
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

    def test_base_manager_keeps_failed_terminal_for_live_room(self):
        """Regression: KVPoll.Failed is 0 and the base manager resolved
        non-Failed writes with max(), so a room retired cross-thread (the
        heartbeat's replacement or max-failures verdict) while its receiver
        was still inside init() was resurrected by init()'s own
        WaitingForInput write — the request then ran against evicted
        endpoints until the waiting timeout. NIXL and Mori pinned Failed
        sticky in their own overrides; the base manager must enforce it for
        every backend that inherits the heartbeat's retirement verdicts."""
        mgr = object.__new__(CommonKVManager)
        mgr.request_status = {}
        mgr.status_lock = threading.Lock()

        mgr.update_status(21, KVPoll.Bootstrapping)
        mgr.update_status(21, KVPoll.Failed)
        mgr.update_status(21, KVPoll.WaitingForInput)

        self.assertEqual(mgr.request_status[21], KVPoll.Failed)

    def test_concurrent_failed_write_is_not_lost_to_racing_promotion(self):
        """Regression: update_status resolved the terminal-Failed check and
        the max() write with separate unsynchronized reads, so a Failed
        written by another thread (the heartbeat's replacement verdict)
        between the two landed only in the max() re-read — and
        KVPoll.Failed is 0, so max() overwrote it with the promotion,
        resurrecting the retired room against evicted endpoints until the
        waiting timeout. Reproduces the exact interleaving
        deterministically: the promoting thread's first status read for
        the room pauses until a concurrent update_status(Failed) has run
        to completion (or, once transitions are serialized, until that
        writer has provably blocked)."""
        mgr = object.__new__(CommonKVManager)
        mgr.status_lock = threading.Lock()
        failed_written = threading.Event()

        def write_failed():
            mgr.update_status(21, KVPoll.Failed)
            failed_written.set()

        failed_thread = threading.Thread(target=write_failed, daemon=True)

        class FirstReadYieldsToFailedWriter(dict):
            """After the first status read for the room completes, lets the
            concurrent Failed writer run before the reader proceeds."""

            def __init__(self, *args):
                super().__init__(*args)
                self._fired = False

            def _yield_to_failed_writer(self):
                if not self._fired:
                    self._fired = True
                    failed_thread.start()
                    failed_written.wait(timeout=1.0)

            def get(self, key, default=None):
                value = super().get(key, default)
                self._yield_to_failed_writer()
                return value

            def __getitem__(self, key):
                value = super().__getitem__(key)
                self._yield_to_failed_writer()
                return value

        mgr.request_status = FirstReadYieldsToFailedWriter({21: KVPoll.Bootstrapping})

        mgr.update_status(21, KVPoll.WaitingForInput)
        failed_thread.join(timeout=2.0)

        self.assertEqual(mgr.request_status[21], KVPoll.Failed)


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
        mgr.status_lock = threading.Lock()

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

    def test_worker_failure_clears_room_outstanding_count(self):
        """Regression: the failure handler marked the room Failed without
        clearing _staging_outstanding, and a last-chunk failure has no later
        pop; a reused room id inherited the count and kept reporting
        Transferring."""
        room = 23
        mgr = self._make_manager(room)
        mgr.send_aux = MagicMock(side_effect=RuntimeError("aux send failed"))
        chunk = self._make_chunk(room, [], is_last_chunk=True)

        self._run_worker_once(mgr, chunk)

        self.assertEqual(mgr.request_status[room], KVPoll.Failed)
        self.assertIn(room, mgr.failure_records)
        self.assertEqual(dict(mgr._staging_outstanding), {})


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
        # "Cannot confirm the prefill either way" — the fail-open default,
        # authorizing cache-only invalidation but not room retirement.
        mgr._probe_bootstrap_instance.return_value = InstanceProbeVerdict.INCONCLUSIVE

        receiver = object.__new__(NixlKVReceiver)
        receiver.kv_mgr = mgr
        receiver.bootstrap_room = 11
        receiver.bootstrap_addr = "prefill:8998"
        receiver.started_transfer = False
        receiver.init_time = None
        receiver.conclude_state = None
        receiver.abort_notified = False
        receiver._used_cached_bootstrap_infos = True
        receiver._aborted = False
        # The topology snapshot this receiver bootstrapped against; eviction
        # is generation-gated on it. The table holds the same snapshot —
        # current generation — unless a test replaces it.
        receiver.prefill_info = object()
        mgr.connection_lock = threading.Lock()
        mgr.prefill_info_table = {"prefill:8998": receiver.prefill_info}
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
        self.assertNotIn(11, mgr.addr_to_rooms_tracker.get("prefill:8998", set()))
        self.assertEqual(receiver.conclude_state, KVPoll.Success)

    def test_clear_removes_room_from_address_tracker(self):
        """Regression: rooms enter addr_to_rooms_tracker at receiver
        construction, but only a successful transfer or a whole-address
        node failure removed them — a terminally failed room (waiting
        timeout, abort, dp-rank resolution deadline) stayed tracked for
        the address's lifetime, growing the set under persistent failing
        traffic. clear() is the terminal cleanup every pop path runs, so
        it must drop the room."""
        receiver, mgr = self._make_receiver()
        mgr.request_status = {11: KVPoll.Failed}
        mgr.required_prefill_response_num_table = {}
        mgr.prefill_response_tracker = {}

        receiver.clear()

        self.assertNotIn(11, mgr.addr_to_rooms_tracker.get("prefill:8998", set()))
        self.assertNotIn(11, mgr.request_status)

    def test_clear_after_address_pop_does_not_recreate_tracker_entry(self):
        """Regression: clear() indexed the tracker defaultdict, so terminal
        cleanup racing a whole-address node-failure pop recreated an empty
        entry, retaining one tracker key per retired bootstrap address for
        the manager's lifetime."""
        receiver, mgr = self._make_receiver()
        mgr.request_status = {11: KVPoll.Failed}
        mgr.required_prefill_response_num_table = {}
        mgr.prefill_response_tracker = {}
        mgr.addr_to_rooms_tracker.pop("prefill:8998")

        receiver.clear()

        self.assertNotIn("prefill:8998", mgr.addr_to_rooms_tracker)

    def test_clear_removes_room_failure_record(self):
        """Regression: when a room's own failure probe confirmed a prefill
        replacement, the retirement it triggered recorded a fresh failure
        for that same room after failure_exception had already consumed
        the original, and terminal cleanup never removed it — the record
        leaked, and a future request reusing the bootstrap_room inherited
        a stale "instance was replaced" reason, misreporting a propagated
        failure as a local one."""
        receiver, mgr = self._make_receiver()
        mgr.request_status = {11: KVPoll.Failed}
        mgr.required_prefill_response_num_table = {}
        mgr.prefill_response_tracker = {}
        mgr.failure_lock = threading.Lock()
        mgr.failure_records = {
            11: "Prefill instance was replaced (bootstrap_addr: prefill:8998)"
        }

        receiver.clear()

        self.assertNotIn(11, mgr.failure_records)

    def test_base_clear_leaves_failure_record_for_backend_consumption(self):
        """Regression: the shared terminal cleanup popped the room's failure
        record, but the Mooncake and Mori receivers consume theirs in
        failure_exception only AFTER calling clear() — the pop erased the
        real reason first, so every local failure on those backends was
        misreported as a propagated one with a generic message. Only NIXL,
        which consumes the record before terminal cleanup, may pop it (in
        its own clear() override)."""
        receiver, mgr = self._make_receiver()
        mgr.request_status = {11: KVPoll.Failed}
        mgr.required_prefill_response_num_table = {}
        mgr.prefill_response_tracker = {}
        mgr.failure_lock = threading.Lock()
        mgr.failure_records = {11: "Mooncake transfer engine failed"}

        CommonKVReceiver.clear(receiver)

        self.assertEqual(mgr.failure_records, {11: "Mooncake transfer engine failed"})

    def test_failure_invalidates_cached_registration_for_next_request(self):
        """#33789: a request that reused cached rank endpoints and then failed
        locally must drop the cache, or every later request keeps talking to a
        replaced prefill until a heartbeat tick notices. An inconclusive
        probe authorizes only the cache-only invalidation — never the
        evict-and-retire path, which would mass-retry every in-flight room
        at an overloaded-but-alive prefill."""
        receiver, mgr = self._make_receiver(status=KVPoll.Failed)
        mgr.failure_records = {11: "peer registration failed"}

        with self.assertRaisesRegex(Exception, "peer registration failed"):
            receiver.failure_exception()

        mgr.invalidate_cached_connections.assert_called_once_with(
            "prefill:8998", expected_prefill_info=receiver.prefill_info
        )
        mgr.handle_instance_replacement.assert_not_called()

    def test_confirmed_replacement_on_failure_probe_retires_rooms(self):
        """Regression: the failure-path probe collapsed a healthy response
        presenting a different instance id — a confirmed replacement — into
        the same verdict as a failed probe, so it only invalidated the
        cache. The rooms already in flight against the evicted snapshot
        were stranded until the waiting timeout: once the next request
        published the replacement's topology, the heartbeat compared new
        against new and could never indict them. A confirmed mismatch must
        take the same evict-and-retire path as the heartbeat's replacement
        verdict."""
        receiver, mgr = self._make_receiver(status=KVPoll.Failed)
        mgr._probe_bootstrap_instance.return_value = (
            InstanceProbeVerdict.CONFIRMED_REPLACED
        )
        mgr.failure_records = {11: "request timed out"}

        with self.assertRaisesRegex(Exception, "request timed out"):
            receiver.failure_exception()

        mgr.handle_instance_replacement.assert_called_once_with(
            "prefill:8998",
            expected_prefill_info=receiver.prefill_info,
            detected_by="failure-path health probe",
        )
        mgr.invalidate_cached_connections.assert_not_called()

    def test_confirmed_unchanged_prefill_keeps_connection_pool(self):
        """A slow-but-healthy prefill that still presents the registered
        instance_id must keep its cache: invalidating would disconnect
        sockets shared with in-flight rooms."""
        receiver, mgr = self._make_receiver(status=KVPoll.Failed)
        mgr._probe_bootstrap_instance.return_value = InstanceProbeVerdict.CONFIRMED_SAME
        mgr.failure_records = {11: "request timed out"}

        with self.assertRaisesRegex(Exception, "request timed out"):
            receiver.failure_exception()

        mgr._probe_bootstrap_instance.assert_called_once_with(
            "prefill:8998", expected_prefill_info=receiver.prefill_info
        )
        mgr.invalidate_cached_connections.assert_not_called()
        mgr.handle_instance_replacement.assert_not_called()

    def test_stale_generation_failure_skips_health_probe(self):
        """Stale failures burst after a replacement and their evictions no-op
        anyway; the generation gate must be checked before the blocking
        /health probe, not after."""
        receiver, mgr = self._make_receiver(status=KVPoll.Failed)
        mgr.failure_records = {11: "request timed out"}
        # The cache was already refreshed past this receiver's snapshot.
        mgr.prefill_info_table = {"prefill:8998": object()}

        with self.assertRaisesRegex(Exception, "request timed out"):
            receiver.failure_exception()

        mgr._probe_bootstrap_instance.assert_not_called()
        mgr.invalidate_cached_connections.assert_not_called()

    def test_abort_does_not_invalidate_cached_registration(self):
        """A client-side abort says nothing about the prefill's health; it must
        not churn cached endpoints shared with concurrent requests."""
        receiver, mgr = self._make_receiver(status=KVPoll.Failed)
        receiver._aborted = True
        mgr.failure_records = {11: "Aborted by AbortReq."}

        with self.assertRaisesRegex(Exception, "Aborted by AbortReq."):
            receiver.failure_exception()

        mgr.invalidate_cached_connections.assert_not_called()

    def test_failure_on_fresh_registration_keeps_connection_pool(self):
        """A failure on a connection registered within this same request is not
        evidence of staleness; the just-created pool entry must survive."""
        receiver, mgr = self._make_receiver(status=KVPoll.Failed)
        receiver._used_cached_bootstrap_infos = False
        mgr.failure_records = {11: "transfer timed out"}

        with self.assertRaisesRegex(Exception, "transfer timed out"):
            receiver.failure_exception()

        mgr.invalidate_cached_connections.assert_not_called()

    def test_propagated_failure_without_local_record_keeps_connection_pool(self):
        """When the failure detail lives on another rank, this rank observed
        nothing locally and must not invalidate on hearsay; the detecting rank
        handles its own cache."""
        receiver, mgr = self._make_receiver(status=KVPoll.Failed)
        mgr.failure_records = {}

        with self.assertRaisesRegex(Exception, "NIXL KVReceiver Exception"):
            receiver.failure_exception()

        mgr.invalidate_cached_connections.assert_not_called()


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
        mgr._instance_confirm_cache = {
            "10.0.0.1:8998": ("instance-1", float("inf")),
            "10.0.0.2:8998": ("instance-2", float("inf")),
        }
        mgr.heartbeat_failures = {}
        mgr.addr_to_rooms_tracker = defaultdict(set)
        mgr.addr_to_rooms_tracker["10.0.0.1:8998"] = {3, 4, 5}
        mgr.request_status = {
            3: KVPoll.WaitingForInput,
            4: KVPoll.Transferring,
            5: KVPoll.Success,
        }
        mgr.failure_records = {}
        mgr.failure_lock = threading.Lock()
        mgr.status_lock = threading.Lock()
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

    def test_node_failure_iterates_room_snapshot_not_live_set(self):
        """Regression: _handle_node_failure iterated the live per-address room
        set after releasing connection_lock while other threads mutate it;
        the RuntimeError killed the heartbeat thread. Reproduced by mutating
        the set from within the first room's failure recording."""
        mgr = self._make_manager()
        tracked_rooms = mgr.addr_to_rooms_tracker["10.0.0.1:8998"]
        real_record_failure = mgr.record_failure

        def record_failure_and_register_new_room(room, failure_reason):
            tracked_rooms.add(99)
            real_record_failure(room, failure_reason)

        mgr.record_failure = record_failure_and_register_new_room

        mgr._handle_node_failure("10.0.0.1:8998")

        self.assertEqual(mgr.request_status[3], KVPoll.Failed)
        self.assertEqual(mgr.request_status[4], KVPoll.Failed)
        self.assertEqual(mgr.request_status[5], KVPoll.Success)

    def test_node_failure_retirement_survives_concurrent_terminal_cleanup(self):
        """Regression: the retirement loop resolved each room with an
        unlocked membership check followed by an indexed status read, so a
        terminal cleanup popping the room between the two raised KeyError —
        killing the heartbeat thread, which has no other exception boundary
        (silently disabling replacement detection for the process lifetime),
        and leaving the rest of the room snapshot unretired. Reproduces the
        interleaving: the membership check's success triggers the concurrent
        pop before the indexed read."""
        mgr = self._make_manager()

        class PopsRoomAfterMembershipCheck(dict):
            def __contains__(self, room):
                present = super().__contains__(room)
                if present and room == 3:
                    self.pop(3)
                return present

        mgr.request_status = PopsRoomAfterMembershipCheck(mgr.request_status)

        mgr._handle_node_failure("10.0.0.1:8998")

        self.assertEqual(mgr.request_status[4], KVPoll.Failed)

    def test_eviction_requires_full_bootstrap_addr_match(self):
        """Connection keys are "{addr}_{dp}_{cp}_{tp}", so "10.0.0.1:899" is a
        string prefix of "10.0.0.1:8998" keys; eviction matching on a bare
        startswith(addr) would tear down an unrelated prefill's connections."""
        mgr = self._make_manager()

        mgr._handle_node_failure("10.0.0.1:899")
        mgr.invalidate_cached_connections("10.0.0.1:899")

        self.assertIn("10.0.0.1:8998_0_0_0", mgr.connection_pool)
        self.assertIn("10.0.0.1:8998_0_0_1", mgr.connection_pool)
        self.assertIn("10.0.0.2:8998_0_0_0", mgr.connection_pool)

    def test_reactive_invalidation_refreshes_topology_snapshot(self):
        """Regression: keeping the prefill_info_table entry left the replaced
        instance_id as the heartbeat baseline, so the next tick failed the
        freshly recovered rooms. The confirm-probe cache is baseline state
        too and is cleared with it."""
        mgr = self._make_manager()

        mgr.invalidate_cached_connections("10.0.0.1:8998")

        self.assertNotIn("10.0.0.1:8998", mgr.prefill_info_table)
        self.assertIn("10.0.0.2:8998", mgr.prefill_info_table)
        self.assertNotIn("10.0.0.1:8998", mgr._instance_confirm_cache)
        self.assertIn("10.0.0.2:8998", mgr._instance_confirm_cache)

    def test_stale_generation_invalidation_keeps_refreshed_cache(self):
        """A stale receiver's failure can surface after the cache was already
        refreshed; unconditional invalidation would tear down the live cache.
        Stale snapshot no-ops, current still evicts."""
        mgr = self._make_manager()
        stale_info = object()

        mgr.invalidate_cached_connections(
            "10.0.0.1:8998", expected_prefill_info=stale_info
        )

        self.assertIn("10.0.0.1:8998_0_0_0", mgr.connection_pool)
        self.assertIn("10.0.0.1:8998", mgr.prefill_info_table)

        current_info = mgr.prefill_info_table["10.0.0.1:8998"]
        mgr.invalidate_cached_connections(
            "10.0.0.1:8998", expected_prefill_info=current_info
        )

        self.assertNotIn("10.0.0.1:8998_0_0_0", mgr.connection_pool)
        self.assertNotIn("10.0.0.1:8998", mgr.prefill_info_table)

    def test_late_failed_update_does_not_resurrect_cleared_room(self):
        mgr = object.__new__(CommonKVManager)
        mgr.request_status = {}
        mgr.status_lock = threading.Lock()

        CommonKVManager.update_status(mgr, 9, KVPoll.Failed)

        self.assertNotIn(9, mgr.request_status)

    def test_detects_prefill_replacement_at_same_healthy_address(self):
        """The detector compares against the snapshot captured before the
        request, not the live table, and hands that snapshot back as the
        caller's eviction gate. No snapshot, no verdict."""
        mgr = self._make_manager()
        old_info = PrefillServerInfo(
            attn_tp_size=1,
            attn_cp_size=1,
            dp_size=1,
            pp_size=1,
            page_size=1,
            kv_cache_dtype="auto",
            follow_bootstrap_room=True,
            instance_id="old-instance",
        )
        response = SimpleNamespace(
            headers={BOOTSTRAP_INSTANCE_ID_HEADER: "new-instance"}
        )

        self.assertIs(mgr._bootstrap_instance_replaced(old_info, response), old_info)

        same_response = SimpleNamespace(
            headers={BOOTSTRAP_INSTANCE_ID_HEADER: "old-instance"}
        )
        self.assertIsNone(mgr._bootstrap_instance_replaced(old_info, same_response))
        self.assertIsNone(mgr._bootstrap_instance_replaced(None, response))

    def test_headerless_baseline_detects_header_emitting_replacement(self):
        """Regression: requiring both ids reported a headerless baseline as
        unchanged forever after its prefill was replaced by a stamping
        version. An id appearing over a headerless baseline is a replacement
        (no upgrade-in-place); an id disappearing stays inconclusive."""
        mgr = self._make_manager()
        headerless_info = SimpleNamespace(instance_id=None)
        id_response = SimpleNamespace(
            headers={BOOTSTRAP_INSTANCE_ID_HEADER: "new-instance"}
        )

        self.assertIs(
            mgr._bootstrap_instance_replaced(headerless_info, id_response),
            headerless_info,
        )

        current_info = SimpleNamespace(instance_id="old-instance")
        no_header_response = SimpleNamespace(headers={})
        self.assertIsNone(
            mgr._bootstrap_instance_replaced(current_info, no_header_response)
        )

    def test_heartbeat_probe_opens_a_fresh_connection_each_tick(self):
        """Regression: a pooled keep-alive heartbeat socket kept reaching a
        draining replaced process, which answers with the expected id,
        masking the replacement until the old process exited. Each probe
        must resolve the address afresh on a non-reusable connection."""
        mgr = self._make_manager()
        addr = "10.0.0.1:8998"
        mgr.prefill_info_table[addr] = SimpleNamespace(instance_id="old-instance")
        mgr.heartbeat_failures = {}
        mgr.max_failures = 1

        with patch(
            "sglang.srt.disaggregation.common.conn.requests.get",
            return_value=SimpleNamespace(
                status_code=200,
                headers={BOOTSTRAP_INSTANCE_ID_HEADER: "old-instance"},
            ),
        ) as health_get:
            mgr._heartbeat_tick(addr)
            mgr._heartbeat_tick(addr)

        self.assertEqual(health_get.call_count, 2)
        for probe in health_get.call_args_list:
            self.assertEqual(probe.kwargs["headers"], {"Connection": "close"})
        self.assertEqual(mgr.heartbeat_failures, {addr: 0})
        self.assertIn(addr, mgr.prefill_info_table)

    def test_topology_fetch_captures_header_id_and_drops_unknown_fields(self):
        """try_ensure_parallel_info bridges the wire contract to the cached
        baseline: the header id must land on the snapshot and unknown
        topology keys must be dropped, not crashed on. Every other test
        injects the id directly, so this is the only guard on the bridge."""
        mgr = object.__new__(CommonKVManager)
        mgr.prefill_info_table = {}
        mgr.kv_args = SimpleNamespace(page_size=1)
        mgr.kv_cache_dtype_str = "auto"
        mgr.dcp_size = 1
        mgr._resolve_rank_mapping = lambda info: None
        response = SimpleNamespace(
            status_code=200,
            headers={BOOTSTRAP_INSTANCE_ID_HEADER: "server-instance"},
            json=lambda: {
                "attn_tp_size": 1,
                "attn_cp_size": 1,
                "dp_size": 1,
                "pp_size": 1,
                "page_size": 1,
                "kv_cache_dtype": "auto",
                "follow_bootstrap_room": True,
                "future_topology_field": 123,
            },
        )

        with patch(
            "sglang.srt.disaggregation.common.conn.requests.get",
            return_value=response,
        ):
            self.assertTrue(mgr.try_ensure_parallel_info("10.0.0.1:8998"))

        cached = mgr.prefill_info_table["10.0.0.1:8998"]
        self.assertEqual(cached.instance_id, "server-instance")
        self.assertEqual(cached.attn_tp_size, 1)

    def _prepare_heartbeat(self, mgr, addr, health_get_side_effect):
        mgr.prefill_info_table[addr] = SimpleNamespace(instance_id="old-instance")
        mgr.heartbeat_failures = {}
        mgr.max_failures = 1
        patcher = patch(
            "sglang.srt.disaggregation.common.conn.requests.get",
            side_effect=health_get_side_effect,
        )
        self.health_get = patcher.start()
        self.addCleanup(patcher.stop)

    def test_heartbeat_tick_ignores_delayed_response_from_replaced_instance(self):
        """A delayed /health response from the replaced instance must not be
        compared against the freshly recovered table entry; the verdict
        anchors to the snapshot captured before the request was sent."""
        mgr = self._make_manager()
        addr = "10.0.0.1:8998"
        new_info = SimpleNamespace(instance_id="new-instance")

        def delayed_response_from_old_instance(url, **kwargs):
            # Reactive recovery finishes while the request is in flight.
            mgr.prefill_info_table[addr] = new_info
            return SimpleNamespace(
                status_code=200,
                headers={BOOTSTRAP_INSTANCE_ID_HEADER: "old-instance"},
            )

        self._prepare_heartbeat(mgr, addr, delayed_response_from_old_instance)

        mgr._heartbeat_tick(addr)

        self.assertIs(mgr.prefill_info_table[addr], new_info)
        self.assertIn("10.0.0.1:8998_0_0_0", mgr.connection_pool)
        self.assertEqual(mgr.request_status[3], KVPoll.WaitingForInput)
        self.assertEqual(mgr.failure_records, {})

    def test_heartbeat_tick_timeout_verdict_is_generation_gated(self):
        """A timeout verdict whose generation was evicted and replaced
        mid-flight must no-op rather than tear down the new generation's
        cache; the stale failure count is discarded with it."""
        mgr = self._make_manager()
        addr = "10.0.0.1:8998"
        new_info = SimpleNamespace(instance_id="new-instance")

        def timeout_after_replacement(url, **kwargs):
            mgr.prefill_info_table[addr] = new_info
            raise OSError("health check timed out")

        self._prepare_heartbeat(mgr, addr, timeout_after_replacement)

        mgr._heartbeat_tick(addr)

        self.assertIs(mgr.prefill_info_table[addr], new_info)
        self.assertIn("10.0.0.1:8998_0_0_0", mgr.connection_pool)
        self.assertEqual(mgr.request_status[3], KVPoll.WaitingForInput)
        self.assertEqual(mgr.failure_records, {})
        self.assertEqual(mgr.heartbeat_failures, {})

    def test_heartbeat_miss_racing_recovery_is_not_charged_to_new_generation(self):
        """A probe miss that raced with eviction plus recovery must not
        increment the address-keyed failure count: carried over, it would
        put the new generation one hiccup away from the threshold."""
        mgr = self._make_manager()
        addr = "10.0.0.1:8998"
        new_info = SimpleNamespace(instance_id="new-instance")

        def timeout_after_replacement(url, **kwargs):
            mgr.prefill_info_table[addr] = new_info
            raise OSError("health check timed out")

        self._prepare_heartbeat(mgr, addr, timeout_after_replacement)
        mgr.max_failures = 2

        mgr._heartbeat_tick(addr)

        self.assertEqual(mgr.heartbeat_failures, {})

        self.health_get.side_effect = OSError("transient miss")

        mgr._heartbeat_tick(addr)

        self.assertIs(mgr.prefill_info_table[addr], new_info)
        self.assertIn("10.0.0.1:8998_0_0_0", mgr.connection_pool)
        self.assertEqual(mgr.request_status[3], KVPoll.WaitingForInput)
        self.assertEqual(mgr.heartbeat_failures, {addr: 1})

    def test_eviction_clears_heartbeat_failure_count(self):
        """Reactive eviction replaces the generation outside the heartbeat
        thread; a count accrued against the old generation must not survive
        it, or the replacement starts partway toward the failure threshold."""
        mgr = self._make_manager()
        addr = "10.0.0.1:8998"
        mgr.heartbeat_failures = {addr: 1}

        mgr.invalidate_cached_connections(addr)

        self.assertEqual(mgr.heartbeat_failures, {})

    def test_heartbeat_tick_verdicts_still_act_on_unreplaced_generation(self):
        """The gates exist to stop cross-generation verdicts, not to weaken
        detection: with no concurrent eviction, a replaced instance id evicts
        and fails pending rooms, and so does a timeout that reaches the
        failure threshold."""
        mgr = self._make_manager()
        addr = "10.0.0.1:8998"
        self._prepare_heartbeat(
            mgr,
            addr,
            lambda url, **kwargs: SimpleNamespace(
                status_code=200,
                headers={BOOTSTRAP_INSTANCE_ID_HEADER: "new-instance"},
            ),
        )

        mgr._heartbeat_tick(addr)

        self.assertNotIn(addr, mgr.prefill_info_table)
        self.assertNotIn("10.0.0.1:8998_0_0_0", mgr.connection_pool)
        self.assertEqual(mgr.request_status[3], KVPoll.Failed)
        self.assertIn(3, mgr.failure_records)

        timing_out = self._make_manager()

        def raise_timeout(url, **kwargs):
            raise OSError("health check timed out")

        self._prepare_heartbeat(timing_out, addr, raise_timeout)

        timing_out._heartbeat_tick(addr)

        self.assertNotIn(addr, timing_out.prefill_info_table)
        self.assertNotIn("10.0.0.1:8998_0_0_0", timing_out.connection_pool)
        self.assertEqual(timing_out.request_status[3], KVPoll.Failed)
        self.assertIn(3, timing_out.failure_records)

    def test_heartbeat_tick_skips_success_bookkeeping_without_snapshot(self):
        """Regression: a healthy response after the entry was evicted fell
        through to the success path, resetting the failure count and firing
        the backend hook for an address the eviction had just cleared."""
        mgr = self._make_manager()
        addr = "10.0.0.1:8998"
        self._prepare_heartbeat(
            mgr,
            addr,
            lambda url, **kwargs: SimpleNamespace(
                status_code=200,
                headers={BOOTSTRAP_INSTANCE_ID_HEADER: "new-instance"},
            ),
        )
        # The eviction lands before the tick captures its snapshot.
        del mgr.prefill_info_table[addr]

        with patch.object(mgr, "_on_heartbeat_success") as on_success:
            mgr._heartbeat_tick(addr)

        on_success.assert_not_called()
        self.assertEqual(mgr.heartbeat_failures, {})

    def test_heartbeat_tick_success_verdict_is_generation_gated(self):
        """Regression: the success path only checked the snapshot was
        non-None, so a delayed healthy response from an evicted-and-replaced
        generation credited the replacement (failure-count entry, backend
        hook). Success must be identity-gated like failure."""
        mgr = self._make_manager()
        addr = "10.0.0.1:8998"
        new_info = SimpleNamespace(instance_id="new-instance")

        def delayed_healthy_response_from_old_instance(url, **kwargs):
            # Reactive recovery evicts the probed generation and installs
            # its replacement while the /health request is in flight; the
            # response still presents the probed generation's id, so it
            # passes the snapshot comparison and reaches the success path.
            mgr.prefill_info_table[addr] = new_info
            return SimpleNamespace(
                status_code=200,
                headers={BOOTSTRAP_INSTANCE_ID_HEADER: "old-instance"},
            )

        self._prepare_heartbeat(mgr, addr, delayed_healthy_response_from_old_instance)

        with patch.object(mgr, "_on_heartbeat_success") as on_success:
            mgr._heartbeat_tick(addr)

        on_success.assert_not_called()
        self.assertEqual(mgr.heartbeat_failures, {})
        self.assertIs(mgr.prefill_info_table[addr], new_info)

    def test_heartbeat_success_hook_runs_under_the_generation_lock(self):
        """Regression: the backend success hook ran after releasing
        connection_lock, so an eviction landing in the gap handed the stale
        verdict's hook the replacement's state. The hook must run inside the
        critical section that authorized it."""
        mgr = self._make_manager()
        addr = "10.0.0.1:8998"
        self._prepare_heartbeat(
            mgr,
            addr,
            lambda url, **kwargs: SimpleNamespace(
                status_code=200,
                headers={BOOTSTRAP_INSTANCE_ID_HEADER: "old-instance"},
            ),
        )

        lock_held_during_hook = []
        with patch.object(
            mgr,
            "_on_heartbeat_success",
            side_effect=lambda a: lock_held_during_hook.append(
                mgr.connection_lock.locked()
            ),
        ):
            mgr._heartbeat_tick(addr)

        self.assertEqual(lock_held_during_hook, [True])
        self.assertEqual(mgr.heartbeat_failures, {addr: 0})

    def test_heartbeat_node_failure_is_generation_gated(self):
        """Between the id comparison and the eviction a request can install a
        new generation; ungated _handle_node_failure would evict the fresh
        cache. Stale snapshot no-ops, current still evicts."""
        mgr = self._make_manager()

        mgr._handle_node_failure("10.0.0.1:8998", expected_prefill_info=object())

        self.assertIn("10.0.0.1:8998_0_0_0", mgr.connection_pool)
        self.assertIn("10.0.0.1:8998", mgr.prefill_info_table)
        self.assertEqual(mgr.request_status[3], KVPoll.WaitingForInput)
        self.assertEqual(mgr.failure_records, {})

        current_info = mgr.prefill_info_table["10.0.0.1:8998"]
        mgr._handle_node_failure("10.0.0.1:8998", expected_prefill_info=current_info)

        self.assertNotIn("10.0.0.1:8998_0_0_0", mgr.connection_pool)
        self.assertNotIn("10.0.0.1:8998", mgr.prefill_info_table)
        self.assertEqual(mgr.request_status[3], KVPoll.Failed)
        self.assertIn(3, mgr.failure_records)


class TestBootstrapInstanceProbe(CustomTestCase):
    ADDR = "10.0.0.1:8998"

    @staticmethod
    def _make_manager():
        mgr = object.__new__(NixlKVManager)
        mgr._instance_confirm_cache = {}
        return mgr

    @staticmethod
    def _snapshot(instance_id):
        return PrefillServerInfo(
            attn_tp_size=1,
            attn_cp_size=1,
            dp_size=1,
            pp_size=1,
            page_size=1,
            kv_cache_dtype="auto",
            follow_bootstrap_room=True,
            instance_id=instance_id,
        )

    @staticmethod
    def _response(instance_id, status_code=200):
        headers = {}
        if instance_id is not None:
            headers[BOOTSTRAP_INSTANCE_ID_HEADER] = instance_id
        return SimpleNamespace(status_code=status_code, headers=headers)

    def _probe(self, expected_id, response):
        mgr = self._make_manager()
        with patch(
            "sglang.srt.disaggregation.common.conn.requests.get",
            return_value=response,
        ):
            return mgr._probe_bootstrap_instance(
                self.ADDR, expected_prefill_info=self._snapshot(expected_id)
            )

    def test_matching_health_header_confirms_instance(self):
        """The only outcome allowed to suppress failure-driven cache recovery
        is a positive /health match against the registered instance_id."""
        self.assertIs(
            self._probe("live", self._response("live")),
            InstanceProbeVerdict.CONFIRMED_SAME,
        )

    def test_healthy_mismatch_is_a_confirmed_replacement(self):
        """Regression: a healthy response presenting a different instance id
        — a confirmed replacement, by the same rule the heartbeat and the
        query path apply — was collapsed into the same verdict as a failed
        probe, so the failure path only invalidated the cache and the rooms
        in flight against the replaced snapshot waited out their timeout.
        An id appearing over a headerless baseline is a replacement too:
        there is no upgrade-in-place."""
        cases = {
            "different id": ("old", self._response("new")),
            "id over headerless baseline": (None, self._response("new")),
        }
        for name, (expected_id, response) in cases.items():
            with self.subTest(name):
                self.assertIs(
                    self._probe(expected_id, response),
                    InstanceProbeVerdict.CONFIRMED_REPLACED,
                )

    def test_uncertain_probe_outcomes_are_inconclusive(self):
        """Anything short of a healthy response with a definite identity —
        an unhealthy status even when it carries the expected id, or an id
        disappearing — neither confirms the instance (which would suppress
        recovery) nor convicts it (which would retire every in-flight room
        at the address on an overload blip)."""
        cases = {
            "id disappeared (downgrade or stripped header)": (
                "old",
                self._response(None),
            ),
            "unhealthy status": ("old", self._response("old", status_code=503)),
            "unhealthy status with mismatched id": (
                "old",
                self._response("new", status_code=503),
            ),
        }
        for name, (expected_id, response) in cases.items():
            with self.subTest(name):
                self.assertIs(
                    self._probe(expected_id, response),
                    InstanceProbeVerdict.INCONCLUSIVE,
                )

    def test_headerless_baseline_is_confirmed_by_live_headerless_probe(self):
        """Regression: any headerless baseline was unconfirmable, so every
        local failure on a reused connection evicted the pool of a healthy
        pre-instance-id prefill. A live headerless response is all the
        confirmation such a baseline can get."""
        self.assertIs(
            self._probe(None, self._response(None)),
            InstanceProbeVerdict.CONFIRMED_SAME,
        )

    def test_probe_error_is_inconclusive(self):
        mgr = self._make_manager()

        with patch(
            "sglang.srt.disaggregation.common.conn.requests.get",
            side_effect=ConnectionError("bootstrap unreachable"),
        ):
            self.assertIs(
                mgr._probe_bootstrap_instance(
                    self.ADDR, expected_prefill_info=self._snapshot("old")
                ),
                InstanceProbeVerdict.INCONCLUSIVE,
            )

    def test_positive_confirm_is_cached_within_ttl(self):
        """The probe blocks the scheduler thread and an overloaded prefill
        fails one local request after another; a just-confirmed instance is
        trusted for a short window instead of re-probing per failure."""
        mgr = self._make_manager()
        snapshot = self._snapshot("same")

        with patch(
            "sglang.srt.disaggregation.common.conn.requests.get",
            return_value=self._response("same"),
        ) as mock_get:
            for _ in range(2):
                self.assertIs(
                    mgr._probe_bootstrap_instance(
                        self.ADDR, expected_prefill_info=snapshot
                    ),
                    InstanceProbeVerdict.CONFIRMED_SAME,
                )

        self.assertEqual(mock_get.call_count, 1)

    def test_confirm_cache_is_keyed_to_the_confirmed_instance_id(self):
        """A cache keyed by address alone would let a confirmation of the old
        instance suppress the probe after the baseline was refreshed to a new
        one; the cached entry must only match the instance_id it confirmed."""
        mgr = self._make_manager()

        with patch(
            "sglang.srt.disaggregation.common.conn.requests.get",
            return_value=self._response("old"),
        ) as mock_get:
            self.assertIs(
                mgr._probe_bootstrap_instance(
                    self.ADDR, expected_prefill_info=self._snapshot("old")
                ),
                InstanceProbeVerdict.CONFIRMED_SAME,
            )
        self.assertEqual(mock_get.call_count, 1)

        with patch(
            "sglang.srt.disaggregation.common.conn.requests.get",
            return_value=self._response("new"),
        ) as mock_get:
            self.assertIs(
                mgr._probe_bootstrap_instance(
                    self.ADDR, expected_prefill_info=self._snapshot("new")
                ),
                InstanceProbeVerdict.CONFIRMED_SAME,
            )
        self.assertEqual(mock_get.call_count, 1)

    def test_replacement_verdict_does_not_populate_confirm_cache(self):
        """A mismatch must leave no trace in the positive-confirmation
        cache: the entry is keyed to the id it confirmed, and caching
        anything here could suppress a later probe's verdict."""
        mgr = self._make_manager()

        with patch(
            "sglang.srt.disaggregation.common.conn.requests.get",
            return_value=self._response("new"),
        ):
            mgr._probe_bootstrap_instance(
                self.ADDR, expected_prefill_info=self._snapshot("old")
            )

        self.assertEqual(mgr._instance_confirm_cache, {})


class TestReceiverTableLocking(CustomTestCase):
    class _LockAssertingDict(dict):
        """Shared-table stand-in that rejects any access performed without
        holding connection_lock."""

        def __init__(self, lock):
            super().__init__()
            self._lock = lock

        def _check(self):
            if not self._lock.locked():
                raise AssertionError("connection_pool accessed without connection_lock")

        def get(self, key, default=None):
            self._check()
            return super().get(key, default)

        def __contains__(self, key):
            self._check()
            return super().__contains__(key)

        def __getitem__(self, key):
            self._check()
            return super().__getitem__(key)

        def __setitem__(self, key, value):
            self._check()
            super().__setitem__(key, value)

    def test_cached_lookup_happens_under_connection_lock(self):
        """The heartbeat thread evicts pool keys under connection_lock; the
        cached-path read must be a single locked snapshot or a concurrent
        eviction turns it into a KeyError."""
        lock = threading.Lock()
        mgr = SimpleNamespace(
            connection_lock=lock,
            connection_pool=self._LockAssertingDict(lock),
        )
        infos = [{"rank_ip": "10.0.0.1", "rank_port": 7000}]
        with lock:
            mgr.connection_pool["10.0.0.1:8998_0_0_0"] = infos

        receiver = object.__new__(NixlKVReceiver)
        receiver.kv_mgr = mgr
        receiver.bootstrap_addr = "10.0.0.1:8998"
        receiver.prefill_dp_rank = 0
        receiver.target_tp_rank = 0
        receiver.target_cp_ranks = [0]
        receiver._used_cached_bootstrap_infos = False

        receiver._setup_bootstrap_infos()

        self.assertEqual(receiver.bootstrap_infos, infos)
        self.assertTrue(receiver._used_cached_bootstrap_infos)

    def test_init_reads_prefill_info_under_connection_lock(self):
        """The heartbeat thread pops prefill_info_table entries under
        connection_lock; init() must take a single locked snapshot, which is
        also the object the generation-gated eviction anchors to."""
        lock = threading.Lock()
        table = self._LockAssertingDict(lock)
        info = SimpleNamespace(
            target_tp_rank=0,
            target_tp_ranks=[0],
            target_cp_ranks=[0],
            target_pp_ranks=[0],
            required_dst_info_num=1,
            required_prefill_response_num=1,
        )
        with lock:
            table["10.0.0.1:8998"] = info
        mgr = SimpleNamespace(
            connection_lock=lock,
            prefill_info_table=table,
            required_prefill_response_num_table={},
            enable_staging=False,
            update_status=MagicMock(),
        )

        receiver = object.__new__(NixlKVReceiver)
        receiver.kv_mgr = mgr
        receiver.bootstrap_addr = "10.0.0.1:8998"
        receiver.bootstrap_room = 7
        receiver.conclude_state = None
        receiver._setup_bootstrap_infos = MagicMock()

        receiver.init(prefill_dp_rank=0)

        self.assertIs(receiver.prefill_info, info)
        self.assertEqual(mgr.required_prefill_response_num_table[7], 1)
        receiver._setup_bootstrap_infos.assert_called_once_with()

    def test_constructor_publishes_room_and_status_in_one_locked_section(self):
        """Regression, twice: the constructor's tracker add ran unlocked
        while node-failure handling snapshots-and-pops the address's set
        under connection_lock — the add could land on the just-popped
        (orphaned) set; then the locked add still published the initial
        Bootstrapping status after releasing the lock, and the node-failure
        retirement loop skips rooms absent from request_status — a
        snapshot-and-pop between the two publications dropped the room's
        tracker membership but skipped retiring it (no status yet), and the
        late status write then installed Bootstrapping for a permanently
        untracked room. Both failure modes strand the request until its
        waiting timeout; both publications must land in one critical
        section."""

        class _CountingLock:
            def __init__(inner):
                inner._lock = threading.Lock()
                inner.acquisitions = 0

            def __enter__(inner):
                inner._lock.acquire()
                inner.acquisitions += 1

            def __exit__(inner, *exc):
                inner._lock.release()

            def locked(inner):
                return inner._lock.locked()

        lock = _CountingLock()
        held_during_add = []
        status_writes = []

        class _LockCheckedSet(set):
            def add(inner, room):
                held_during_add.append(lock.locked())
                set.add(inner, room)

        mgr = MagicMock()
        mgr.connection_lock = lock
        mgr.addr_to_rooms_tracker = defaultdict(_LockCheckedSet)
        mgr.update_status = MagicMock(
            side_effect=lambda room, status: status_writes.append(
                (room, status, lock.locked())
            )
        )

        receiver = object.__new__(NixlKVReceiver)
        CommonKVReceiver.__init__(
            receiver, mgr, bootstrap_addr="10.0.0.1:8998", bootstrap_room=7
        )

        self.assertEqual(held_during_add, [True])
        self.assertEqual(status_writes, [(7, KVPoll.Bootstrapping, True)])
        self.assertEqual(lock.acquisitions, 1)
        self.assertIn(7, mgr.addr_to_rooms_tracker["10.0.0.1:8998"])

    def _make_fetching_receiver(self, mgr, prefill_info):
        receiver = object.__new__(NixlKVReceiver)
        receiver.kv_mgr = mgr
        receiver.bootstrap_addr = "10.0.0.1:8998"
        receiver.bootstrap_room = 7
        receiver.prefill_dp_rank = 0
        receiver.target_tp_rank = 0
        receiver.target_tp_ranks = [0]
        receiver.target_cp_ranks = [0]
        receiver.target_pp_ranks = [0]
        receiver.prefill_info = prefill_info
        receiver._used_cached_bootstrap_infos = False
        receiver._get_bootstrap_info_from_server = MagicMock(
            return_value={"rank_ip": "10.0.0.1", "rank_port": 7000}
        )
        receiver._register_kv_args = MagicMock(return_value=True)
        return receiver

    def test_stale_receiver_does_not_republish_evicted_endpoints(self):
        """A replacement can complete inside a cache-miss receiver's fetch
        window; an unconditional publish would overwrite the new
        generation's pool entry with dead endpoints. Stale snapshot skips,
        current publishes."""
        current_info = object()
        mgr = SimpleNamespace(
            connection_lock=threading.Lock(),
            connection_pool={},
            prefill_info_table={"10.0.0.1:8998": current_info},
            is_mla_backend=False,
        )

        stale = self._make_fetching_receiver(mgr, prefill_info=object())
        stale._setup_bootstrap_infos()

        self.assertEqual(mgr.connection_pool, {})
        self.assertIsNotNone(stale.bootstrap_infos)

        fresh = self._make_fetching_receiver(mgr, prefill_info=current_info)
        fresh._setup_bootstrap_infos()

        self.assertIn("10.0.0.1:8998_0_0_0", mgr.connection_pool)


class TestRankEndpointFetchValidation(CustomTestCase):
    ENDPOINT = {"rank_ip": "10.0.0.1", "rank_port": 7000}

    def _make_receiver(self, instance_id):
        receiver = object.__new__(NixlKVReceiver)
        receiver.bootstrap_addr = "10.0.0.1:8998"
        receiver.prefill_info = SimpleNamespace(instance_id=instance_id)
        receiver.kv_mgr = MagicMock()
        return receiver

    def _response(self, instance_id):
        headers = {}
        if instance_id is not None:
            headers[BOOTSTRAP_INSTANCE_ID_HEADER] = instance_id
        return SimpleNamespace(
            status_code=200, headers=headers, json=lambda: self.ENDPOINT
        )

    def _fetch(self, receiver, *responses):
        session = MagicMock()
        session.get.side_effect = list(responses)
        with patch(
            "sglang.srt.disaggregation.common.conn._get_bootstrap_session",
            return_value=session,
        ), patch(
            "sglang.srt.disaggregation.common.conn._drop_bootstrap_session"
        ) as drop_session:
            result = receiver._get_bootstrap_info_from_server(0, 0, 0, 0)
        return result, drop_session, session

    def test_rank_endpoint_fetch_rejects_persistently_mismatched_instance(self):
        """Regression: an answer from a draining replaced process cached dead
        rank endpoints under the fresh topology, and the /health veto made
        the stall permanent. A second mismatch on the fresh connection
        indicts the snapshot instead, which is evicted generation-gated;
        matching and both-headerless pairs keep working."""
        stale = self._make_receiver(instance_id="new-instance")
        result, drop_session, session = self._fetch(
            stale,
            self._response("old-instance"),
            self._response("old-instance"),
        )
        self.assertIsNone(result)
        self.assertEqual(session.get.call_count, 2)
        self.assertEqual(drop_session.call_count, 2)
        # Retirement, not cache-only invalidation: rooms in flight against
        # the evicted snapshot would otherwise ride out the waiting timeout
        # (the heartbeat can no longer indict them once the replacement's
        # topology is published).
        stale.kv_mgr.handle_instance_replacement.assert_called_once_with(
            "10.0.0.1:8998",
            expected_prefill_info=stale.prefill_info,
            detected_by="rank endpoint fetch",
        )

        # An unstamped response against a stamped snapshot is the same
        # process disagreement in the other direction (a stamped topology
        # means the server versions its responses).
        unstamped_rx = self._make_receiver(instance_id="new-instance")
        unstamped, drop_session, _ = self._fetch(
            unstamped_rx, self._response(None), self._response(None)
        )
        self.assertIsNone(unstamped)
        unstamped_rx.kv_mgr.handle_instance_replacement.assert_called_once()

        matching_rx = self._make_receiver(instance_id="new-instance")
        matching, drop_session, _ = self._fetch(
            matching_rx, self._response("new-instance")
        )
        self.assertEqual(matching, self.ENDPOINT)
        drop_session.assert_not_called()
        matching_rx.kv_mgr.handle_instance_replacement.assert_not_called()

        headerless, drop_session, _ = self._fetch(
            self._make_receiver(instance_id=None), self._response(None)
        )
        self.assertEqual(headerless, self.ENDPOINT)
        drop_session.assert_not_called()

    def test_rank_endpoint_fetch_retries_once_on_a_fresh_connection(self):
        """Regression: a single mismatch failed the whole fetch even though
        its usual cause is the just-dropped keep-alive socket; one retry on
        the fresh connection resolves the draining-pod case."""
        receiver = self._make_receiver(instance_id="new-instance")
        result, drop_session, session = self._fetch(
            receiver,
            self._response("old-instance"),
            self._response("new-instance"),
        )
        self.assertEqual(result, self.ENDPOINT)
        self.assertEqual(session.get.call_count, 2)
        drop_session.assert_called_once_with("10.0.0.1:8998")
        receiver.kv_mgr.handle_instance_replacement.assert_not_called()

    def test_rank_endpoint_fetch_drops_session_on_error_status(self):
        """Regression: only 200 responses rotated the session, so a draining
        process 404ing new-topology ranks kept the socket pinned and every
        later receiver repeated the error."""
        receiver = self._make_receiver(instance_id="new-instance")
        error = SimpleNamespace(status_code=404, headers={}, text="not found")
        result, drop_session, session = self._fetch(receiver, error)
        self.assertIsNone(result)
        self.assertEqual(session.get.call_count, 1)
        drop_session.assert_called_once_with("10.0.0.1:8998")
        receiver.kv_mgr.handle_instance_replacement.assert_not_called()


class TestDpRankQueryValidation(CustomTestCase):
    ADDR = "10.0.0.1:8998"

    def _query(self, *responses, expected_instance_id):
        session = MagicMock()
        session.post.side_effect = list(responses)
        with patch(
            "sglang.srt.disaggregation.common.conn._get_bootstrap_session",
            return_value=session,
        ), patch(
            "sglang.srt.disaggregation.common.conn._drop_bootstrap_session"
        ) as drop_session:
            result = CommonKVReceiver.query_prefill_dp_ranks(
                self.ADDR, [7], expected_instance_id=expected_instance_id
            )
        return result, drop_session, session

    def _response(self, instance_id, mapping):
        headers = {}
        if instance_id is not None:
            headers[BOOTSTRAP_INSTANCE_ID_HEADER] = instance_id
        return SimpleNamespace(status_code=200, headers=headers, json=lambda: mapping)

    def test_dp_rank_query_rejects_mismatched_instance(self):
        """Regression: the dp-rank query was the one bootstrap response
        accepted without instance validation, and a draining process
        omitting unregistered rooms parked those requests with no timeout
        armed. A mismatch discards answer and session; a second mismatch
        returns None so the caller evicts the snapshot. Matching and
        both-headerless pairs keep working."""
        mapping = {"7": 2}

        cases_rejected = {
            "stale socket answers for the old instance": ("old", "new"),
            "stamped answer against a headerless snapshot": ("new", None),
            "headerless answer against a stamped snapshot": (None, "new"),
        }
        for name, (observed, expected) in cases_rejected.items():
            with self.subTest(name):
                result, drop_session, session = self._query(
                    self._response(observed, mapping),
                    self._response(observed, mapping),
                    expected_instance_id=expected,
                )
                self.assertIsNone(result)
                self.assertEqual(session.post.call_count, 2)
                self.assertEqual(drop_session.call_count, 2)

        cases_accepted = {
            "matching instance": ("same", "same"),
            "both headerless (pre-instance-id server)": (None, None),
        }
        for name, (observed, expected) in cases_accepted.items():
            with self.subTest(name):
                result, drop_session, session = self._query(
                    self._response(observed, mapping), expected_instance_id=expected
                )
                self.assertEqual(result, mapping)
                self.assertEqual(session.post.call_count, 1)
                drop_session.assert_not_called()

    def test_dp_rank_query_retries_once_on_a_fresh_connection(self):
        """Regression: a single mismatch discarded the answer even though its
        usual cause is the just-dropped keep-alive socket; only a second
        mismatch on the fresh connection becomes the evict-the-snapshot
        signal (None)."""
        mapping = {"7": 2}
        result, drop_session, session = self._query(
            self._response("old", mapping),
            self._response("new", mapping),
            expected_instance_id="new",
        )
        self.assertEqual(result, mapping)
        self.assertEqual(session.post.call_count, 2)
        drop_session.assert_called_once_with(self.ADDR)


class TestNixlRemotePeerRegistration(CustomTestCase):
    def _make_manager(self):
        mgr = object.__new__(NixlKVManager)
        mgr.agent = MagicMock()
        mgr.decode_kv_args_table = {}
        mgr.prep_handles = {}
        mgr.prep_handles_slice_dst = {}
        mgr.disaggregation_mode = DisaggregationMode.PREFILL
        mgr.requires_dcp_relayout = MagicMock(return_value=False)
        mgr._prepare_payload_xfer = MagicMock()
        return mgr

    @staticmethod
    def _peer(agent_name):
        return SimpleNamespace(
            agent_name=agent_name,
            agent_metadata=f"metadata-{agent_name}".encode("ascii"),
            dst_dcp_size=1,
            dst_dcp_rank=0,
        )

    def test_failed_registration_is_atomic_and_retryable(self):
        mgr = self._make_manager()
        peer = self._peer("decode-agent")
        mgr.agent.add_remote_agent.side_effect = [RuntimeError("UCX failed"), None]

        with self.assertRaisesRegex(RuntimeError, "UCX failed"):
            mgr._add_remote_peer(peer)

        self.assertNotIn(peer.agent_name, mgr.decode_kv_args_table)

        mgr._add_remote_peer(peer)

        self.assertIs(mgr.decode_kv_args_table[peer.agent_name], peer)
        self.assertEqual(mgr.agent.add_remote_agent.call_count, 2)
        mgr._prepare_payload_xfer.assert_called_once_with(peer)

    def test_invalidated_remote_metadata_is_registered_again(self):
        mgr = self._make_manager()
        old_peer = self._peer("decode-agent")
        replacement = self._peer("decode-agent")
        mgr.decode_kv_args_table[old_peer.agent_name] = old_peer
        mgr.agent.check_remote_metadata.return_value = False

        mgr._add_remote_peer(replacement)

        mgr.agent.remove_remote_agent.assert_called_once_with("decode-agent")
        mgr.agent.add_remote_agent.assert_called_once_with(replacement.agent_metadata)
        self.assertIs(mgr.decode_kv_args_table[replacement.agent_name], replacement)

    def test_new_decode_agent_registers_alongside_stale_agent(self):
        mgr = self._make_manager()
        old_peer = self._peer("old-decode-agent")
        replacement = self._peer("new-decode-agent")
        mgr.decode_kv_args_table[old_peer.agent_name] = old_peer

        mgr._add_remote_peer(replacement)

        self.assertEqual(
            set(mgr.decode_kv_args_table),
            {"old-decode-agent", "new-decode-agent"},
        )
        mgr.agent.add_remote_agent.assert_called_once_with(replacement.agent_metadata)


class TestNixlBootstrapThread(CustomTestCase):
    def test_bootstrap_thread_survives_recv_errors_and_exits_on_shutdown(self):
        """The bootstrap thread is the only receiver of peer registrations:
        recv errors must not kill it, and only ETERM/ENOTSOCK may end it.
        Drives a non-zmq error, a transient zmq error, one message, and
        shutdown."""
        mgr = object.__new__(NixlKVManager)
        handled = []
        mgr._handle_bootstrap_message = handled.append
        mgr.server_socket = MagicMock()
        mgr.server_socket.recv_multipart.side_effect = [
            ValueError("malformed frame"),
            zmq.ZMQError(zmq.EAGAIN),
            [b"frame"],
            zmq.ZMQError(zmq.ETERM),
        ]
        threads = []
        real_thread = threading.Thread

        def capture_thread(*args, **kwargs):
            # Daemonize so a regressed shutdown path (thread never exits)
            # fails the assertion below instead of wedging the test process.
            kwargs.setdefault("daemon", True)
            thread = real_thread(*args, **kwargs)
            threads.append(thread)
            return thread

        with patch(
            "sglang.srt.disaggregation.nixl.conn.threading.Thread",
            side_effect=capture_thread,
        ), patch("sglang.srt.disaggregation.nixl.conn.time.sleep"):
            mgr._start_bootstrap_thread()
            threads[0].join(timeout=10)

        self.assertFalse(threads[0].is_alive())
        self.assertEqual(handled, [[b"frame"]])


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

    def test_prefetch_predicate_survives_concurrent_peer_discard(self):
        """The staging predicate reads decode_kv_args_table while the
        bootstrap thread pops entries; check-then-index raced to a KeyError.
        Reproduced with a table whose membership test discards the entry."""

        class DiscardOnContains(dict):
            def __contains__(self, key):
                present = super().__contains__(key)
                if present:
                    super().pop(key)
                return present

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
        mgr.decode_kv_args_table = DiscardOnContains(
            {"agent": SimpleNamespace(decode_tp_size=2)}
        )

        mgr._prefetch_staging_reqs(3)

        self.assertIn(3, mgr._staging_ctx.prefetched_rooms)

    def test_missing_peer_leaves_room_retryable(self):
        """Regression: classifying a mid-re-registration peer as "no staging
        needed" permanently marked the room prefetched with no STAGING_REQ
        ever sent. An unresolved peer leaves the room unmarked; the fan-out
        must then run."""
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
        mgr.decode_kv_args_table = {}

        mgr._prefetch_staging_reqs(3)

        self.assertNotIn(3, mgr._staging_ctx.prefetched_rooms)

        mgr.decode_kv_args_table["agent"] = SimpleNamespace(decode_tp_size=4)
        with patch(
            "sglang.srt.disaggregation.nixl.conn.get_schedule",
            return_value=SimpleNamespace(chunked_prefill_size=4),
        ), patch(
            "sglang.srt.disaggregation.common.staging_handler.prefetch_staging_reqs"
        ) as fan_out:
            mgr._prefetch_staging_reqs(3)

        fan_out.assert_called_once()
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

    def test_last_chunk_of_unprefetched_room_fails_instead_of_spinning(self):
        """Regression: a last chunk enqueued while the prefetch bailed on a
        re-registering peer can never become ready, and no later chunk
        re-runs the prefetch -- requeueing waits forever. It must fail the
        room so the retry re-runs the prefetch; non-last chunks may requeue."""
        kv_chunk = TransferKVChunk(
            room=3,
            prefill_kv_indices=np.array([10, 11], dtype=np.int32),
            index_slice=slice(0, 2),
            is_last_chunk=True,
            chunk_id=0,
            prefill_aux_index=0,
            state_indices=None,
        )
        req = SimpleNamespace(room=3, agent_name="decode_agent")
        strategy = MagicMock()
        strategy.check_ready.return_value = (False, 0, -1, 0, -1)

        mgr = self._make_manager()
        mgr._staging_ctx = PrefillStagingContext()  # room 3 never prefetched
        queue = FakeQueue()
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.disaggregation.common.staging_buffer": (
                    _fake_staging_buffer_module()
                )
            },
        ):
            with self.assertRaisesRegex(RuntimeError, "No staging request"):
                mgr._do_staging_transfer(
                    strategy,
                    kv_chunk,
                    kv_chunk.prefill_kv_indices,
                    req,
                    SimpleNamespace(),
                    queue,
                )
        self.assertEqual(queue.items, [])

        # Control: the same last chunk keeps requeueing normally when the
        # prefetch did run — a pending allocation is legitimately in flight.
        mgr = self._make_manager()
        mgr._staging_ctx = PrefillStagingContext()
        mgr._staging_ctx.prefetched_rooms.add(3)
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
