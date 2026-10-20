"""CPU-buffer PD data integrity and ownership regressions (no GPU required)."""

import ctypes
import struct
import threading
import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.arg_groups.pd_disaggregation_hook import handle_pd_disaggregation
from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.bootstrap import DeferredBootstrap
from sglang.srt.disaggregation.common.conn import KVTransferError
from sglang.srt.disaggregation.nixl.conn import (
    NixlKVManager,
    NixlKVReceiver,
    NixlKVSender,
    TransferInfo,
)
from sglang.srt.disaggregation.nixl.cpu_buffer import (
    CPU_BUFFER_VERSION,
    CpuBufferPool,
    chunk_descriptors,
    pack_pages,
    page_views,
    scatter_pages,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def manager(size=1024):
    mgr = NixlKVManager.__new__(NixlKVManager)
    mgr._cpu_pool = CpuBufferPool(size)
    mgr._cpu_receives = {}
    mgr._cpu_receive_lock = threading.RLock()
    mgr._deferred_abort_generation = 0
    mgr._deferred_abort_ack_tracker = {}
    mgr.attn_tp_size = 1
    mgr.kv_args = SimpleNamespace(kv_item_lens=[4], kv_layer_ids=[0])
    mgr._cpu_views = [torch.arange(32, dtype=torch.uint8).reshape(8, 4)]
    return mgr


def lifecycle_manager():
    mgr = manager()
    mgr._cpu_bootstrap = DeferredBootstrap(30)
    mgr.request_status = {}
    mgr.failure_lock = threading.Lock()
    mgr.failure_records = {}
    mgr.exceptions = {}
    mgr.deferred_bootstrap = None
    mgr.defer_decode_allocation = False
    mgr.enable_deferred_decode_kv_release = False
    mgr.enable_staging = False
    mgr._staging_ctx = None
    mgr.bootstrap_timeout = 600
    mgr.is_dummy_cp_rank = False
    mgr._deferred_ack_targets = {}
    mgr._deferred_ack_poisoned_rooms = set()
    mgr._staging_outstanding = {}
    mgr.addr_to_rooms_tracker = defaultdict(set)
    mgr.required_prefill_response_num_table = {}
    mgr.prefill_response_tracker = {}
    mgr.transfer_statuses = {}
    mgr.transfer_infos = {}
    mgr.req_to_decode_prefix_len = {}
    mgr.local_ip = "127.0.0.1"
    mgr.rank_port = 1234
    mgr._maybe_ack_drained_abort = MagicMock()
    return mgr


class TestCpuBufferAllocationFailure(CustomTestCase):
    def setUp(self):
        self.enterContext(
            patch(
                "sglang.srt.disaggregation.common.conn.get_parallel",
                return_value=SimpleNamespace(num_dp_ranks=1),
            )
        )

    def test_exhaustion_notifies_prefill_before_metadata_and_preserves_cause(self):
        mgr = lifecycle_manager()
        receiver = NixlKVReceiver(mgr, "peer", 7)
        receiver.bootstrap_infos = [{"rank_ip": "peer", "rank_port": 1234}]
        sock = MagicMock()
        receiver._connect_to_bootstrap_server = MagicMock(
            return_value=(sock, threading.Lock())
        )
        receiver.send_metadata(np.arange(300, dtype=np.int32), 0)
        self.assertEqual(receiver.poll(), KVPoll.Failed)
        self.assertEqual(mgr._cpu_pool._ranges, {})
        with self.assertRaisesRegex(KVTransferError, "pool exhausted"):
            receiver.failure_exception()
        receiver.clear()
        sock.send_multipart.assert_called_once()
        self.assertEqual(sock.send_multipart.call_args.args[0][0], b"ABORT")
        self.assertNotIn(7, mgr.request_status)
        self.assertEqual(mgr._cpu_receives, {})
        self.assertEqual(mgr._deferred_abort_ack_tracker, {})

        # The ABORT arrives before the prefill HTTP request creates a sender.
        prefill = lifecycle_manager()
        prefill._handle_abort_notification(sock.send_multipart.call_args.args[0])
        sender = NixlKVSender(prefill, "peer", 7, True)
        self.assertEqual(sender.poll(), KVPoll.Failed)
        sender.clear()
        self.assertNotIn(7, prefill.request_status)
        self.assertIsNone(prefill._cpu_bootstrap.rooms[7].owner)

        # A later, distinct request remains usable and releases its room.
        healthy = NixlKVSender(prefill, "peer", 8, True)
        self.assertEqual(healthy.poll(), KVPoll.Bootstrapping)
        prefill.update_status(8, KVPoll.Success)
        healthy.clear()
        self.assertNotIn(8, prefill._cpu_bootstrap.rooms)

    def test_abort_after_sender_creation_is_terminal(self):
        mgr = lifecycle_manager()
        sender = NixlKVSender(mgr, "peer", 9, True)
        mgr._handle_abort_notification([b"ABORT", b"9", b"127.0.0.1", b"1234"])
        self.assertEqual(sender.poll(), KVPoll.Failed)
        mgr.update_status(9, KVPoll.WaitingForInput)
        self.assertEqual(sender.poll(), KVPoll.Failed)
        sender.clear()
        sender.clear()
        self.assertNotIn(9, mgr.request_status)

    def test_sender_uses_common_chunk_arguments(self):
        mgr = lifecycle_manager()
        mgr.kv_shard_size = 1
        mgr.enable_all_cp_ranks_for_transfer = False
        mgr.prepare_send_state_indices = lambda indices: indices
        mgr.add_transfer_request = MagicMock()
        sender = NixlKVSender(mgr, "peer", 9, True)
        sender.init(2, aux_index=4)
        indices = np.array([1, 2], dtype=np.int32)
        states = [[3]]
        sender.send(indices, states, num_kv_tokens=2)
        args, kwargs = mgr.add_transfer_request.call_args
        self.assertEqual(args[0], 9)
        self.assertIs(args[1], indices)
        self.assertEqual(args[2:], (slice(0, 2), True))
        self.assertEqual(kwargs["aux_index"], 4)
        self.assertEqual(kwargs["state_indices"], states)
        self.assertEqual(kwargs["num_kv_tokens"], 2)

    def test_success_restores_cpu_state_once_before_returning(self):
        mgr = lifecycle_manager()
        receiver = NixlKVReceiver(mgr, "peer", 7)
        region = mgr._allocate_cpu_receive(7, np.array([2]), 1)
        region.buffer.fill_(91)
        mgr.update_status(7, KVPoll.Success)
        with patch.object(
            mgr, "_finish_cpu_receive", wraps=mgr._finish_cpu_receive
        ) as finish:
            self.assertEqual(receiver.poll(), KVPoll.Success)
            self.assertTrue(torch.all(mgr._cpu_views[0][2] == 91))
            self.assertEqual(mgr._cpu_pool._ranges, {})
            self.assertEqual(receiver.poll(), KVPoll.Success)
            finish.assert_called_once_with(7)

    def test_restore_failure_is_terminal(self):
        mgr = lifecycle_manager()
        receiver = NixlKVReceiver(mgr, "peer", 7)
        mgr.update_status(7, KVPoll.Success)
        with patch.object(
            mgr, "_finish_cpu_receive", side_effect=RuntimeError("restore failed")
        ) as finish:
            self.assertEqual(receiver.poll(), KVPoll.Failed)
            self.assertEqual(receiver.poll(), KVPoll.Failed)
            finish.assert_called_once_with(7)
        self.assertIn("restore failed", mgr.failure_records[7])


class TestCpuBufferConfiguration(CustomTestCase):
    def test_matching_local_topology_defaults_to_pageable_retraction(self):
        for mode in ("prefill", "decode"):
            with self.subTest(mode=mode):
                args = ServerArgs(
                    model_path="dummy",
                    device="cuda",
                    disaggregation_mode=mode,
                    disaggregation_transfer_backend="nixl",
                    nixl_use_cpu_buffer=True,
                )
                handle_pd_disaggregation(args)
                self.assertEqual(
                    resolving_view(args).disaggregation_decode_retraction_backup,
                    "cpu_tensor" if mode == "decode" else None,
                )
                self.assertIsNone(args.disaggregation_decode_retraction_backup)

        for backend in ("none", "cpu_tensor"):
            with self.subTest(backend=backend):
                args = ServerArgs(
                    model_path="dummy",
                    device="cuda",
                    disaggregation_mode="decode",
                    disaggregation_transfer_backend="nixl",
                    nixl_use_cpu_buffer=True,
                    disaggregation_decode_retraction_backup=backend,
                )
                handle_pd_disaggregation(args)
                self.assertEqual(
                    resolving_view(args).disaggregation_decode_retraction_backup,
                    backend,
                )

    def test_unsupported_combinations_fail(self):
        for change in (
            {"disaggregation_mode": "null"},
            {"disaggregation_transfer_backend": "mooncake"},
            {"nixl_cpu_buffer_size_gb": float("nan")},
            {"nixl_cpu_buffer_size_gb": 0},
            {"pp_size": 2},
            {"attn_cp_size": 2},
            {"dcp_size": 2},
            {"speculative_algorithm": "EAGLE"},
            {"enable_hierarchical_cache": True},
            {"disaggregation_decode_enable_offload_kvcache": True},
            {"disaggregation_decode_host_receive_threshold": 0.5},
            {
                "disaggregation_mode": "decode",
                "disaggregation_decode_retraction_backup": "host_pool",
            },
        ):
            with self.subTest(change=change):
                values = dict(
                    model_path="dummy",
                    device="cuda",
                    disaggregation_mode="prefill",
                    disaggregation_transfer_backend="nixl",
                    nixl_use_cpu_buffer=True,
                )
                values.update(change)
                with self.assertRaisesRegex(ValueError, "CPU.buffer|cpu.buffer"):
                    handle_pd_disaggregation(ServerArgs(**values))

    def test_optional_receive_region_preserves_existing_wire_frames(self):
        msg = [
            b"7",
            b"127.0.0.1",
            b"1234",
            b"peer",
            np.array([2, 5], dtype=np.int32).tobytes(),
            b"0",
            b"1",
            b"",
            b"0",
            b"0",
        ]
        old = TransferInfo.from_zmq(msg)
        new = TransferInfo.from_zmq(msg + [struct.pack("!QQ", 4096, 2)])
        self.assertEqual(old.cpu_buffer_ptr, 0)
        self.assertEqual(new.cpu_buffer_ptr, 4096)
        self.assertEqual(new.cpu_buffer_pages, 2)
        np.testing.assert_array_equal(old.dst_kv_indices, new.dst_kv_indices)


class TestCpuBufferPool(CustomTestCase):
    def test_alignment_exhaustion_and_reuse(self):
        pool = CpuBufferPool(768)
        a, _ = pool.allocate(1)
        b, _ = pool.allocate(257)
        self.assertEqual((a, b), (0, 256))
        with self.assertRaises(BufferError):
            pool.allocate(1)
        pool.release(a)
        c, _ = pool.allocate(255)
        self.assertEqual(c, a)
        pool.release(b)
        pool.release(c)
        self.assertEqual(pool.allocate(768)[0], 0)
        self.assertFalse(pool.buffer.is_pinned())

    def test_pending_or_unknown_state_keeps_source_owned(self):
        pool = CpuBufferPool(256)
        offset, _ = pool.allocate(16)
        pool.retain("handle", offset)
        agent = MagicMock()
        agent.check_xfer_state.side_effect = ["PROC", RuntimeError("peer lost"), "DONE"]
        for _ in range(2):
            pool.reap(agent)
            with self.assertRaises(BufferError):
                pool.allocate(1)
        pool.reap(agent)
        self.assertEqual(pool.allocate(1)[0], offset)

    def test_registration_only_uses_host_memory(self):
        with torch.device("meta"):
            pool = CpuBufferPool(256)
        self.assertEqual(pool.buffer.device.type, "cpu")
        mgr = manager()
        mgr._cpu_pool = pool
        mgr.kv_args = SimpleNamespace(
            kv_data_ptrs=[0xDEAD],
            kv_data_lens=[16],
            kv_data_mem_kinds=["VRAM"],
            aux_data_ptrs=[0xBEEF],
            aux_data_lens=[8],
            gpu_id=0,
            state_data_ptrs=[],
            state_data_lens=[],
        )
        mgr.agent = MagicMock()
        mgr.register_buffer_to_engine()
        calls = mgr.agent.register_memory.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(call.args[1] == "DRAM" for call in calls))
        self.assertEqual(calls[0].args[0], [(pool.buffer.data_ptr(), 256, 0, "")])
        self.assertEqual(calls[1].args[0], [(0xBEEF, 8, 0, "")])


class TestCpuBufferData(CustomTestCase):
    def test_chunked_layer_major_roundtrip_preserves_bytes(self):
        # Unequal widths also cover MLA-like single-buffer layouts. Sparse source
        # and destination pages must not alter untouched destination pages.
        for dtype in (torch.float16, torch.bfloat16, torch.uint8):
            with self.subTest(dtype=dtype):
                source = [
                    torch.arange(48).to(dtype).reshape(8, 6),
                    torch.arange(80).to(dtype).reshape(8, 10),
                ]
                target = [torch.zeros_like(t) for t in source]
                widths = [t[0].numel() * t.element_size() for t in source]
                src_views = page_views(source, [t.data_ptr() for t in source], widths)
                dst_views = page_views(target, [t.data_ptr() for t in target], widths)
                receive = torch.empty(3 * sum(widths), dtype=torch.uint8)
                src_ids, dst_ids = np.array([5, 1, 7]), np.array([2, 6, 0])
                for start, end in ((0, 2), (2, 3)):
                    packed = torch.empty((end - start) * sum(widths), dtype=torch.uint8)
                    pack_pages(src_views, src_ids[start:end], packed)
                    src, dst = chunk_descriptors(
                        packed.data_ptr(),
                        receive.data_ptr(),
                        widths,
                        end - start,
                        3,
                        start,
                    )
                    for (sp, size, _), (dp, dst_size, _) in zip(src, dst):
                        self.assertEqual(size, dst_size)
                        ctypes.memmove(dp, sp, size)
                scatter_pages(dst_views, dst_ids, receive)
                for original, restored in zip(source, target):
                    self.assertTrue(torch.equal(original[src_ids], restored[dst_ids]))
                    self.assertEqual(torch.count_nonzero(restored[[1, 3, 4, 5, 7]]), 0)

    def test_invalid_ranges_fail_before_transfer(self):
        with self.assertRaises(ValueError):
            chunk_descriptors(100, 200, [8], 2, 3, 2)
        with self.assertRaises(ValueError):
            pack_pages([torch.zeros((4, 8), dtype=torch.uint8)], [-1], torch.empty(8))
        with self.assertRaises(ValueError):
            page_views([torch.zeros(4)], [123], [4])


class TestCpuBufferOwnership(CustomTestCase):
    def test_abort_timeout_does_not_recycle_published_region(self):
        mgr = manager(256)
        region = mgr._allocate_cpu_receive(7, np.array([1]), 2)
        region.published = True
        generation = mgr.register_deferred_abort_room(7)
        mgr._clear_cpu_receive(7)
        mgr.clear_deferred_abort_state(7)  # scheduler timeout
        mgr.note_abort_ack(7, 0, generation - 1)  # stale generation
        mgr.note_abort_ack(7, 0, generation)
        mgr.note_abort_ack(7, 0, generation)  # duplicate rank
        with self.assertRaises(BufferError):
            mgr._cpu_pool.allocate(1)
        self.assertIn(7, mgr._deferred_abort_ack_tracker)
        mgr.note_abort_ack(7, 1, generation)
        self.assertNotIn(7, mgr._cpu_receives)
        self.assertNotIn(7, mgr._deferred_abort_ack_tracker)
        self.assertEqual(mgr._cpu_pool.allocate(1)[0], 0)

    def test_unpublished_region_and_success_recycle(self):
        mgr = manager(256)
        mgr._allocate_cpu_receive(7, np.array([2]), 1)
        mgr._clear_cpu_receive(7)
        region = mgr._allocate_cpu_receive(8, np.array([2]), 1)
        region.buffer.fill_(99)
        mgr._finish_cpu_receive(8)
        self.assertTrue(
            torch.equal(mgr._cpu_views[0][2], torch.full((4,), 99, dtype=torch.uint8))
        )
        self.assertEqual(mgr._cpu_pool.allocate(1)[0], 0)

    def test_partial_post_keeps_handle_and_source_for_drain(self):
        mgr = manager(256)
        mgr.agent = MagicMock()
        mgr.agent.initialize_xfer.return_value = "handle"
        mgr.agent.transfer.side_effect = RuntimeError("partial post")
        mgr.agent.check_xfer_state.return_value = "PROC"
        peer = SimpleNamespace(
            cpu_buffer_version=CPU_BUFFER_VERSION,
            decode_tp_size=1,
            dst_dcp_size=1,
            dst_kv_item_lens=[4],
            dst_kv_layer_ids=[0],
            dst_state_data_ptrs=[],
        )
        req = SimpleNamespace(
            cpu_buffer_pages=1,
            dst_kv_indices=[2],
            cpu_buffer_ptr=4096,
            agent_name="peer",
        )
        chunk = SimpleNamespace(
            index_slice=slice(0, 1), prefill_kv_indices=np.array([1])
        )
        handles = []
        with self.assertRaisesRegex(RuntimeError, "partial post"):
            mgr._send_cpu_chunk(req, peer, chunk, "7_kv_0_1_0", handles)
        self.assertEqual(handles, ["handle"])
        mgr._cpu_pool.reap(mgr.agent)
        with self.assertRaises(BufferError):
            mgr._cpu_pool.allocate(1)
        mgr.agent.check_xfer_state.return_value = "ERR"
        mgr._cpu_pool.reap(mgr.agent)
        self.assertEqual(mgr._cpu_pool.allocate(1)[0], 0)

    def test_aux_partial_post_is_in_drain_list(self):
        mgr = manager()
        mgr.kv_args.aux_data_ptrs = [128]
        mgr.kv_args.aux_item_lens = [4]
        mgr.agent = MagicMock()
        mgr.agent.initialize_xfer.return_value = "aux"
        mgr.agent.transfer.side_effect = RuntimeError("partial aux post")
        handles = []
        with self.assertRaisesRegex(RuntimeError, "partial aux post"):
            mgr.send_aux("peer", 0, [256], 0, handles=handles)
        self.assertEqual(handles, ["aux"])

    def test_mixed_protocol_and_tp_are_rejected(self):
        mgr = manager()
        peer = SimpleNamespace(cpu_buffer_version=b"")
        with self.assertRaisesRegex(ValueError, "Both NIXL peers"):
            mgr._validate_cpu_peer(peer)
        peer.cpu_buffer_version = CPU_BUFFER_VERSION
        peer.decode_tp_size = 2
        with self.assertRaisesRegex(ValueError, "matching TP"):
            mgr._validate_cpu_peer(peer)


if __name__ == "__main__":
    unittest.main()
