"""CPU-buffer PD data integrity and ownership regressions (no GPU required)."""

import ctypes
import json
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
from sglang.srt.disaggregation.base.conn import KVPoll, StateType
from sglang.srt.disaggregation.common.bootstrap import DeferredBootstrap
from sglang.srt.disaggregation.common.conn import KVTransferError
from sglang.srt.disaggregation.nixl.conn import (
    KVArgsRegisterInfo,
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
    mgr.kv_args = SimpleNamespace(
        kv_item_lens=[4],
        kv_layer_ids=[0],
        state_types=[],
        state_data_ptrs=[],
        state_item_lens=[],
    )
    mgr._cpu_state_views = []
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

    def test_kv_sharding_rejected_only_when_cpu_buffer_enabled(self):
        for mode in ("prefill", "decode"):
            for enabled in (False, True):
                with self.subTest(mode=mode, cpu_buffer=enabled):
                    args = ServerArgs(
                        model_path="dummy",
                        device="cuda",
                        tp_size=2,
                        disaggregation_mode=mode,
                        disaggregation_transfer_backend="nixl",
                        nixl_use_cpu_buffer=enabled,
                        enable_kv_cache_sharding=True,
                    )
                    if enabled:
                        with self.assertRaisesRegex(
                            ValueError,
                            "--nixl-use-cpu-buffer.*--enable-kv-cache-sharding",
                        ):
                            handle_pd_disaggregation(args)
                    else:
                        # This hook must not reject the existing NIXL path.
                        # Sharding's own hook validates its role/topology rules.
                        handle_pd_disaggregation(args)

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

    def test_state_types_wire_extension_preserves_normal_registration(self):
        msg = [
            b"0",
            b"localhost",
            b"1",
            b"peer",
            b"metadata",
            struct.pack("Q", 4096),
            b"",
            b"",
            b"0",
            b"2",
            b"0",
            b"4",
            b"",
            b"",
            b"",
            b"0",
            b"",
            b"VRAM",
            struct.pack("Q", 4),
            b"",
            b"",
            b"1",
            b"0",
            b"",
        ]
        old = KVArgsRegisterInfo.from_zmq(msg)
        types = [StateType.MAMBA, StateType.QSA_PENDING, StateType.QSA_COMPRESSED]
        new = KVArgsRegisterInfo.from_zmq(
            msg + [CPU_BUFFER_VERSION, json.dumps(types).encode()]
        )
        self.assertEqual(old.cpu_buffer_version, b"")
        self.assertEqual(old.dst_state_types, [])
        self.assertEqual(new.cpu_buffer_version, CPU_BUFFER_VERSION)
        self.assertEqual(new.dst_state_types, types)
        self.assertEqual(old.dst_kv_ptrs, new.dst_kv_ptrs)

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
            state_data_ptrs=[[0xFACE]],
            state_data_lens=[[32]],
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

    def test_disjoint_regions_share_backing_without_overwrite(self):
        source = torch.arange(96, dtype=torch.uint8).reshape(12, 8)
        target = torch.full_like(source, 255)
        # SWA rows and compressed pages occupy disjoint regions of one tensor.
        src = page_views(
            [source], [source.data_ptr(), source.data_ptr() + 32], [8, 16], [32, 64]
        )
        dst = page_views(
            [target], [target.data_ptr(), target.data_ptr() + 32], [8, 16], [32, 64]
        )
        for src_view, dst_view, src_ids, dst_ids in (
            (src[0], dst[0], [0, 2], [1, 3]),
            (src[1], dst[1], [1, 3], [0, 2]),
        ):
            packed = torch.empty(len(src_ids) * src_view.shape[1], dtype=torch.uint8)
            pack_pages([src_view], src_ids, packed)
            scatter_pages([dst_view], dst_ids, packed)
            self.assertTrue(torch.equal(src_view[src_ids], dst_view[dst_ids]))
        self.assertTrue(torch.all(target[[0, 2, 6, 7, 10, 11]] == 255))

    def test_physical_region_bounds(self):
        tensor = torch.zeros(64, dtype=torch.uint8)
        for offset, width, length in ((32, 8, 40), (0, 8, 63), (0, 0, 64), (0, 8, -8)):
            with self.subTest(offset=offset, width=width, length=length):
                with self.assertRaises(ValueError):
                    page_views(
                        [tensor], [tensor.data_ptr() + offset], [width], [length]
                    )
        with self.assertRaises(ValueError):
            page_views([tensor], [tensor.data_ptr()], [8], [])

    def test_indexer_tail_padding_is_not_addressable(self):
        tensor = torch.arange(68, dtype=torch.uint8)
        views = page_views(
            [tensor], [tensor.data_ptr()], [8], [68], allow_trailing_padding=True
        )
        self.assertEqual(views[0].shape, (8, 8))
        with self.assertRaises(ValueError):
            pack_pages(views, [8], torch.empty(8, dtype=torch.uint8))
        for length in (7, 72):
            with self.subTest(length=length), self.assertRaises(ValueError):
                page_views(
                    [tensor],
                    [tensor.data_ptr()],
                    [8],
                    [length],
                    allow_trailing_padding=True,
                )

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


class TestCpuBufferMambaState(CustomTestCase):
    def test_hybrid_tensor_mapping_matches_registered_pointer_order(self):
        mgr = manager()
        key = torch.zeros((8, 4), dtype=torch.uint8)
        value = torch.zeros_like(key)
        conv = torch.zeros((4, 3), dtype=torch.bfloat16)
        recurrent = torch.zeros((4, 2, 2), dtype=torch.float32)
        mgr.kv_args.kv_data_ptrs = [value.data_ptr(), key.data_ptr()]
        mgr.kv_args.kv_item_lens = [4, 4]
        mgr.kv_args.state_types = [StateType.MAMBA]
        mgr.kv_args.state_data_ptrs = [[recurrent.data_ptr(), conv.data_ptr()]]
        mgr.kv_args.state_item_lens = [[16, 6]]
        pool = SimpleNamespace(
            full_kv_pool=SimpleNamespace(k_buffer=[key], v_buffer=[value]),
            mamba_pool=SimpleNamespace(
                _iter_transfer_state_entries=lambda: iter(
                    [("conv", conv, 0, 1), ("ssm", recurrent, 0, 1)]
                )
            ),
        )
        mgr.set_cpu_kv_buffers(pool)
        self.assertEqual(
            [v.data_ptr() for v in mgr._cpu_views], mgr.kv_args.kv_data_ptrs
        )
        self.assertEqual([v.shape[1] for v in mgr._cpu_state_views[0]], [16, 6])
        self.assertEqual(
            [v.data_ptr() for v in mgr._cpu_state_views[0]],
            mgr.kv_args.state_data_ptrs[0],
        )

    def test_state_roundtrip_and_abort_share_receive_ownership(self):
        mgr = manager(512)
        mgr.kv_args.state_types = [StateType.MAMBA]
        mgr._cpu_state_views = [
            [
                torch.zeros((4, 6), dtype=torch.uint8),
                torch.zeros((4, 8), dtype=torch.uint8),
            ]
        ]
        region = mgr._allocate_cpu_receive(9, np.array([2]), 2, [[3]])
        region.buffer.copy_(torch.arange(18, dtype=torch.uint8))
        mgr._finish_cpu_receive(9)
        self.assertEqual(mgr._cpu_views[0][2].tolist(), list(range(4)))
        self.assertEqual(mgr._cpu_state_views[0][0][3].tolist(), list(range(4, 10)))
        self.assertEqual(mgr._cpu_state_views[0][1][3].tolist(), list(range(10, 18)))
        self.assertEqual(mgr._cpu_state_views[0][0][:3].count_nonzero(), 0)
        region = mgr._allocate_cpu_receive(10, np.array([], dtype=np.int32), 2, [[1]])
        self.assertEqual(region.buffer.numel(), 14)  # Final state without KV pages.
        region.published = True
        generation = mgr.register_deferred_abort_room(10)
        mgr._clear_cpu_receive(10)
        mgr.note_abort_ack(10, 0, generation)
        self.assertIn(10, mgr._cpu_receives)
        mgr.note_abort_ack(10, 1, generation)
        self.assertNotIn(10, mgr._cpu_receives)

    def test_state_post_failure_retains_source_and_validates_layout(self):
        mgr = manager(256)
        mgr.kv_args.state_types = [StateType.MAMBA]
        mgr._cpu_state_views = [[torch.arange(16, dtype=torch.uint8).reshape(4, 4)]]
        mgr.kv_args.state_item_lens = [[4]]
        mgr.kv_args.state_layer_ids = [[1]]
        mgr.kv_args.state_dim_per_tensor = [[2]]
        peer = SimpleNamespace(
            cpu_buffer_version=CPU_BUFFER_VERSION,
            decode_tp_size=1,
            dst_dcp_size=1,
            dst_kv_item_lens=[4],
            dst_kv_layer_ids=[0],
            dst_state_types=[StateType.MAMBA],
            dst_state_item_lens=[[4]],
            dst_state_layer_ids=[[1]],
            dst_state_dim_per_tensor=[[2]],
        )
        req = SimpleNamespace(
            cpu_buffer_ptr=4096,
            cpu_buffer_pages=2,
            dst_kv_indices=[2, 3],
            dst_state_indices=[[1]],
            agent_name="peer",
        )
        chunk = SimpleNamespace(state_indices=[[2]])
        mgr.agent = MagicMock()
        mgr.agent.initialize_xfer.return_value = "state"
        mgr.agent.transfer.side_effect = RuntimeError("partial state post")
        mgr.agent.check_xfer_state.return_value = "PROC"
        handles = []
        with self.assertRaisesRegex(RuntimeError, "partial state post"):
            mgr._send_cpu_state(req, peer, chunk, "9_state_0", handles)
        self.assertEqual(handles, ["state"])
        self.assertEqual(
            mgr.agent.get_xfer_descs.call_args_list[1].args, ([(4104, 4, 0)], "DRAM")
        )
        mgr._cpu_pool.reap(mgr.agent)
        with self.assertRaises(BufferError):
            mgr._cpu_pool.allocate(1)
        peer.dst_state_layer_ids = [[3]]
        with self.assertRaisesRegex(ValueError, "state layouts"):
            mgr._validate_cpu_peer(peer)
        for indices in (None, [], [[1, 2]], [[-1]], [[4]]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                mgr._allocate_cpu_receive(11, np.array([0]), 1, indices)
        self.assertNotIn(11, mgr._cpu_receives)


class TestCpuBufferQsaState(CustomTestCase):
    def make_manager(self, size=2048):
        mgr = manager(size)
        mgr.kv_args.state_types = [
            StateType.MAMBA,
            StateType.QSA_PENDING,
            StateType.QSA_COMPRESSED,
        ]
        mgr._cpu_state_views = [
            [torch.zeros((8, 6), dtype=torch.uint8)],
            [
                torch.zeros((8, 16), dtype=torch.uint8),
                torch.zeros((8, 32), dtype=torch.uint8),
            ],
            [
                torch.zeros((8, 8), dtype=torch.uint8),
                torch.zeros((8, 4), dtype=torch.uint8),
            ],
        ]
        mgr.kv_args.state_item_lens = [[6], [16, 32], [8, 4]]
        return mgr

    def test_pending_ring_and_compressed_page_mapping(self):
        mgr = manager()
        ratio, page_size = 4, 64
        key = torch.zeros((128, 2), dtype=torch.bfloat16)
        pending = torch.zeros((8 * ratio, 2), dtype=torch.bfloat16)
        positions = torch.zeros((8 * ratio,), dtype=torch.int64)
        compressed = torch.zeros((128 // ratio, 3), dtype=torch.bfloat16)
        mgr.kv_args.kv_data_ptrs = [key.data_ptr()]
        mgr.kv_args.kv_item_lens = [page_size * key[0].nbytes]
        mgr.kv_args.state_types = [StateType.QSA_PENDING, StateType.QSA_COMPRESSED]
        mgr.kv_args.state_data_ptrs = [
            [positions.data_ptr(), pending.data_ptr()],
            [compressed.data_ptr()],
        ]
        mgr.kv_args.state_item_lens = [
            [ratio * 8, ratio * pending[0].nbytes],
            [page_size // ratio * compressed[0].nbytes],
        ]
        pool = SimpleNamespace(
            kv_buffer=[key],
            qsa_key_state_buffer_pool=[pending],
            qsa_rope_position_buffer=positions,
            qsa_compressed_k_buffer_pool=[compressed],
        )
        mgr.set_cpu_kv_buffers(pool)
        self.assertEqual(
            [tuple(v.shape) for v in mgr._cpu_state_views[0]], [(8, 32), (8, 16)]
        )
        self.assertEqual(tuple(mgr._cpu_state_views[1][0].shape), (2, 96))
        # Pending slots are request rows; compressed slots are full-KV pages.
        mgr._cpu_state_views[0][0][3].fill_(1)
        self.assertEqual(
            positions.view(torch.uint8)[3 * 32 : 4 * 32].tolist(), [1] * 32
        )
        self.assertEqual(positions[: 3 * ratio].count_nonzero(), 0)

    def test_sparse_multicomponent_transfer_and_reclamation(self):
        sender, receiver = self.make_manager(), self.make_manager()
        src_ids = [np.array([5]), np.array([2]), np.array([6, 1, 4])]
        dst_ids = [np.array([1]), np.array([7]), np.array([0, 5, 2])]
        for group in sender._cpu_state_views:
            for view in group:
                view.copy_(
                    torch.arange(view.numel(), dtype=torch.uint8).reshape(view.shape)
                )
        region = receiver._allocate_cpu_receive(9, np.array([3]), 1, dst_ids)
        # Exercise sender packing and the exact advertised remote address.
        sender.agent = MagicMock()
        sender.agent.get_xfer_descs.side_effect = lambda descs, kind: descs

        def copy_to_remote(op, src, dst, agent, notif):
            self.assertEqual(op, "WRITE")
            self.assertEqual(src[0][1], dst[0][1])
            ctypes.memmove(dst[0][0], src[0][0], src[0][1])
            return "state"

        sender.agent.initialize_xfer.side_effect = copy_to_remote
        sender.agent.check_xfer_state.return_value = "DONE"
        peer = SimpleNamespace(
            cpu_buffer_version=CPU_BUFFER_VERSION,
            decode_tp_size=1,
            dst_dcp_size=1,
            dst_kv_item_lens=[4],
            dst_kv_layer_ids=[0],
            dst_state_types=sender.kv_args.state_types,
            dst_state_item_lens=sender.kv_args.state_item_lens,
        )
        req = SimpleNamespace(
            cpu_buffer_pages=1,
            dst_kv_indices=[3],
            dst_state_indices=dst_ids,
            cpu_buffer_ptr=region.buffer.data_ptr(),
            agent_name="peer",
        )
        handles = []
        sender._send_cpu_state(
            req, peer, SimpleNamespace(state_indices=src_ids), "9_state_0", handles
        )
        self.assertEqual(handles, ["state"])
        receiver._finish_cpu_receive(9)
        for source, target, si, di in zip(
            sender._cpu_state_views, receiver._cpu_state_views, src_ids, dst_ids
        ):
            for before, after in zip(source, target):
                self.assertTrue(torch.equal(before[si], after[di]))
                untouched = [i for i in range(8) if i not in di]
                self.assertEqual(after[untouched].count_nonzero(), 0)
        sender._cpu_pool.reap(sender.agent)
        self.assertFalse(sender._cpu_pool._ranges)
        self.assertFalse(receiver._cpu_pool._ranges)
        peer.dst_state_types = list(reversed(peer.dst_state_types))
        with self.assertRaisesRegex(ValueError, "state layouts"):
            sender._validate_cpu_peer(peer)
        peer.dst_state_types = sender.kv_args.state_types
        peer.cpu_buffer_version = b"cpu-buffer-v2"
        with self.assertRaises(ValueError):
            sender._validate_cpu_peer(peer)

    def test_state_pressure_and_all_rank_abort_barrier(self):
        mgr = self.make_manager(256)
        indices = [[1], [2], [0, 3, 5]]
        region = mgr._allocate_cpu_receive(1, np.array([1]), 2, indices)
        with self.assertRaises(BufferError):
            mgr._allocate_cpu_receive(2, np.array([2]), 2, indices)
        self.assertNotIn(2, mgr._cpu_receives)
        region.published = True
        generation = mgr.register_deferred_abort_room(1)
        mgr._clear_cpu_receive(1)
        mgr.note_abort_ack(1, 0, generation)
        self.assertIn(1, mgr._cpu_receives)
        mgr.note_abort_ack(1, 1, generation)
        self.assertFalse(mgr._cpu_pool._ranges)
        mgr._allocate_cpu_receive(3, np.array([2]), 2, indices)
        mgr._clear_cpu_receive(3)
        self.assertFalse(mgr._cpu_pool._ranges)

    def test_invalid_components_never_allocate(self):
        mgr = self.make_manager()
        for ids in (
            [[1], [2]],
            [[1, 2], [2], [0]],
            [[1], [8], [0]],
            [[1], [2], [-1]],
            [[1], [2], [0.5]],
        ):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                mgr._allocate_cpu_receive(1, np.array([0]), 1, ids)
        self.assertFalse(mgr._cpu_pool._ranges)
        self.assertFalse(mgr._cpu_receives)

    def test_qsa_retraction_restores_to_new_physical_slots(self):
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
        from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool

        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            with self.subTest(dtype=dtype):
                pool = QSATokenToKVPool.__new__(QSATokenToKVPool)
                pool.qsa_compress_ratio = 4
                pool.qsa_key_state_buffer_pool = [
                    torch.arange(64).to(torch.bfloat16).reshape(32, 2)
                ]
                pool.qsa_rope_position_buffer = torch.arange(
                    96, dtype=torch.int64
                ).reshape(32, 3)
                pool.qsa_compressed_k_buffer_pool = [
                    torch.arange(48).to(dtype).reshape(16, 3)
                ]
                src = torch.tensor([20, 21, 22, 23, 24, 25, 26, 27, 4, 5, 6])
                dst = torch.tensor([32, 33, 34, 35, 36, 37, 38, 39, 12, 13, 14])
                tensors = [
                    *pool.qsa_key_state_buffer_pool,
                    pool.qsa_rope_position_buffer,
                    *pool.qsa_compressed_k_buffer_pool,
                ]
                original = [t.clone() for t in tensors]
                with patch.object(
                    HybridLinearKVPool, "get_cpu_copy", return_value="base"
                ) as get_base:
                    backup = pool.get_cpu_copy(src, mamba_indices=5, req_pool_index=2)
                get_base.assert_called_once_with(src, 5, 2)
                for tensor in tensors:
                    tensor.zero_()
                with patch.object(HybridLinearKVPool, "load_cpu_copy") as load_base:
                    pool.load_cpu_copy(backup, dst, mamba_indices=1, req_pool_index=6)
                load_base.assert_called_once_with("base", dst, 1, 6)
                for before, after in zip(original[:2], tensors[:2]):
                    self.assertTrue(torch.equal(before[8:12], after[24:28]))
                    self.assertEqual(after[:24].count_nonzero(), 0)
                self.assertTrue(
                    torch.equal(
                        original[2].view(torch.uint8)[[5, 6, 1]],
                        tensors[2].view(torch.uint8)[[8, 9, 3]],
                    )
                )
                self.assertEqual(
                    tensors[2]
                    .view(torch.uint8)[[0, 1, 2, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15]]
                    .count_nonzero(),
                    0,
                )
                with self.assertRaisesRegex(ValueError, "request pool index"):
                    pool.get_cpu_copy(src)

    def test_cpu_buffer_model_override_keeps_ple_on_gpu(self):
        from sglang.srt.arg_groups.model_overrides.qwen4_exp import _qwen4_exp_overrides

        for enabled in (False, True):
            args = ServerArgs(
                model_path="dummy", device="cuda", nixl_use_cpu_buffer=enabled
            )
            with (
                patch(
                    "sglang.srt.arg_groups.model_overrides.qwen4_exp.get_platform",
                    return_value=SimpleNamespace(is_cuda=True, is_sm100=False),
                ),
                patch(
                    "sglang.srt.arg_groups.model_overrides.qwen4_exp.model_config_of",
                    return_value=SimpleNamespace(dtype=torch.bfloat16),
                ),
                patch(
                    "sglang.srt.layers.attention.qsa.config.parse_qsa_profile",
                    return_value=None,
                ),
            ):
                overrides = _qwen4_exp_overrides(args, SimpleNamespace())
            self.assertEqual(overrides["ple_offload_embedding"], not enabled)

    def test_ple_offload_rejected_only_for_cpu_buffer(self):
        from sglang.srt.arg_groups.memory_hook import handle_offload_compatibility

        for enabled in (False, True):
            args = ServerArgs(
                model_path="dummy",
                device="cuda",
                nixl_use_cpu_buffer=enabled,
                ple_offload_embedding=True,
            )
            if enabled:
                with self.assertRaisesRegex(ValueError, "CPU-buffer PD.*ple-offload"):
                    handle_offload_compatibility(args)
            else:
                handle_offload_compatibility(args)


class TestCpuBufferDeepSeekState(CustomTestCase):
    def test_unified_fp8_compressed_regions_exclude_swa_ring(self):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

        pool = object.__new__(DeepSeekV4TokenToKVPool)
        pool._unified_kv = True
        pool._unified_kv_fp8 = True
        pool._stage_start, pool._stage_end = 0, 2
        pool.compression_ratios = [2, 1]
        pool.page_size = 4
        pool.request_window = pool.swa_kv_pool = None
        pool.compress_state_pools = pool.indexer_compress_state_pools = []
        pool.kv_pools = {2: None, 1: None}
        pool.index_pools = {}
        kv = [torch.zeros((20, 8), dtype=torch.uint8) for _ in range(2)]
        rope = [torch.zeros((20, 4), dtype=torch.uint8) for _ in range(2)]
        pool.unified_kv_pool = SimpleNamespace(
            swa_pages=4, kv_buffer=kv, kv_buffer_rope=rope
        )
        pointers, lengths, widths = pool.get_contiguous_buf_infos()
        views = page_views(
            pool.cpu_buffer_transfer_tensors(), pointers, widths, lengths
        )
        self.assertEqual(
            [tuple(v.shape) for v in views], [(8, 16), (8, 8), (4, 32), (4, 16)]
        )
        for view in views:
            view.fill_(7)
        for tensor in kv + rope:
            self.assertTrue(torch.all(tensor[:4] == 0))
            self.assertTrue(torch.all(tensor[4:] == 7))
        pointers, lengths, widths = pool.get_unified_swa_ring_buf_infos()
        views = page_views(
            pool.cpu_buffer_transfer_tensors(), pointers, widths, lengths
        )
        for view in views:
            view.fill_(9)
        for tensor in kv + rope:
            self.assertTrue(torch.all(tensor[:4] == 9))
            self.assertTrue(torch.all(tensor[4:] == 7))

    def test_ratio_one_two_buffers_and_request_state(self):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

        pool = object.__new__(DeepSeekV4TokenToKVPool)
        pool._unified_kv = False
        pool._unified_kv_fp8 = False
        pool._stage_start, pool._stage_end = 0, 2
        pool.compression_ratios = [2, 1]
        pool.page_size = 4
        pool.swa_page_size = 4
        pool.request_window = None
        pool.kv_pools = {
            ratio: SimpleNamespace(
                kv_buffer=[torch.arange(64, dtype=torch.uint8).view(8, 8)]
            )
            for ratio in (2, 1)
        }
        # One extra physical index page does not form a logical transfer page.
        index = torch.arange(68, dtype=torch.uint8).view(17, 4)
        pool.index_pools = {
            1: SimpleNamespace(page_size=2, contiguous_page_row_buffers=lambda: [index])
        }
        pool.swa_kv_pool = SimpleNamespace(
            page_size=2, kv_buffer=[torch.zeros((16, 4), dtype=torch.uint8)]
        )
        pending = torch.arange(128, dtype=torch.uint8).view(16, 8)
        pool.compress_state_pools = [
            SimpleNamespace(
                request_scoped=True,
                ratio=2,
                ring_size=2,
                kv_score_buffer=SimpleNamespace(kv_score=pending),
            )
        ]
        pool.indexer_compress_state_pools = []
        mgr = manager(2048)
        ptrs, lengths, widths = pool.get_contiguous_buf_infos()
        mgr.kv_args.kv_data_ptrs, mgr.kv_args.kv_data_lens, mgr.kv_args.kv_item_lens = (
            ptrs,
            lengths,
            widths,
        )
        states = [pool.get_state_buf_infos(), pool.get_request_state_buf_infos()]
        mgr.kv_args.state_types = [StateType.SWA, StateType.DSV4_REQUEST_STATE]
        mgr.kv_args.state_data_ptrs = [v[0] for v in states]
        mgr.kv_args.state_data_lens = [v[1] for v in states]
        mgr.kv_args.state_item_lens = [v[2] for v in states]
        mgr.set_cpu_kv_buffers(pool)
        self.assertEqual([tuple(v.shape) for v in mgr._cpu_views], [(8, 8)] * 3)
        self.assertEqual(tuple(mgr._cpu_state_views[0][0].shape), (8, 8))
        self.assertEqual(tuple(mgr._cpu_state_views[1][0].shape), (8, 16))
        for request_slots in ([], [3], [1, 4]):
            ids = mgr._cpu_state_index(
                [
                    np.array([2, 4], dtype=np.int32),
                    request_slots,  # Wire decoding loses the dtype of [].
                ]
            )
            region = mgr._allocate_cpu_receive(19, np.array([1, 3]), 1, ids)
            self.assertEqual(ids[1].dtype.kind, "i")
            self.assertEqual(region.buffer.numel(), 48 + 16 + len(request_slots) * 16)
            mgr._release_cpu_receive(19)
        self.assertEqual(mgr._cpu_pool._ranges, {})
        for invalid in ([1.0], [[1]], None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                mgr._cpu_state_index([[2], invalid])


if __name__ == "__main__":
    unittest.main()
