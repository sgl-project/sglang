"""Bulk KV transfer: NEXTN slot negotiation, byte parity, fallback and completion."""

import struct
import threading
import time
import unittest
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
import zmq

from sglang.srt.disaggregation.base.conn import KVArgs, StateType
from sglang.srt.disaggregation.common import staging_handler as handler
from sglang.srt.disaggregation.common.staging_buffer import (
    StagingBuffer,
    build_staging_kv_entry_ids,
    compute_staging_layout,
    gather_all_layers_to_staging,
    scatter_staging_to_kv,
)
from sglang.srt.disaggregation.common.staging_handler import (
    DecodeStagingHandler,
    PrefillStagingStrategy,
    StagingRegisterInfo,
    staging_enabled,
)
from sglang.srt.disaggregation.mooncake import conn as mooncake_conn
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.disaggregation.nixl.conn import NixlKVManager
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=35, stage="stage-b", runner_config="1-gpu-large")


class TestBulkKVStaging(CustomTestCase):
    def test_mamba_slot_owner_maps_logical_index_before_tp_slice(self):
        for logical in (1, 2, 76, 77, 303, 304):
            for rank in range(4):
                manager = MooncakeKVManager.__new__(MooncakeKVManager)
                manager.kv_args = KVArgs()
                manager.kv_args.state_types = [StateType.MAMBA]
                manager.kv_args.state_data_ptrs = [[200]]
                manager.kv_args.state_item_lens = [[1024]]
                manager.kv_args.state_dim_per_tensor = [[64]]
                manager.kv_args.state_layer_ids = [[0]]
                manager.kv_args.state_cp_size = 4
                manager.kv_args.state_cp_rank = rank
                manager.attn_tp_size = manager.state_tp_size = 1
                manager._send_mamba_state_slice = Mock(return_value=0)
                manager._send_mamba_state = Mock(return_value=0)
                peer = SimpleNamespace(
                    dst_attn_tp_size=8,
                    dst_tp_rank=3,
                    dst_state_item_lens=[[1024]],
                    dst_state_dim_per_tensor=[[64]],
                    dst_state_data_ptrs=[[300]],
                    dst_state_layer_ids=[[0]],
                )
                self.assertEqual(
                    manager.maybe_send_extra(
                        SimpleNamespace(dst_state_indices=[[1000]]),
                        [[logical]],
                        None,
                        peer,
                    ),
                    0,
                )
                if rank == (logical - 1) % 4:
                    self.assertEqual(
                        manager._send_mamba_state_slice.call_args.args[1],
                        [(logical - 1) // 4 + 1],
                    )
                else:
                    manager._send_mamba_state_slice.assert_not_called()
                manager._send_mamba_state.assert_not_called()

                nixl = NixlKVManager.__new__(NixlKVManager)
                nixl.kv_args = manager.kv_args
                nixl.attn_tp_size = nixl.state_tp_size = 1
                nixl._state_layout_rejections = {}
                nixl._prepared_state_plans = {}
                nixl.agent = Mock()
                nixl._send_mamba_state_slice = Mock(return_value="state-transfer")
                nixl._send_mamba_state = Mock()
                transfer = list(
                    nixl.maybe_send_extra(
                        "decode",
                        [[logical]],
                        [[1000]],
                        [[7]],
                        0,
                        "state-done",
                        8,
                        3,
                        [[1024]],
                        [[64]],
                        [[0]],
                    )
                )
                if rank == (logical - 1) % 4:
                    self.assertEqual(transfer, ["state-transfer"])
                    self.assertEqual(
                        nixl._send_mamba_state_slice.call_args.args[1],
                        [(logical - 1) // 4 + 1],
                    )
                    nixl.agent.send_notif.assert_not_called()
                else:
                    self.assertEqual(transfer, [])
                    nixl._send_mamba_state_slice.assert_not_called()
                    nixl.agent.send_notif.assert_not_called()
                nixl._send_mamba_state.assert_not_called()

    @unittest.skipIf(torch.cuda.device_count() < 2, "requires a nonzero CUDA device")
    def test_gather_binds_fresh_transfer_thread_to_kv_owner(self):
        # A worker starts on device zero even when the scheduler allocated its
        # staging buffer and preloaded the Triton kernel on another device.
        original_device = torch.cuda.current_device()
        page_size, pages, heads, dim = 32, 8, 2, 16
        try:
            for owner in range(1, min(torch.cuda.device_count(), 4)):
                for dtype in (torch.bfloat16, torch.float8_e4m3fn):
                    for use_triton in (False, True):
                        for precreated_stream in (False, True):
                            with self.subTest(
                                owner=owner,
                                dtype=dtype,
                                triton=use_triton,
                                precreated_stream=precreated_stream,
                            ):
                                device = torch.device("cuda", owner)
                                values = torch.arange(
                                    pages * page_size * heads * dim, device=device
                                ).reshape(pages * page_size, heads, dim)
                                buffers = [
                                    ((values * 7 + slot * 13) % 31 - 15).to(dtype)
                                    for slot in range(4)
                                ]
                                size = 3 * page_size * heads * dim * dtype.itemsize * 4
                                staging = StagingBuffer(size, str(device), owner)
                                if precreated_stream:
                                    staging.get_gather_stream()
                                torch.cuda.synchronize(device)

                                def worker():
                                    stream_id = None
                                    for selected in ([5, 1, 7], [7, 5, 1]):
                                        torch.cuda.set_device(0)
                                        self.assertEqual(torch.cuda.current_device(), 0)
                                        written = gather_all_layers_to_staging(
                                            buffers[:2],
                                            buffers[2:],
                                            np.array(selected, dtype=np.int32),
                                            staging,
                                            0,
                                            heads,
                                            page_size,
                                            owner,
                                        )
                                        self.assertEqual(
                                            torch.cuda.current_device(), owner
                                        )
                                        self.assertEqual(written, size)
                                        current_stream_id = (
                                            staging.get_gather_stream().cuda_stream
                                        )
                                        if stream_id is not None:
                                            self.assertEqual(
                                                current_stream_id, stream_id
                                            )
                                        stream_id = current_stream_id
                                        expected = torch.cat(
                                            [
                                                buf[
                                                    page * page_size : (page + 1)
                                                    * page_size
                                                ]
                                                .view(torch.uint8)
                                                .reshape(-1)
                                                for buf in buffers
                                                for page in selected
                                            ]
                                        )
                                        self.assertTrue(
                                            torch.equal(staging.buffer, expected)
                                        )

                                with (
                                    patch(
                                        "sglang.srt.disaggregation.common.staging_buffer._USE_TRITON_STAGING",
                                        use_triton,
                                    ),
                                    ThreadPoolExecutor(max_workers=1) as executor,
                                ):
                                    executor.submit(worker).result(timeout=120)
        finally:
            torch.cuda.set_device(original_device)

    def _round_trip(
        self, *, cp, tp, dtype, direct_writer, dst_tp=1, dst_rank=0, heads=8
    ):
        page_size, pages, layers, dim = 64, 8, 3, 16
        dst_heads = max(1, heads // dst_tp)
        head_start = (dst_rank // max(1, dst_tp // heads)) * dst_heads
        src_pages = np.array([5, 1, 7], dtype=np.int32)
        dst_pages = torch.tensor([2, 6, 0], device="cuda", dtype=torch.int64)
        # Distinct layer/page/token/head values detect permutations, unlike
        # constant-per-rank fills. Values are representable in all tested dtypes.
        logical = [
            (
                (
                    torch.arange(
                        pages * page_size * heads * dim, device="cuda"
                    ).reshape(pages, page_size, heads, dim)
                    * 7
                    + slot * 13
                )
                % 31
                - 15
            ).to(dtype)
            for slot in range(2 * layers)
        ]
        peer_logical = [
            x[:, :, head_start : head_start + dst_heads].contiguous() for x in logical
        ]
        destination = [torch.full_like(x, -24) for x in peer_logical]
        expected = [x.clone() for x in destination]
        src_index = torch.tensor(src_pages, device="cuda", dtype=torch.int64)
        for dst, full in zip(expected, peer_logical):
            # FP8 indexing kernels are not available in every torch build.
            dst.view(torch.uint8)[dst_pages] = full.view(torch.uint8)[src_index]
        writers, sizes, total = compute_staging_layout(
            tp,
            dst_tp,
            dst_rank,
            heads,
            len(src_pages) * page_size,
            dim * logical[0].element_size(),
            layers,
            cp_size=cp,
        )
        received = torch.full((total,), 0xFF, dtype=torch.uint8, device="cuda")
        mask = torch.ones((writers, 2 * layers), dtype=torch.bool, device="cuda")
        for rank in range(writers):
            if cp > 1:
                source = [x[:, rank::cp].contiguous().flatten(0, 1) for x in logical]
            else:
                nh = heads // tp
                source = [
                    x[:, :, rank * nh : (rank + 1) * nh].contiguous().flatten(0, 1)
                    for x in logical
                ]
            if rank == direct_writer:
                mask[rank] = False
                for dst, full in zip(destination, peer_logical):
                    for sp, dp in zip(src_pages, dst_pages.tolist()):
                        if cp > 1:
                            dst[dp, rank::cp].view(torch.uint8).copy_(
                                full[sp, rank::cp].view(torch.uint8)
                            )
                        else:
                            dst[dp, :, rank * nh : (rank + 1) * nh].view(
                                torch.uint8
                            ).copy_(
                                full[sp, :, rank * nh : (rank + 1) * nh].view(
                                    torch.uint8
                                )
                            )
                continue
            staging = StagingBuffer(sizes[rank], "cuda:0", 0)
            manager = MooncakeKVManager.__new__(MooncakeKVManager)
            manager.kv_buffer_tensors = dict(
                k_buffers=source[:layers],
                v_buffers=source[layers:],
                page_size=page_size // cp,
            )
            manager.kv_args = SimpleNamespace(
                engine_rank=rank,
                page_size=page_size,
                gpu_id=0,
                total_kv_head_num=heads,
                num_target_kv_layers=layers,
                prefill_start_layer=0,
            )
            manager.attn_tp_size, manager.attn_cp_size = tp, cp
            manager.attn_cp_rank, manager.pp_size = rank, 1
            manager.ring_cp_sharded = cp > 1
            calls = []

            def copy_rdma(_session, blocks):
                calls.append(blocks)
                for src, dst, length in blocks:
                    self.assertEqual(src, staging.get_ptr())
                    start = dst - received.data_ptr()
                    received[start : start + length].copy_(staging.buffer[:length])
                return 0

            manager._transfer_data = copy_rdma
            self.assertEqual(
                manager.send_kvcache_staged(
                    "peer",
                    src_pages,
                    received.data_ptr(),
                    total,
                    dst_rank,
                    dst_tp,
                    page_size * dst_heads * dim * logical[0].element_size(),
                    staging_buffer=staging,
                    num_dst_target_kv_layers=layers,
                    dst_num_entries=2 * layers,
                ),
                0,
            )
            self.assertEqual(len(calls), 1)
            self.assertEqual(len(calls[0]), 1)
            self.assertEqual(calls[0][0][2], sizes[rank])

        scatter_staging_to_kv(
            received,
            [x.flatten(0, 1) for x in destination[:layers]],
            [x.flatten(0, 1) for x in destination[layers:]],
            dst_pages,
            page_size,
            tp,
            dst_tp,
            dst_rank,
            heads,
            cp_size=cp,
            writer_slot_mask=mask,
        )
        torch.cuda.synchronize()
        for actual, oracle in zip(destination, expected):
            self.assertTrue(
                torch.equal(actual.view(torch.uint8), oracle.view(torch.uint8))
            )

    def test_shuffled_pages_and_partial_oom_are_bit_exact(self):
        for cp, tp in [(2, 1), (4, 1), (1, 4)]:
            for dtype in [torch.float32, torch.bfloat16, torch.float8_e4m3fn]:
                for direct_writer in [-1, 1]:
                    with self.subTest(cp=cp, tp=tp, dtype=dtype, direct=direct_writer):
                        self._round_trip(
                            cp=cp, tp=tp, dtype=dtype, direct_writer=direct_writer
                        )

    def test_cp_to_decode_head_shards_and_gqa_replicas(self):
        # The 397B topology has two KV heads and eight decode ranks. Adjacent
        # groups of four ranks must receive the same head, with every CP row.
        for dst_tp in (2, 8):
            for dst_rank in range(dst_tp):
                for dtype in (torch.bfloat16, torch.float8_e4m3fn):
                    for direct_writer in (-1, 1):
                        with self.subTest(
                            dst_tp=dst_tp,
                            dst_rank=dst_rank,
                            dtype=dtype,
                            direct=direct_writer,
                        ):
                            self._round_trip(
                                cp=2,
                                tp=1,
                                dtype=dtype,
                                direct_writer=direct_writer,
                                dst_tp=dst_tp,
                                dst_rank=dst_rank,
                                heads=2,
                            )


class DescriptorCopyEngine:
    """GPU copy at the transport boundary; this deliberately does not run RDMA."""

    def __init__(self, sources, destinations):
        self.sources = [t.view(torch.uint8).reshape(-1) for t in sources]
        self.destinations = [t.view(torch.uint8).reshape(-1) for t in destinations]
        self.calls = []

    @staticmethod
    def locate(buffers, address, length):
        for index, buf in enumerate(buffers):
            offset = address - buf.data_ptr()
            if 0 <= offset and offset + length <= buf.numel():
                return index, offset
        raise AssertionError(f"Descriptor outside registered buffers: {length} bytes")

    def batch_transfer_sync(self, session, sources, destinations, lengths):
        self.calls.append(list(lengths))
        groups = {}
        for source, destination, length in zip(sources, destinations, lengths):
            si, so = self.locate(self.sources, source, length)
            di, do = self.locate(self.destinations, destination, length)
            groups.setdefault((si, di, length), []).append((so, do))
        for (si, di, width), offsets in groups.items():
            src, dst = self.sources[si], self.destinations[di]
            if len(offsets) == 1:
                so, do = offsets[0]
                dst[do : do + width].copy_(src[so : so + width])
                continue
            # Execute exactly the supplied descriptors, batching equal-width
            # slices so a direct fallback does not launch a kernel per token.
            offsets = torch.tensor(offsets, device=src.device, dtype=torch.int64)
            byte = torch.arange(width, device=src.device)
            src_index = offsets[:, 0, None] + byte
            dst_index = offsets[:, 1, None] + byte
            dst[dst_index] = src[src_index]
        return 0


class TestNextNBulkKVStaging(CustomTestCase):
    """Real gather/scatter and descriptor generation, simulated transport only."""

    def _nextn_round_trip(
        self,
        cp,
        dst_tp,
        dst_rank,
        target_dtype,
        draft_dtype,
        direct,
        *,
        chunk_tokens=None,
    ):
        layers, pages, page, heads, dim = 15, 8, 64, 2, 32
        dst_heads = max(1, heads // dst_tp)
        head_start = (dst_rank // max(1, dst_tp // heads)) * dst_heads
        src_pages = np.array([5, 1, 7], dtype=np.int32)
        dst_pages_np = np.array([2, 6, 0], dtype=np.int32)
        if chunk_tokens is not None:
            page, dim = 128, 256
            self.assertEqual(chunk_tokens % page, 0)
            count = chunk_tokens // page
            pages = count + 4
            rng = np.random.default_rng(397)
            src_pages = (rng.permutation(count) + 2).astype(np.int32)
            dst_pages_np = rng.permutation(count).astype(np.int32)
        dst_pages = torch.tensor(dst_pages_np, device="cuda", dtype=torch.int64)
        values = torch.arange(pages * page * heads * dim, device="cuda").reshape(
            pages, page, heads, dim
        )
        logical = [
            ((values * 7 + i * 13) % 31 - 15).to(target_dtype)
            for i in range(2 * layers)
        ]
        target = [
            torch.full(
                (pages, page, dst_heads, dim), -24, device="cuda", dtype=target_dtype
            )
            for _ in range(2 * layers)
        ]
        draft = [
            torch.full(
                (pages, page, dst_heads, dim), -7, device="cuda", dtype=draft_dtype
            )
            for _ in range(2)
        ]
        destination = target[:layers] + draft[:1] + target[layers:] + draft[1:]
        expected = [x.clone() for x in destination]
        expected_target = expected[:layers] + expected[layers + 1 : 2 * layers + 1]
        src_index = torch.tensor(src_pages, device="cuda", dtype=torch.int64)
        for oracle, full in zip(expected_target, logical):
            peer = full[:, :, head_start : head_start + dst_heads].contiguous()
            oracle.view(torch.uint8)[dst_pages] = peer.view(torch.uint8)[src_index]

        slot_ids = build_staging_kv_entry_ids(layers + 1, layers)
        slot_lens = [page * dst_heads * dim * x.element_size() for x in destination]
        writers, sizes, total = compute_staging_layout(
            1,
            dst_tp,
            dst_rank,
            heads,
            len(src_pages) * page,
            dim * logical[0].element_size(),
            layers + 1,
            cp_size=cp,
            slot_head_bytes=[dim * x.element_size() for x in destination],
        )
        received = torch.full((total,), 0xFF, dtype=torch.uint8, device="cuda")
        peer = SimpleNamespace(
            dst_tp_rank=dst_rank,
            dst_attn_tp_size=dst_tp,
            dst_kv_item_len=slot_lens[0],
            dst_num_target_kv_layers=layers,
            dst_kv_layer_ids=list(range(layers)) * 2 + [layers, layers],
            # Existing registration uses target-K,target-V,draft-K,draft-V.
            dst_kv_ptrs=[x.data_ptr() for x in target + draft],
            staging=StagingRegisterInfo(
                base_ptr=received.data_ptr(),
                total_size=total,
                layout_version=1,
                slot_ids=slot_ids,
                slot_item_lens=slot_lens,
            ),
        )
        arrivals = []
        for rank in range(writers):
            source = [x[:, rank::cp].contiguous().flatten(0, 1) for x in logical]
            staging = (
                None if rank == direct else StagingBuffer(sizes[rank], "cuda:0", 0)
            )
            manager = MooncakeKVManager.__new__(MooncakeKVManager)
            manager.kv_buffer_tensors = dict(
                k_buffers=source[:layers],
                v_buffers=source[layers:],
                page_size=page // cp,
            )
            manager.kv_args = SimpleNamespace(
                engine_rank=rank,
                page_size=page,
                gpu_id=0,
                total_kv_head_num=heads,
                kv_head_num=heads,
                num_target_kv_layers=layers,
                kv_layer_ids=list(range(layers)) * 2,
                prefill_start_layer=0,
                kv_item_lens=[
                    page // cp * heads * dim * x.element_size() for x in source
                ],
                kv_data_ptrs=[x.data_ptr() for x in source],
            )
            manager.attn_tp_size, manager.attn_cp_size = 1, cp
            manager.attn_cp_rank, manager.pp_size = rank, 1
            manager.ring_cp_sharded = True
            manager.enable_transfer_bandwidth_log = False
            manager.engine = DescriptorCopyEngine(
                source + ([staging.buffer] if staging is not None else []),
                [received] + target + draft,
            )
            strategy = PrefillStagingStrategy.__new__(PrefillStagingStrategy)
            strategy.kv_manager, strategy.staging_buffer = manager, staging
            strategy.check_ready = lambda *args: (True, 0, 0, 0, total)

            def ready(*args, target_info, staged):
                writer, slots = manager._staging_writer_metadata(target_info)
                arrivals.append(("peer", rank, staged, writer, slots))

            manager.send_chunk_ready = ready
            chunk = SimpleNamespace(
                prefill_kv_indices=src_pages, index_slice=slice(0, len(src_pages))
            )
            req = SimpleNamespace(mooncake_session_id="gpu-copy-boundary", room=7)
            with patch(
                "sglang.srt.disaggregation.mooncake.conn.get_disagg",
                return_value=SimpleNamespace(disaggregation_require_bulk=direct < 0),
            ):
                self.assertEqual(
                    manager._do_staging_transfer(
                        strategy, chunk, req, peer, dst_pages_np, None, None, rank
                    ),
                    (0, False),
                )
            lengths = [length for call in manager.engine.calls for length in call]
            expected_bytes = (
                len(src_pages)
                * (page // cp)
                * dst_heads
                * dim
                * logical[0].element_size()
                * 2
                * layers
            )
            self.assertEqual(sum(lengths), expected_bytes)
            if rank != direct:
                self.assertEqual(len(manager.engine.calls), 1)
                # Appended NEXTN slots create one hole between target K and V.
                self.assertEqual(len(lengths), 2)
                self.assertEqual(lengths, [expected_bytes // 2] * 2)
            else:
                self.assertEqual(
                    len(lengths), len(src_pages) * (page // cp) * 2 * layers
                )

        decoder = SimpleNamespace(
            decode_tp=dst_tp, kv_buffer_info={"k_buffers": destination[: layers + 1]}
        )
        receiver = SimpleNamespace(
            prefill_info=SimpleNamespace(
                attn_cp_size=cp, ring_cp_sharded=True, attn_tp_size=1
            )
        )
        mask = DecodeStagingHandler._writer_slot_mask(decoder, receiver, arrivals)
        for rank, row in enumerate(mask):
            self.assertFalse(row[layers])
            self.assertFalse(row[2 * layers + 1])
            self.assertEqual(sum(row), 0 if rank == direct else 2 * layers)
        scatter_staging_to_kv(
            received,
            [x.flatten(0, 1) for x in destination[: layers + 1]],
            [x.flatten(0, 1) for x in destination[layers + 1 :]],
            dst_pages,
            page,
            1,
            dst_tp,
            dst_rank,
            heads,
            cp_size=cp,
            writer_slot_mask=torch.tensor(mask, device="cuda", dtype=torch.bool),
        )
        torch.cuda.synchronize()
        for slot, (actual, oracle) in enumerate(zip(destination, expected)):
            self.assertTrue(
                torch.equal(actual.view(torch.uint8), oracle.view(torch.uint8)),
                f"slot {slot} differs; draft poison or target bytes changed",
            )

    def test_nextn_mixed_dtype_gqa_and_partial_oom(self):
        for cp in (2, 4):
            for dst_tp in (1, 2, 8):
                for dst_rank in range(dst_tp):
                    for target, draft in (
                        (torch.bfloat16, torch.bfloat16),
                        (torch.float8_e4m3fn, torch.float8_e4m3fn),
                        (torch.float8_e4m3fn, torch.bfloat16),
                    ):
                        for direct in (-1, 1):
                            with self.subTest(
                                cp=cp,
                                dst_tp=dst_tp,
                                dst_rank=dst_rank,
                                target=target,
                                draft=draft,
                                direct=direct,
                            ):
                                self._nextn_round_trip(
                                    cp, dst_tp, dst_rank, target, draft, direct
                                )

    def test_nextn_16k_cp4_keeps_two_bulk_descriptors(self):
        # TP8+DP8 serving uses attention TP1; also cover replicated GQA at TP8.
        for dst_tp, dst_rank in ((1, 0), (8, 7)):
            with self.subTest(dst_tp=dst_tp):
                self._nextn_round_trip(
                    4,
                    dst_tp,
                    dst_rank,
                    torch.float8_e4m3fn,
                    torch.bfloat16,
                    -1,
                    chunk_tokens=16384,
                )


class TestBulkStagingProtocol(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(disaggregation_require_bulk=False)
        for module in (mooncake_conn, handler):
            context = patch.object(module, "get_disagg", return_value=self.args)
            context.start()
            self.addCleanup(context.stop)

    def _run_worker_failure(
        self, error=None, *, missing_peer_staging=False, state_failure=False
    ):
        class StopWorker(BaseException):
            pass

        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager.enable_staging = True
        manager.enable_trace = False
        manager.customized_info_enabled = False
        manager.enable_deferred_decode_kv_release = False
        manager.state_layout_rejections = {}
        manager.failure_lock = threading.Lock()
        manager.failure_records = {}
        manager.deferred_bootstrap = None
        manager.send_kv_status_message = Mock()
        manager.req_to_decode_prefix_len = {}
        manager._staging_ctx = SimpleNamespace(
            prefetch_requested=set(), prefetched_rooms=set()
        )
        manager.kv_args = SimpleNamespace(kv_data_ptrs=[100], aux_item_lens=[8])
        manager._get_dsa_cache_transfer_skip_flags = Mock(return_value=(False, False))
        manager.poison_deferred_ack_room = Mock()
        manager._poison_unsettled_send = Mock()
        manager._staging_outstanding = defaultdict(int)
        manager.ring_cp_sharded = True
        manager.attn_tp_size = 1
        manager.attn_tp_rank = manager.attn_cp_rank = manager.pp_rank = 0
        manager.attn_cp_size = 2
        manager.pp_size = 1
        manager.bootstrap_port = 12345
        manager.session_lock = threading.Lock()
        manager.failed_sessions = set()
        manager.session_failures = defaultdict(int)
        manager.request_status = {
            12: mooncake_conn.KVPoll.WaitingForInput,
            13: mooncake_conn.KVPoll.WaitingForInput,
        }
        manager.check_status = manager.request_status.__getitem__
        manager.update_status = lambda room, status: manager.request_status.__setitem__(
            room, status
        )
        manager.record_failure = Mock()
        manager.sync_status_to_decode_endpoint = Mock()
        manager.send_aux = Mock(return_value=0)
        manager.maybe_send_extra = Mock(side_effect=[-1, 0])
        manager._try_create_staging_strategy = Mock(return_value=object())
        manager.prefill_created_times = {12: 1.0, 13: 1.0}
        manager.pending_transfer_chunks = {12: [], 13: []}
        manager.transfer_infos = {}
        manager.decode_kv_args_table = {}
        chunks = []
        for room, session in ((12, "failed-peer"), (13, "healthy-peer")):
            req = SimpleNamespace(
                room=room,
                mooncake_session_id=session,
                endpoint="127.0.0.1",
                dst_port=12345,
                is_dummy=False,
                dst_kv_indices=[0, 1],
                dst_device_kv_indices=None,
                required_dst_info_num=1,
            )
            manager.transfer_infos[room] = {session: req}
            manager.decode_kv_args_table[session] = SimpleNamespace(
                staging=None if missing_peer_staging and room == 12 else object(),
                requires_dcp_relayout=False,
                dst_aux_ptrs=[100],
                dst_aux_item_lens=None,
                dst_state_data_ptrs=[200],
            )
            chunks.append(
                SimpleNamespace(
                    room=room,
                    prefill_kv_indices=[2, 0],
                    index_slice=slice(0, 2),
                    is_last_chunk=True,
                    state_indices=[5] if state_failure else None,
                    prefill_aux_index=0,
                    staging_counted=False,
                    wait_event=None,
                    indices_event=None,
                )
            )
        if missing_peer_staging:
            error = RuntimeError(
                "--disaggregation-require-bulk forbids direct transfer"
            )
        manager._do_staging_transfer = Mock(
            side_effect=(
                [error]
                if error is not None
                else [(0, False), (0, False)]
                if state_failure
                else [(-1, False), (0, False)]
            )
        )
        queue = SimpleNamespace(get=Mock(side_effect=[*chunks, StopWorker()]))
        if error is not None:
            # The target worker poisons deferred release when an exception may
            # follow submitted writes; it must not acknowledge or recycle them.
            with self.assertRaisesRegex(RuntimeError, str(error)):
                manager.transfer_worker(queue, None)
            manager.poison_deferred_ack_room.assert_called_once_with(12)
            manager.send_aux.assert_not_called()
            manager.send_kv_status_message.assert_not_called()
            return
        with self.assertRaises(StopWorker):
            manager.transfer_worker(queue, None)
        self.assertEqual(queue.get.call_count, 3)
        self.assertEqual(manager.request_status[12], mooncake_conn.KVPoll.Failed)
        self.assertEqual(manager.request_status[13], mooncake_conn.KVPoll.Success)
        self.assertIn(12, manager.failure_records)
        statuses = [
            call.kwargs["status"]
            for call in manager.send_kv_status_message.call_args_list
        ]
        self.assertEqual(
            statuses, [mooncake_conn.KVPoll.Failed, mooncake_conn.KVPoll.Success]
        )
        manager._poison_unsettled_send.assert_called_once_with(12)
        manager.send_aux.assert_called_once()
        self.assertEqual(manager.send_aux.call_args.args[0].room, 13)
        self.assertFalse(manager.transfer_infos)

    def test_transfer_worker_preserves_failure_and_deferred_release_contract(self):
        for error in (
            torch.cuda.OutOfMemoryError("gather allocation failed"),
            ValueError("incompatible slot layout"),
            None,
        ):
            with self.subTest(error=error):
                self._run_worker_failure(error)

    def test_transfer_worker_reports_strict_peer_without_staging(self):
        self._run_worker_failure(missing_peer_staging=True)

    def test_transfer_worker_does_not_ack_failed_state_transfer(self):
        self._run_worker_failure(state_failure=True)

    def test_late_bootstrap_requests_allocation_before_retry(self):
        # Bootstrap overlap can make metadata arrive after pre-forward prefetch.
        # A transfer retry must request staging, then complete when it arrives.
        events = []
        manager = SimpleNamespace(
            pp_size=1,
            attn_tp_rank=0,
            _staging_writer_metadata=lambda peer: (0, [0, 1]),
            ring_cp_sharded=False,
            _staging_ctx=SimpleNamespace(
                watermark_cv=threading.Condition(),
                prefetch_requested=set(),
                prefetched_rooms=set(),
            ),
            _prefetch_staging_reqs=lambda room: events.append(("allocate", room)),
            send_chunk_ready=lambda *a, **kw: events.append(("ready", kw["staged"])),
        )
        strategy = SimpleNamespace(
            check_ready=Mock(side_effect=[(False, 0, -1, 0, -1), (True, 0, 0, 0, 32)]),
            transfer=lambda *a: 0,
        )
        chunk = SimpleNamespace(index_slice=slice(0, 2), prefill_kv_indices=[2, 0])
        peer = SimpleNamespace(staging=SimpleNamespace(total_size=128, base_ptr=100))
        req = SimpleNamespace(
            room=12, mooncake_session_id="peer", endpoint="127.0.0.1", dst_port=12345
        )
        queue = SimpleNamespace(put=lambda item: events.append(("retry", item)))
        args = (manager, strategy, chunk, req, peer, [0, 1], None, queue)
        self.assertEqual(MooncakeKVManager._do_staging_transfer(*args), (-1, True))
        self.assertEqual(events, [("allocate", 12), ("retry", chunk)])
        self.assertEqual(MooncakeKVManager._do_staging_transfer(*args), (0, False))
        self.assertEqual(events[-1], ("ready", True))

    def test_cp_fan_in_deduplicates_and_excludes_direct_writer(self):
        h = handler.DecodeStagingHandler.__new__(handler.DecodeStagingHandler)
        h._lock = threading.RLock()
        h._staging_lifecycle = threading.Condition(h._lock)
        h._room_tokens = {12: object()}
        h._defer_scatter = False
        h._pending_scatters = deque()
        receiver = SimpleNamespace(
            prefill_info=SimpleNamespace(
                attn_cp_size=4, ring_cp_sharded=True, attn_tp_size=1, pp_size=1
            )
        )
        h._room_to_receiver = {12: receiver}
        h._completed_chunks = {}
        h._writer_counts = {}
        h.decode_tp = 1
        h.kv_manager = SimpleNamespace(pp_size=1)
        h.kv_buffer_info = {"k_buffers": [None, None]}
        submitted = []
        h.submit_chunk_scatter = lambda *args, **kwargs: (
            submitted.append(kwargs["writer_slot_mask"]) or True
        )
        for rank in [2, 2, 0, 3]:
            self.assertFalse(
                h.handle_chunk_arrived(
                    12,
                    0,
                    0,
                    8,
                    str(rank),
                    staged=rank != 0,
                    writer_index=rank,
                    slot_indices=[0, 1, 2, 3],
                )
            )
        self.assertFalse(submitted)
        self.assertTrue(
            h.handle_chunk_arrived(
                12, 0, 0, 8, "1", writer_index=1, slot_indices=[0, 1, 2, 3]
            )
        )
        self.assertEqual(submitted[0], [[False] * 4] + [[True] * 4] * 3)
        self.assertFalse(
            h.handle_chunk_arrived(
                12, 0, 0, 8, "1", writer_index=1, slot_indices=[0, 1, 2, 3]
            )
        )
        self.assertEqual(len(submitted), 1)

    def test_only_capacity_failure_uses_direct_transfer(self):
        # The transport is the mocked boundary; assert completion payload and
        # fallback behavior, not just a call to the mocked transport itself.
        for ret in [0, handler.STAGING_MEMORY_UNAVAILABLE, -1]:
            events = []
            manager = SimpleNamespace(
                pp_size=1,
                attn_tp_rank=0,
                _staging_writer_metadata=lambda peer: (0, [0, 1]),
                ring_cp_sharded=False,
                _staging_ctx=SimpleNamespace(
                    watermark_cv=threading.Condition(),
                    prefetch_requested=set(),
                    prefetched_rooms=set(),
                ),
            )
            manager._send_direct_kvcache = lambda *a: events.append("direct") or 0
            manager.send_chunk_ready = lambda *a, **kw: events.append(
                ("ready", kw["staged"])
            )
            strategy = SimpleNamespace(
                check_ready=lambda *a: (True, 0, 0, 0, 32), transfer=lambda *a: ret
            )
            chunk = SimpleNamespace(index_slice=slice(0, 2), prefill_kv_indices=[2, 0])
            peer = SimpleNamespace(
                staging=SimpleNamespace(total_size=128, base_ptr=100)
            )
            rc, deferred = MooncakeKVManager._do_staging_transfer(
                manager,
                strategy,
                chunk,
                SimpleNamespace(
                    room=12,
                    mooncake_session_id="peer",
                    endpoint="127.0.0.1",
                    dst_port=12345,
                ),
                peer,
                [0, 1],
                None,
                None,
            )
            self.assertFalse(deferred)
            self.assertEqual(rc, -1 if ret == -1 else 0)
            self.assertEqual(
                events,
                {0: [("ready", True)], -2: ["direct", ("ready", False)], -1: []}[ret],
            )

    def test_receiver_oom_needs_no_chunk_allocation(self):
        events = []
        manager = SimpleNamespace(
            pp_size=1,
            attn_tp_rank=0,
            _staging_writer_metadata=lambda peer: (0, [0, 1]),
            ring_cp_sharded=False,
            _send_direct_kvcache=lambda *a: events.append("direct") or 0,
        )
        peer = SimpleNamespace(
            staging=SimpleNamespace(total_size=handler.STAGING_MEMORY_UNAVAILABLE)
        )
        self.assertEqual(
            MooncakeKVManager._do_staging_transfer(
                manager, None, None, None, peer, None, None, None
            ),
            (0, False),
        )
        self.assertEqual(events, ["direct"])

    def _transfer_fixture(self, *, pp_size=1, cp_size=1, tp_rank=0):
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager.pp_size = pp_size
        manager.pp_rank = 0
        manager.attn_tp_size, manager.attn_tp_rank = 2, tp_rank
        manager.attn_cp_size, manager.attn_cp_rank = cp_size, 0
        manager.ring_cp_sharded = cp_size > 1
        manager._connect = Mock(return_value=Mock())
        manager._socket_send_locks = defaultdict(threading.Lock)
        manager.send_kvcache_slice = Mock(return_value=0)
        manager._send_ring_cp_kvcache = Mock(return_value=0)
        manager._staging_writer_metadata = Mock(
            side_effect=AssertionError("Direct completion must not resolve bulk slots")
        )
        manager._prefetch_staging_reqs = Mock()
        manager._staging_ctx = SimpleNamespace(watermark_cv=threading.Condition())
        peer = SimpleNamespace(
            dst_attn_tp_size=1,
            dst_tp_rank=0,
            dst_kv_ptrs=[100, 200],
            dst_kv_layer_ids=[0, 0],
            dst_num_target_kv_layers=1,
            dst_kv_item_len=512,
            # Deliberately lacks slot metadata: direct PP/OOM needs none.
            staging=SimpleNamespace(total_size=128, base_ptr=100),
        )
        req = SimpleNamespace(
            endpoint="127.0.0.1",
            dst_port=12345,
            room=12,
            mooncake_session_id="peer",
        )
        chunk = SimpleNamespace(index_slice=slice(0, 2), prefill_kv_indices=[2, 0])
        strategy = SimpleNamespace(
            check_ready=Mock(return_value=(True, 0, 0, 0, 32)),
            transfer=Mock(return_value=0),
        )
        return manager, peer, req, chunk, strategy

    def test_pp_direct_fallback_keeps_completion_without_scatter_slots(self):
        h = handler.DecodeStagingHandler.__new__(handler.DecodeStagingHandler)
        h._lock = threading.RLock()
        h._staging_lifecycle = threading.Condition(h._lock)
        h._room_tokens = {12: object()}
        h._defer_scatter = False
        h._pending_scatters = deque()
        h.decode_tp = 1
        h.kv_manager = SimpleNamespace(pp_size=1)
        h.kv_buffer_info = {"k_buffers": [None, None]}
        h._room_to_receiver = {
            12: SimpleNamespace(
                prefill_info=SimpleNamespace(
                    attn_tp_size=2, attn_cp_size=1, pp_size=2, ring_cp_sharded=False
                )
            )
        }
        h._writer_counts, h._completed_chunks = {}, {}
        h.submit_chunk_scatter = Mock(return_value=True)
        arrivals = []
        with patch.object(self.args, "disaggregation_require_bulk", False):
            for tp_rank, pp_rank in ((1, 1), (0, 1), (1, 0), (0, 0)):
                manager, peer, req, chunk, strategy = self._transfer_fixture(
                    pp_size=2, tp_rank=tp_rank
                )
                manager.pp_rank = pp_rank
                strategy.check_ready.return_value = (
                    False,
                    0,
                    handler.STAGING_MEMORY_UNAVAILABLE,
                    0,
                    -1,
                )
                self.assertEqual(
                    manager._do_staging_transfer(
                        strategy, chunk, req, peer, [4, 5], None, None
                    ),
                    (0, False),
                )
                strategy.transfer.assert_not_called()
                manager.send_kvcache_slice.assert_called_once()
                manager._staging_writer_metadata.assert_not_called()
                message = manager._connect.return_value.send_multipart.call_args.args[0]
                self.assertEqual(message[0], b"CHUNK_READY")
                self.assertEqual(message[7], b"0")
                self.assertEqual(int(message[8]), tp_rank)
                self.assertEqual(message[9], b"")
                arrivals.append(
                    h.handle_chunk_arrived(
                        int(message[1]),
                        int(message[2]),
                        int(message[3]),
                        int(message[4]),
                        message[6].decode(),
                        staged=False,
                        writer_index=int(message[8]),
                        slot_indices=[],
                    )
                )
        self.assertEqual(arrivals, [False, False, False, True])
        h.submit_chunk_scatter.assert_called_once_with(
            12,
            0,
            0,
            2,
            room_token=h._room_tokens[12],
            writer_slot_mask=[[False] * 4, [False] * 4],
        )

    def test_pp_strict_bulk_uses_negotiated_layer_slots(self):
        manager, peer, req, chunk, strategy = self._transfer_fixture(pp_size=2)
        manager._staging_writer_metadata = Mock(return_value=(0, [1, 3]))
        with patch.object(self.args, "disaggregation_require_bulk", True):
            self.assertEqual(
                manager._do_staging_transfer(
                    strategy, chunk, req, peer, [4, 5], None, None
                ),
                (0, False),
            )
        strategy.transfer.assert_called_once()
        manager.send_kvcache_slice.assert_not_called()
        message = manager._connect.return_value.send_multipart.call_args.args[0]
        self.assertEqual(message[7:], [b"1", b"0", struct.pack("2I", 1, 3)])

    def test_cp_oom_completion_does_not_require_bulk_layout(self):
        for source_oom in (False, True):
            with self.subTest(source_oom=source_oom):
                manager, peer, req, chunk, strategy = self._transfer_fixture(cp_size=4)
                manager.attn_tp_size = 1
                manager.attn_cp_rank = 3
                if source_oom:
                    strategy.transfer.return_value = handler.STAGING_MEMORY_UNAVAILABLE
                else:
                    strategy.check_ready.return_value = (
                        False,
                        0,
                        handler.STAGING_MEMORY_UNAVAILABLE,
                        0,
                        -1,
                    )
                with patch.object(self.args, "disaggregation_require_bulk", False):
                    self.assertEqual(
                        manager._do_staging_transfer(
                            strategy, chunk, req, peer, [4, 5], None, None
                        ),
                        (0, False),
                    )
                manager._send_ring_cp_kvcache.assert_called_once()
                manager._staging_writer_metadata.assert_not_called()
                message = manager._connect.return_value.send_multipart.call_args.args[0]
                self.assertEqual(message[7:], [b"0", b"3", b""])

    def test_pp1_bulk_completion_keeps_exact_staging_slots(self):
        manager, peer, req, chunk, strategy = self._transfer_fixture()
        manager._staging_writer_metadata = Mock(return_value=(1, [0, 2]))
        with patch.object(self.args, "disaggregation_require_bulk", True):
            self.assertEqual(
                manager._do_staging_transfer(
                    strategy, chunk, req, peer, [4, 5], None, None
                ),
                (0, False),
            )
        strategy.transfer.assert_called_once()
        manager.send_kvcache_slice.assert_not_called()
        manager._staging_writer_metadata.assert_called_once_with(peer)
        message = manager._connect.return_value.send_multipart.call_args.args[0]
        self.assertEqual(message[7:], [b"1", b"1", struct.pack("2I", 0, 2)])

    def test_allocation_oom_falls_back_but_registration_error_propagates(self):
        args = SimpleNamespace(page_size=64, kv_item_lens=[512], gpu_id=0)
        with patch.object(handler, "_get_custom_mem_pool", return_value=(None, None)):
            with patch(
                "sglang.srt.disaggregation.common.staging_buffer.StagingBuffer",
                side_effect=torch.OutOfMemoryError("injected GPU OOM"),
            ):
                buffers = handler.init_staging_buffers(
                    lambda *a: None,
                    args,
                    2,
                    64,
                    allow_oom_fallback=True,
                    device_type="cuda",
                )
                self.assertEqual(buffers, [None, None])
            with patch(
                "sglang.srt.disaggregation.common.staging_buffer.StagingBuffer",
                return_value=SimpleNamespace(get_ptr=lambda: 100, get_size=lambda: 512),
            ):

                def failed_registration(*args):
                    raise RuntimeError("RDMA registration failure")

                with self.assertRaisesRegex(RuntimeError, "RDMA registration"):
                    handler.init_staging_buffers(
                        failed_registration,
                        args,
                        1,
                        64,
                        allow_oom_fallback=True,
                        device_type="cuda",
                    )

    def test_cp_uses_logical_page_grid(self):
        manager = SimpleNamespace(
            kv_args=SimpleNamespace(page_size=64), kv_buffer_tensors={"page_size": 16}
        )
        with patch.object(
            handler,
            "get_schedule",
            return_value=SimpleNamespace(chunked_prefill_size=16384),
        ):
            self.assertEqual(
                handler.PrefillStagingStrategy(manager, None).full_chunk_pages, 256
            )

    def test_invalid_geometry_is_not_an_oom_fallback(self):
        # A dtype/stride mismatch would otherwise write bytes into wrong KV rows.
        manager = SimpleNamespace(
            kv_buffer_tensors=dict(
                k_buffers=[torch.empty((64, 8, 16))],
                v_buffers=[torch.empty((64, 8, 16))],
                page_size=16,
            ),
            kv_args=SimpleNamespace(total_kv_head_num=8, page_size=64),
            attn_tp_size=1,
            attn_cp_size=4,
            ring_cp_sharded=True,
        )
        with self.assertRaisesRegex(ValueError, "page/head/dtype"):
            MooncakeKVManager.send_kvcache_staged(
                manager, "peer", [0], 100, 128, 0, 1, 512, None
            )

    def test_success_waits_for_late_arrival_and_scatter_event(self):
        # Transport Success may overtake CHUNK_READY. Neither the allocation
        # nor the destination KV may be reused until scatter's CUDA event fires.
        h = handler.DecodeStagingHandler.__new__(handler.DecodeStagingHandler)
        h._lock = threading.RLock()
        h._staging_lifecycle = threading.Condition(h._lock)
        h._room_tokens = {12: object()}
        h._defer_scatter = False
        h._pending_scatters = deque()
        h._pending_watermarks = {}
        receiver = SimpleNamespace(chunk_staging_infos=[(7, 0, 0, 128, 2)])
        req = SimpleNamespace(
            req=SimpleNamespace(bootstrap_room=12),
            _chunk_events=[],
            _staging_all_success=False,
            _staging_scatter_done=False,
        )
        h._room_to_receiver = {12: receiver}
        h._room_to_decode_req = {12: req}
        h.completion_timeout = 30
        h._free_and_send_watermark = Mock()
        self.assertTrue(h.submit_last_scatter_async(12))
        h.advance_scatter(req)
        self.assertFalse(h.is_done(req))
        h._free_and_send_watermark.assert_not_called()
        event = Mock()
        event.query.return_value = False
        req._chunk_events.append((event, 7))
        receiver.chunk_staging_infos[0] = (-1, -1, 0, -1, 0)
        h.advance_scatter(req)
        self.assertFalse(h.is_done(req))
        h._free_and_send_watermark.assert_not_called()
        event.query.return_value = True
        h.advance_scatter(req)
        self.assertTrue(h.is_done(req))
        h._free_and_send_watermark.assert_called_once_with(7, req)

    def test_watermark_backpressure_does_not_block_decode(self):
        context = zmq.Context()
        push, pull = context.socket(zmq.PUSH), context.socket(zmq.PULL)
        for socket in (push, pull):
            socket.setsockopt(zmq.LINGER, 0)
        push.setsockopt(zmq.SNDHWM, 1)
        push.setsockopt(zmq.SNDTIMEO, 1000)
        pull.setsockopt(zmq.RCVHWM, 1)
        pull.bind("inproc://staging-watermark-backpressure")
        push.connect("inproc://staging-watermark-backpressure")
        receiver = SimpleNamespace(
            bootstrap_infos=[{"rank_ip": "peer", "rank_port": 1}],
            _connect_to_bootstrap_server=lambda _: (push, threading.Lock()),
        )
        try:
            with self.assertRaises(zmq.Again):
                for _ in range(64):
                    push.send_multipart([b"queued"], flags=zmq.NOBLOCK)
            start = time.monotonic()
            self.assertIs(
                DecodeStagingHandler._send_watermark(receiver, "session", (1, 32)),
                False,
            )
            self.assertLess(time.monotonic() - start, 0.5)
        finally:
            push.close()
            pull.close()
            context.term()

    def test_watermark_retries_latest_value_and_retires_last_room(self):
        allocator = Mock()
        allocator.get_watermark.return_value = (1, 32)
        manager = SimpleNamespace(
            _staging_ctx=SimpleNamespace(room_receivers={}, room_bootstrap={})
        )
        h = DecodeStagingHandler(
            manager,
            allocator,
            {"k_buffers": [SimpleNamespace(device=torch.device("cuda:0"))]},
            1,
            1,
            0,
            None,
        )
        lock = threading.Lock()
        socket = Mock()
        socket.send_multipart.side_effect = zmq.Again()
        receiver = SimpleNamespace(
            bootstrap_infos=[{"rank_ip": "peer", "rank_port": 1}],
            _connect_to_bootstrap_server=lambda _: (socket, lock),
        )
        req = SimpleNamespace(_chunk_events=[], _staging_all_success=False)
        h._room_to_receiver = {12: receiver, 13: receiver}
        h._room_to_decode_req = {12: req, 13: req}
        h.release_room = Mock()
        h.register_wm_subscriber(receiver, "session")
        allocator.get_watermark.return_value = (1, 128)
        h._free_and_send_watermark(7, req)
        self.assertEqual(list(h._pending_watermarks.values()), [(1, 128)])

        # Retry on a poll with no newly completed CUDA events.
        socket.send_multipart.side_effect = None
        h.advance_scatter(req)
        socket.send_multipart.assert_called_with(
            [b"WATERMARK", b"1", b"128", b"session"], flags=zmq.NOBLOCK
        )
        self.assertFalse(h._pending_watermarks)
        h.unregister_decode_req(12)
        self.assertEqual(len(h._wm_subscribers), 1)
        h.unregister_decode_req(13)
        self.assertFalse(h._wm_subscribers)
        before = socket.send_multipart.call_count
        h._free_and_send_watermark(8, req)
        self.assertEqual(socket.send_multipart.call_count, before)

        # Rejoining after an idle interval receives the current allocator state.
        allocator.get_watermark.return_value = (2, 64)
        h.register_wm_subscriber(receiver, "new-session")
        socket.send_multipart.assert_called_with(
            [b"WATERMARK", b"2", b"64", b"new-session"], flags=zmq.NOBLOCK
        )

    def test_watermark_busy_socket_does_not_hold_scheduler(self):
        lock = threading.Lock()
        socket = Mock()
        receiver = SimpleNamespace(
            bootstrap_infos=[{"rank_ip": "peer", "rank_port": 1}],
            _connect_to_bootstrap_server=lambda _: (socket, lock),
        )
        with ThreadPoolExecutor(max_workers=1) as pool:
            lock.acquire()
            try:
                future = pool.submit(
                    DecodeStagingHandler._send_watermark, receiver, "session", (0, 0)
                )
                self.assertIs(future.result(timeout=1), False)
                socket.send_multipart.assert_not_called()
            finally:
                lock.release()

    def test_bulk_protocol_requires_explicit_rollout(self):
        with envs.SGLANG_DISAGG_STAGING_BUFFER.override(None):
            self.assertFalse(staging_enabled(is_mla=False))
            self.assertFalse(staging_enabled(is_mla=True))
        with envs.SGLANG_DISAGG_STAGING_BUFFER.override(True):
            self.assertTrue(staging_enabled(is_mla=False))
            with self.assertRaises(ValueError):
                staging_enabled(is_mla=True)
        with envs.SGLANG_DISAGG_STAGING_BUFFER.override(False):
            self.assertFalse(staging_enabled(is_mla=False))


class StopWorker(BaseException):
    pass


class TestStagingTransferCompletion(unittest.TestCase):
    def setUp(self):
        context = patch.object(
            mooncake_conn,
            "get_disagg",
            return_value=SimpleNamespace(disaggregation_require_bulk=True),
        )
        context.start()
        self.addCleanup(context.stop)

    @staticmethod
    def chunk(start, final=False):
        return SimpleNamespace(
            room=12,
            prefill_kv_indices=list(range(start, start + 128)),
            index_slice=slice(start, start + 128),
            is_last_chunk=final,
            state_indices=None,
            prefill_aux_index=0,
            staging_counted=False,
            wait_event=None,
            indices_event=None,
        )

    def manager(self):
        req = SimpleNamespace(
            room=12,
            mooncake_session_id="peer",
            endpoint="127.0.0.1",
            dst_port=12345,
            is_dummy=False,
            dst_kv_indices=list(range(256)),
            dst_device_kv_indices=None,
            required_dst_info_num=1,
            staging=SimpleNamespace(offsets=[-1, -1], rounds=[0, 0], ends=[1000, 2000]),
        )
        manager = SimpleNamespace(
            enable_staging=True,
            enable_trace=False,
            customized_info_enabled=False,
            enable_deferred_decode_kv_release=False,
            defer_decode_allocation=False,
            reports_transfer_latency=False,
            state_layout_rejections={},
            req_to_decode_prefix_len={},
            failure_lock=threading.Lock(),
            failure_records={},
            deferred_bootstrap=None,
            send_kv_status_message=Mock(),
            poison_deferred_ack_room=Mock(),
            kv_args=SimpleNamespace(kv_data_ptrs=[100], aux_item_lens=[8]),
            _get_dsa_cache_transfer_skip_flags=lambda peer: (False, False),
            ring_cp_sharded=True,
            attn_tp_size=1,
            attn_tp_rank=0,
            attn_cp_size=4,
            attn_cp_rank=0,
            pp_rank=0,
            pp_size=1,
            session_lock=threading.Lock(),
            failed_sessions=set(),
            session_failures=defaultdict(int),
            request_status={12: mooncake_conn.KVPoll.WaitingForInput},
            transfer_infos={12: {"peer": req}},
            decode_kv_args_table={
                "peer": SimpleNamespace(
                    staging=SimpleNamespace(base_ptr=10000, total_size=100000),
                    dst_aux_ptrs=[42],
                    dst_aux_item_lens=None,
                    requires_dcp_relayout=False,
                )
            },
            prefill_created_times={12: 1.0},
            pending_transfer_chunks={},
            _staging_outstanding=defaultdict(int),
            _staging_ctx=SimpleNamespace(
                watermark_cv=threading.Condition(),
                prefetch_requested=set(),
                prefetched_rooms=set(),
            ),
            bootstrap_port=1234,
            record_failure=Mock(),
            _is_watermark_ready=Mock(return_value=True),
            send_chunk_ready=Mock(),
            _staging_writer_metadata=lambda peer: (0, [0, 1]),
            send_aux=Mock(return_value=0),
            sync_status_to_decode_endpoint=Mock(),
        )
        manager.check_status = manager.request_status.__getitem__
        manager.update_status = manager.request_status.__setitem__
        for name in (
            "_do_staging_transfer",
            "_prefill_unique_rank",
            "conclude_transfer",
            "validate_aux_buffers",
        ):
            setattr(
                manager, name, MethodType(getattr(MooncakeKVManager, name), manager)
            )
        return manager, req

    def run_deferred(self, deferrals):
        manager, req = self.manager()
        strategy = SimpleNamespace(full_chunk_pages=128, kv_manager=manager)
        strategy.check_ready = MethodType(PrefillStagingStrategy.check_ready, strategy)
        strategy.transfer = Mock(return_value=0)
        manager._try_create_staging_strategy = Mock(return_value=strategy)
        replies = []

        def make_replies_ready(room):
            replies.append(room)
            if len(replies) == deferrals:
                # Both ordered STAGING_RSPs arrive after the pending check.
                # The unconditional tail requeue still lets the last item pass.
                req.staging.offsets[:] = [0, 1000]

        manager._prefetch_staging_reqs = make_replies_ready
        sender = object.__new__(mooncake_conn.MooncakeKVSender)
        sender.conclude_state = None
        sender.kv_mgr = manager
        sender.bootstrap_room = 12
        sender.trace_ctx = Mock()
        snapshots = []
        initial, final = self.chunk(0), self.chunk(128, final=True)
        items = deque([initial, final])

        def get():
            if manager.request_status[12] == mooncake_conn.KVPoll.Success and items:
                snapshots.append(
                    (
                        manager._staging_outstanding[12],
                        mooncake_conn.MooncakeKVSender.poll(sender),
                        12 in manager.transfer_infos,
                        sender.conclude_state,
                    )
                )
            if not items:
                raise StopWorker()
            return items.popleft()

        queue = SimpleNamespace(get=get, put=items.append)
        with self.assertRaises(StopWorker):
            MooncakeKVManager.transfer_worker(
                manager, queue, None, staging_buffer=object()
            )
        self.assertEqual(len(replies), deferrals)
        self.assertEqual(
            snapshots, [(1, mooncake_conn.KVPoll.Transferring, True, None)]
        )
        self.assertEqual(
            [call.args[1] for call in strategy.transfer.call_args_list],
            [list(range(128, 256)), list(range(128))],
        )
        self.assertEqual(
            [
                call.kwargs["chunk_idx"]
                for call in manager.send_chunk_ready.call_args_list
            ],
            [1, 0],
        )
        self.assertEqual(dict(manager._staging_outstanding), {})
        self.assertNotIn(12, manager.transfer_infos)
        self.assertEqual(
            mooncake_conn.MooncakeKVSender.poll(sender), mooncake_conn.KVPoll.Success
        )
        manager.send_aux.assert_called_once()
        manager.send_kv_status_message.assert_called_once()

    def test_last_chunk_overtakes_pending_chunk_without_releasing_source(self):
        self.run_deferred(deferrals=1)

    def test_repeated_deferrals_do_not_double_count_queue_items(self):
        self.run_deferred(deferrals=3)

    def test_failed_or_cleared_room_retires_deferred_count_once(self):
        for room_exists in (True, False):
            with self.subTest(room_exists=room_exists):
                manager, _ = self.manager()
                if room_exists:
                    manager.request_status[12] = mooncake_conn.KVPoll.Failed
                else:
                    manager.request_status.clear()
                deferred = self.chunk(0)
                deferred.staging_counted = True
                manager._staging_outstanding[12] = 1
                queue = SimpleNamespace(
                    get=Mock(
                        side_effect=[deferred, self.chunk(128, True), StopWorker()]
                    )
                )
                with self.assertRaises(StopWorker):
                    MooncakeKVManager.transfer_worker(
                        manager, queue, None, staging_buffer=object()
                    )
                self.assertEqual(dict(manager._staging_outstanding), {})
                manager.send_chunk_ready.assert_not_called()
                manager.send_aux.assert_not_called()

    def test_non_staging_success_without_outstanding_chunks(self):
        manager = SimpleNamespace(
            enable_staging=False,
            check_status=Mock(return_value=4),
            request_status={12: mooncake_conn.KVPoll.Success},
            _staging_outstanding={},
            defer_decode_allocation=False,
            reports_transfer_latency=False,
        )
        sender = object.__new__(mooncake_conn.MooncakeKVSender)
        sender.conclude_state = None
        sender.kv_mgr = manager
        sender.bootstrap_room = 12
        sender.trace_ctx = Mock()
        self.assertEqual(
            mooncake_conn.MooncakeKVSender.poll(sender), mooncake_conn.KVPoll.Success
        )
        self.assertEqual(sender.conclude_state, mooncake_conn.KVPoll.Success)

    def test_missing_bootstrap_attribute_keeps_watermark_unregistered(self):
        staging = DecodeStagingHandler.__new__(DecodeStagingHandler)
        staging._wm_subscribers = {}
        staging._send_watermark = Mock()
        for receiver in (None, SimpleNamespace(), SimpleNamespace(bootstrap_infos=[])):
            staging.register_wm_subscriber(receiver, "session")
        self.assertEqual(staging._wm_subscribers, {})
        staging._send_watermark.assert_not_called()


if __name__ == "__main__":
    unittest.main()
