import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.dcp_pack import (
    dcp_pack_buffer_bytes,
    init_dcp_pack_buffers,
)
from sglang.srt.disaggregation.common.utils import build_dcp_token_transfer_plan
from sglang.srt.disaggregation.nixl.conn import NixlKVManager, NixlKVSender
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.mem_cache.kv.alloc.id_space import IdSpace
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def _plan(*, src, dst, page_size, dcp_size, dcp_rank, **kwargs):
    return build_dcp_token_transfer_plan(
        np.asarray(src, dtype=np.int32),
        np.asarray(dst, dtype=np.int32),
        physical_page_size=page_size,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
        **kwargs,
    )


class TestDcpCollapsedDraftPlan(CustomTestCase):
    def test_collapsed_draft_takes_the_target_rows(self):
        plan = _plan(
            src=[5, 2, 11, 4],
            dst=[7],
            page_size=2,
            dcp_size=4,
            dcp_rank=2,
            num_kv_tokens=8,
            draft_rows_collapsed=True,
        )
        self.assertEqual(plan.target_src_token_indices.tolist(), [4, 8])
        self.assertEqual(plan.target_dst_token_indices.tolist(), [14, 15])
        self.assertEqual(plan.draft_src_token_indices.tolist(), [4, 8])
        self.assertEqual(plan.draft_dst_token_indices.tolist(), [14, 15])


class TestDcpCachedPrefixSend(CustomTestCase):
    def test_cached_prefix_fits_pack_capacity_and_preserves_pages_and_state(self):
        """Bound DCP sends by the allocation without splitting TP sends on the same worker."""
        total, page_size, token_bytes = 2055, 64, 16
        mgr = object.__new__(NixlKVManager)
        mgr.kv_args = SimpleNamespace(
            kv_item_lens=[page_size * token_bytes],
            num_draft_entries=0,
            page_size=page_size,
            gpu_id=0,
            state_types=[StateType.MAMBA],
            dsa_index_k_token_bytes=None,
            state_sub_pools=[None],
        )
        mgr._dcp_pack_buffers = None
        mgr._dcp_pack_max_tokens = None
        mgr.transfer_queues = [None]
        mgr._register_staging_memory = Mock()
        mgr.request_status = {}
        mgr._deferred_ack_targets = {}
        mgr._deferred_ack_poisoned_rooms = set()
        mgr._staging_outstanding = {}
        mgr.is_dummy_cp_rank = False
        mgr.enable_all_cp_ranks_for_transfer = False
        mgr.decode_kv_args_table = {
            peer: SimpleNamespace(requires_dcp_relayout=relayout)
            for peer, relayout in (("dcp", True), ("tp", False))
        }

        def allocate(size, *args, **kwargs):
            return SimpleNamespace(get_ptr=lambda: 0x1000, get_size=lambda: size)

        with (
            get_context().override_server_args(chunked_prefill_size=250, device="cuda"),
            patch(
                "sglang.srt.disaggregation.common.staging_handler._get_custom_mem_pool",
                return_value=(None, None),
            ),
            patch(
                "sglang.srt.disaggregation.common.dcp_pack.StagingBuffer",
                side_effect=allocate,
            ),
        ):
            mgr._init_dcp_pack_buffers_once(dcp_size=4)
        limit = mgr._dcp_pack_buffers[0].get_size() // token_bytes

        for peer, prefix in (("dcp", 0), ("dcp", 256), ("tp", 0), ("tp", 256)):
            with self.subTest(peer=peer, decode_prefix=prefix):
                mgr.transfer_infos = {
                    1: {
                        "dummy": SimpleNamespace(is_dummy=True),
                        peer: SimpleNamespace(is_dummy=False),
                    }
                }
                mgr.add_transfer_request = Mock()
                with get_context().override_server_args(dp_size=1):
                    sender = NixlKVSender(mgr, "unused", 1)
                sender.init((total - prefix + page_size - 1) // page_size, 3)
                req = SimpleNamespace(
                    rid="cached-prefix",
                    kv=SimpleNamespace(req_pool_idx=0),
                    origin_input_ids=[0] * total,
                    extend_end=total,
                    start_send_idx=prefix,
                    disagg_decode_prefix_len=prefix,
                    disagg_kv_sender=sender,
                )
                scheduler = SimpleNamespace(
                    enable_staging=False,
                    token_to_kv_pool_allocator=SimpleNamespace(
                        page_size=page_size,
                        translate_kv_indices_for_transfer=lambda x: x,
                        state_ids=lambda sub_pool: IdSpace(key=(sub_pool, None)),
                        storage=None,
                    ),
                    req_to_token_pool=SimpleNamespace(
                        req_to_token=torch.arange(total).reshape(1, -1),
                        req_index_to_mamba_index_mapping=torch.tensor([17]),
                        state_sub_pool="mamba",
                    ),
                    disagg_metadata_buffers=Mock(),
                    disagg_prefill_bootstrap_queue=SimpleNamespace(kv_manager=mgr),
                )
                SchedulerDisaggregationPrefillMixin._send_kv_chunk(
                    scheduler, req, last_chunk=True
                )
                calls = mgr.add_transfer_request.call_args_list
                token_counts = [c.kwargs["num_kv_tokens"] for c in calls]
                if peer == "dcp":
                    self.assertLessEqual(max(token_counts), limit)
                else:
                    self.assertEqual(len(calls), 1)
                self.assertEqual(sum(token_counts), total - prefix)
                np.testing.assert_array_equal(
                    np.concatenate([c.args[1] for c in calls]),
                    np.arange(prefix // 64, 33),
                )
                page_offset = 0
                for call in calls:
                    pages = len(call.args[1])
                    self.assertEqual(
                        call.args[2], slice(page_offset, page_offset + pages)
                    )
                    page_offset += pages
                self.assertEqual(
                    [c.args[3] for c in calls], [False] * (len(calls) - 1) + [True]
                )
                self.assertTrue(
                    all(c.kwargs["state_indices"] is None for c in calls[:-1])
                )
                self.assertEqual(int(calls[-1].kwargs["state_indices"][0][0]), 17)


class TestHostTensorIndicesEvent(CustomTestCase):
    def test_indices_event_comes_from_the_scheduler_device_module(self):
        """The D2H indices event is created and recorded on the scheduler's device.

        A sender that takes host-tensor indices gets a pinned host copy plus an
        event the transfer worker waits on. Building that event with torch.cuda
        crashes every non-CUDA prefill scheduler on its first KV send.
        """
        page_size, total = 64, 128
        stream = object()

        class _Event:
            def record(self, recorded_on):
                self.recorded_on = recorded_on

        sender = SimpleNamespace(
            supports_host_tensor_indices=True,
            kv_mgr=SimpleNamespace(
                enable_all_cp_ranks_for_transfer=False, kv_shard_size=1
            ),
            get_max_transfer_tokens=lambda: None,
            should_send_kv_chunk=lambda num_pages, is_last: True,
            send=Mock(),
        )
        req = SimpleNamespace(
            rid="host-indices",
            kv=SimpleNamespace(req_pool_idx=0, cache_protected_len=total),
            skip_radix_cache_insert=False,
            origin_input_ids=[0] * total,
            extend_end=total,
            start_send_idx=0,
            disagg_decode_prefix_len=0,
            disagg_kv_sender=sender,
        )
        scheduler = SimpleNamespace(
            enable_staging=False,
            tree_cache=SimpleNamespace(supports_prefix_sharing=lambda: True),
            device_module=SimpleNamespace(Event=_Event, current_stream=lambda: stream),
            token_to_kv_pool_allocator=SimpleNamespace(
                page_size=page_size,
                translate_kv_indices_for_transfer=lambda x: x,
                storage=None,
            ),
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.arange(total).reshape(1, -1)
            ),
        )
        real_empty = torch.empty

        def empty_unpinned(*args, pin_memory=False, **kwargs):
            # Pinned host memory needs an accelerator, which the CPU runner lacks.
            return real_empty(*args, **kwargs)

        with patch("torch.empty", side_effect=empty_unpinned):
            SchedulerDisaggregationPrefillMixin._send_kv_chunk(
                scheduler, req, last_chunk=False
            )

        event = sender.send.call_args.kwargs["indices_event"]
        self.assertIsInstance(event, _Event)
        self.assertIs(event.recorded_on, stream)
        np.testing.assert_array_equal(sender.send.call_args.args[0], [0, 1])

    def test_sharded_sender_gets_completed_indices_before_filtering(self):
        # CP-TP sharing does not set enable_all_cp_ranks_for_transfer. Shard
        # filtering still runs in send(), before a worker could wait for D2H.
        page_size, total = 64, 5
        manager = SimpleNamespace(
            enable_all_cp_ranks_for_transfer=False,
            kv_shard_size=4,
            kv_args=SimpleNamespace(state_types=[], state_sub_pools=[]),
        )
        sender = SimpleNamespace(
            supports_host_tensor_indices=True,
            kv_mgr=manager,
            get_max_transfer_tokens=lambda: None,
            should_send_kv_chunk=lambda num_pages, is_last: True,
            send=Mock(),
        )
        req = SimpleNamespace(
            rid="sharded-host-indices",
            kv=SimpleNamespace(req_pool_idx=0),
            origin_input_ids=[0] * total,
            extend_end=total,
            start_send_idx=0,
            disagg_kv_sender=sender,
        )
        scheduler = SimpleNamespace(
            enable_staging=False,
            token_to_kv_pool_allocator=SimpleNamespace(
                page_size=page_size,
                translate_kv_indices_for_transfer=lambda x: x,
                storage=None,
            ),
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.arange(5 * page_size, 5 * page_size + total).reshape(
                    1, -1
                )
            ),
            disagg_metadata_buffers=Mock(),
            disagg_prefill_bootstrap_queue=SimpleNamespace(kv_manager=manager),
            disagg_prefill_pending_chunk_rids=set(),
        )
        with patch(
            "sglang.srt.disaggregation.prefill.async_d2h",
            side_effect=AssertionError("shard filtering cannot read an async copy"),
        ):
            SchedulerDisaggregationPrefillMixin._send_kv_chunk(
                scheduler, req, last_chunk=True
            )
        self.assertIsInstance(sender.send.call_args.args[0], np.ndarray)
        np.testing.assert_array_equal(sender.send.call_args.args[0], [5])
        self.assertNotIn("indices_event", sender.send.call_args.kwargs)


class TestDcpPackBufferBytes(CustomTestCase):
    def test_sizes_fixed_regions_for_each_dcp_rank(self):
        self.assertEqual(
            dcp_pack_buffer_bytes(
                [64 * 16, 64 * 16],
                page_size=64,
                max_tokens=10,
                dcp_size=4,
            ),
            4 * 3 * (16 + 16),
        )

    def test_rejects_invalid_item_lens(self):
        with self.assertRaisesRegex(ValueError, "at least one page"):
            dcp_pack_buffer_bytes([0], page_size=64, max_tokens=8)
        with self.assertRaisesRegex(ValueError, "page-aligned"):
            dcp_pack_buffer_bytes([100], page_size=64, max_tokens=8)


class TestInitDcpPackBuffers(CustomTestCase):
    """The pack kernel is Triton-only: a non-CUDA device must get no buffers
    (per-token fallback) rather than crash, and an indexed "cuda:N" device must
    still get them, not silently fall back."""

    _KV_ARGS = SimpleNamespace(
        kv_item_lens=[64 * 16],
        page_size=64,
        num_draft_entries=0,
        gpu_id=3,
        dsa_index_k_token_bytes=None,
    )

    def test_non_triton_device_gets_no_buffers(self):
        register_fn = Mock()
        with patch(
            "sglang.srt.disaggregation.common.dcp_pack.StagingBuffer"
        ) as staging_buffer:
            buffers = init_dcp_pack_buffers(
                register_fn, self._KV_ARGS, 2, 4, 64, device_type="xpu"
            )

        self.assertEqual(buffers, [])
        staging_buffer.assert_not_called()
        register_fn.assert_not_called()

    def test_cuda_buffers_use_the_bare_device_type_and_gpu_id(self):
        with (
            patch(
                "sglang.srt.disaggregation.common.dcp_pack.StagingBuffer"
            ) as staging_buffer,
            patch(
                "sglang.srt.disaggregation.common.staging_handler._get_custom_mem_pool",
                return_value=(None, None),
            ),
        ):
            buffers = init_dcp_pack_buffers(
                Mock(), self._KV_ARGS, 2, 4, 64, device_type="cuda:0"
            )

        self.assertEqual(len(buffers), 2)
        self.assertEqual(staging_buffer.call_args.args[1], "cuda:3")


class TestTryDcpPack(CustomTestCase):
    def test_try_pack_uses_requested_region_and_dense_indices(self):
        """A gather must fit its rank region even when the total buffer has space."""
        dim = 4
        kv = torch.arange(16 * dim, dtype=torch.float32).view(16, 1, dim)
        item_len = int(kv[0].nbytes)
        pack = torch.zeros(8 * item_len, dtype=torch.uint8)
        gather_stream = Mock()
        buf = type(
            "Buf",
            (),
            {
                "buffer": pack,
                "fits": lambda self, n: n <= pack.numel(),
                "get_ptr": lambda self: 0x1000,
                "get_size": lambda self: pack.numel(),
                "get_gather_stream": lambda self: gather_stream,
                "producer_stream": lambda self: Mock(),
            },
        )()
        src = np.array([1, 5, 9, 13], dtype=np.int64)
        pack_offset = 4 * item_len
        mgr = object.__new__(NixlKVManager)
        mgr.kv_args = SimpleNamespace(kv_data_ptrs=[kv.data_ptr()], num_draft_entries=0)
        dst = SimpleNamespace(
            dst_dcp_rank=1, dst_dcp_size=2, dcp_token_item_lens=[item_len]
        )
        with (
            patch(
                "sglang.srt.disaggregation.common.dcp_pack.device_stream_context",
                return_value=nullcontext(),
            ),
            patch(
                "sglang.srt.disaggregation.common.dcp_pack.copy_mla_rows_into_pack"
            ) as copy_mock,
        ):
            packed = mgr._pack_dcp_rank_once(buf, dst, src, {})
            dst.dst_dcp_size = 4
            overflow = mgr._pack_dcp_rank_once(buf, dst, src, {})

        self.assertIsNone(overflow)
        gather_stream.synchronize.assert_called_once_with()
        self.assertIsNotNone(packed)
        ptrs, indices = packed
        self.assertEqual(ptrs, [0x1000 + pack_offset])
        np.testing.assert_array_equal(indices, np.arange(4))
        pack_view = copy_mock.call_args.args[2]
        self.assertEqual(pack_view.storage_offset(), pack_offset)
        self.assertEqual(pack_view.numel(), src.size * item_len)


class TestDcpMToNRelayout(CustomTestCase):
    """End-to-end CPU checks for DCP-M -> DCP-N PD transfer planning, rank
    mapping, and hybrid-MLA (Mamba/KDA + draft) state ownership."""

    def test_plan_exact_roundtrip_fan_in_and_fan_out(self):
        from sglang.srt.disaggregation.common.conn import CommonKVManager

        for src_dcp, dst_dcp, page_size, prefix_len, total_len, chunk_pages in (
            # DP4 x DCP2 -> DEP32 (DCP1): fan-in, multi-chunk, unaligned tail
            (2, 1, 4, 0, 29, 2),
            (2, 1, 4, 8, 35, 1),
            # DP4 x DCP2 -> DP4 x DCP8: fan-out, multi-chunk, unaligned tail
            (2, 8, 4, 0, 51, 3),
            (2, 8, 4, 32, 83, 2),
            # DCP4 -> DCP2 and DCP8 -> DCP2 fan-in
            (4, 2, 2, 0, 27, 2),
            (8, 2, 2, 16, 45, 1),
        ):
            with self.subTest(
                src_dcp=src_dcp,
                dst_dcp=dst_dcp,
                page_size=page_size,
                prefix_len=prefix_len,
                total_len=total_len,
            ):
                src_vpage = page_size * src_dcp
                dst_vpage = page_size * dst_dcp
                delta = total_len - prefix_len
                num_src_pages = (delta + src_vpage - 1) // src_vpage
                num_dst_pages = (delta + dst_vpage - 1) // dst_vpage

                # Non-contiguous physical page ids on both sides.
                src_pages = np.array(
                    [10 + 3 * i for i in range(num_src_pages)], dtype=np.int32
                )
                dst_pages = np.array(
                    [5 + 2 * i for i in range(num_dst_pages)], dtype=np.int32
                )
                src_pool_slots = (int(src_pages.max()) + 1) * page_size
                dst_pool_slots = (int(dst_pages.max()) + 1) * page_size

                # Populate each prefill DCP rank's physical KV pool with global pos.
                src_pools = [
                    np.full(src_pool_slots, -1, dtype=np.int64) for _ in range(src_dcp)
                ]
                for off in range(delta):
                    pos = prefix_len + off
                    r_s = pos % src_dcp
                    local = off // src_dcp
                    row = src_pages[local // page_size] * page_size + local % page_size
                    src_pools[r_s][row] = pos

                dst_pools = [
                    np.full(dst_pool_slots, -1, dtype=np.int64) for _ in range(dst_dcp)
                ]
                dst_writes = [
                    np.zeros(dst_pool_slots, dtype=np.int64) for _ in range(dst_dcp)
                ]

                # Also check replicated draft KV (written once per decode rank).
                src_draft_pools = [
                    np.full((int(src_pages.max()) + 1) * src_vpage, -1, dtype=np.int64)
                    for _ in range(src_dcp)
                ]
                for off in range(delta):
                    pos = prefix_len + off
                    row = src_pages[off // src_vpage] * src_vpage + off % src_vpage
                    for r_s in range(src_dcp):
                        src_draft_pools[r_s][row] = pos
                dst_draft_pools = [
                    np.full((int(dst_pages.max()) + 1) * dst_vpage, -1, dtype=np.int64)
                    for _ in range(dst_dcp)
                ]
                dst_draft_writes = [
                    np.zeros((int(dst_pages.max()) + 1) * dst_vpage, dtype=np.int64)
                    for _ in range(dst_dcp)
                ]

                for page_off in range(0, num_src_pages, chunk_pages):
                    seg_pages = src_pages[page_off : page_off + chunk_pages]
                    seg_start = page_off * src_vpage
                    seg_tokens = min(delta - seg_start, len(seg_pages) * src_vpage)
                    for r_s in range(src_dcp):
                        mgr = object.__new__(CommonKVManager)
                        mgr.dcp_size = src_dcp
                        mgr.dcp_rank = r_s
                        for r_d in range(dst_dcp):
                            plan = build_dcp_token_transfer_plan(
                                seg_pages,
                                dst_pages,
                                physical_page_size=page_size,
                                dcp_size=dst_dcp,
                                dcp_rank=r_d,
                                src_page_offset=page_off,
                                decode_prefix_len=prefix_len,
                                num_kv_tokens=seg_tokens,
                                src_dcp_size=src_dcp,
                                src_dcp_rank=r_s,
                                send_replicated_draft=mgr._should_send_replicated_dcp_draft(
                                    dst_dcp, r_d
                                ),
                            )
                            for s_row, d_row in zip(
                                plan.target_src_token_indices,
                                plan.target_dst_token_indices,
                            ):
                                dst_pools[r_d][d_row] = src_pools[r_s][s_row]
                                dst_writes[r_d][d_row] += 1
                            for s_row, d_row in zip(
                                plan.draft_src_token_indices,
                                plan.draft_dst_token_indices,
                            ):
                                dst_draft_pools[r_d][d_row] = src_draft_pools[r_s][
                                    s_row
                                ]
                                dst_draft_writes[r_d][d_row] += 1

                for off in range(delta):
                    pos = prefix_len + off
                    r_d = pos % dst_dcp
                    local = off // dst_dcp
                    row = dst_pages[local // page_size] * page_size + local % page_size
                    self.assertEqual(dst_pools[r_d][row], pos)
                    self.assertEqual(dst_writes[r_d][row], 1)

                    d_draft_row = (
                        dst_pages[off // dst_vpage] * dst_vpage + off % dst_vpage
                    )
                    for r_all in range(dst_dcp):
                        self.assertEqual(dst_draft_pools[r_all][d_draft_row], pos)
                        self.assertEqual(dst_draft_writes[r_all][d_draft_row], 1)

    def test_plan_rejects_non_divisible_dcp_sizes(self):
        with self.assertRaisesRegex(ValueError, "must divide"):
            _plan(
                src=[0],
                dst=[0],
                page_size=4,
                dcp_size=3,
                dcp_rank=0,
                src_dcp_size=2,
                src_dcp_rank=0,
            )

    def test_rank_mapping_and_registration_mla_and_hybrid(self):
        from sglang.srt.disaggregation.common.conn import (
            CommonKVManager,
            PrefillServerInfo,
        )

        def _make_decode(attn_tp, dcp, engine_rank, *, is_mla, is_hybrid):
            mgr = object.__new__(CommonKVManager)
            mgr.attn_tp_size = attn_tp
            mgr.attn_cp_size = 1
            mgr.attn_cp_rank = 0
            mgr.pp_size = 1
            mgr.pp_rank = 0
            mgr.dcp_size = dcp
            mgr.dcp_rank = engine_rank % dcp
            mgr.is_mla_backend = is_mla
            mgr.is_hybrid_mla_backend = is_hybrid
            mgr.enable_all_cp_ranks_for_transfer = False
            mgr.kv_args = SimpleNamespace(engine_rank=engine_rank)
            return mgr

        def _info(src_tp, src_dcp):
            return PrefillServerInfo(
                attn_tp_size=src_tp,
                attn_cp_size=1,
                dp_size=4,
                pp_size=1,
                dcp_size=src_dcp,
                page_size=64,
                kv_cache_dtype="bf16",
                follow_bootstrap_room=True,
            )

        # 1. Pure MLA: DP4 x DCP2 (attn_tp=2, dcp=2) -> DEP32 (attn_tp=1, dcp=1)
        info = _info(2, 2)
        dec = _make_decode(1, 1, 0, is_mla=True, is_hybrid=False)
        dec._resolve_rank_mapping(info)
        self.assertEqual(info.target_tp_ranks, [0, 1])
        self.assertEqual(info.real_tp_ranks, [0, 1])
        self.assertEqual(info.required_prefill_response_num, 2)
        self.assertEqual(info.required_dst_info_num_by_tp, {0: 1, 1: 1})

        # 2. Pure MLA: DP4 x DCP2 -> DP4 x DCP8 (modulo-2 pairing)
        for d in range(8):
            info = _info(2, 2)
            dec = _make_decode(8, 8, d, is_mla=True, is_hybrid=False)
            dec._resolve_rank_mapping(info)
            self.assertEqual(info.target_tp_ranks, [d % 2])
            self.assertEqual(info.real_tp_ranks, [d % 2])
            self.assertEqual(info.required_prefill_response_num, 1)
            self.assertEqual(info.required_dst_info_num_by_tp, {d % 2: 4})

        # 3. Hybrid-MLA (Kimi K3 / K3.1): DP4 x DCP2 -> DP4 x DCP8
        # Odd/even KV parity conflicts with contiguous KDA head blocks on d in {1,3,4,6}.
        expected_real = {
            0: [0],
            1: [0, 1],
            2: [0],
            3: [0, 1],
            4: [0, 1],
            5: [1],
            6: [0, 1],
            7: [1],
        }
        for d in range(8):
            info = _info(2, 2)
            dec = _make_decode(8, 8, d, is_mla=False, is_hybrid=True)
            dec._resolve_rank_mapping(info)
            self.assertEqual(info.target_tp_ranks, expected_real[d])
            self.assertEqual(info.real_tp_ranks, expected_real[d])
            self.assertEqual(info.required_prefill_response_num, len(expected_real[d]))
            for p in expected_real[d]:
                self.assertEqual(info.required_dst_info_num_by_tp[p], 6)

        # 4. Pure MLA with unused prefill ranks: attn_tp=4, dcp=2 -> attn_tp=1, dcp=1
        info = _info(4, 2)
        dec = _make_decode(1, 1, 0, is_mla=True, is_hybrid=False)
        dec._resolve_rank_mapping(info)
        self.assertEqual(info.target_tp_ranks, [0, 1, 2, 3])
        self.assertEqual(info.real_tp_ranks, [0, 1])
        self.assertEqual(info.required_prefill_response_num, 2)

        # 5. requires_dcp_relayout + prepare_dcp_token_item_lens with src_dcp=2
        prefill_mgr = object.__new__(CommonKVManager)
        prefill_mgr.dcp_size = 2
        prefill_mgr.dcp_rank = 0
        prefill_mgr.is_mla_backend = False
        prefill_mgr.is_hybrid_mla_backend = True
        prefill_mgr.kv_args = SimpleNamespace(
            page_size=64,
            num_draft_entries=1,
            draft_rows_dcp_collapsed=False,
            kv_item_lens=[64 * 576, 64 * 2 * 128],
        )
        self.assertTrue(prefill_mgr.requires_dcp_relayout(1, 0))
        self.assertTrue(prefill_mgr.requires_dcp_relayout(8, 0))
        self.assertTrue(prefill_mgr.requires_dcp_relayout(8, 1))
        with self.assertRaisesRegex(RuntimeError, "Unsupported PD DCP topology"):
            prefill_mgr.requires_dcp_relayout(3, 0)
        self.assertEqual(
            prefill_mgr.prepare_dcp_token_item_lens([64 * 576, 64 * 8 * 128], 8),
            [576, 128],
        )
        self.assertEqual(
            prefill_mgr.prepare_dcp_token_item_lens([64 * 576, 64 * 1 * 128], 1),
            [576, 128],
        )

    def test_hybrid_mamba_state_and_draft_slicing_under_dcp_m_to_n(self):
        from sglang.srt.disaggregation.common.conn import (
            CommonKVManager,
            PrefillServerInfo,
        )
        from sglang.srt.disaggregation.mooncake.conn import (
            MooncakeBufferKey,
            MooncakeKVManager,
        )
        from sglang.srt.disaggregation.utils import (
            compute_mamba_state_slice_byte_blocks,
        )

        for src_tp, src_dcp, dst_tp, dst_dcp in (
            (2, 2, 8, 8),
            (2, 2, 1, 1),
            (4, 4, 2, 1),
            (4, 4, 4, 2),
        ):
            with self.subTest(
                src_tp=src_tp, src_dcp=src_dcp, dst_tp=dst_tp, dst_dcp=dst_dcp
            ):
                # 16 heads total, 4 bytes per head, 3 outer conv rows.
                total_heads = 16
                elem_bytes = 4
                outer_count = 3
                src_dim = total_heads // src_tp
                dst_dim = total_heads // dst_tp
                src_item_len = outer_count * src_dim * elem_bytes
                dst_item_len = outer_count * dst_dim * elem_bytes

                full_state = np.arange(
                    outer_count * total_heads * elem_bytes, dtype=np.uint8
                ).reshape(outer_count, total_heads, elem_bytes)
                src_bufs = [
                    full_state[:, p * src_dim : (p + 1) * src_dim, :].reshape(-1).copy()
                    for p in range(src_tp)
                ]
                dst_bufs = [
                    np.zeros(dst_item_len, dtype=np.uint8) for _ in range(dst_tp)
                ]
                dst_writes = [
                    np.zeros(dst_item_len, dtype=np.int32) for _ in range(dst_tp)
                ]

                for d in range(dst_tp):
                    info = PrefillServerInfo(
                        attn_tp_size=src_tp,
                        attn_cp_size=1,
                        dp_size=1,
                        pp_size=1,
                        dcp_size=src_dcp,
                        page_size=64,
                        kv_cache_dtype="bf16",
                        follow_bootstrap_room=True,
                    )
                    dec = object.__new__(CommonKVManager)
                    dec.attn_tp_size = dst_tp
                    dec.attn_cp_size = 1
                    dec.attn_cp_rank = 0
                    dec.pp_size = 1
                    dec.pp_rank = 0
                    dec.dcp_size = dst_dcp
                    dec.dcp_rank = d % dst_dcp
                    dec.is_mla_backend = False
                    dec.is_hybrid_mla_backend = True
                    dec.enable_all_cp_ranks_for_transfer = False
                    dec.kv_args = SimpleNamespace(engine_rank=d)
                    dec._resolve_rank_mapping(info)

                    for p in info.real_tp_ranks:
                        blocks = compute_mamba_state_slice_byte_blocks(
                            src_item_len=src_item_len,
                            dst_item_len=dst_item_len,
                            src_dim=src_dim,
                            dst_dim=dst_dim,
                            outer_count=outer_count,
                            src_attn_tp_size=src_tp,
                            dst_attn_tp_size=dst_tp,
                            dst_tp_rank_in_group=d,
                            local_tp_rank_in_group=p,
                        )
                        for s_off, d_off, nbytes in blocks:
                            dst_bufs[d][d_off : d_off + nbytes] = src_bufs[p][
                                s_off : s_off + nbytes
                            ]
                            dst_writes[d][d_off : d_off + nbytes] += 1

                    expected = full_state[
                        :, d * dst_dim : (d + 1) * dst_dim, :
                    ].reshape(-1)
                    np.testing.assert_array_equal(dst_bufs[d], expected)
                    np.testing.assert_array_equal(
                        dst_writes[d], np.ones(dst_item_len, dtype=np.int32)
                    )

        # Mooncake skip_kv under hybrid-MLA DCP2 -> DCP1 must keep both p0 and p1.
        for p_rank, src_tp, src_dcp, expected_skip in (
            (0, 2, 2, False),
            (1, 2, 2, False),
            (2, 4, 2, True),
            (3, 4, 2, True),
        ):
            mc = object.__new__(MooncakeKVManager)
            mc.attn_cp_size = 1
            mc.attn_cp_rank = 0
            mc.attn_tp_size = src_tp
            mc.dcp_size = src_dcp
            mc.is_hybrid_mla_backend = True
            mc.kv_args = SimpleNamespace(engine_rank=p_rank)
            skip_kv, _ = mc._get_dsa_cache_transfer_skip_flags(
                SimpleNamespace(dst_attn_tp_size=1)
            )
            self.assertEqual(skip_kv, expected_skip)

        # Mooncake _dcp_draft_plan: for d=1 in (attn_tp=2, dcp=2 -> attn_tp=8, dcp=8),
        # head-sliced draft goes from p0 (head owner) only; replicated draft goes from p1 (KV owner) only.
        plan = _plan(
            src=[0],
            dst=[0],
            page_size=2,
            dcp_size=8,
            dcp_rank=1,
            src_dcp_size=2,
            src_dcp_rank=0,
            num_kv_tokens=4,
        )
        dummy_key = MooncakeBufferKey(0, 0)
        for p_rank, expect_sliced, expect_replicated in ((0, 1, 0), (1, 0, 1)):
            mc = object.__new__(MooncakeKVManager)
            mc.attn_tp_size = 2
            mc.dcp_size = 2
            mc.dcp_rank = p_rank
            mc.is_mla_backend = False
            mc.kv_args = SimpleNamespace(
                engine_rank=p_rank,
                page_size=2,
                draft_rows_dcp_collapsed=False,
            )
            copies, slices = mc._dcp_draft_plan(
                plan=plan,
                draft_entries=[0],
                token_bytes=[64],
                dst_kv_item_lens=[2 * 8 * 16],
                dst_dcp_size=8,
                dst_dcp_rank=1,
                dst_tp_rank=1,
                dst_attn_tp_size=8,
                draft_rows_collapsed=False,
                src_keys=[dummy_key],
                dst_keys=[dummy_key],
            )
            self.assertEqual(len(copies), 0)
            self.assertEqual(len(slices), expect_sliced)
            copies_rep, slices_rep = mc._dcp_draft_plan(
                plan=plan,
                draft_entries=[0],
                token_bytes=[64],
                dst_kv_item_lens=[2 * 8 * 64],
                dst_dcp_size=8,
                dst_dcp_rank=1,
                dst_tp_rank=1,
                dst_attn_tp_size=8,
                draft_rows_collapsed=False,
                src_keys=[dummy_key],
                dst_keys=[dummy_key],
            )
            self.assertEqual(len(copies_rep), expect_replicated)
            self.assertEqual(len(slices_rep), 0)

        # NIXL _send_mamba_state_slice returns None (no transfer) for KV-only rank p1 -> d1.
        nixl = object.__new__(NixlKVManager)
        nixl.state_tp_size = 2
        nixl.pp_size = 1
        nixl.kv_args = SimpleNamespace(engine_rank=1, gpu_id=0)
        nixl.agent = Mock()
        handle = nixl._send_mamba_state_slice(
            peer_name="d1",
            prefill_state_indices=[0],
            src_state_data_ptrs=[0x1000],
            src_state_item_lens=[32],
            src_state_dim_per_tensor=[8],
            dst_state_data_ptrs=[0x2000],
            dst_state_indices=[0],
            dst_state_item_lens=[8],
            dst_state_dim_per_tensor=[2],
            dst_gpu_id=1,
            decode_tp_size=8,
            decode_tp_rank=1,
        )
        self.assertIsNone(handle)
        nixl.agent.initialize_xfer.assert_not_called()

        # Equal-attn_tp hybrid-MLA (4/4 -> 4/2): KV-only rank p2 connected to d0
        # must skip equal-TP Mamba state writes on both Mooncake and NIXL.
        nixl_eq = object.__new__(NixlKVManager)
        nixl_eq.state_tp_size = 4
        nixl_eq.attn_tp_size = 4
        nixl_eq.pp_size = 1
        nixl_eq.is_mla_backend = False
        nixl_eq._state_layout_rejections = {}
        nixl_eq._prepared_state_plans = {}
        nixl_eq.kv_args = SimpleNamespace(
            engine_rank=2,
            gpu_id=0,
            state_types=[StateType.MAMBA],
            state_data_ptrs=[[0x1000]],
            state_item_lens=[[32]],
            state_dim_per_tensor=[[8]],
            state_conv_shard_groups=[[]],
            state_slice_outer_counts=[[]],
            state_layer_ids=[[]],
        )
        nixl_eq._send_mamba_state = Mock()
        nixl_handles = list(
            nixl_eq.maybe_send_extra(
                peer_name="d0",
                prefill_state_indices=[[0]],
                dst_state_data_ptrs=[[0x2000]],
                dst_state_indices=[[0]],
                dst_gpu_id=0,
                decode_tp_size=4,
                decode_tp_rank=0,
                dst_state_item_lens=[[32]],
                dst_state_dim_per_tensor=[[8]],
            )
        )
        self.assertEqual(nixl_handles, [])
        nixl_eq._send_mamba_state.assert_not_called()

        # Non-divisible attn_tp sizes fail cleanly in _resolve_dcp_tp_rank_mapping.
        with self.assertRaisesRegex(RuntimeError, "divisible"):
            CommonKVManager._resolve_dcp_tp_rank_mapping(
                src_attn_tp=2,
                src_dcp=2,
                dst_attn_tp=3,
                dst_dcp=1,
                dst_tp_rank=0,
                is_hybrid_mla=False,
            )


if __name__ == "__main__":
    unittest.main()
