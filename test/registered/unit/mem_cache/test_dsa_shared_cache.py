import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import numpy as np
import torch

from sglang.srt.disaggregation.common.conn import CommonKVSender, PrefillServerInfo
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    filter_dsa_shared_pages_for_cp_rank,
    prepare_dsa_shared_state_indices,
)
from sglang.srt.layers.attention.dsa.dsa_indexer import (
    Indexer,
    _get_index_cache_write_owner,
    _prepare_paged_index_page_table,
    _synchronize_shared_cache_writes,
)
from sglang.srt.layers.attention.dsa_backend import (
    _synchronize_pool_main_cache,
    _translate_pool_main_page_table,
)
from sglang.srt.layers.cp import utils as cp_utils
from sglang.srt.layers.cp.utils import (
    get_glm_dsa_shared_effective_num_layers,
    get_glm_dsa_shared_info,
    is_glm_dsa_cache_shared_enabled,
)
from sglang.srt.mem_cache.dsa_cache_shared import (
    SharedDSAPageLayout,
    SharedDSATokenToKVPool,
    _export_dsa_shareable_handles,
    _release_partial_vmm_mapping,
    _synchronize_vmm_stage,
    _validate_same_host_group,
)
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestSharedDSAPageLayout(CustomTestCase):
    def setUp(self):
        self.layout = SharedDSAPageLayout(cp_size=4, page_size=4, pages_per_rank=3)

    def test_logical_pages_map_to_rank_major_segments(self):
        logical_pages = torch.arange(12, dtype=torch.int64)
        expected = torch.tensor(
            [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11], dtype=torch.int64
        )

        self.assertTrue(
            torch.equal(self.layout.translate_pages(logical_pages), expected)
        )

    def test_rejects_cross_host_cp_group(self):
        def gather(hosts, _local_host, group):
            hosts[:] = ["host-a", "host-b"]

        with (
            patch("torch.distributed.get_world_size", return_value=2),
            patch("torch.distributed.all_gather_object", side_effect=gather),
            self.assertRaisesRegex(ValueError, "same host"),
        ):
            _validate_same_host_group("cp-cpu")

    def test_logical_pages_map_to_rank_local_segments(self):
        logical_pages = torch.arange(12, dtype=torch.int64)
        expected = torch.tensor(
            [6, 9, 0, 3, 7, 10, 1, 4, 8, 11, 2, 5], dtype=torch.int64
        )

        self.assertTrue(
            torch.equal(
                self.layout.translate_pages_for_rank(logical_pages, rank=2),
                expected,
            )
        )

    def test_main_slots_and_indexer_pages_have_the_same_owner(self):
        logical_pages = torch.arange(12, dtype=torch.int64)
        logical_slots = logical_pages * self.layout.page_size

        shared_pages = self.layout.translate_pages(logical_pages)
        shared_slots = self.layout.translate_slots(logical_slots)

        self.assertTrue(
            torch.equal(shared_slots // self.layout.page_size, shared_pages)
        )

    def test_main_slots_map_to_rank_local_segments(self):
        logical_slots = torch.arange(12, dtype=torch.int64) * self.layout.page_size
        expected_pages = torch.tensor(
            [6, 9, 0, 3, 7, 10, 1, 4, 8, 11, 2, 5], dtype=torch.int64
        )

        translated = self.layout.translate_slots_for_rank(logical_slots, rank=2)

        self.assertTrue(
            torch.equal(translated // self.layout.page_size, expected_pages)
        )

    def test_padding_value_is_preserved(self):
        pages = torch.tensor([0, -1, 5], dtype=torch.int32)
        slots = torch.tensor([0, -1, 21], dtype=torch.int32)

        self.assertEqual(self.layout.translate_pages(pages).tolist(), [0, -1, 4])
        self.assertEqual(self.layout.translate_slots(slots).tolist(), [0, -1, 17])

    def test_owner_mask_and_local_slots_match(self):
        slots = torch.tensor([0, 4, 8, 12, 16, 20], dtype=torch.int64)

        self.assertEqual(
            self.layout.owned_slot_mask(slots, owner_rank=1).tolist(),
            [False, True, False, False, False, True],
        )
        self.assertEqual(
            self.layout.translate_local_slots(slots).tolist(), [0, 0, 0, 0, 4, 4]
        )

    def test_uneven_tail_stays_inside_owner_segment(self):
        layout = SharedDSAPageLayout(cp_size=4, page_size=64, pages_per_rank=3)
        logical_pages = torch.arange(10, dtype=torch.int64)

        translated = layout.translate_pages(logical_pages)

        self.assertEqual(translated.tolist(), [0, 3, 6, 9, 1, 4, 7, 10, 2, 5])
        self.assertTrue(torch.all(translated < layout.cp_size * layout.pages_per_rank))

    def test_different_segment_strides_keep_the_same_owner(self):
        main_layout = SharedDSAPageLayout(4, 64, pages_per_rank=128)
        index_layout = SharedDSAPageLayout(4, 64, pages_per_rank=256)
        slots = torch.arange(0, 16 * 64, 64)

        for owner_rank in range(4):
            self.assertTrue(
                torch.equal(
                    main_layout.owned_slot_mask(slots, owner_rank=owner_rank),
                    index_layout.owned_slot_mask(slots, owner_rank=owner_rank),
                )
            )

    def test_slab_rank_stride_is_independent_from_layer_capacity(self):
        layout = SharedDSAPageLayout(
            cp_size=4,
            page_size=4,
            pages_per_rank=12,
            local_pages_per_layer=3,
        )
        logical_pages = torch.arange(12, dtype=torch.int64)

        self.assertEqual(
            layout.translate_pages(logical_pages).tolist(),
            [0, 12, 24, 36, 1, 13, 25, 37, 2, 14, 26, 38],
        )


class TestSharedDSATokenToKVPoolHelpers(CustomTestCase):
    def test_shared_main_k_writer_receives_quantized_fp8_bytes(self):
        pool = SimpleNamespace(
            use_dsa=True,
            dtype=torch.float8_e4m3fn,
            dsa_kv_cache_store_fp8=True,
            store_dtype=torch.uint8,
        )
        dst = torch.empty((4, 8), dtype=torch.uint8)
        loc = torch.tensor([0, 1], dtype=torch.int64)
        nope = torch.ones((2, 4), dtype=torch.bfloat16)
        rope = torch.ones((2, 2), dtype=torch.bfloat16)
        nope_fp8 = torch.ones((2, 5), dtype=torch.uint8)
        rope_fp8 = torch.ones((2, 3), dtype=torch.uint8)
        writer = MagicMock()

        with patch(
            "sglang.srt.mem_cache.memory_pool.quantize_k_cache_separate",
            return_value=(nope_fp8, rope_fp8),
        ):
            MLATokenToKVPool._write_mla_kv_buffer(
                pool, dst, loc, nope, rope, write_fn=writer
            )

        writer.assert_called_once_with(dst, loc, nope_fp8, rope_fp8)

    def test_shared_main_k_uses_owner_writer_without_row_selection(self):
        pool = object.__new__(SharedDSATokenToKVPool)
        pool.size = 64
        pool.page_size = 4
        pool.start_layer = 0
        pool.local_kv_buffer = [torch.empty((4, 8), dtype=torch.uint8)]
        pool._write_mla_kv_buffer = MagicMock()
        loc = torch.tensor([0, 4, 8, 12], dtype=torch.int64)
        nope = torch.ones((4, 5), dtype=torch.uint8)
        rope = torch.ones((4, 3), dtype=torch.uint8)

        SharedDSATokenToKVPool.set_mla_kv_buffer(
            pool, SimpleNamespace(layer_id=0), loc, nope, rope
        )

        pool._write_mla_kv_buffer.assert_called_once_with(
            pool.local_kv_buffer[0],
            loc,
            nope,
            rope,
            write_fn=pool._write_owned_mla_kv_buffer,
        )

    def test_vmm_stage_reports_local_rank_failure_with_cause(self):
        local_error = RuntimeError("cuMemCreate(POSIX_FD): CUDA_ERROR_OUT_OF_MEMORY")

        with (
            patch(
                "sglang.srt.mem_cache.dsa_cache_shared.dist.get_world_size",
                return_value=2,
            ),
            patch(
                "sglang.srt.mem_cache.dsa_cache_shared.dist.all_gather_object",
                side_effect=lambda output, value, group: output.__setitem__(0, value),
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "DSA shared VMM allocation failed on rank 0.*OUT_OF_MEMORY",
            ) as raised,
        ):
            _synchronize_vmm_stage("group", 0, "allocation", local_error)

        self.assertIs(raised.exception.__cause__, local_error)

    def test_vmm_stage_reports_remote_rank_failure(self):
        gathered_errors = [None, "cuMemMap(rank=0): CUDA_ERROR_INVALID_VALUE"]

        with (
            patch(
                "sglang.srt.mem_cache.dsa_cache_shared.dist.get_world_size",
                return_value=2,
            ),
            patch(
                "sglang.srt.mem_cache.dsa_cache_shared.dist.all_gather_object",
                side_effect=lambda output, _value, group: output.__setitem__(
                    slice(None), gathered_errors
                ),
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "DSA shared VMM mapping failed on rank 1.*cuMemMap",
            ),
        ):
            _synchronize_vmm_stage("group", 0, "mapping", None)

    def test_partial_vmm_cleanup_uses_recorded_addresses(self):
        drv = MagicMock()
        mapped_addresses = [0x3000, 0x1000]

        _release_partial_vmm_mapping(
            drv,
            base_va=0x1000,
            total_bytes=0x4000,
            mapped_addresses=mapped_addresses,
            segment_bytes=0x1000,
        )

        self.assertEqual(
            drv.cuMemUnmap.call_args_list,
            [call(0x1000, 0x1000), call(0x3000, 0x1000)],
        )
        drv.cuMemAddressFree.assert_called_once_with(0x1000, 0x4000)

    def test_posix_transport_is_reused_after_fabric_fallback(self):
        import sglang.srt.mem_cache.dsa_cache_shared as shared

        shared._shared_vmm_use_fabric = None
        with patch.object(
            shared,
            "export_shareable_handles",
            return_value=([], [7], False),
        ) as export:
            _export_dsa_shareable_handles([1], "group", 0)
            _export_dsa_shareable_handles([2], "group", 0)

        self.assertEqual(
            export.call_args_list,
            [
                call([1], "group", 0, try_fabric=True, log_fallback=True),
                call([2], "group", 0, try_fabric=False, log_fallback=False),
            ],
        )

    def test_shared_allocation_supports_fabric_and_posix_export(self):
        import sglang.srt.mem_cache.dsa_cache_shared as shared

        handle_types = SimpleNamespace(
            CU_MEM_HANDLE_TYPE_FABRIC=0x8,
            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR=0x1,
        )
        drv = SimpleNamespace(CUmemAllocationHandleType=handle_types)

        with patch.object(shared, "_shared_vmm_use_fabric", None):
            self.assertEqual(shared._shareable_allocation_handle_types(drv), 0x9)
        with patch.object(shared, "_shared_vmm_use_fabric", False):
            self.assertEqual(shared._shareable_allocation_handle_types(drv), 0x1)

    def test_index_pages_use_index_segment_stride(self):
        pool = SimpleNamespace(
            index_layout=SharedDSAPageLayout(4, 64, pages_per_rank=256)
        )
        pages = torch.tensor([0, 1, 4, 5, -1], dtype=torch.int32)

        translated = SharedDSATokenToKVPool.translate_index_pages(pool, pages)

        self.assertEqual(translated.tolist(), [0, 256, 1, 257, -1])

    def test_index_slots_use_index_segment_stride(self):
        pool = SimpleNamespace(
            index_layout=SharedDSAPageLayout(4, 64, pages_per_rank=256)
        )
        slots = torch.tensor([0, 64, 256, 320, -1], dtype=torch.int64)

        translated = SharedDSATokenToKVPool.translate_index_slots(pool, slots)

        self.assertEqual(translated.tolist(), [0, 16384, 64, 16448, -1])

    def test_write_fence_is_collective_and_mandatory(self):
        calls = []
        fence = torch.zeros(1, dtype=torch.int32)
        pool = SimpleNamespace(
            shared_write_fence=fence,
            shared_cp_group=SimpleNamespace(
                _all_reduce_in_place=lambda tensor: calls.append(tensor.clone())
            ),
        )

        SharedDSATokenToKVPool.synchronize_shared_writes(pool)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].item(), 1)

    def test_indexer_runs_one_shared_write_fence_before_read(self):
        calls = []
        pool = SimpleNamespace(synchronize_shared_writes=lambda: calls.append("fence"))

        _synchronize_shared_cache_writes(pool)

        self.assertEqual(calls, ["fence"])

    def test_paged_indexer_uses_rank_local_view_and_page_table(self):
        calls = []
        expected = torch.zeros((2, 8), dtype=torch.uint8)
        pool = SimpleNamespace(
            shared_rank=2,
            layer_transfer_counter=None,
            start_layer=0,
            index_layout=SharedDSAPageLayout(4, 64, pages_per_rank=3),
            rank_local_index_k_with_scale_buffer=[expected],
            synchronize_shared_writes=lambda: calls.append("fence"),
        )
        pages = torch.tensor([0, 1, 2, 3, 4, -1], dtype=torch.int32)

        translated = SharedDSATokenToKVPool.prepare_paged_index_page_table(pool, pages)
        actual = SharedDSATokenToKVPool.get_paged_index_k_with_scale_buffer(pool, 0)

        self.assertEqual(translated.tolist(), [6, 9, 0, 3, 7, -1])
        self.assertIs(actual, expected)
        self.assertEqual(calls, [])

    def test_pd_transfer_buffers_use_local_layer_views(self):
        main_local = torch.empty((6, 1, 8), dtype=torch.uint8)
        index_local = torch.empty((3, 8), dtype=torch.uint8)
        pool = SimpleNamespace(
            page_size=4,
            local_kv_buffer=[main_local],
            local_index_k_with_scale_buffer=[index_local],
        )

        main_ptrs, main_lens, main_item_lens = (
            SharedDSATokenToKVPool.get_contiguous_buf_infos(pool)
        )
        state_ptrs, state_lens, state_item_lens = (
            SharedDSATokenToKVPool.get_state_buf_infos(pool)
        )

        self.assertEqual(main_ptrs, [main_local.data_ptr()])
        self.assertEqual(main_lens, [main_local.nbytes])
        self.assertEqual(main_item_lens, [main_local[0].nbytes * 4])
        self.assertEqual(state_ptrs, [index_local.data_ptr()])
        self.assertEqual(state_lens, [index_local.nbytes])
        self.assertEqual(state_item_lens, [index_local[0].nbytes])

    def test_attention_reads_canonical_main_view(self):
        calls = []
        global_view = torch.empty((24, 1, 8), dtype=torch.uint8)
        rank_local_view = torch.empty((24, 1, 8), dtype=torch.uint8)
        pool = SimpleNamespace(
            layer_transfer_counter=None,
            start_layer=0,
            dtype=torch.uint8,
            store_dtype=torch.uint8,
            rank_local_kv_buffer=[rank_local_view],
            kv_buffer=[global_view],
            synchronize_shared_writes=lambda: calls.append("fence"),
        )

        actual = SharedDSATokenToKVPool.get_key_buffer(pool, 0)

        self.assertIs(actual, global_view)
        self.assertEqual(calls, [])

        _synchronize_pool_main_cache(pool)

        self.assertEqual(calls, ["fence"])

    def test_attention_translates_main_slots_to_canonical_layout(self):
        pool = SimpleNamespace(
            main_layout=SharedDSAPageLayout(4, 64, pages_per_rank=3),
        )
        slots = torch.tensor([0, 64, 128, 192, 256, -1], dtype=torch.int64)

        translated = SharedDSATokenToKVPool.translate_main_slots(pool, slots)

        self.assertEqual(translated.tolist(), [0, 192, 384, 576, 64, -1])

    def test_pd_transfer_pages_are_filtered_and_translated_by_owner(self):
        pages = np.arange(8, 16, dtype=np.int32)

        local_pages, positions = filter_dsa_shared_pages_for_cp_rank(
            pages,
            cp_rank=2,
            cp_size=4,
            position_offset=20,
        )

        np.testing.assert_array_equal(local_pages, np.array([2, 3], np.int32))
        np.testing.assert_array_equal(positions, np.array([22, 26], np.int64))


class TestSharedDSAPDTransfer(CustomTestCase):
    def test_prefill_registration_skips_shared_vmm_buffers(self):
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager.disaggregation_mode = DisaggregationMode.PREFILL
        manager.server_args = SimpleNamespace(enable_dsa_shared_kv_cache=True)
        manager.kv_args = SimpleNamespace(
            kv_data_ptrs=[100],
            kv_data_lens=[10],
            aux_data_ptrs=[200],
            aux_data_lens=[20],
            state_data_ptrs=[[300]],
            state_data_lens=[[30]],
        )
        manager.engine = MagicMock()

        manager.register_buffer_to_engine()

        manager.engine.batch_register.assert_called_once_with([200], [20])

    def test_shared_staging_packs_owner_rows_before_transfer(self):
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager._transfer_data = MagicMock(return_value=0)
        staging = SimpleNamespace(
            buffer=torch.empty(8, dtype=torch.uint8),
            get_ptr=lambda: 1000,
            get_size=lambda: 8,
        )
        src_buffers = [
            torch.tensor([[10, 11], [20, 21], [30, 31], [40, 41]], dtype=torch.uint8),
            torch.tensor([[50, 51], [60, 61], [70, 71], [80, 81]], dtype=torch.uint8),
        ]
        packed = []

        def capture_transfer(_session_id, blocks):
            packed.append(staging.buffer[:4].clone())
            return 0

        manager._transfer_data.side_effect = capture_transfer

        status = manager._send_dsa_shared_staged(
            mooncake_session_id="session",
            src_buffers=src_buffers,
            item_lens=[2, 2],
            src_indices=np.array([3, 1], dtype=np.int32),
            dst_ptrs=[2000, 3000],
            dst_indices=np.array([5, 8], dtype=np.int32),
            staging_buffer=staging,
        )

        self.assertEqual(status, 0)
        self.assertEqual(
            manager._transfer_data.call_args_list,
            [
                call("session", [(1000, 2010, 2), (1002, 2016, 2)]),
                call("session", [(1000, 3010, 2), (1002, 3016, 2)]),
            ],
        )
        torch.testing.assert_close(
            packed[0], torch.tensor([40, 41, 20, 21], dtype=torch.uint8)
        )
        torch.testing.assert_close(
            packed[1], torch.tensor([80, 81, 60, 61], dtype=torch.uint8)
        )

    def test_all_cp_owners_transfer_indexer_state(self):
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager.is_hybrid_mla_backend = True
        manager.attn_tp_size = 1
        manager.attn_cp_size = 8
        manager.attn_cp_rank = 3
        manager.kv_args = SimpleNamespace(engine_rank=3)
        manager.server_args = SimpleNamespace(
            enable_dsa_cache_layer_split=False,
            enable_dsa_shared_kv_cache=True,
        )

        self.assertEqual(
            manager._get_dsa_cache_transfer_skip_flags(None),
            (False, False),
        )

    def test_sender_uses_interleaved_local_pages_and_destination_positions(self):
        sender = SimpleNamespace(
            curr_idx=0,
            num_kv_indices=8,
            kv_mgr=SimpleNamespace(
                enable_all_cp_ranks_for_transfer=True,
                attn_cp_rank=2,
                attn_cp_size=4,
                server_args=SimpleNamespace(
                    enable_dsa_cache_layer_split=False,
                    enable_dsa_shared_kv_cache=True,
                ),
            ),
        )

        pages, destination_positions, is_last, should_skip = (
            CommonKVSender._prepare_send_indices(
                sender, np.arange(8, 16, dtype=np.int32)
            )
        )

        np.testing.assert_array_equal(pages, np.array([2, 3], np.int32))
        np.testing.assert_array_equal(destination_positions, np.array([2, 6], np.int64))
        self.assertTrue(is_last)
        self.assertFalse(should_skip)

    def test_indexer_state_uses_the_same_owner_mapping(self):
        src, dst = prepare_dsa_shared_state_indices(
            np.arange(8, 16, dtype=np.int32),
            np.arange(40, 48, dtype=np.int32),
            cp_rank=2,
            cp_size=4,
        )

        np.testing.assert_array_equal(src, np.array([2, 3], np.int32))
        np.testing.assert_array_equal(dst, np.array([42, 46], np.int32))

    def test_prefill_metadata_identifies_shared_cp_pages(self):
        info = PrefillServerInfo(
            attn_tp_size=8,
            attn_cp_size=8,
            dp_size=1,
            pp_size=1,
            page_size=64,
            kv_cache_dtype="fp8_e4m3",
            follow_bootstrap_room=True,
            enable_dsa_shared_kv_cache=True,
        )

        self.assertTrue(info.enable_dsa_shared_kv_cache)


class TestSharedDSAIntegrationHooks(CustomTestCase):
    def test_indexer_keeps_existing_read_buffer_helper(self):
        self.assertTrue(hasattr(Indexer, "_get_index_k_read_buffer"))

    def test_index_write_uses_pool_owner(self):
        pool = SimpleNamespace(get_index_k_write_owner=lambda: (3, 8))

        self.assertEqual(_get_index_cache_write_owner(pool), (3, 8))

    def test_page_tables_delegate_to_pool(self):
        pool = SimpleNamespace(
            prepare_paged_index_page_table=lambda table: table + 15,
            translate_main_slots=lambda table: table + 20,
        )
        table = torch.tensor([[1, -1]], dtype=torch.int32)

        self.assertEqual(
            _prepare_paged_index_page_table(pool, table).tolist(), [[16, 14]]
        )
        self.assertEqual(
            _translate_pool_main_page_table(pool, table).tolist(), [[21, 19]]
        )

    def test_plain_pool_is_unchanged(self):
        pool = SimpleNamespace()
        table = torch.tensor([[3]], dtype=torch.int32)

        self.assertEqual(_get_index_cache_write_owner(pool), (0, 1))
        self.assertIs(_prepare_paged_index_page_table(pool, table), table)
        self.assertIs(_translate_pool_main_page_table(pool, table), table)


class TestSharedDSAPoolSelection(CustomTestCase):
    def setUp(self):
        self.runner = SimpleNamespace(
            is_draft_worker=False,
            use_mla_backend=True,
            server_args=SimpleNamespace(
                enable_dsa_shared_kv_cache=True,
                enable_dsa_cache_layer_split=False,
            ),
            model_config=SimpleNamespace(hf_config=SimpleNamespace()),
        )

    @patch("sglang.srt.configs.model_config.is_deepseek_dsa", return_value=True)
    def test_shared_pool_info_uses_attention_cp_group(self, _):
        with get_parallel().override(attn_cp_size=8, attn_cp_rank=3):
            self.assertTrue(is_glm_dsa_cache_shared_enabled(self.runner))
            self.assertEqual(get_glm_dsa_shared_info(self.runner), (3, 8))

    @patch("sglang.srt.configs.model_config.is_deepseek_dsa", return_value=True)
    def test_shared_capacity_counts_only_local_physical_pages(self, _):
        with get_parallel().override(attn_cp_size=8, attn_cp_rank=3):
            self.assertEqual(
                get_glm_dsa_shared_effective_num_layers(self.runner, 61), 8
            )

    @patch("sglang.srt.configs.model_config.is_deepseek_dsa", return_value=True)
    def test_layer_split_capacity_api_remains_compatible(self, _):
        self.runner.server_args.enable_dsa_shared_kv_cache = False
        self.runner.server_args.enable_dsa_cache_layer_split = True
        with get_parallel().override(attn_cp_size=8, attn_cp_rank=3):
            self.assertTrue(
                hasattr(cp_utils, "get_glm_dsa_layer_split_effective_num_layers")
            )
            self.assertEqual(
                cp_utils.get_glm_dsa_layer_split_effective_num_layers(self.runner, 61),
                9,
            )


class TestOwnerAwareHiCacheAllocation(CustomTestCase):
    def test_host_pool_group_owner_hooks_preserve_default_allocation(self):
        from sglang.srt.mem_cache.hicache_storage import PoolName
        from sglang.srt.mem_cache.memory_pool_host import HostPoolGroup, PoolEntry

        host = MagicMock()
        host.layout = "page_first"
        host.page_size = 4
        host.device = "cpu"
        host.size = 16
        host.can_use_write_back_jit = False
        host.alloc.return_value = torch.arange(8)
        group = HostPoolGroup(
            [
                PoolEntry(
                    name=PoolName.KV,
                    host_pool=host,
                    device_pool=MagicMock(),
                    layer_mapper=lambda layer_id: layer_id,
                    is_primary_index_anchor=True,
                )
            ]
        )
        allocator = SimpleNamespace(alloc=MagicMock(return_value=torch.arange(8)))

        host_indices = group.allocate_host_for_device(torch.arange(8))
        device_indices = group.allocate_device_for_host(allocator, torch.arange(8))

        self.assertTrue(torch.equal(host_indices, torch.arange(8)))
        self.assertTrue(torch.equal(device_indices, torch.arange(8)))
        host.alloc.assert_called_once_with(8)
        allocator.alloc.assert_called_once_with(8)

    def test_controller_uses_host_group_allocation_hooks(self):
        from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
            HybridCacheController,
        )

        host_group = MagicMock()
        host_group.allocate_host_for_device.return_value = torch.arange(8)
        allocator = SimpleNamespace(alloc=MagicMock(return_value=torch.arange(8)))
        host_group.allocate_device_for_host.return_value = torch.arange(8)
        controller = SimpleNamespace(
            mem_pool_host=host_group,
            mem_pool_device_allocator=allocator,
            device="cpu",
            write_queue=[],
            load_queue=[],
            start_writing=MagicMock(),
            _resolve_pool_transfers_allocation=MagicMock(return_value=None),
        )

        HybridCacheController.write(controller, torch.arange(8))
        HybridCacheController.load(controller, torch.arange(8))

        host_group.allocate_host_for_device.assert_called_once()
        host_group.allocate_device_for_host.assert_called_once()
        alloc_args = host_group.allocate_device_for_host.call_args.args
        self.assertIs(alloc_args[0], allocator)
        self.assertTrue(torch.equal(alloc_args[1], torch.arange(8)))
        host_group.alloc.assert_not_called()
        allocator.alloc.assert_not_called()

    def test_paged_allocator_allocates_requested_page_owners_atomically(self):
        from sglang.srt.mem_cache.allocator.paged import (
            PagedTokenToKVPoolAllocator,
        )

        allocator = PagedTokenToKVPoolAllocator.__new__(PagedTokenToKVPoolAllocator)
        allocator.page_size = 4
        allocator.device = "cpu"
        allocator.need_sort = False
        allocator.free_pages = torch.tensor([7, 0, 5, 2, 4, 3, 1, 6])
        allocator.release_pages = torch.empty(0, dtype=torch.int64)
        owners = torch.tensor([2, 1, 0, 3, 1], dtype=torch.int64)

        indices = allocator.alloc_for_page_owners(owners, owner_count=4)

        pages = indices.reshape(-1, 4)[:, 0] // 4
        self.assertTrue(torch.equal(pages % 4, owners))
        self.assertEqual(pages.tolist(), [2, 5, 0, 7, 1])
        self.assertEqual(allocator.free_pages.tolist(), [4, 3, 6])

        before = allocator.free_pages.clone()
        self.assertIsNone(
            allocator.alloc_for_page_owners(torch.tensor([0, 0]), owner_count=4)
        )
        self.assertTrue(torch.equal(allocator.free_pages, before))

    def test_paged_owner_allocation_merges_releases_and_reuses_cp8_pages(self):
        from sglang.srt.mem_cache.allocator.paged import (
            PagedTokenToKVPoolAllocator,
        )

        allocator = PagedTokenToKVPoolAllocator.__new__(PagedTokenToKVPoolAllocator)
        allocator.page_size = 2
        allocator.device = "cpu"
        allocator.need_sort = True
        allocator.debug_mode = False
        allocator.is_not_in_free_group = True
        allocator.free_pages = torch.tensor([8, 15, 18, 21])
        allocator.release_pages = torch.tensor([9, 10, 11, 12, 13, 14])
        owners = torch.tensor([7, 1, 6, 0, 5, 2, 4, 3])

        indices = allocator.alloc_for_page_owners(owners, owner_count=8)

        pages = indices.reshape(-1, 2)[:, 0] // 2
        self.assertTrue(torch.equal(pages % 8, owners))
        self.assertEqual(allocator.release_pages.numel(), 0)
        allocator.free(indices)
        reused = allocator.alloc_for_page_owners(owners.flip(0), owner_count=8)
        reused_pages = reused.reshape(-1, 2)[:, 0] // 2
        self.assertTrue(torch.equal(reused_pages % 8, owners.flip(0)))

        before_free = allocator.free_pages.clone()
        before_release = allocator.release_pages.clone()
        self.assertIsNone(
            allocator.alloc_for_page_owners(torch.tensor([0, 0]), owner_count=8)
        )
        self.assertTrue(torch.equal(allocator.free_pages, before_free))
        self.assertTrue(torch.equal(allocator.release_pages, before_release))


class TestSharedDSAHostCache(CustomTestCase):
    def test_owner_local_host_allocator_preserves_global_owner_ids(self):
        from sglang.srt.mem_cache.dsa_shared_host import (
            _SharedDSAHostPageAllocator,
        )

        allocator = _SharedDSAHostPageAllocator(
            local_page_num=2, page_size=4, cp_size=2
        )
        device_pages = torch.tensor([1, 0, 3])
        device_indices = (device_pages[:, None] * 4 + torch.arange(4)).reshape(-1)

        host_indices = allocator.allocate_for_device(device_indices)

        host_pages = host_indices.reshape(-1, 4)[:, 0] // 4
        self.assertEqual(allocator.global_page_num, 4)
        self.assertEqual(allocator.available_size(), 4)
        self.assertEqual(host_pages.tolist(), [1, 0, 3])
        self.assertTrue(torch.equal(host_pages % 2, device_pages % 2))

        before = allocator.free_pages.clone()
        self.assertIsNone(
            allocator.allocate_for_device(
                (torch.tensor([1, 3])[:, None] * 4 + torch.arange(4)).reshape(-1)
            )
        )
        self.assertTrue(torch.equal(allocator.free_pages, before))
        self.assertEqual(allocator.free(host_indices), 12)
        self.assertEqual(allocator.available_size(), 16)

    def test_local_device_pool_exposes_only_local_physical_capacity(self):
        from sglang.srt.mem_cache.dsa_shared_host import _LocalDSADevicePool

        pool = SimpleNamespace(
            local_kv_buffer=[torch.empty((16, 1, 8))],
            local_index_k_with_scale_buffer=[torch.empty((4, 8))],
            size=128,
        )

        local_pool = _LocalDSADevicePool(pool)

        self.assertEqual(local_pool.size, 16)
        self.assertIs(local_pool.kv_buffer, pool.local_kv_buffer)
        self.assertIs(
            local_pool.index_k_with_scale_buffer,
            pool.local_index_k_with_scale_buffer,
        )

    def test_owner_local_l2_has_no_posix_shared_memory_contract(self):
        from sglang.srt.mem_cache import dsa_shared_host

        source = inspect.getsource(dsa_shared_host)
        self.assertNotIn("SharedDSAHostTensorAllocator", source)
        self.assertNotIn("multiprocessing import shared_memory", source)
        self.assertNotIn("/dev/shm", source)

    def test_shared_host_group_backs_up_only_local_main_and_indexer_pages(self):
        from sglang.srt.mem_cache.dsa_shared_host import SharedDSAHostPoolGroup
        from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
        from sglang.srt.mem_cache.memory_pool_host import PoolEntry

        calls = []

        class FakeHost:
            layout = "page_first"
            page_size = 4
            device = "cpu"
            size = 16
            page_num = 4
            can_use_write_back_jit = False
            size_per_token = 1
            dtype = torch.uint8
            start_layer = 0
            end_layer = 1
            allocator = None

            def get_ksize_per_token(self):
                return 1

            def backup_from_device_all_layer(
                self, device_pool, host_indices, device_indices, io_backend
            ):
                calls.append(
                    (
                        device_pool.kv_buffer,
                        device_pool.index_k_with_scale_buffer,
                        host_indices.clone(),
                        device_indices.clone(),
                    )
                )

        pool = SimpleNamespace(
            local_kv_buffer=["local-main"],
            local_index_k_with_scale_buffer=["local-index"],
        )
        main_host = FakeHost()
        index_host = FakeHost()
        group = SharedDSAHostPoolGroup(
            [
                PoolEntry(
                    name=PoolName.KV,
                    host_pool=main_host,
                    device_pool=pool,
                    layer_mapper=lambda layer_id: layer_id,
                    is_primary_index_anchor=True,
                ),
                PoolEntry(
                    name=PoolName.INDEXER,
                    host_pool=index_host,
                    device_pool=pool,
                    layer_mapper=lambda layer_id: layer_id,
                ),
            ],
            shared_rank=1,
            shared_size=2,
            cpu_sync_group="cp-cpu",
            device_sync_group="cp-device",
            sync_device="cpu",
        )
        group._synchronize = lambda error: (
            calls.append(("sync", error is None)) or (error is None, error)
        )
        indices = torch.arange(16)
        group.backup_from_device_all_layer(
            pool,
            indices,
            indices,
            "kernel",
            pool_transfers=[
                PoolTransfer(
                    name=PoolName.INDEXER,
                    host_indices=indices,
                    device_indices=indices,
                )
            ],
        )

        self.assertEqual(calls[0][0], ["local-main"])
        self.assertEqual(calls[0][1], ["local-index"])
        self.assertEqual(calls[0][2].tolist(), list(range(8)))
        self.assertEqual(calls[0][3].tolist(), list(range(8)))
        self.assertEqual(calls[1][2].tolist(), list(range(8)))
        self.assertEqual(calls[1][3].tolist(), list(range(8)))
        self.assertEqual(calls[2], ("sync", True))

    def test_shared_host_group_empty_owner_participates_in_backup_fence(self):
        from sglang.srt.mem_cache.dsa_shared_host import SharedDSAHostPoolGroup
        from sglang.srt.mem_cache.hicache_storage import PoolName
        from sglang.srt.mem_cache.memory_pool_host import PoolEntry

        calls = []
        host = SimpleNamespace(
            layout="page_first",
            page_size=4,
            device="cpu",
            size=16,
            page_num=4,
            can_use_write_back_jit=False,
            size_per_token=1,
            dtype=torch.uint8,
            start_layer=0,
            end_layer=1,
            allocator=None,
            get_ksize_per_token=lambda: 1,
            backup_from_device_all_layer=lambda *args: calls.append("copy"),
        )
        pool = SimpleNamespace(
            local_kv_buffer=["local-main"],
            local_index_k_with_scale_buffer=["local-index"],
        )
        group = SharedDSAHostPoolGroup(
            [
                PoolEntry(
                    name=PoolName.KV,
                    host_pool=host,
                    device_pool=pool,
                    layer_mapper=lambda layer_id: layer_id,
                    is_primary_index_anchor=True,
                )
            ],
            shared_rank=1,
            shared_size=2,
            cpu_sync_group="cp-cpu",
            device_sync_group="cp-device",
            sync_device="cpu",
        )
        group._synchronize = lambda error: (
            calls.append(("sync", error is None)) or (error is None, error)
        )

        indices = torch.arange(4)
        group.backup_from_device_all_layer(pool, indices, indices, "kernel")

        self.assertEqual(calls, [("sync", True)])

    def test_shared_host_group_loads_only_local_owner_then_fences(self):
        from sglang.srt.mem_cache.dsa_shared_host import SharedDSAHostPoolGroup
        from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
        from sglang.srt.mem_cache.memory_pool_host import PoolEntry

        calls = []

        class FakeHost:
            layout = "page_first"
            page_size = 4
            device = "cpu"
            size = 32
            page_num = 8
            can_use_write_back_jit = False
            size_per_token = 1
            dtype = torch.uint8
            start_layer = 0
            end_layer = 1
            allocator = None

            def get_ksize_per_token(self):
                return 1

            def load_to_device_per_layer(
                self,
                device_pool,
                host_indices,
                device_indices,
                layer_id,
                io_backend,
            ):
                calls.append(
                    (
                        device_pool.kv_buffer,
                        device_pool.index_k_with_scale_buffer,
                        host_indices.clone(),
                        device_indices.clone(),
                    )
                )

        pool = SimpleNamespace(
            shared_rank=1,
            kv_buffer=["global-main"],
            local_kv_buffer=["local-main"],
            index_k_with_scale_buffer=["global-index"],
            local_index_k_with_scale_buffer=["local-index"],
        )
        main_host = FakeHost()
        index_host = FakeHost()
        group = SharedDSAHostPoolGroup(
            [
                PoolEntry(
                    name=PoolName.KV,
                    host_pool=main_host,
                    device_pool=pool,
                    layer_mapper=lambda layer_id: layer_id,
                    is_primary_index_anchor=True,
                ),
                PoolEntry(
                    name=PoolName.INDEXER,
                    host_pool=index_host,
                    device_pool=pool,
                    layer_mapper=lambda layer_id: layer_id,
                ),
            ],
            shared_rank=1,
            shared_size=2,
            cpu_sync_group="cp-cpu",
            device_sync_group="cp-device",
            sync_device="cpu",
        )
        group._synchronize = lambda error: (
            calls.append(("sync", error is None)) or (error is None, error)
        )
        indices = torch.arange(16)
        group.load_to_device_per_layer(
            pool,
            indices,
            indices,
            0,
            "kernel",
            pool_transfers=[
                PoolTransfer(
                    name=PoolName.INDEXER,
                    host_indices=indices,
                    device_indices=indices,
                )
            ],
        )

        self.assertEqual(calls[0][0], ["local-main"])
        self.assertEqual(calls[0][1], ["local-index"])
        self.assertEqual(calls[0][2].tolist(), list(range(8)))
        self.assertEqual(calls[0][3].tolist(), list(range(8)))
        self.assertEqual(calls[1][3].tolist(), list(range(8)))
        self.assertEqual(calls[2], ("sync", True))

    def test_shared_host_group_uses_cpu_agreement_and_device_fence(self):
        from sglang.srt.mem_cache import dsa_shared_host
        from sglang.srt.mem_cache.dsa_shared_host import SharedDSAHostPoolGroup

        group = SharedDSAHostPoolGroup.__new__(SharedDSAHostPoolGroup)
        group.cpu_sync_group = "cp-cpu"
        group.device_sync_group = "cp-device"
        group._sync_flag = torch.ones(1, dtype=torch.int32)
        with (
            patch.object(dsa_shared_host, "all_ranks_ok", return_value=True) as all_ok,
            patch.object(dsa_shared_host.dist, "all_reduce") as all_reduce,
        ):
            success, error = group._synchronize(None)

        all_ok.assert_called_once_with("cp-cpu", True)
        all_reduce.assert_called_once_with(group._sync_flag, group="cp-device")
        self.assertTrue(success)
        self.assertIsNone(error)

    def test_shared_host_group_reports_transfer_failure(self):
        from sglang.srt.mem_cache.dsa_shared_host import SharedDSAHostPoolGroup

        group = SharedDSAHostPoolGroup.__new__(SharedDSAHostPoolGroup)
        group.cpu_sync_group = "cp-cpu"
        group.device_sync_group = "cp-device"
        group._sync_flag = torch.ones(1, dtype=torch.int32)
        transfer_error = RuntimeError("copy failed")
        with (
            patch(
                "sglang.srt.mem_cache.dsa_shared_host.all_ranks_ok",
                return_value=False,
            ) as all_ok,
            patch("sglang.srt.mem_cache.dsa_shared_host.dist.all_reduce") as all_reduce,
        ):
            success, error = group._synchronize(transfer_error)

        all_ok.assert_called_once_with("cp-cpu", False)
        all_reduce.assert_not_called()
        self.assertFalse(success)
        self.assertIs(error, transfer_error)


class TestSharedDSAHiCacheDispatch(CustomTestCase):
    def test_unified_dsa_strategy_selects_shared_stack(self):
        from sglang.srt.mem_cache.hicache_storage import PoolName
        from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler
        from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
            _DsaStrategy,
        )

        host_pool = object()
        host_group = MagicMock()
        host_group.get_pool.return_value = host_pool
        controller = object()
        cache = SimpleNamespace(page_size=64)
        kvcache = SimpleNamespace(layer_num=2, kv_cache_dim=656)
        params = SimpleNamespace(
            token_to_kv_pool_allocator=object(),
            tp_cache_group="tp",
            pp_cache_group="pp",
        )
        server_args = SimpleNamespace(enable_dsa_shared_kv_cache=True)

        with patch.object(
            hybrid_pool_assembler,
            "build_shared_dsa_hicache_stack",
            return_value=(host_group, controller),
            create=True,
        ) as build:
            result = _DsaStrategy().build(
                cache=cache,
                kvcache=kvcache,
                params=params,
                server_args=server_args,
                load_cache_event=object(),
                attn_cp_group="cp",
            )

        build.assert_called_once()
        self.assertIs(result.host_pool_group, host_group)
        self.assertIs(result.cache_controller, controller)
        self.assertIs(result.component_host_pools.popitem()[1], host_pool)
        self.assertEqual(result.sidecars[0].pool_name, PoolName.INDEXER)

    def test_hiradix_dsa_attach_selects_shared_stack(self):
        from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler

        host_pool = object()
        host_group = MagicMock()
        host_group.get_pool.return_value = host_pool
        controller = object()
        kv = SimpleNamespace(layer_num=2, kv_cache_dim=656)
        radix_cache = SimpleNamespace(
            kv_cache=kv,
            page_size=64,
            tp_group="tp",
            pp_group="pp",
        )
        params = SimpleNamespace(token_to_kv_pool_allocator=object())
        server_args = SimpleNamespace(
            enable_dsa_shared_kv_cache=True,
            hicache_storage_backend=None,
            served_model_name="model",
        )

        with patch.object(
            hybrid_pool_assembler,
            "build_shared_dsa_hicache_stack",
            return_value=(host_group, controller),
            create=True,
        ) as build:
            hybrid_pool_assembler.attach_hybrid_dsa_pool_to_hiradix_cache(
                radix_cache,
                params,
                server_args,
                extra_config={},
                prefetch_threshold=256,
                enable_storage_metrics=False,
                load_cache_event=object(),
                attn_cp_group="cp",
            )

        build.assert_called_once()
        self.assertIs(radix_cache.full_kv_pool_host, host_pool)
        self.assertIs(radix_cache.token_to_kv_pool_host, host_group)
        self.assertIs(radix_cache.cache_controller, controller)


if __name__ == "__main__":
    unittest.main()
