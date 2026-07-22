import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.jit_kernel.dsv4 import compress as dsv4_compress
from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
    dequantize_k_cache_paged,
)
from sglang.srt.arg_groups.deepseek_v4_hook import get_dsv4_shared_info
from sglang.srt.layers.attention import deepseek_v4_backend as dsv4_backend
from sglang.srt.layers.attention.deepseek_v4_backend import (
    _stage_dsv4_shared_extra_slots,
    _synchronize_dsv4_shared_writes,
    _translate_dsv4_shared_slots,
)
from sglang.srt.layers.attention.dsv4.shared_cache_access import (
    DSV4SharedCacheAccess,
)
from sglang.srt.mem_cache import deepseek_v4_shared as dsv4_shared
from sglang.srt.mem_cache.deepseek_v4_shared import (
    SharedCompressStatePool,
    SharedDeepSeekV4Family,
    SharedDeepSeekV4IndexerPool,
    SharedDeepSeekV4SingleKVPool,
    SharedDeepSeekV4TokenToKVPool,
    _build_shared_page_stage_plan,
    _build_shared_slot_stage_plan,
    _build_single_request_page_stage_plan,
    _build_single_request_slot_stage_plan,
    _stage_shared_cache_from_plan,
    _translate_shared_slots_fused,
    build_dsv4_shared_page_layout,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _attach_shared_access(pool):
    pool.shared_cache_access = DSV4SharedCacheAccess(pool)
    return pool


class TestDeepSeekV4SharedPageLayout(CustomTestCase):
    def test_dsv4_family_descriptor_separates_ownership_from_storage_rows(self):
        captured_specs = []

        def create_storage(*, spec, cp_size, **_kwargs):
            captured_specs.append(spec)
            layout = build_dsv4_shared_page_layout(
                logical_size=1023,
                page_size=spec.ownership_granule,
                cp_size=cp_size,
            ).owner_layout
            return SimpleNamespace(layout=layout, slab=SimpleNamespace())

        with patch.object(
            dsv4_shared.OwnerShardedFamily,
            "create",
            side_effect=create_storage,
        ):
            SharedDeepSeekV4Family.create(
                name="swa",
                logical_size=1023,
                page_size=256,
                layer_num=43,
                dtype=torch.uint8,
                row_shape=(584,),
                rows_per_page=1,
                cp_size=8,
                cpu_group="group",
            )
            SharedDeepSeekV4Family.create(
                name="c4_attn_state",
                logical_size=1023,
                page_size=4,
                layer_num=21,
                dtype=torch.bfloat16,
                row_shape=(1024,),
                rows_per_page=4,
                cp_size=8,
                cpu_group="group",
                zero_initialize=False,
            )

        packed_kv, c4_state = captured_specs
        self.assertEqual(
            (packed_kv.ownership_granule, packed_kv.storage_rows_per_granule),
            (256, 1),
        )
        self.assertEqual(
            (c4_state.ownership_granule, c4_state.storage_rows_per_granule),
            (4, 4),
        )

    def test_shared_compress_state_uses_rank_major_alias_layout(self):
        layout = build_dsv4_shared_page_layout(
            logical_size=31,
            page_size=4,
            cp_size=2,
        )
        family = SimpleNamespace(
            layout=layout,
            rank_local_views=[torch.empty((64, 16), dtype=torch.float32)],
        )
        pool = SharedCompressStatePool(
            size=23,
            ring_size=8,
            overlap=True,
            head_dim=4,
            dtype=torch.float32,
            device="cpu",
            enable_memory_saver=False,
            ratio=4,
            shared_family=family,
            shared_layer_id=0,
            shared_rank=1,
        )

        self.assertEqual(
            pool.get_shared_state_layout(),
            (1, 2, layout.pages_per_rank),
        )
        logical = torch.tensor([0, 4, 8, 12], dtype=torch.int64)
        self.assertTrue(
            torch.equal(
                pool.translate_state_slots_for_read(logical),
                layout.translate_slots_for_rank(logical, rank=1),
            )
        )

    def test_offline_compressor_forwards_shared_state_layout_to_jit(self):
        module = SimpleNamespace(prefill=Mock(), decode=Mock())
        plan = dsv4_compress.CompressorPrefillPlan(
            4,
            torch.empty((0, 16), dtype=torch.uint8),
            torch.empty((0, 8), dtype=torch.uint8),
        )
        buffer = torch.empty((8, 4, 16), dtype=torch.float32)
        kv_input = torch.empty((0, 16), dtype=torch.float32)
        ape = torch.empty((8, 4), dtype=torch.float32)

        with patch.object(
            dsv4_compress,
            "_jit_compress_module",
            return_value=module,
        ):
            dsv4_compress.compress_forward(
                buffer,
                kv_input,
                ape,
                plan,
                head_dim=4,
                compress_ratio=4,
                shared_state_layout=(3, 8, 101),
            )

        module.prefill.assert_called_once()
        self.assertEqual(module.prefill.call_args.args[-3:], (3, 8, 101))

    def test_single_request_plan_uses_request_batch_not_query_rows(self):
        metadata = SimpleNamespace(shared_single_request=False)
        forward_batch = SimpleNamespace(batch_size=1)

        self.assertTrue(
            dsv4_backend._use_single_request_shared_plan(metadata, forward_batch)
        )

    def test_single_request_page_plan_uses_one_repeated_row(self):
        layout = build_dsv4_shared_page_layout(
            logical_size=64,
            page_size=4,
            cp_size=2,
        )
        pages = torch.tensor(
            [[0, 3, 7, 8], [0, 3, 7, 8], [0, 3, 7, 8]],
            dtype=torch.int32,
        )

        physical, remapped = _build_single_request_page_stage_plan(layout, pages)

        self.assertTrue(torch.equal(physical, layout.translate_pages(pages[0]).long()))
        self.assertTrue(
            torch.equal(
                remapped,
                torch.tensor(
                    [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]],
                    dtype=torch.int32,
                ),
            )
        )

    def test_single_request_slot_plan_avoids_dynamic_unique(self):
        layout = build_dsv4_shared_page_layout(
            logical_size=64,
            page_size=4,
            cp_size=2,
        )
        slots = torch.tensor(
            [
                [21, 20, 19, 18],
                [23, 22, 21, 20],
                [25, 24, 23, 22],
            ],
            dtype=torch.int32,
        )

        with (
            patch(
                "torch.unique",
                side_effect=AssertionError(
                    "single-request SWA plan must stay fixed-size"
                ),
            ),
            patch(
                "torch.full",
                side_effect=AssertionError(
                    "single-request SWA plan must not extract a CUDA fill scalar"
                ),
            ),
        ):
            physical, remapped = _build_single_request_slot_stage_plan(layout, slots)

        logical_pages = torch.tensor([4, 5, 6, 6], dtype=torch.int64)
        self.assertTrue(
            torch.equal(physical, layout.translate_pages(logical_pages).long())
        )
        self.assertTrue(
            torch.equal(
                remapped,
                torch.tensor(
                    [
                        [5, 4, 3, 2],
                        [7, 6, 5, 4],
                        [13, 12, 7, 6],
                    ],
                    dtype=torch.int32,
                ),
            )
        )

    def test_single_request_slot_plan_preserves_fragmented_allocator_pages(self):
        layout = build_dsv4_shared_page_layout(
            logical_size=128,
            page_size=4,
            cp_size=2,
        )
        # One logical request can span allocator pages that are not contiguous
        # after radix-cache reuse.  The first row/column still identify every
        # page, but filling the numeric min/max interval would invent pages.
        slots = torch.tensor(
            [
                [5, 4, 43, 42],
                [7, 6, 5, 4],
                [41, 40, 7, 6],
            ],
            dtype=torch.int32,
        )

        physical, remapped = _build_single_request_slot_stage_plan(layout, slots)

        compact_pages = torch.div(remapped, layout.page_size, rounding_mode="floor")
        staged_physical_pages = physical.index_select(0, compact_pages.flatten())
        expected_physical_pages = layout.translate_pages(
            torch.div(slots, layout.page_size, rounding_mode="floor")
        ).flatten()
        self.assertTrue(torch.equal(staged_physical_pages, expected_physical_pages))
        self.assertTrue(
            torch.equal(
                torch.remainder(remapped, layout.page_size),
                torch.remainder(slots, layout.page_size),
            )
        )

    def test_single_request_slot_plan_reserves_page_for_cp_padding(self):
        layout = build_dsv4_shared_page_layout(
            logical_size=128 * 16,
            page_size=128,
            cp_size=8,
        )
        # A 17-token extend is padded to 24 tokens for CP8.  A rank can have
        # two real rows followed by a token-zero padding row.  With fragmented
        # allocator pages, row zero's SWA window, the second real row, and the
        # padding row can cover four distinct pages even though the continuous
        # real-token span alone is bounded by three.
        first_window = torch.cat(
            (
                torch.arange(128 * 10 + 120, 128 * 11, dtype=torch.int32),
                torch.arange(128 * 12 - 1, 128 * 11 + 7, -1, dtype=torch.int32),
            )
        )
        second_row = first_window.clone()
        second_row[0] = 128 * 12
        padded_row = torch.full_like(first_window, -1)
        padded_row[0] = 0
        slots = torch.stack((first_window, second_row, padded_row))

        physical, remapped = _build_single_request_slot_stage_plan(layout, slots)

        expected_pages = torch.tensor([0, 10, 11, 12], dtype=torch.int64)
        self.assertTrue(
            torch.equal(physical[:4], layout.translate_pages(expected_pages).long())
        )
        self.assertEqual(remapped[2, 0].item() // layout.page_size, 0)

    def test_single_request_slot_plan_maps_clamped_tail_to_last_page(self):
        layout = build_dsv4_shared_page_layout(
            logical_size=64,
            page_size=4,
            cp_size=2,
        )
        slots = torch.tensor(
            [
                [1, 0, -1, -1],
                [3, 2, 1, 0],
                [5, 4, 3, 2],
            ],
            dtype=torch.int32,
        )

        physical, remapped = _build_single_request_slot_stage_plan(layout, slots)

        logical_pages = torch.tensor([0, 1, 1, 1], dtype=torch.int64)
        self.assertTrue(
            torch.equal(physical, layout.translate_pages(logical_pages).long())
        )
        self.assertEqual(remapped[2].tolist(), [13, 12, 3, 2])

    def test_each_logical_page_has_exactly_one_owner(self):
        page_size = 64
        layout = build_dsv4_shared_page_layout(
            logical_size=page_size * 17,
            page_size=page_size,
            cp_size=8,
        )
        slots = torch.arange(page_size * 17, dtype=torch.int64)
        owners = (slots // page_size) % 8

        owner_count = torch.zeros_like(slots)
        for rank in range(8):
            mask = layout.owner_layout.owned_row_mask(slots, rank=rank)
            self.assertTrue(torch.equal(mask, owners == rank))
            owner_count += mask

        self.assertTrue(torch.equal(owner_count, torch.ones_like(owner_count)))

    def test_family_page_sizes_use_the_same_owner_rule(self):
        for page_size in (2, 64, 256):
            layout = build_dsv4_shared_page_layout(
                logical_size=page_size * 17,
                page_size=page_size,
                cp_size=8,
            )
            logical_pages = torch.arange(17, dtype=torch.int64)
            slots = logical_pages * page_size

            for rank in range(8):
                expected = (logical_pages % 8) == rank
                self.assertTrue(
                    torch.equal(
                        layout.owner_layout.owned_row_mask(slots, rank=rank),
                        expected,
                    )
                )

    def test_padding_sentinel_is_preserved(self):
        layout = build_dsv4_shared_page_layout(
            logical_size=64 * 17,
            page_size=64,
            cp_size=8,
        )
        slots = torch.tensor([-1, 0, 64 * 16], dtype=torch.int64)
        pages = torch.tensor([-1, 0, 16], dtype=torch.int64)

        self.assertEqual(layout.translate_slots(slots)[0].item(), -1)
        self.assertEqual(layout.translate_pages(pages)[0].item(), -1)

    def test_tail_pages_fit_inside_rank_segments(self):
        layout = build_dsv4_shared_page_layout(
            logical_size=64 * 17,
            page_size=64,
            cp_size=8,
        )
        translated = layout.translate_pages(torch.arange(18, dtype=torch.int64))

        self.assertTrue(torch.all(translated < layout.cp_size * layout.pages_per_rank))
        self.assertGreaterEqual(layout.owner_layout.minimum_blocks_per_rank, 3)

    def test_stage_pages_compacts_remote_alias_once_and_remaps_table(self):
        layout = build_dsv4_shared_page_layout(
            logical_size=64 * 17,
            page_size=64,
            cp_size=8,
        )
        cache = torch.arange(
            layout.cp_size * layout.pages_per_rank * 3, dtype=torch.int64
        ).view(-1, 3)
        logical_pages = torch.tensor([[0, 8, 1, -1, 8]], dtype=torch.int64)

        physical_pages, remapped = _build_shared_page_stage_plan(layout, logical_pages)
        staged = _stage_shared_cache_from_plan(cache, physical_pages)

        physical = layout.translate_pages(torch.tensor([0, 8, 1]))
        expected_pages = torch.unique(physical, sorted=True)
        self.assertTrue(torch.equal(staged, cache.index_select(0, expected_pages)))
        self.assertEqual(remapped.tolist(), [[0, 1, 2, -1, 1]])

    def test_stage_slots_preserves_offsets_and_negative_sentinels(self):
        layout = build_dsv4_shared_page_layout(
            logical_size=64 * 17,
            page_size=64,
            cp_size=8,
        )
        cache = torch.arange(
            layout.cp_size * layout.pages_per_rank * 2, dtype=torch.int64
        ).view(-1, 2)
        logical_slots = torch.tensor([0, 65, 64 * 8 + 7, -1], dtype=torch.int64)

        physical_pages, remapped = _build_shared_slot_stage_plan(layout, logical_slots)
        staged = _stage_shared_cache_from_plan(cache, physical_pages)

        physical_pages = layout.translate_pages(torch.tensor([0, 1, 8]))
        expected_pages = torch.unique(physical_pages, sorted=True)
        self.assertTrue(torch.equal(staged, cache.index_select(0, expected_pages)))
        # Physical pages sort as logical pages 0, 8, 1.  Preserve the in-page
        # offsets while replacing only the page id.
        self.assertEqual(remapped.tolist(), [0, 2 * 64 + 1, 64 + 7, -1])

    def test_page_stage_plan_is_reusable_across_layer_caches(self):
        layout = build_dsv4_shared_page_layout(
            logical_size=64 * 17,
            page_size=64,
            cp_size=8,
        )
        logical_pages = torch.tensor([[0, 8, 1, -1, 8]], dtype=torch.int64)
        physical_pages, remapped = _build_shared_page_stage_plan(layout, logical_pages)
        layer0 = torch.arange(
            layout.cp_size * layout.pages_per_rank * 2, dtype=torch.int64
        ).view(-1, 2)
        layer1 = layer0 + 10_000

        staged0 = _stage_shared_cache_from_plan(layer0, physical_pages)
        staged1 = _stage_shared_cache_from_plan(layer1, physical_pages)

        self.assertEqual(remapped.tolist(), [[0, 1, 2, -1, 1]])
        self.assertTrue(torch.equal(staged0, layer0.index_select(0, physical_pages)))
        self.assertTrue(torch.equal(staged1, layer1.index_select(0, physical_pages)))

    def test_slot_stage_plan_preserves_offsets_when_reused(self):
        layout = build_dsv4_shared_page_layout(
            logical_size=64 * 17,
            page_size=64,
            cp_size=8,
        )
        logical_slots = torch.tensor([0, 65, 64 * 8 + 7, -1], dtype=torch.int64)

        physical_pages, remapped = _build_shared_slot_stage_plan(layout, logical_slots)

        self.assertEqual(remapped.tolist(), [0, 2 * 64 + 1, 64 + 7, -1])
        self.assertEqual(physical_pages.numel(), 3)

    def test_slot_stage_plan_keeps_dummy_page_for_all_padding_indices(self):
        layout = build_dsv4_shared_page_layout(
            logical_size=64 * 17,
            page_size=64,
            cp_size=8,
        )
        logical_slots = torch.full((8, 64), -1, dtype=torch.int32)

        physical_pages, remapped = _build_shared_slot_stage_plan(layout, logical_slots)

        self.assertEqual(physical_pages.tolist(), [0])
        self.assertTrue(torch.equal(remapped, logical_slots))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_dequant_fuses_logical_to_shared_slot_translation(self):
        page_size = 2
        layout = build_dsv4_shared_page_layout(
            logical_size=page_size * 17,
            page_size=page_size,
            cp_size=8,
        )
        bytes_per_page = 3 * 576
        cache = torch.zeros(
            (layout.cp_size * layout.pages_per_rank, bytes_per_page),
            dtype=torch.uint8,
            device="cuda",
        )
        # Valid FP8 1.0 values and ue8m0 scale exponent 0.  Distinguish pages
        # through the RoPE bf16 tail without introducing NaNs.
        for page in range(cache.shape[0]):
            for token in range(page_size):
                token_base = token * 576
                cache[page, token_base : token_base + 448] = 0x38
                rope = cache[page, token_base + 448 : token_base + 576].view(
                    torch.bfloat16
                )
                rope.fill_(float(page * 10 + token))
                scale_base = page_size * 576 + token * 8
                cache[page, scale_base : scale_base + 7] = 127

        logical_slots = torch.tensor([0, 1, 16, 17, 2, 18], device="cuda")
        physical_slots = layout.translate_slots(logical_slots)
        expected = dequantize_k_cache_paged(cache, physical_slots, page_size)

        actual = dequantize_k_cache_paged(
            cache,
            logical_slots,
            page_size,
            shared_cp_size=layout.cp_size,
            shared_pages_per_rank=layout.pages_per_rank,
        )

        self.assertTrue(torch.equal(actual, expected))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_fused_slot_translation_matches_layout(self):
        layout = build_dsv4_shared_page_layout(
            logical_size=64 * 17,
            page_size=64,
            cp_size=8,
        )
        for dtype in (torch.int32, torch.int64):
            logical_slots = torch.tensor(
                [-1, 0, 1, 63, 64, 64 * 8 + 7, 64 * 16 + 9],
                dtype=dtype,
                device="cuda",
            )

            expected = layout.translate_slots(logical_slots)
            actual = _translate_shared_slots_fused(layout, logical_slots)

            self.assertTrue(torch.equal(actual, expected))


class TestDeepSeekV4SharedOwnerWrites(CustomTestCase):
    def test_compressor_flashmla_write_keeps_one_owner_writer(self):
        pool = object.__new__(SharedDeepSeekV4SingleKVPool)
        pool.page_size = 64
        pool.shared_rank = 3
        pool.shared_size = 8
        local_cache = torch.empty((4, 64), dtype=torch.uint8)
        pool.local_kv_buffer = [local_cache]
        loc = torch.tensor([0, 64 * 3, 64 * 11], dtype=torch.int64)
        cache_k = torch.empty((3, 512), dtype=torch.bfloat16)

        with patch(
            "sglang.srt.mem_cache.deepseek_v4_shared.fused_store_cache_shared"
        ) as store:
            pool.set_key_buffer_fused(0, loc, cache_k)

        store.assert_called_once()
        kwargs = store.call_args.kwargs
        self.assertIs(kwargs["input"], cache_k)
        self.assertIs(kwargs["cache"], local_cache)
        self.assertIs(kwargs["indices"], loc)
        self.assertEqual(kwargs["page_size"], 64)
        self.assertEqual(kwargs["type"], "flashmla")
        self.assertEqual(kwargs["owner_rank"], 3)
        self.assertEqual(kwargs["owner_size"], 8)

    def test_compressor_indexer_write_keeps_one_owner_writer(self):
        pool = object.__new__(SharedDeepSeekV4IndexerPool)
        pool.page_size = 64
        pool.start_layer = 0
        pool.shared_rank = 5
        pool.shared_size = 8
        local_cache = torch.empty((4, 64), dtype=torch.uint8)
        pool.local_index_k_with_scale_buffer = [local_cache]
        loc = torch.tensor([64 * 5, 64 * 13], dtype=torch.int64)
        cache_k = torch.empty((2, 128), dtype=torch.bfloat16)

        with patch(
            "sglang.srt.mem_cache.deepseek_v4_shared.fused_store_cache_shared"
        ) as store:
            pool.set_index_fused(0, loc, cache_k)

        store.assert_called_once()
        kwargs = store.call_args.kwargs
        self.assertIs(kwargs["input"], cache_k)
        self.assertIs(kwargs["cache"], local_cache)
        self.assertIs(kwargs["indices"], loc)
        self.assertEqual(kwargs["page_size"], 64)
        self.assertEqual(kwargs["type"], "indexer")
        self.assertEqual(kwargs["owner_rank"], 5)
        self.assertEqual(kwargs["owner_size"], 8)

    def test_read_indices_translate_to_rank_major_shared_alias(self):
        pool = object.__new__(SharedDeepSeekV4SingleKVPool)
        pool.shared_family = SimpleNamespace(
            layout=build_dsv4_shared_page_layout(
                logical_size=64 * 17,
                page_size=64,
                cp_size=8,
            )
        )
        logical = torch.tensor([-1, 0, 64, 64 * 8], dtype=torch.int64)

        translated = pool.translate_slots_for_read(logical)

        self.assertEqual(translated[0].item(), -1)
        self.assertEqual(translated[1:].tolist(), [0, 256, 64])

    def test_indexer_slot_read_alias_is_relative_to_the_current_rank(self):
        pool = object.__new__(SharedDeepSeekV4IndexerPool)
        pool.shared_rank = 3
        pool.shared_family = SimpleNamespace(
            layout=build_dsv4_shared_page_layout(
                logical_size=64 * 17,
                page_size=64,
                cp_size=8,
            )
        )
        logical_slots = torch.tensor(
            [-1, 0, 64 * 3, 64 * 4, 64 * 11], dtype=torch.int32
        )

        translated = pool.translate_slots_for_read(logical_slots)
        expected = pool.shared_family.layout.owner_layout.rank_relative_rows(
            logical_slots, rank=3
        )

        self.assertTrue(torch.equal(translated, expected))


class TestDeepSeekV4SharedTokenPool(CustomTestCase):
    def test_explicit_close_is_idempotent_and_closes_each_family_once(self):
        family = Mock()
        kv_pool = SimpleNamespace(
            shared_family=family,
            kv_buffer=[object()],
            local_kv_buffer=[object()],
        )
        state_pool = SimpleNamespace(
            shared_family=family,
            _shared_buffer=object(),
            kv_score_buffer=object(),
        )
        pool = object.__new__(SharedDeepSeekV4TokenToKVPool)
        pool.swa_kv_pool = kv_pool
        pool.c4_kv_pool = None
        pool.c128_kv_pool = None
        pool.c4_indexer_kv_pool = None
        pool.compress_state_pools = [state_pool]
        pool.indexer_compress_state_pools = []
        pool.shared_state_families = {"state": family}
        pool.shared_write_publisher = object()
        pool.shared_cache_access = object()

        pool.close()
        pool.close()

        family.close.assert_called_once_with()
        self.assertEqual(kv_pool.kv_buffer, [])
        self.assertEqual(kv_pool.local_kv_buffer, [])
        self.assertIsNone(kv_pool.shared_family)
        self.assertIsNone(state_pool._shared_buffer)
        self.assertIsNone(state_pool.kv_score_buffer)
        self.assertIsNone(state_pool.shared_family)

    def test_pool_construction_failure_closes_completed_shared_families(self):
        completed_family = Mock()
        completed_pool = SimpleNamespace(
            shared_family=completed_family,
            kv_buffer=[object()],
            local_kv_buffer=[object()],
        )

        def fail_after_first_family(instance, *_args, **_kwargs):
            instance.swa_kv_pool = completed_pool
            raise RuntimeError("injected later-family failure")

        with (
            patch.object(
                dsv4_shared.DeepSeekV4TokenToKVPool,
                "__init__",
                new=fail_after_first_family,
            ),
            self.assertRaisesRegex(RuntimeError, "later-family failure"),
        ):
            SharedDeepSeekV4TokenToKVPool(shared_rank=0, shared_size=8)

        completed_family.close.assert_called_once_with()
        self.assertEqual(completed_pool.kv_buffer, [])
        self.assertEqual(completed_pool.local_kv_buffer, [])
        self.assertIsNone(completed_pool.shared_family)

    def test_capacity_log_reports_each_family_and_alignment_overhead(self):
        def family(name, granule, mapped_bytes, alignment_overhead):
            accounting = SimpleNamespace(
                name=name,
                logical_blocks_per_layer=32,
                minimum_blocks_per_rank=4,
                physical_blocks_per_rank=8,
                mapped_bytes_per_rank=mapped_bytes,
                alignment_overhead_bytes_per_rank=alignment_overhead,
            )
            return SimpleNamespace(
                name=name,
                storage=SimpleNamespace(
                    spec=SimpleNamespace(
                        logical_rows_per_layer=8192,
                        ownership_granule=granule,
                    ),
                    accounting=lambda: accounting,
                ),
            )

        pool = object.__new__(SharedDeepSeekV4TokenToKVPool)
        pool.shared_rank = 0
        pool.swa_kv_pool = SimpleNamespace(shared_family=family("swa", 256, 1024, 64))
        pool.c4_kv_pool = SimpleNamespace(shared_family=family("c4", 64, 512, 32))
        pool.c128_kv_pool = SimpleNamespace(shared_family=None)
        pool.c4_indexer_kv_pool = SimpleNamespace(shared_family=None)
        pool.shared_state_families = {}

        with patch.object(dsv4_shared.logger, "info") as log_info:
            pool._log_shared_family_accounting()

        self.assertEqual(log_info.call_count, 3)
        rendered = [call.args[0] % call.args[1:] for call in log_info.call_args_list]
        self.assertIn("family=swa", rendered[0])
        self.assertIn("ownership_granule=256", rendered[0])
        self.assertIn("mapped_bytes_per_rank=1024", rendered[0])
        self.assertIn("alignment_overhead_bytes_per_rank=64", rendered[0])
        self.assertIn("family=c4", rendered[1])
        self.assertIn("mapped_bytes_per_rank=1536", rendered[2])

    def test_short_c4_uses_direct_shared_slots_even_with_common_plan(self):
        stage_calls = []
        translate_calls = []
        pool = _attach_shared_access(
            SimpleNamespace(
                stage_compressed_pages_with_indexer_plan=lambda layer_id, pages: (
                    stage_calls.append((layer_id, pages.clone()))
                    or torch.tensor([[99]])
                ),
                translate_extra_slots_for_read=lambda layer_id, slots: (
                    translate_calls.append((layer_id, slots.clone())) or slots + 200
                ),
            )
        )
        indices = torch.tensor([3, 7, -1])
        fallback_cache = torch.tensor([[-1]])
        metadata = SimpleNamespace(
            shared_compressed_physical_pages={
                "indexer": torch.tensor([1]),
                4: torch.tensor([2]),
                128: torch.tensor([3]),
            },
            shared_stage_c4=False,
        )

        actual_cache, actual_indices = _stage_dsv4_shared_extra_slots(
            pool,
            metadata,
            layer_id=4,
            compress_ratio=4,
            extra_k_cache=fallback_cache,
            extra_indices=indices,
        )

        self.assertIs(actual_cache, fallback_cache)
        self.assertTrue(torch.equal(actual_indices, indices + 200))
        self.assertEqual(stage_calls, [])
        self.assertEqual(len(translate_calls), 1)

    def test_extra_stage_plan_reuses_c128_and_published_c4_page_plans(self):
        prepare_calls = []
        stage_calls = []
        c4_stage_calls = []
        translate_calls = []

        def prepare(layer_id, slots):
            prepare_calls.append((layer_id, slots.clone()))
            return torch.tensor([layer_id]), slots + 100

        def stage(layer_id, pages):
            stage_calls.append((layer_id, pages.clone()))
            return torch.tensor([[layer_id, pages.item()]])

        pool = _attach_shared_access(
            SimpleNamespace(
                prepare_extra_slots_for_read=prepare,
                stage_extra_slots_with_plan=stage,
                stage_compressed_pages_with_indexer_plan=lambda layer_id, pages: (
                    c4_stage_calls.append((layer_id, pages.clone()))
                    or torch.tensor([[layer_id, pages.numel()]])
                ),
                translate_extra_slots_for_read=lambda layer_id, slots: (
                    translate_calls.append((layer_id, slots.clone())) or slots + 200
                ),
            )
        )
        metadata = SimpleNamespace()
        fallback_cache = torch.tensor([[-1]])

        c128_indices = torch.tensor([3, -1])
        first_cache, first_indices = _stage_dsv4_shared_extra_slots(
            pool,
            metadata,
            layer_id=3,
            compress_ratio=128,
            extra_k_cache=fallback_cache,
            extra_indices=c128_indices,
        )
        second_cache, second_indices = _stage_dsv4_shared_extra_slots(
            pool,
            metadata,
            layer_id=5,
            compress_ratio=128,
            extra_k_cache=fallback_cache,
            extra_indices=c128_indices,
        )

        self.assertEqual(len(prepare_calls), 1)
        self.assertEqual([call[0] for call in stage_calls], [3, 5])
        self.assertTrue(torch.equal(first_indices, c128_indices + 100))
        self.assertIs(second_indices, first_indices)
        self.assertEqual(first_cache.tolist(), [[3, 3]])
        self.assertEqual(second_cache.tolist(), [[5, 3]])

        first_c4 = torch.tensor([7, -1])
        second_c4 = torch.tensor([9, -1])
        c4_physical_pages = torch.tensor([4, 7, 9])
        metadata._dsv4_shared_c4_stage_plan = c4_physical_pages
        _, remapped_c4_first = _stage_dsv4_shared_extra_slots(
            pool,
            metadata,
            layer_id=4,
            compress_ratio=4,
            extra_k_cache=fallback_cache,
            extra_indices=first_c4,
        )
        _, remapped_c4_second = _stage_dsv4_shared_extra_slots(
            pool,
            metadata,
            layer_id=6,
            compress_ratio=4,
            extra_k_cache=fallback_cache,
            extra_indices=second_c4,
        )

        self.assertEqual(len(prepare_calls), 1)
        self.assertEqual([call[0] for call in c4_stage_calls], [4, 6])
        self.assertTrue(
            all(torch.equal(call[1], c4_physical_pages) for call in c4_stage_calls)
        )
        self.assertEqual(translate_calls, [])
        self.assertIs(remapped_c4_first, first_c4)
        self.assertIs(remapped_c4_second, second_c4)

        _, fallback_indices = _stage_dsv4_shared_extra_slots(
            pool,
            SimpleNamespace(),
            layer_id=8,
            compress_ratio=4,
            extra_k_cache=fallback_cache,
            extra_indices=first_c4,
        )
        self.assertTrue(torch.equal(fallback_indices, first_c4 + 200))

    def test_c128_stage_propagates_single_request_plan(self):
        physical_pages = torch.tensor([4, 7])
        remapped = torch.tensor([0, 1, -1])
        staged = torch.tensor([[10, 11]])
        prepare = unittest.mock.Mock(return_value=(physical_pages, remapped))
        pool = _attach_shared_access(
            SimpleNamespace(
                prepare_extra_slots_for_read=prepare,
                stage_extra_slots_with_plan=unittest.mock.Mock(return_value=staged),
            )
        )
        metadata = SimpleNamespace()

        actual_cache, actual_indices = _stage_dsv4_shared_extra_slots(
            pool,
            metadata,
            layer_id=3,
            compress_ratio=128,
            extra_k_cache=torch.tensor([[-1]]),
            extra_indices=torch.tensor([3, 2, -1]),
            single_request=True,
        )

        self.assertIs(actual_cache, staged)
        self.assertIs(actual_indices, remapped)
        prepare.assert_called_once_with(
            3,
            unittest.mock.ANY,
            single_request=True,
        )

    def test_compressed_kv_stage_accepts_family_specific_physical_pages(self):
        pool = object.__new__(SharedDeepSeekV4TokenToKVPool)
        gathered = torch.tensor([[1, 2, 3]])
        c4_pool = object.__new__(SharedDeepSeekV4SingleKVPool)
        c4_pool.shared_family = SimpleNamespace(
            layout=SimpleNamespace(cp_size=8, page_size=64, pages_per_rank=17)
        )
        c4_pool.stage_slots_with_plan = lambda layer_id, pages: gathered
        pool.c4_indexer_kv_pool = SimpleNamespace(
            shared_family=SimpleNamespace(
                layout=SimpleNamespace(cp_size=8, page_size=64, pages_per_rank=29)
            )
        )
        pool.layer_mapping = {
            4: SimpleNamespace(
                compress_ratio=4,
                compress_layer_id=2,
                compress_kv_pool=c4_pool,
            )
        }

        pages = torch.tensor([1, 7])
        actual = pool.stage_compressed_pages_with_indexer_plan(4, pages)

        self.assertIs(actual, gathered)

    def test_common_compressed_plan_translates_each_family_stride(self):
        pool = object.__new__(SharedDeepSeekV4TokenToKVPool)
        indexer_layout = SimpleNamespace(cp_size=2, pages_per_rank=10)
        physical_indexer = torch.tensor([1, 12, 19], dtype=torch.int64)
        compact_page_table = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        pool.c4_indexer_kv_pool = SimpleNamespace(
            shared_family=SimpleNamespace(layout=indexer_layout),
            prepare_pages_for_read=lambda pages: (
                physical_indexer,
                compact_page_table,
            ),
        )

        translated_logical_pages = []

        def make_compressed_pool(offset):
            def translate_pages(logical_pages):
                translated_logical_pages.append(logical_pages.clone())
                return logical_pages + offset

            compressed_pool = object.__new__(SharedDeepSeekV4SingleKVPool)
            compressed_pool.shared_family = SimpleNamespace(
                layout=SimpleNamespace(
                    cp_size=2,
                    translate_pages=translate_pages,
                )
            )
            return compressed_pool

        pool.c4_kv_pool = make_compressed_pool(100)
        pool.c128_kv_pool = make_compressed_pool(1000)

        physical_by_family, actual_table = pool.prepare_compressed_pages_for_read(
            torch.tensor([[3, 7, 9]], dtype=torch.int64)
        )

        # owner = physical // 10, local = physical % 10,
        # logical = local * cp + owner.
        expected_logical = torch.tensor([2, 5, 19], dtype=torch.int64)
        self.assertTrue(torch.equal(physical_by_family["indexer"], physical_indexer))
        self.assertTrue(torch.equal(physical_by_family[4], expected_logical + 100))
        self.assertTrue(torch.equal(physical_by_family[128], expected_logical + 1000))
        self.assertTrue(torch.equal(actual_table, compact_page_table))
        self.assertEqual(len(translated_logical_pages), 2)
        self.assertTrue(
            all(
                torch.equal(logical_pages, expected_logical)
                for logical_pages in translated_logical_pages
            )
        )

    def test_swa_stage_plan_preserves_compact_page_indices(self):
        pool = object.__new__(SharedDeepSeekV4TokenToKVPool)
        pool._stage_start = 0
        gathered = torch.tensor([[10, 11, 12], [20, 21, 22]], dtype=torch.int64)
        physical_pages = torch.tensor([4, 7], dtype=torch.int64)
        logical_remap = torch.tensor([0, 65, -1], dtype=torch.int64)
        pool.swa_kv_pool = SimpleNamespace(
            page_size=64,
            prepare_slots_for_read=lambda slots: (physical_pages, logical_remap),
            stage_slots_with_plan=lambda layer_id, pages: gathered,
        )

        actual_pages, remapped = pool.prepare_swa_slots_for_read(
            torch.tensor([0, 65, -1], dtype=torch.int64)
        )
        staged = pool.stage_swa_slots_with_plan(3, actual_pages)

        self.assertIs(actual_pages, physical_pages)
        self.assertIs(remapped, logical_remap)
        self.assertIs(staged, gathered)

    def test_single_row_decode_uses_fixed_swa_stage_plan(self):
        pool = object.__new__(SharedDeepSeekV4TokenToKVPool)
        pool.swa_kv_pool = SimpleNamespace(
            shared_family=SimpleNamespace(layout=object()),
            prepare_slots_for_read=unittest.mock.Mock(
                side_effect=AssertionError("decode must not use dynamic unique")
            ),
        )
        slots = torch.tensor([[3, 2, 1, 0]], dtype=torch.int32)
        expected = (torch.tensor([0]), torch.tensor([[3, 2, 1, 0]]))

        with patch(
            "sglang.srt.mem_cache.deepseek_v4_shared."
            "_build_single_request_slot_stage_plan",
            return_value=expected,
        ) as build_fixed:
            actual = pool.prepare_swa_slots_for_read(slots)

        self.assertIs(actual, expected)
        build_fixed.assert_called_once_with(
            pool.swa_kv_pool.shared_family.layout,
            slots,
        )

    def test_runtime_selects_shared_pool_only_for_target_dsv4_cp(self):
        parallel = SimpleNamespace(attn_cp_rank=3, attn_cp_size=8)
        runner = SimpleNamespace(
            is_draft_worker=False,
            # DSV4 owns a dedicated attention backend rather than the generic
            # MLA backend; this must not disable its shared pool selector.
            use_mla_backend=False,
            server_args=SimpleNamespace(enable_dsa_shared_kv_cache=True),
            model_config=SimpleNamespace(hf_config=SimpleNamespace()),
        )
        with (
            patch(
                "sglang.srt.configs.model_config.is_deepseek_v4",
                return_value=True,
            ),
            patch(
                "sglang.srt.arg_groups.deepseek_v4_hook.get_parallel",
                return_value=parallel,
            ),
        ):
            self.assertEqual(get_dsv4_shared_info(runner), (3, 8))
            runner.is_draft_worker = True
            self.assertEqual(get_dsv4_shared_info(runner), (None, 1))

    def test_flash_and_pro_profiles_select_the_same_shared_runtime(self):
        parallel = SimpleNamespace(attn_cp_rank=5, attn_cp_size=8)
        profiles = {
            "flash": [0, 0] + [4, 128] * 20 + [4],
            "pro": [128, 128] + [4, 128] * 29 + [4],
        }

        with (
            patch(
                "sglang.srt.configs.model_config.is_deepseek_v4",
                return_value=True,
            ) as is_deepseek_v4,
            patch(
                "sglang.srt.arg_groups.deepseek_v4_hook.get_parallel",
                return_value=parallel,
            ),
        ):
            for name, compression_ratios in profiles.items():
                with self.subTest(name=name):
                    hf_config = SimpleNamespace(
                        model_type="deepseek_v4",
                        compress_ratios=compression_ratios,
                    )
                    runner = SimpleNamespace(
                        is_draft_worker=False,
                        server_args=SimpleNamespace(enable_dsa_shared_kv_cache=True),
                        model_config=SimpleNamespace(hf_config=hf_config),
                    )

                    self.assertEqual(get_dsv4_shared_info(runner), (5, 8))
                    is_deepseek_v4.assert_called_with(hf_config)

    def test_backend_slot_helper_preserves_base_and_translates_shared_indices(self):
        pages = torch.tensor([-1, 0, 1], dtype=torch.int32)
        base_pool = SimpleNamespace()
        self.assertIs(
            _translate_dsv4_shared_slots(base_pool, "extra", pages, layer_id=3),
            pages,
        )

        shared_pool = _attach_shared_access(
            SimpleNamespace(
                translate_extra_slots_for_read=lambda layer_id, value: value
                + 100
                + layer_id,
            )
        )
        self.assertTrue(
            torch.equal(
                _translate_dsv4_shared_slots(shared_pool, "extra", pages, layer_id=3),
                pages + 103,
            )
        )

    def test_backend_fence_helper_is_mandatory_for_shared_pool_only(self):
        synchronize = unittest.mock.Mock()
        _synchronize_dsv4_shared_writes(SimpleNamespace())
        _synchronize_dsv4_shared_writes(
            _attach_shared_access(
                SimpleNamespace(synchronize_shared_writes=synchronize)
            )
        )
        synchronize.assert_called_once_with()

    def test_backend_does_not_reuse_indexer_fence_after_main_compressor_writes(self):
        synchronize = unittest.mock.Mock()
        pool = _attach_shared_access(
            SimpleNamespace(synchronize_shared_writes=synchronize)
        )
        metadata = SimpleNamespace(_dsv4_shared_write_fence_layer=7)

        _synchronize_dsv4_shared_writes(pool, core_attn_metadata=metadata, layer_id=7)

        synchronize.assert_called_once_with()

    def test_slot_translation_dispatches_to_the_matching_cache_family(self):
        pool = object.__new__(SharedDeepSeekV4TokenToKVPool)
        pool.swa_kv_pool = SimpleNamespace(
            translate_slots_for_read=lambda slots: slots + 200,
        )
        c4_pool = SimpleNamespace(
            translate_slots_for_read=lambda slots: slots + 400,
        )
        pool.layer_mapping = [SimpleNamespace(compress_kv_pool=c4_pool)]

        values = torch.tensor([0, 1], dtype=torch.int32)
        self.assertTrue(
            torch.equal(pool.translate_swa_slots_for_read(values), values + 200)
        )
        self.assertTrue(
            torch.equal(pool.translate_extra_slots_for_read(0, values), values + 400)
        )

    def test_compressor_direct_write_keeps_one_owner_writer(self):
        pool = object.__new__(SharedDeepSeekV4TokenToKVPool)
        pool.shared_rank = 3
        pool.shared_size = 8
        extra_local = torch.empty((4, 64), dtype=torch.uint8)
        indexer_local = torch.empty((4, 64), dtype=torch.uint8)
        pool.layer_mapping = [
            SimpleNamespace(
                compress_kv_pool=SimpleNamespace(local_kv_buffer=[extra_local]),
                compress_layer_id=0,
            )
        ]
        pool.c4_indexer_kv_pool = SimpleNamespace(
            start_layer=0,
            local_index_k_with_scale_buffer=[indexer_local],
        )

        self.assertEqual(
            pool.get_compressor_write_info(0, is_indexer=False),
            (extra_local, 3, 8),
        )
        self.assertEqual(
            pool.get_compressor_write_info(0, is_indexer=True),
            (indexer_local, 3, 8),
        )

    def test_main_norm_rope_store_keeps_one_owner_writer(self):
        pool = object.__new__(SharedDeepSeekV4TokenToKVPool)
        pool.shared_rank = 2
        pool.shared_size = 8
        local_cache = torch.empty((4, 64), dtype=torch.uint8)
        pool.swa_kv_pool = SimpleNamespace(
            page_size=64,
            local_kv_buffer=[local_cache],
        )
        pool._swa_local_layer_id = lambda layer_id: 0
        kv = torch.empty((2, 512), dtype=torch.bfloat16)
        weight = torch.empty((512,), dtype=torch.bfloat16)
        freqs = torch.empty((4, 32), dtype=torch.complex64)
        positions = torch.tensor([0, 1], dtype=torch.int64)
        loc = torch.tensor([0, 64], dtype=torch.int32)

        with patch(
            "sglang.srt.mem_cache.deepseek_v4_shared.fused_k_norm_rope_flashmla"
        ) as store:
            pool.set_swa_key_buffer_radix_fused_norm_rope(
                layer_id=0,
                swa_loc=loc,
                kv=kv,
                kv_weight=weight,
                eps=1.0e-6,
                freqs_cis=freqs,
                positions=positions,
            )

        store.assert_called_once()
        kwargs = store.call_args.kwargs
        self.assertIs(kwargs["kv"], kv)
        self.assertIs(kwargs["kv_weight"], weight)
        self.assertEqual(kwargs["eps"], 1.0e-6)
        self.assertIs(kwargs["freqs_cis"], freqs)
        self.assertIs(kwargs["positions"], positions)
        self.assertIs(kwargs["out_loc"], loc)
        self.assertIs(kwargs["kvcache"], local_cache)
        self.assertEqual(kwargs["page_size"], 64)
        self.assertEqual(kwargs["owner_rank"], 2)
        self.assertEqual(kwargs["owner_size"], 8)

    def test_shared_write_publication_delegates_to_mandatory_publisher(self):
        pool = object.__new__(SharedDeepSeekV4TokenToKVPool)
        pool.shared_write_publisher = unittest.mock.Mock()

        pool.synchronize_shared_writes()

        pool.shared_write_publisher.publish.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
