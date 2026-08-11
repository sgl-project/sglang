import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.kernels.ops.attention.dsv4 import compress as dsv4_compress
from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
    dequantize_k_cache_paged,
)
from sglang.srt.arg_groups.deepseek_v4_hook import get_dsv4_shared_info
from sglang.srt.layers.attention import deepseek_v4_backend as dsv4_backend
from sglang.srt.layers.attention.deepseek_v4_backend import (
    _synchronize_dsv4_shared_writes,
)
from sglang.srt.layers.attention.dsv4.shared_cache_access import (
    DSV4SharedCacheAccess,
)
from sglang.srt.mem_cache import deepseek_v4_shared as dsv4_shared
from sglang.srt.mem_cache.deepseek_v4_shared import (
    SharedCompressStatePool,
    SharedDeepSeekV4Family,
    SharedDeepSeekV4SingleKVPool,
    SharedDeepSeekV4TokenToKVPool,
    _translate_shared_slots_fused,
    build_dsv4_shared_page_layout,
    dsv4_prefill_demand_cache_fits_direct_slots,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _attach_shared_access(pool):
    pool.shared_cache_access = DSV4SharedCacheAccess(pool)
    return pool


class TestDeepSeekV4SharedPageLayout(CustomTestCase):
    def test_prefill_demand_cache_supports_hopper_and_blackwell(self):
        cases = (
            (True, False, True),
            (False, True, True),
            (False, False, False),
        )
        for sm90, sm100, expected in cases:
            with (
                self.subTest(sm90=sm90, sm100=sm100),
                patch(
                    "sglang.srt.utils.is_sm90_supported",
                    return_value=sm90,
                ),
                patch(
                    "sglang.srt.utils.is_sm100_supported",
                    return_value=sm100,
                ),
            ):
                self.assertEqual(
                    dsv4_shared.supports_dsv4_shared_demand_cache(),
                    expected,
                )

    def test_prefill_demand_direct_slots_require_both_families_to_fit(self):
        self.assertTrue(
            dsv4_prefill_demand_cache_fits_direct_slots(
                swa_rows=223_232,
                c4_rows=275_968,
                c128_rows=8_624,
                cache_rows=1_048_576,
            )
        )
        self.assertFalse(
            dsv4_prefill_demand_cache_fits_direct_slots(
                swa_rows=524_289,
                c4_rows=1,
                c128_rows=1,
                cache_rows=1_048_576,
            )
        )
        self.assertFalse(
            dsv4_prefill_demand_cache_fits_direct_slots(
                swa_rows=1,
                c4_rows=524_289,
                c128_rows=1,
                cache_rows=1_048_576,
            )
        )

    @staticmethod
    def _make_dsv4_capacity_configurator(shared_cache_size):
        from sglang.srt.model_executor.pool_configurator import DSV4PoolConfigurator

        ratios = [0, 0] + [4, 128] * 20 + [4]
        configurator = object.__new__(DSV4PoolConfigurator)
        configurator.qk_nope_head_dim = 448
        configurator.qk_rope_head_dim = 64
        configurator.indexer_head_dim = 128
        configurator.swa_ratio = 0.1
        configurator.swa_page_size = 128
        configurator.c4_ring_size = 8
        configurator.c128_ring_size = 128
        configurator.c4_shrink_factor = 1
        configurator.num_layers_total = len(ratios)
        configurator.num_layers_ca4 = sum(ratio == 4 for ratio in ratios)
        configurator.num_layers_ca128 = sum(ratio == 128 for ratio in ratios)
        configurator.shared_cache_size = shared_cache_size
        return configurator

    def test_capacity_shards_attention_but_keeps_indexer_replicated(self):
        replicated = self._make_dsv4_capacity_configurator(1)
        shared = self._make_dsv4_capacity_configurator(8)

        with (
            patch(
                "sglang.srt.model_executor.pool_configurator._get_dsv4_compress_state_dtype_sizes",
                return_value=(4, 4),
            ),
            patch.object(
                dsv4_backend.envs.SGLANG_OPT_USE_ONLINE_COMPRESS,
                "get",
                return_value=False,
            ),
        ):
            replicated_bytes = replicated._get_bytes_per_full_token()
            shared_bytes = shared._get_bytes_per_full_token()

        self.assertAlmostEqual(replicated_bytes, 7705.45, places=5)
        self.assertAlmostEqual(shared_bytes, 1804.75625, places=5)
        self.assertAlmostEqual(replicated_bytes / shared_bytes, 4.269522, places=5)

    def test_demand_cache_is_limited_to_pure_prefill_modes(self):
        self.assertTrue(
            dsv4_backend._use_dsv4_shared_prefill_demand_cache(ForwardMode.EXTEND)
        )
        self.assertTrue(
            dsv4_backend._use_dsv4_shared_prefill_demand_cache(
                ForwardMode.SPLIT_PREFILL
            )
        )
        for mode in (
            ForwardMode.DECODE,
            ForwardMode.MIXED,
            ForwardMode.TARGET_VERIFY,
            ForwardMode.IDLE,
        ):
            with self.subTest(mode=mode):
                self.assertFalse(
                    dsv4_backend._use_dsv4_shared_prefill_demand_cache(mode)
                )

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

    def test_flashmla_page_view_preserves_tma_aligned_page_stride(self):
        padded_c4_pages = torch.empty((3, 37440), dtype=torch.uint8)

        actual = dsv4_backend._view_dsv4_flashmla_pages(
            padded_c4_pages,
            page_size=64,
            token_bytes=584,
        )

        self.assertEqual(actual.shape, (3, 64, 1, 584))
        self.assertEqual(actual.stride(0) * actual.element_size(), 37440)
        self.assertEqual(
            (actual.stride(0) * actual.element_size()) % 576,
            0,
        )

    def test_flashmla_page_view_rejects_non_byte_or_unaligned_storage(self):
        invalid = (
            torch.empty((1, 18720), dtype=torch.float32),
            torch.empty((1, 37441), dtype=torch.uint8)[:, 1:],
        )

        for cache in invalid:
            with self.subTest(dtype=cache.dtype, offset=cache.storage_offset()):
                with self.assertRaisesRegex(ValueError, "byte-addressable|16-byte"):
                    dsv4_backend._view_dsv4_flashmla_pages(
                        cache,
                        page_size=64,
                        token_bytes=584,
                    )

    def test_shared_prefill_uses_flashmla_kv_direct_vmm_when_sparse_is_enabled(self):
        def forbid_staging(*_args, **_kwargs):
            raise AssertionError(
                "Shared FlashMLA-KV Prefill must not stage attention KV"
            )

        swa_cache = torch.empty((3, 74880), dtype=torch.uint8)
        pool = object.__new__(SharedDeepSeekV4TokenToKVPool)
        pool._stage_start = 0
        pool.page_size = 256
        pool.swa_window_size = 128
        pool.wait_layer_transfer = lambda _layer_id: None
        pool.swa_kv_pool = SimpleNamespace(
            kv_cache_total_dim=584,
            get_key_buffer=lambda _layer_id: swa_cache,
        )
        pool.synchronize_shared_writes = lambda: None
        pool.translate_swa_slots_for_read = lambda slots: slots + 64
        pool.translate_extra_slots_for_read = lambda _layer_id, slots: slots + 128
        pool.shared_cache_access = DSV4SharedCacheAccess(pool)

        indices = torch.zeros((1, 64), dtype=torch.int32)
        lengths = torch.ones((1,), dtype=torch.int32)
        core = dsv4_backend.DSV4AttnMetadata(
            page_size=256,
            page_table=torch.zeros((1, 1), dtype=torch.int32),
            raw_out_loc=torch.zeros((1,), dtype=torch.int32),
            cuda_int32_kwargs={},
            seq_lens_casual=torch.ones((1,), dtype=torch.int32),
            positions_casual=torch.zeros((1,), dtype=torch.int32),
            swa_page_indices=indices,
            swa_topk_lengths=lengths,
            c4_sparse_topk=512,
        )
        core.c1_flashmla_metadata = object()
        metadata = dsv4_backend.DSV4Metadata(core, indexer_metadata=None)

        backend = object.__new__(dsv4_backend.DeepseekV4AttnBackend)
        backend.mtp_enabled = False
        backend.forward_metadata = metadata
        backend.token_to_kv_pool = pool
        backend.head_dim_v = 512
        backend.softmax_scale = 1.0

        q = torch.empty((1, 1, 576), dtype=torch.bfloat16)
        kv = torch.empty((1, 1, 576), dtype=torch.bfloat16)
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            batch_size=1,
        )
        layer = SimpleNamespace(layer_id=0, v_head_dim=512)
        flashmla_output = torch.empty((1, 1, 1, 512), dtype=torch.bfloat16)

        with (
            patch.object(
                dsv4_backend.envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL,
                "get",
                return_value=True,
            ),
            patch.object(
                backend,
                "_forward_prefill_sparse",
                side_effect=AssertionError("Shared release must not dispatch Sparse"),
            ),
            patch(
                "sgl_kernel.flash_mla.flash_mla_with_kvcache",
                return_value=(flashmla_output, None),
            ) as flashmla_kv,
        ):
            output = backend.forward(
                q=q,
                k=kv,
                v=kv,
                layer=layer,
                forward_batch=forward_batch,
                compress_ratio=0,
                save_kv_cache=False,
                attn_sink=torch.empty((1,), dtype=torch.float32),
            )

        self.assertEqual(output.shape, (1, 1, 512))
        call = flashmla_kv.call_args.kwargs
        self.assertTrue(torch.equal(call["indices"].squeeze(1), indices + 64))
        self.assertEqual(call["k_cache"].stride(0), 74880)

    def test_flashmla_single_page_view_preserves_physical_stride(self):
        cases = (
            (128, 584, 74880),
            (64, 584, 37440),
            (2, 584, 1728),
        )

        for page_size, token_bytes, physical_stride in cases:
            with self.subTest(page_size=page_size):
                cache = torch.empty((1, physical_stride), dtype=torch.uint8)

                actual = dsv4_backend._view_dsv4_flashmla_pages(
                    cache,
                    page_size=page_size,
                    token_bytes=token_bytes,
                )

                self.assertEqual(
                    actual.stride(0) * actual.element_size(), physical_stride
                )

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

    def test_shared_pool_builds_an_ordinary_replicated_indexer(self):
        token_pool = object.__new__(SharedDeepSeekV4TokenToKVPool)
        expected = object()

        with patch.object(
            dsv4_shared, "DeepSeekV4IndexerPool", return_value=expected
        ) as constructor:
            actual = token_pool._make_indexer_pool(
                1024, 64, torch.bfloat16, 128, 21, "cuda", False
            )

        self.assertIs(actual, expected)
        constructor.assert_called_once_with(
            1024, 64, torch.bfloat16, 128, 21, "cuda", False
        )

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
            index_k_with_scale_buffer=[indexer_local],
        )

        self.assertEqual(
            pool.get_compressor_write_info(0, is_indexer=False),
            (extra_local, 3, 8),
        )
        self.assertEqual(
            pool.get_compressor_write_info(0, is_indexer=True),
            (indexer_local, 0, 1),
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
