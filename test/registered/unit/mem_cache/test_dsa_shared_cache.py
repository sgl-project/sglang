import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import sglang.srt.layers.attention.dsa_backend as dsa_backend_module
import sglang.srt.mem_cache.dsa_cache_shared as dsa_cache_shared_module
import sglang.srt.mem_cache.dsa_shared_demand as dsa_shared_demand_module
from sglang.srt.layers.attention.dsa.dsa_indexer import (
    Indexer,
    _full_batch_indexer_cache_inputs,
    _get_index_cache_write_targets,
    _prepare_paged_index_page_table,
    _synchronize_shared_cache_writes,
)
from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod
from sglang.srt.layers.attention.dsa.shared_cache_access import DSASharedCacheAccess
from sglang.srt.layers.attention.dsa_backend import (
    DeepseekSparseAttnBackend,
    DSAFlashMLAMetadata,
    DSAIndexerMetadata,
    _get_pool_main_owner_translation_args,
    _prepare_pool_index_page_table,
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
    ReplicatedIndexKeyCache,
    SharedDSAPageLayout,
    SharedDSATokenToKVPool,
    gather_shared_index_rows,
    indexer_pool_cache_shape,
)
from sglang.srt.mem_cache.dsa_shared_demand import (
    DSAFlashMLADemandCacheManager,
    SharedMLACurrentRows,
    SlotDemandCache,
    _build_slot_demand_cache,
    _expand_flashmla_shared_cache_request_slots,
    get_indexer_pool_cache_layer_ids,
)
from sglang.srt.mem_cache.kv_cache_configurator import calculate_mla_kv_cache_dim
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.mem_cache.shared_kv.demand_cache import PoolDemandCache
from sglang.srt.mem_cache.shared_kv.layout import OwnerShardedLayout
from sglang.srt.mem_cache.shared_kv.synchronization import SharedWritePublisher
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestPoolDemandCache(CustomTestCase):
    def test_pool_layer_storage_epoch_and_accounting(self):
        cache = PoolDemandCache.create(
            keys=(2, 6),
            entries_per_key=3,
            entry_bytes=4,
            dtype=torch.uint8,
            device="cpu",
        )

        layer2_data, layer2_tags = cache.storage_for(2)
        layer6_data, layer6_tags = cache.storage_for(6)

        self.assertEqual(layer2_data.shape, (3, 4))
        self.assertEqual(layer2_tags.shape, (3,))
        self.assertNotEqual(layer2_data.data_ptr(), layer6_data.data_ptr())
        self.assertNotEqual(layer2_tags.data_ptr(), layer6_tags.data_ptr())
        self.assertEqual(cache.allocated_bytes, 2 * 3 * 4 + 2 * 3 * 8 + 4)
        self.assertFalse(hasattr(cache, "nbytes"))
        self.assertEqual(cache.epoch, 1)
        self.assertEqual(cache.epoch_tensor.item(), 1)

        cache.invalidate()

        self.assertEqual(cache.epoch, 2)
        self.assertEqual(cache.epoch_tensor.item(), 2)
        with self.assertRaisesRegex(ValueError, "key 10 is not configured"):
            cache.storage_for(10)

    def test_glm52_pool_keys_cover_only_target_topk_producers(self):
        config = SimpleNamespace(
            architectures=["GlmMoeDsaForCausalLM"],
            index_topk=2048,
            index_head_dim=128,
            index_topk_freq=4,
            index_skip_topk_offset=3,
        )

        self.assertEqual(
            get_indexer_pool_cache_layer_ids(config, 0, 78),
            (
                0,
                1,
                2,
                6,
                10,
                14,
                18,
                22,
                26,
                30,
                34,
                38,
                42,
                46,
                50,
                54,
                58,
                62,
                66,
                70,
                74,
            ),
        )


class TestFullBatchIndexerPoolCacheGeometry(CustomTestCase):
    def test_cp8_keeps_all_requests_and_all_verify_rows_on_every_rank(self):
        page_tables = torch.arange(32 * 5, dtype=torch.int32).view(32, 5)
        lengths = torch.arange(100, 132, dtype=torch.int32)

        pool_pages = page_tables + 1000
        source_pages, target_pages, cache_lengths = _full_batch_indexer_cache_inputs(
            page_tables,
            pool_pages,
            lengths,
            batch_size=8,
            next_n=4,
        )

        torch.testing.assert_close(source_pages, page_tables[::4])
        torch.testing.assert_close(target_pages, pool_pages[::4])
        torch.testing.assert_close(cache_lengths, lengths.view(8, 4)[:, -1])

    def test_incomplete_request_geometry_falls_back(self):
        page_tables = torch.arange(31 * 5, dtype=torch.int32).view(31, 5)
        lengths = torch.arange(31, dtype=torch.int32)

        self.assertIsNone(
            _full_batch_indexer_cache_inputs(
                page_tables,
                page_tables,
                lengths,
                batch_size=8,
                next_n=4,
            )
        )

    def test_single_layer_cache_covers_logical_pool_page_domain(self):
        self.assertEqual(
            indexer_pool_cache_shape(
                index_buf_size=1_070_336,
                page_size=64,
                page_bytes=8_448,
            ),
            (16_725, 8_448),
        )

    def test_mixed_lengths_and_batch_reorder_keep_pool_page_identity(self):
        source = torch.tensor(
            [[100, 107, 111, -1], [100, 103, 104, 108]], dtype=torch.int32
        )
        pool_pages = torch.tensor([[0, 7, 11, -1], [0, 3, 4, 8]], dtype=torch.int32)
        reordered_source = source.flip(0)
        reordered_pool_pages = pool_pages.flip(0)
        cache = object.__new__(
            __import__(
                "sglang.srt.mem_cache.dsa_cache_shared",
                fromlist=["SharedIndexKeyCache"],
            ).SharedIndexKeyCache
        )
        cache.pool = SimpleNamespace(page_size=64)
        cache.get_global_buffer = lambda _layer_id: torch.empty((16, 8))
        cache.pool_cache = SimpleNamespace(
            storage_for=lambda _layer_id: (
                torch.empty((16, 8)),
                torch.zeros(16, dtype=torch.int64),
            ),
            epoch_tensor=torch.ones((), dtype=torch.int32),
        )

        with patch(
            "sglang.srt.mem_cache.dsa_cache_shared.materialize_indexer_pages_triton"
        ) as materialize:
            _, first_table = cache.materialize_pages(
                0,
                source,
                pool_pages,
                torch.tensor([129, 257], dtype=torch.int32),
            )
            _, reordered_table = cache.materialize_pages(
                0,
                reordered_source,
                reordered_pool_pages,
                torch.tensor([257, 129], dtype=torch.int32),
            )

        self.assertIs(first_table, pool_pages)
        self.assertIs(reordered_table, reordered_pool_pages)
        self.assertEqual(first_table[0, 0].item(), first_table[1, 0].item())
        self.assertEqual(materialize.call_count, 2)
        first_call = materialize.call_args_list[0]
        self.assertIs(first_call.args[2], source)
        self.assertIs(first_call.args[3], pool_pages)

    def test_one_indexer_invalidation_advances_one_epoch(self):
        cache = PoolDemandCache.create(
            keys=(0,),
            entries_per_key=4,
            entry_bytes=8,
            dtype=torch.uint8,
            device="cpu",
        )
        pool = SimpleNamespace(
            index_key_cache=SimpleNamespace(invalidate_cache=cache.invalidate)
        )

        SharedDSATokenToKVPool.invalidate_indexer_cache(pool)

        self.assertEqual(cache.epoch, 2)


def _build_mla_current_row_geometry(
    logical_locs: torch.Tensor,
    *,
    rows_per_request: int,
    max_current_rows: int,
    owner_size: int,
    page_size: int,
    pages_per_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference geometry used only to validate the fused production writer."""
    if logical_locs.dim() != 1:
        raise ValueError("Shared MLA current-row locations must be one-dimensional")
    query_rows = logical_locs.numel()
    if rows_per_request <= 0 or query_rows % rows_per_request:
        raise ValueError(
            "Shared MLA current rows require complete request groups: "
            f"query_rows={query_rows} rows_per_request={rows_per_request}"
        )
    if max_current_rows < rows_per_request:
        raise ValueError("Shared MLA per-request row capacity exceeded")
    if owner_size <= 0 or page_size <= 0 or pages_per_rank <= 0:
        raise ValueError("Shared MLA current-row layout dimensions must be positive")
    if query_rows == 0:
        return (
            torch.empty(
                (0, max_current_rows), dtype=torch.int32, device=logical_locs.device
            ),
            torch.empty((0,), dtype=torch.int32, device=logical_locs.device),
        )

    query_indices = torch.arange(query_rows, device=logical_locs.device)
    positions = query_indices.remainder(rows_per_request)
    counts = torch.where(
        logical_locs >= 0, positions + 1, torch.zeros_like(positions)
    ).to(torch.int32)
    current_indices = torch.arange(max_current_rows, device=logical_locs.device)
    source_indices = query_indices.sub(positions).unsqueeze(
        1
    ) + current_indices.unsqueeze(0)
    valid = current_indices.unsqueeze(0) < counts.unsqueeze(1)
    source_locs = logical_locs[source_indices.clamp(max=query_rows - 1)]
    valid &= source_locs >= 0

    pages = torch.div(source_locs, page_size, rounding_mode="floor")
    page_offsets = source_locs.remainder(page_size)
    owners = pages.remainder(owner_size)
    owner_pages = torch.div(pages, owner_size, rounding_mode="floor")
    physical_rows = (owners * pages_per_rank + owner_pages) * page_size + page_offsets
    return (
        torch.where(valid, physical_rows, -1).to(torch.int32),
        counts,
    )


class TestSharedDSAKVCacheDimension(CustomTestCase):
    @staticmethod
    def _model_config():
        return SimpleNamespace(
            hf_config=object(),
            kv_lora_rank=512,
            qk_rope_head_dim=64,
        )

    @staticmethod
    def _server_args(*, shared: bool, decode_backend: str):
        return SimpleNamespace(
            enable_dsa_shared_kv_cache=shared,
            dsa_prefill_backend="flashmla_sparse_q8",
            dsa_decode_backend=decode_backend,
        )

    @patch(
        "sglang.srt.mem_cache.kv_cache_configurator.is_deepseek_dsa",
        return_value=True,
    )
    def test_shared_flashmla_uses_scaled_fp8_layout(self, _):
        dim = calculate_mla_kv_cache_dim(
            model_config=self._model_config(),
            kv_cache_dtype=torch.float8_e4m3fn,
            server_args=self._server_args(shared=True, decode_backend="flashmla_kv"),
        )

        self.assertEqual(dim, 656)

    @patch(
        "sglang.srt.mem_cache.kv_cache_configurator.is_deepseek_dsa",
        return_value=True,
    )
    def test_base_flashmla_retains_scaled_fp8_layout(self, _):
        dim = calculate_mla_kv_cache_dim(
            model_config=self._model_config(),
            kv_cache_dtype=torch.float8_e4m3fn,
            server_args=self._server_args(shared=False, decode_backend="flashmla_kv"),
        )

        self.assertEqual(dim, 656)


def make_shared_dsa_layout(
    cp_size: int,
    page_size: int,
    pages_per_rank: int,
    *,
    local_pages_per_layer: int | None = None,
    rank_stride_pages: int | None = None,
) -> SharedDSAPageLayout:
    return SharedDSAPageLayout(
        OwnerShardedLayout(
            cp_size=cp_size,
            ownership_granule=page_size,
            logical_rows=cp_size * pages_per_rank * page_size,
            physical_blocks_per_rank=rank_stride_pages or pages_per_rank,
        ),
        local_pages_per_layer=local_pages_per_layer,
    )


class TestSharedMLACurrentRows(CustomTestCase):
    @staticmethod
    def _make_write_pool(device: torch.device | str) -> SharedDSATokenToKVPool:
        pool = object.__new__(SharedDSATokenToKVPool)
        pool.start_layer = 0
        pool.shared_rank = 0
        pool.shared_size = 2
        pool.page_size = 64
        pool.size = 256
        pool.dsa_kv_cache_store_fp8 = True
        pool.main_layout = make_shared_dsa_layout(
            2,
            64,
            2,
            local_pages_per_layer=2,
        )
        pool.local_kv_buffer = [
            torch.full((128, 1, 656), 0xCD, dtype=torch.uint8, device=device)
        ]
        return pool

    def test_current_rows_normal_decode_geometry_is_one_rank_major_row_per_query(self):
        physical_rows, counts = _build_mla_current_row_geometry(
            torch.tensor([0, 64], dtype=torch.int64),
            rows_per_request=1,
            max_current_rows=1,
            owner_size=2,
            page_size=64,
            pages_per_rank=2,
        )

        torch.testing.assert_close(
            physical_rows, torch.tensor([[0], [128]], dtype=torch.int32)
        )
        torch.testing.assert_close(counts, torch.tensor([1, 1], dtype=torch.int32))

    def test_current_rows_target_verify_geometry_is_request_major_and_causal(self):
        physical_rows, counts = _build_mla_current_row_geometry(
            torch.tensor([0, 1, 2, 3, 64, 65, 66, 67], dtype=torch.int64),
            rows_per_request=4,
            max_current_rows=4,
            owner_size=2,
            page_size=64,
            pages_per_rank=2,
        )

        self.assertEqual(counts.tolist(), [1, 2, 3, 4, 1, 2, 3, 4])
        self.assertEqual(physical_rows[3].tolist(), [0, 1, 2, 3])
        self.assertEqual(physical_rows[7].tolist(), [128, 129, 130, 131])
        self.assertEqual(physical_rows[0].tolist(), [0, -1, -1, -1])

    def test_current_rows_three_token_target_verify_is_request_major_and_causal(self):
        physical_rows, counts = _build_mla_current_row_geometry(
            torch.tensor([0, 1, 2, 64, 65, 66], dtype=torch.int64),
            rows_per_request=3,
            max_current_rows=3,
            owner_size=2,
            page_size=64,
            pages_per_rank=2,
        )

        self.assertEqual(counts.tolist(), [1, 2, 3, 1, 2, 3])
        self.assertEqual(physical_rows[2].tolist(), [0, 1, 2])
        self.assertEqual(physical_rows[5].tolist(), [128, 129, 130])
        self.assertEqual(physical_rows[0].tolist(), [0, -1, -1])

    def test_current_rows_graph_padding_has_zero_count_and_no_physical_rows(self):
        physical_rows, counts = _build_mla_current_row_geometry(
            torch.tensor([0, 1, -1, -1], dtype=torch.int64),
            rows_per_request=4,
            max_current_rows=4,
            owner_size=2,
            page_size=64,
            pages_per_rank=2,
        )

        self.assertEqual(counts.tolist(), [1, 2, 0, 0])
        self.assertEqual(physical_rows[2:].tolist(), [[-1] * 4, [-1] * 4])

    def test_current_rows_geometry_rejects_partial_groups_and_insufficient_width(self):
        kwargs = dict(owner_size=2, page_size=64, pages_per_rank=2)
        with self.assertRaisesRegex(ValueError, "complete request groups"):
            _build_mla_current_row_geometry(
                torch.tensor([0, 1, 2]),
                rows_per_request=2,
                max_current_rows=2,
                **kwargs,
            )
        with self.assertRaisesRegex(ValueError, "capacity"):
            _build_mla_current_row_geometry(
                torch.tensor([0, 1, 2, 3]),
                rows_per_request=4,
                max_current_rows=2,
                **kwargs,
            )

    def test_current_rows_write_rejects_mismatched_kv_shapes_before_dispatch(self):
        pool = self._make_write_pool("cpu")
        current_rows = SharedMLACurrentRows.create(
            max_query_rows=2, max_current_rows=1, device="cpu"
        )
        loc = torch.tensor([0, 1], dtype=torch.int64)
        good_nope = torch.empty((2, 1, 512), dtype=torch.bfloat16)
        good_rope = torch.empty((2, 1, 64), dtype=torch.bfloat16)

        with self.assertRaisesRegex(ValueError, "NoPE shape"):
            pool.set_mla_kv_buffer_with_current_rows(
                SimpleNamespace(layer_id=0),
                loc,
                good_nope[:1],
                good_rope,
                current_rows,
                query_rows=2,
                rows_per_request=1,
            )
        with self.assertRaisesRegex(ValueError, "RoPE shape"):
            pool.set_mla_kv_buffer_with_current_rows(
                SimpleNamespace(layer_id=0),
                loc,
                good_nope,
                good_rope[..., :32],
                current_rows,
                query_rows=2,
                rows_per_request=1,
            )

    def test_current_rows_target_verify_shadow_has_exact_scaled_fp8_bytes(self):
        if not torch.cuda.is_available():
            self.skipTest("scaled-FP8 current-row test requires CUDA")

        from sglang.kernels.ops.attention.dsa.quant_k_cache import (
            quantize_k_cache_separate,
        )

        device = torch.device("cuda")
        current_rows = SharedMLACurrentRows.create(
            max_query_rows=4,
            max_current_rows=4,
            device=device,
        )
        pool = self._make_write_pool(device)

        loc = torch.tensor([63, 64, 65, 66], dtype=torch.int64, device=device)
        torch.manual_seed(20260809)
        k_nope = torch.randn((4, 1, 512), dtype=torch.bfloat16, device=device)
        k_rope = torch.randn((4, 1, 64), dtype=torch.bfloat16, device=device)
        expected_nope, expected_rope = quantize_k_cache_separate(k_nope, k_rope)
        expected = torch.cat(
            [expected_nope.view(4, -1), expected_rope.view(4, -1)], dim=1
        )

        pool.set_mla_kv_buffer_with_current_rows(
            SimpleNamespace(layer_id=0),
            loc,
            k_nope,
            k_rope,
            current_rows,
            query_rows=4,
            rows_per_request=4,
        )
        torch.cuda.synchronize()

        self.assertEqual(current_rows.counts[:4].cpu().tolist(), [1, 2, 3, 4])
        expected_physical_rows = [63, 128, 129, 130]
        for query_row in range(4):
            count = query_row + 1
            self.assertEqual(
                current_rows.physical_rows[query_row, :count].cpu().tolist(),
                expected_physical_rows[:count],
            )
            torch.testing.assert_close(
                current_rows.encoded_rows[query_row, :count],
                expected[:count],
                rtol=0,
                atol=0,
            )
        expected_owner = torch.full_like(pool.local_kv_buffer[0][:, 0], 0xCD)
        expected_owner[63] = expected[0]
        torch.testing.assert_close(
            pool.local_kv_buffer[0][:, 0], expected_owner, rtol=0, atol=0
        )


class TestSharedDSAPageLayout(CustomTestCase):
    def setUp(self):
        self.layout = make_shared_dsa_layout(4, 4, 3)

    def test_owner_mask_and_local_slots_match(self):
        slots = torch.tensor([0, 4, 8, 12, 16, 20], dtype=torch.int64)

        self.assertEqual(
            self.layout.owned_slot_mask(slots, owner_rank=1).tolist(),
            [False, True, False, False, False, True],
        )
        self.assertEqual(
            self.layout.translate_local_slots(slots).tolist(), [0, 0, 0, 0, 4, 4]
        )

    def test_different_segment_strides_keep_the_same_owner(self):
        main_layout = make_shared_dsa_layout(4, 64, 128)
        index_layout = make_shared_dsa_layout(4, 64, 256)
        slots = torch.arange(0, 16 * 64, 64)

        for owner_rank in range(4):
            self.assertTrue(
                torch.equal(
                    main_layout.owned_slot_mask(slots, owner_rank=owner_rank),
                    index_layout.owned_slot_mask(slots, owner_rank=owner_rank),
                )
            )

    def test_layout_wraps_model_neutral_owner_sharding(self):
        owner_layout = OwnerShardedLayout(
            cp_size=4,
            ownership_granule=64,
            logical_rows=12 * 64,
        )

        layout = SharedDSAPageLayout(owner_layout)

        self.assertIs(layout.owner_layout, owner_layout)
        self.assertEqual(layout.pages_per_rank, 3)


class TestSlotDemandCache(CustomTestCase):
    def test_decode_slot_lifecycle_preserves_stable_request_cache(self):
        tags = torch.zeros((2, 4, 8), dtype=torch.int64)
        lifecycle = dsa_shared_demand_module.FlashMLASharedDecodeSlotLifecycle.create(
            tags=tags, num_request_slots=4
        )
        slots = torch.tensor([1, 3], dtype=torch.int64)
        generations = torch.tensor([7, 11], dtype=torch.int64)
        lifecycle.refresh(
            active_request_slots=slots,
            request_generations=generations,
        )
        tags.fill_(29)

        cleared = lifecycle.refresh(
            active_request_slots=slots,
            request_generations=generations,
        )

        self.assertEqual(cleared, [])
        self.assertTrue(torch.all(tags == 29))

    def test_decode_slot_lifecycle_invalidates_only_reused_request(self):
        tags = torch.zeros((1, 4, 8), dtype=torch.int64)
        lifecycle = dsa_shared_demand_module.FlashMLASharedDecodeSlotLifecycle.create(
            tags=tags, num_request_slots=4
        )
        slots = torch.tensor([1, 2], dtype=torch.int64)
        lifecycle.refresh(
            active_request_slots=slots,
            request_generations=torch.tensor([3, 5], dtype=torch.int64),
        )
        tags[:, 0, :] = 31
        tags[:, 1, :] = 47

        cleared = lifecycle.refresh(
            active_request_slots=slots,
            request_generations=torch.tensor([3, 6], dtype=torch.int64),
        )

        self.assertEqual(cleared, [2])
        self.assertTrue(torch.all(tags[:, 0, :] == 31))
        self.assertTrue(torch.all(tags[:, 1, :] == 0))

    def test_decode_cache_uses_fixed_4k_request_slots(self):
        self.assertEqual(
            dsa_shared_demand_module.FLASHMLA_SHARED_DECODE_ROWS_PER_REQUEST,
            1 << 12,
        )
        self.assertEqual(
            dsa_shared_demand_module.FLASHMLA_SHARED_PREFILL_ROWS,
            1 << 18,
        )

    def test_decode_epoch_advances_once_per_64_steps(self):
        tags = torch.zeros((2, 4, 8), dtype=torch.int64)
        state = dsa_shared_demand_module.FlashMLASharedDecodeGeneration.create(
            device="cpu",
            tags=tags,
            calls_per_generation=(
                dsa_shared_demand_module._FLASHMLA_SHARED_DECODE_CALLS_PER_GENERATION
            ),
        )

        observed = [int(state.advance().item()) for _ in range(65)]

        self.assertEqual(observed[:63], [1] * 63)
        self.assertEqual(observed[63:], [2, 2])

    def test_decode_epoch_wrap_clears_all_tags_before_reuse(self):
        tags = torch.ones((2, 4, 8), dtype=torch.int64)
        state = dsa_shared_demand_module.FlashMLASharedDecodeGeneration.create(
            device="cpu",
            tags=tags,
            max_generation=2,
            calls_per_generation=1,
        )

        self.assertEqual(int(state.advance().item()), 2)
        self.assertEqual(int(state.advance().item()), 1)
        torch.testing.assert_close(tags, torch.zeros_like(tags))

    def test_decode_factory_uses_one_slice_per_request_and_layer(self):
        self.assertEqual(
            dsa_shared_demand_module.FLASHMLA_SHARED_DECODE_ROWS_PER_REQUEST,
            1 << 12,
        )
        pool = SimpleNamespace(
            main_layout=make_shared_dsa_layout(
                8,
                64,
                100,
                local_pages_per_layer=100,
                rank_stride_pages=800,
            ),
            shared_rank=0,
        )

        caches, generation, lifecycle = (
            dsa_shared_demand_module._build_decode_slot_demand_caches(
                pool,
                device="cpu",
                num_layers=2,
                rows_per_request=8,
                num_request_slots=4,
            )
        )

        self.assertEqual(lifecycle.tags.shape, (2, 4, 8))
        self.assertEqual(len(caches), 2)
        self.assertEqual(caches[0].row_cache.shape, (32, 656))
        self.assertEqual(caches[1].row_cache.shape, (32, 656))
        self.assertEqual(caches[0].num_request_slots, 4)
        self.assertEqual(lifecycle.installed_generations.shape, (5,))
        self.assertIs(caches[0].generation_tensor, generation.tensor)
        self.assertIs(caches[1].generation_tensor, generation.tensor)

    def test_decode_request_slots_keep_one_slot_per_query_row(self):
        request_slots = torch.tensor([3, 7], dtype=torch.int64)

        expanded = _expand_flashmla_shared_cache_request_slots(
            request_slots, num_query_rows=2
        )

        torch.testing.assert_close(expanded, torch.tensor([3, 7]))

    def test_target_verify_request_slots_repeat_in_request_major_order(self):
        request_slots = torch.tensor([3, 7], dtype=torch.int64)

        expanded = _expand_flashmla_shared_cache_request_slots(
            request_slots, num_query_rows=6
        )

        torch.testing.assert_close(expanded, torch.tensor([3, 3, 3, 7, 7, 7]))

    def test_request_slot_expansion_rejects_partial_request_rows(self):
        with self.assertRaisesRegex(ValueError, "multiple of request count"):
            _expand_flashmla_shared_cache_request_slots(
                torch.tensor([3, 7], dtype=torch.int64), num_query_rows=5
            )

    def test_factory_uses_request_scoped_direct_map_and_rank_major_range(self):
        pool = SimpleNamespace(
            main_layout=make_shared_dsa_layout(
                8,
                64,
                100,
                local_pages_per_layer=100,
                rank_stride_pages=800,
            ),
            shared_rank=3,
        )

        state = _build_slot_demand_cache(
            pool,
            device="cpu",
            enabled=True,
            rows_per_request=8,
            num_request_slots=2,
        )
        request_slots = torch.tensor([3, 7], dtype=torch.int64)

        self.assertIsNotNone(state)
        self.assertEqual(state.row_cache.shape, (16, 656))
        self.assertEqual(state.tags.shape, (2, 8))
        self.assertEqual(state.rows_per_request, 8)
        self.assertEqual(state.num_request_slots, 2)
        self.assertEqual(state.local_row_begin, 3 * 800 * 64)
        self.assertEqual(state.local_row_end, 3 * 800 * 64 + 100 * 64)
        self.assertEqual(
            state.next_call_kwargs(request_slots=request_slots),
            {
                "shared_kv_row_cache": state.row_cache,
                "shared_kv_cache_tags": state.tags,
                "shared_kv_request_slots": request_slots,
                "shared_kv_cache_rows_per_request": 8,
                "shared_kv_num_request_slots": 2,
                "shared_kv_cache_epoch": 1,
                "shared_kv_cache_generation_tensor": None,
                "shared_kv_local_row_begin": 3 * 800 * 64,
                "shared_kv_local_row_end": 3 * 800 * 64 + 100 * 64,
            },
        )

    def test_persistent_cache_keeps_one_epoch_across_decode_steps(self):
        pool = SimpleNamespace(
            main_layout=make_shared_dsa_layout(
                8,
                64,
                100,
                local_pages_per_layer=100,
                rank_stride_pages=800,
            ),
            shared_rank=0,
        )
        state = _build_slot_demand_cache(
            pool,
            device="cpu",
            enabled=True,
            rows_per_request=8,
        )

        epochs = [
            state.next_call_kwargs(persistent=True)["shared_kv_cache_epoch"]
            for _ in range(65)
        ]

        self.assertEqual(epochs, [1] * 65)

    def test_factory_is_disabled_for_non_shared_pool(self):
        state = _build_slot_demand_cache(
            SimpleNamespace(main_layout=None),
            device="cpu",
            enabled=True,
            rows_per_request=8,
        )

        self.assertIsNone(state)


class TestSharedDSATokenToKVPoolHelpers(CustomTestCase):
    def test_capacity_ledger_reports_logical_and_per_rank_physical_storage(self):
        pool = SimpleNamespace(
            size=1000,
            page_size=64,
            shared_size=8,
            share_indexer=True,
            main_family=SimpleNamespace(
                accounting=lambda: SimpleNamespace(
                    minimum_blocks_per_rank=3,
                    physical_blocks_per_rank=30,
                    mapped_bytes_per_rank=1100,
                )
            ),
            index_key_cache=SimpleNamespace(
                family=SimpleNamespace(
                    accounting=lambda: SimpleNamespace(
                        minimum_blocks_per_rank=4,
                        physical_blocks_per_rank=40,
                        mapped_bytes_per_rank=1300,
                    )
                ),
                pool_cache=SimpleNamespace(allocated_bytes=310),
            ),
            shared_write_publisher=SimpleNamespace(mapped_bytes_per_rank=17),
        )

        with self.assertLogs(
            "sglang.srt.mem_cache.dsa_cache_shared", level="INFO"
        ) as captured:
            accounting = dsa_cache_shared_module.log_shared_dsa_capacity_accounting(
                pool, main_demand_workspace_bytes=700
            )

        self.assertEqual(
            accounting,
            dsa_cache_shared_module.SharedDSACapacityAccounting(
                logical_tokens=1000,
                logical_pages=16,
                cp_size=8,
                main_physical_token_slots_per_layer_per_rank=192,
                indexer_mode="shared",
                indexer_physical_token_slots_per_layer_per_rank=256,
                authoritative_bytes_per_rank=2417,
                indexer_demand_bytes_per_rank=310,
                main_demand_bytes_per_rank=700,
                tracked_total_bytes_per_rank=3427,
            ),
        )
        self.assertEqual(len(captured.records), 1)
        self.assertIn("logical_tokens=1000", captured.output[0])
        self.assertIn(
            "main_physical_token_slots_per_layer_per_rank=192", captured.output[0]
        )
        self.assertIn("tracked_total_bytes_per_rank=3427", captured.output[0])

    def test_main_demand_workspace_accounting_counts_each_tensor_once(self):
        prefill = SlotDemandCache(
            row_cache=torch.empty((2, 5), dtype=torch.uint8),
            tags=torch.empty((2,), dtype=torch.int64),
            local_row_begin=0,
            local_row_end=2,
            rows_per_request=2,
            num_request_slots=1,
        )
        decode_tags = torch.empty((1, 2), dtype=torch.int64)
        decode = SlotDemandCache(
            row_cache=torch.empty((2, 5), dtype=torch.uint8),
            tags=decode_tags,
            local_row_begin=0,
            local_row_end=2,
            rows_per_request=2,
            num_request_slots=1,
            generation_tensor=torch.empty((), dtype=torch.int32),
        )
        current = SharedMLACurrentRows(
            encoded_rows=torch.empty((1, 1, 5), dtype=torch.uint8),
            physical_rows=torch.empty((1, 1), dtype=torch.int32),
            counts=torch.empty((1,), dtype=torch.int32),
        )
        manager = DSAFlashMLADemandCacheManager(
            prefill_cache=prefill,
            decode_caches=[decode],
            decode_generation=SimpleNamespace(tensor=decode.generation_tensor),
            decode_lifecycle=None,
            current_rows_by_layer=[current],
            max_current_rows=1,
        )

        self.assertEqual(manager.allocated_bytes, 69)

    def test_capacity_ledger_reports_replicated_indexer_storage(self):
        pool = SimpleNamespace(
            size=1000,
            page_size=64,
            shared_size=8,
            share_indexer=False,
            main_family=SimpleNamespace(
                accounting=lambda: SimpleNamespace(
                    minimum_blocks_per_rank=3,
                    physical_blocks_per_rank=30,
                    mapped_bytes_per_rank=1100,
                )
            ),
            index_key_cache=SimpleNamespace(
                buffer=[
                    torch.empty((17, 2), dtype=torch.uint8),
                    torch.empty((17, 2), dtype=torch.uint8),
                ]
            ),
            shared_write_publisher=SimpleNamespace(mapped_bytes_per_rank=17),
        )

        accounting = dsa_cache_shared_module.log_shared_dsa_capacity_accounting(
            pool, main_demand_workspace_bytes=700
        )

        self.assertEqual(accounting.indexer_mode, "replicated")
        self.assertEqual(
            accounting.indexer_physical_token_slots_per_layer_per_rank, 1088
        )
        self.assertEqual(accounting.authoritative_bytes_per_rank, 1185)
        self.assertEqual(accounting.indexer_demand_bytes_per_rank, 0)
        self.assertEqual(accounting.tracked_total_bytes_per_rank, 1885)

    def test_replicated_indexer_factory_skips_shared_vmm_and_demand_cache(self):
        pool = object.__new__(SharedDSATokenToKVPool)
        pool.share_indexer = False
        pool.index_buf_size = 256

        with patch(
            "sglang.srt.mem_cache.dsa_cache_shared.ReplicatedIndexKeyCache"
        ) as replicated:
            cache = SharedDSATokenToKVPool._create_index_key_cache(pool)

        replicated.assert_called_once_with(pool, 256)
        self.assertIs(cache, replicated.return_value)

    def test_replicated_indexer_snapshots_all_sources_before_overlap_restore(self):
        pool = SimpleNamespace(page_size=2, index_head_dim=4, quant_block_size=4)
        cache = object.__new__(ReplicatedIndexKeyCache)
        cache.pool = pool
        cache.buffer = [torch.arange(32, dtype=torch.uint8).view(2, 16)]
        targets = torch.tensor([0, 2], dtype=torch.int64)
        sources = torch.tensor([2, 0], dtype=torch.int64)

        owned_targets, snapshots = cache.snapshot_move(targets, sources)

        torch.testing.assert_close(owned_targets, targets)
        expected_k, expected_scale = gather_shared_index_rows(
            pool, cache.buffer[0], sources
        )
        torch.testing.assert_close(snapshots[0][0], expected_k)
        torch.testing.assert_close(snapshots[0][1], expected_scale)

        with patch(
            "sglang.srt.mem_cache.dsa_cache_shared.index_buf_accessor.SetKAndS.execute"
        ) as restore:
            cache.restore_move(owned_targets, snapshots)

        restore.assert_called_once()
        self.assertIs(restore.call_args.kwargs["buf"], cache.buffer[0])
        torch.testing.assert_close(restore.call_args.kwargs["loc"], targets)
        torch.testing.assert_close(
            restore.call_args.kwargs["index_k"], expected_k.view(torch.float8_e4m3fn)
        )
        torch.testing.assert_close(
            restore.call_args.kwargs["index_k_scale"],
            expected_scale.view(torch.float32),
        )

    def test_indexer_write_target_is_owner_local_shard(self):
        local = torch.empty((8, 8), dtype=torch.uint8)
        pool = SimpleNamespace(
            start_layer=0,
            shared_rank=2,
            shared_size=4,
            index_key_cache=SimpleNamespace(get_local_buffer=lambda _layer: local),
        )

        targets = SharedDSATokenToKVPool.get_index_k_write_targets(pool, 0)

        self.assertEqual(targets, ((local, 2, 4),))

    def test_replicated_indexer_write_target_uses_full_local_buffer(self):
        local = torch.empty((8, 8), dtype=torch.uint8)
        pool = SimpleNamespace(
            start_layer=0,
            shared_rank=2,
            shared_size=4,
            share_indexer=False,
            index_key_cache=SimpleNamespace(get_local_buffer=lambda _layer: local),
        )

        targets = SharedDSATokenToKVPool.get_index_k_write_targets(pool, 0)

        self.assertEqual(targets, ((local, 0, 1),))

    def test_replicated_indexer_keeps_native_page_table_and_disables_demand(self):
        pool = SimpleNamespace(share_indexer=False, index_layout=None)
        pool.shared_cache_access = DSASharedCacheAccess(pool)
        pages = torch.tensor([[0, 4, -1]], dtype=torch.int32)

        prepared = SharedDSATokenToKVPool.prepare_paged_index_page_table(pool, pages)

        self.assertIs(prepared, pages)
        self.assertFalse(pool.shared_cache_access.uses_shared_indexer)

    def test_kv_size_counts_main_indexer_and_publication_mappings(self):
        pool = SimpleNamespace(
            main_family=SimpleNamespace(
                accounting=lambda: SimpleNamespace(mapped_bytes_per_rank=11)
            ),
            shared_write_publisher=SimpleNamespace(mapped_bytes_per_rank=17),
            index_key_cache=SimpleNamespace(
                family=SimpleNamespace(
                    accounting=lambda: SimpleNamespace(mapped_bytes_per_rank=13)
                ),
                pool_cache=SimpleNamespace(allocated_bytes=31),
            ),
        )

        self.assertEqual(SharedDSATokenToKVPool.get_kv_size_bytes(pool), 72)

    def test_kv_size_counts_full_replicated_indexer_buffers(self):
        index_buffers = [
            torch.empty((3, 5), dtype=torch.uint8),
            torch.empty((2, 7), dtype=torch.uint8),
        ]
        pool = SimpleNamespace(
            share_indexer=False,
            main_family=SimpleNamespace(
                accounting=lambda: SimpleNamespace(mapped_bytes_per_rank=11)
            ),
            shared_write_publisher=SimpleNamespace(mapped_bytes_per_rank=17),
            index_key_cache=SimpleNamespace(buffer=index_buffers),
        )

        self.assertEqual(SharedDSATokenToKVPool.get_kv_size_bytes(pool), 57)

    def test_clear_buffers_closes_publisher_main_and_local_index_cache(self):
        events = []
        publisher = SimpleNamespace(close=lambda: events.append("publisher"))
        main_family = SimpleNamespace(close=lambda: events.append("main"))
        index_key_cache = SimpleNamespace(clear=lambda: events.append("index"))
        pool = SimpleNamespace(
            kv_buffer=["global-main"],
            local_kv_buffer=["local-main"],
            main_family=main_family,
            index_key_cache=index_key_cache,
            shared_write_publisher=publisher,
        )

        SharedDSATokenToKVPool._clear_buffers(pool)

        self.assertEqual(events, ["publisher", "main", "index"])
        self.assertEqual(pool.kv_buffer, [])
        self.assertEqual(pool.local_kv_buffer, [])
        self.assertIsNone(pool.shared_write_publisher)
        self.assertIsNone(pool.main_family)

        SharedDSATokenToKVPool._clear_buffers(pool)
        self.assertEqual(events, ["publisher", "main", "index"])

    def test_main_family_preserves_allocation_contract_and_publisher_order(self):
        events = []
        family = SimpleNamespace(
            slab=SimpleNamespace(
                global_views=[torch.empty((4 * 6 * 64, 1, 656), dtype=torch.uint8)],
                local_views=[torch.empty((6 * 64, 1, 656), dtype=torch.uint8)],
            ),
            layout=OwnerShardedLayout(
                cp_size=4,
                ownership_granule=64,
                logical_rows=4 * 6 * 64,
                physical_blocks_per_rank=6,
            ),
        )
        pool = SimpleNamespace(
            size=1000,
            page_size=64,
            shared_size=4,
            layer_num=2,
            kv_cache_dim=656,
            store_dtype=torch.uint8,
            device="cpu",
            shared_rank=0,
            memory_saver_adapter=SimpleNamespace(
                region=lambda _memory_type: nullcontext()
            ),
            _get_cp_group=lambda: SimpleNamespace(cpu_group="cpu"),
        )

        def create_family(**_kwargs):
            events.append("family")
            return family

        def create_publisher(_group):
            events.append("publisher")
            return "publisher"

        with (
            patch(
                "sglang.srt.mem_cache.dsa_cache_shared.OwnerShardedFamily.create",
                side_effect=create_family,
            ) as create,
            patch(
                "sglang.srt.mem_cache.dsa_cache_shared.SharedWritePublisher",
                side_effect=create_publisher,
            ),
        ):
            SharedDSATokenToKVPool._create_buffers(pool)

        spec = create.call_args.kwargs["spec"]
        self.assertEqual(events, ["family", "publisher"])
        self.assertEqual(spec.num_layers, 2)
        self.assertEqual(spec.logical_rows_per_layer, 4 * 6 * 64)
        self.assertEqual(spec.ownership_granule, 64)
        self.assertEqual(spec.storage_rows_per_granule, 64)
        self.assertEqual(spec.row_shape, (1, 656))
        self.assertEqual(spec.dtype, torch.uint8)
        self.assertFalse(spec.map_rank_local)
        self.assertFalse(create.call_args.kwargs["zero_initialize"])
        self.assertEqual(pool.shared_write_publisher, "publisher")

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

    def test_shared_index_pages_translate_to_rank_major_vmm(self):
        pool = SimpleNamespace(
            shared_rank=2,
            index_layout=make_shared_dsa_layout(4, 1, 256).owner_layout,
        )
        pages = torch.tensor([0, 1, 4, 5, -1], dtype=torch.int32)

        translated = SharedDSATokenToKVPool.translate_index_pages(pool, pages)

        self.assertEqual(translated.tolist(), [512, 768, 513, 769, -1])

    def test_shared_index_slots_translate_page_owner_and_keep_token_offset(self):
        pool = SimpleNamespace(
            page_size=64,
            shared_rank=2,
            index_layout=make_shared_dsa_layout(4, 1, 256).owner_layout,
        )
        slots = torch.tensor([0, 64, 256, 320, -1], dtype=torch.int64)

        translated = SharedDSATokenToKVPool.translate_index_slots(pool, slots)

        self.assertEqual(translated.tolist(), [32768, 49152, 32832, 49216, -1])

    def test_eagle_move_copies_main_and_indexer_only_on_destination_owner(self):
        page_size = 4
        main_layout = make_shared_dsa_layout(2, page_size, 4)
        index_layout = make_shared_dsa_layout(2, 1, 4).owner_layout
        local_main = torch.zeros((16, 1, 1), dtype=torch.uint8)
        shared_main = torch.zeros((32, 1, 1), dtype=torch.uint8)
        tgt_loc = torch.tensor([1], dtype=torch.int64)
        src_loc = torch.tensor([5], dtype=torch.int64)
        index_cache = MagicMock()
        index_cache.snapshot_move.return_value = (tgt_loc, [(object(), object())])
        publications = []

        pool = SimpleNamespace(
            size=32,
            page_size=page_size,
            index_head_dim=128,
            quant_block_size=128,
            shared_rank=0,
            main_layout=main_layout,
            index_layout=index_layout,
            local_kv_buffer=[local_main],
            kv_buffer=[shared_main],
            index_key_cache=index_cache,
            synchronize_shared_writes=lambda: publications.append("publish"),
        )
        shared_main[main_layout.translate_slots(src_loc)] = 11

        SharedDSATokenToKVPool.move_kv_cache(pool, tgt_loc, src_loc)

        self.assertEqual(local_main[1].item(), 11)
        index_cache.snapshot_move.assert_called_once_with(tgt_loc, src_loc)
        index_cache.restore_move.assert_called_once_with(
            tgt_loc, index_cache.snapshot_move.return_value[1]
        )
        self.assertEqual(publications, ["publish", "publish"])

    def test_write_fence_uses_model_neutral_publisher(self):
        publisher = MagicMock(spec=SharedWritePublisher)
        pool = SimpleNamespace(shared_write_publisher=publisher)

        SharedDSATokenToKVPool.synchronize_shared_writes(pool)

        publisher.publish.assert_called_once_with()

    def test_indexer_runs_one_shared_write_fence_before_read(self):
        calls = []
        pool = SimpleNamespace(
            share_indexer=True,
            synchronize_shared_writes=lambda: calls.append("fence"),
        )
        pool.shared_cache_access = DSASharedCacheAccess(pool)

        _synchronize_shared_cache_writes(pool)

        self.assertEqual(calls, ["fence"])

    def test_replicated_indexer_has_no_shared_write_fence(self):
        calls = []
        pool = SimpleNamespace(
            share_indexer=False,
            synchronize_shared_writes=lambda: calls.append("shared"),
        )
        pool.shared_cache_access = DSASharedCacheAccess(pool)

        _synchronize_shared_cache_writes(pool)

        self.assertEqual(calls, [])

    def test_paged_indexer_uses_global_vmm_view_and_translated_page_table(self):
        calls = []
        expected = torch.zeros((2, 8), dtype=torch.uint8)
        pool = SimpleNamespace(
            shared_rank=2,
            layer_transfer_counter=None,
            start_layer=0,
            index_layout=make_shared_dsa_layout(4, 1, 256).owner_layout,
            index_key_cache=SimpleNamespace(
                get_global_buffer=lambda _layer_id: expected
            ),
            synchronize_shared_writes=lambda: calls.append("fence"),
        )
        pool.shared_cache_access = DSASharedCacheAccess(pool)
        pages = torch.tensor([0, 1, 2, 3, 4, -1], dtype=torch.int32)

        translated = SharedDSATokenToKVPool.prepare_paged_index_page_table(pool, pages)
        actual = SharedDSATokenToKVPool.get_paged_index_k_with_scale_buffer(pool, 0)

        self.assertEqual(translated.tolist(), [512, 768, 0, 256, 513, -1])
        self.assertIs(actual, expected)
        self.assertEqual(calls, [])

    def test_indexer_write_targets_use_only_owner_local_view(self):
        owner_local = torch.empty((8, 8), dtype=torch.uint8)
        pool = SimpleNamespace(
            start_layer=0,
            shared_rank=2,
            shared_size=4,
            index_key_cache=SimpleNamespace(
                get_local_buffer=lambda _layer_id: owner_local
            ),
        )

        targets = SharedDSATokenToKVPool.get_index_k_write_targets(pool, 0)

        self.assertEqual(targets, ((owner_local, 2, 4),))

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
        pool.shared_cache_access = DSASharedCacheAccess(pool)

        actual = SharedDSATokenToKVPool.get_key_buffer(pool, 0)

        self.assertIs(actual, global_view)
        self.assertEqual(calls, [])

        _synchronize_pool_main_cache(pool)

        self.assertEqual(calls, ["fence"])

    def test_attention_translates_main_slots_to_canonical_layout(self):
        pool = SimpleNamespace(
            main_layout=make_shared_dsa_layout(4, 64, 3),
        )
        slots = torch.tensor([0, 64, 128, 192, 256, -1], dtype=torch.int64)

        translated = SharedDSATokenToKVPool.translate_main_slots(pool, slots)

        self.assertEqual(translated.tolist(), [0, 192, 384, 576, 64, -1])


class TestSharedDSAIntegrationHooks(CustomTestCase):
    @staticmethod
    def _flashmla_backend(*, kv_cache_dim: int, stores_scaled_fp8: bool):
        backend = object.__new__(DeepseekSparseAttnBackend)
        backend.real_page_size = 64
        backend.kv_cache_dim = kv_cache_dim
        backend.dsa_kv_cache_store_fp8 = stores_scaled_fp8
        backend.flashmla_kv_num_q_heads = 64
        backend.dsa_index_topk = 2048
        return backend

    def test_decode_forwards_shared_demand_cache_to_flashmla(self):
        backend = object.__new__(DeepseekSparseAttnBackend)
        backend.dsa_decode_impl = "flashmla_kv"
        request_slots = torch.tensor([3], dtype=torch.int64)
        backend.forward_metadata = SimpleNamespace(
            shared_cache_request_slots=request_slots,
            shared_mla_current_rows=None,
            shared_mla_current_rows_per_request=0,
        )
        kv_cache = torch.empty((64, 1, 656), dtype=torch.uint8)
        backend.token_to_kv_pool = SimpleNamespace(
            start_layer=0,
            synchronize_shared_writes=lambda: None,
            get_key_buffer=lambda _: kv_cache,
        )
        backend.hisparse_coordinator = None
        backend.use_fused_topk = True
        backend._get_fused_topk_page_table = lambda _: torch.zeros(
            (1, 2048), dtype=torch.int32
        )
        backend._pad_topk_indices = lambda indices, _: indices
        decode_cache = object()
        backend._shared_main_demand_cache = SimpleNamespace(
            has_decode_cache=True,
            cache_for_layer=lambda **_: (decode_cache, True),
        )
        observed = {}

        def run_flashmla(**kwargs):
            observed.update(kwargs)
            return torch.tensor([7])

        backend._forward_flashmla_kv = run_flashmla
        layer = SimpleNamespace(
            is_cross_attention=False,
            layer_id=0,
            tp_q_head_num=16,
            v_head_dim=512,
            head_dim=576,
            scaling=576**-0.5,
        )
        q = torch.empty((1, 16, 576), dtype=torch.bfloat16)
        topk_indices = torch.zeros((1, 2048), dtype=torch.int32)

        with patch(
            "sglang.srt.layers.attention.dsa_backend.concat_mla_absorb_q_general",
            side_effect=lambda q_nope, q_rope: torch.cat([q_nope, q_rope], dim=-1),
        ):
            result = backend.forward_decode(
                q=q,
                k=None,
                v=None,
                layer=layer,
                forward_batch=SimpleNamespace(req_pool_indices=request_slots),
                topk_indices=topk_indices,
            )

        torch.testing.assert_close(result, torch.tensor([7]))
        self.assertIs(observed["shared_demand_cache"], decode_cache)
        self.assertTrue(observed["persistent"])
        self.assertIs(observed["shared_cache_request_slots"], request_slots)

    def test_decode_graph_replay_refreshes_current_row_locations(self):
        backend = object.__new__(DeepseekSparseAttnBackend)
        backend.real_page_size = 1
        backend.device = "cpu"
        backend.dsa_index_topk = 2
        backend.dsa_decode_impl = "test"
        backend.speculative_num_draft_tokens = 4
        backend.req_to_token = torch.tensor([[41, 42], [51, 52]], dtype=torch.int64)
        backend.set_dsa_prefill_impl = lambda forward_batch: None

        capture_locs = torch.full((8,), 7, dtype=torch.int64)
        page_table = torch.zeros((8, 2), dtype=torch.int64)
        metadata = SimpleNamespace(
            cache_seqlens_int32=torch.zeros(2, dtype=torch.int32),
            cu_seqlens_k=torch.zeros(3, dtype=torch.int32),
            page_table_1=page_table,
            dsa_cache_seqlens_int32=torch.zeros(8, dtype=torch.int32),
            dsa_cu_seqlens_k=torch.zeros(9, dtype=torch.int32),
            dsa_seqlens_expanded=torch.zeros(8, dtype=torch.int32),
            shared_cache_request_slots=None,
            shared_mla_current_row_locs=capture_locs,
            page_size=1,
            real_page_table=page_table,
            prepared_paged_index_page_table=page_table,
        )
        backend.decode_cuda_graph_metadata = {2: metadata}
        replay_locs = torch.tensor([99, 100, 101, 102], dtype=torch.int64)

        expanded_lens = torch.tensor([3, 4, 5, 6, 2, 3, 4, 5], dtype=torch.int32)
        with (
            patch.object(dsa_backend_module, "is_cuda", return_value=False),
            patch.object(
                dsa_backend_module,
                "seqlens_expand_triton",
                return_value=expanded_lens,
            ),
        ):
            backend._apply_cuda_graph_metadata(
                bs=2,
                req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
                seq_lens=torch.tensor([2, 1], dtype=torch.int32),
                seq_lens_cpu=torch.tensor([2, 1], dtype=torch.int32),
                forward_mode=dsa_backend_module.ForwardMode.TARGET_VERIFY,
                spec_info=None,
                out_cache_loc=replay_locs,
            )

        torch.testing.assert_close(
            metadata.shared_mla_current_row_locs,
            torch.tensor([99, 100, 101, 102, 0, 0, 0, 0], dtype=torch.int64),
        )

    def test_base_flashmla_keeps_scaled_layout_and_legacy_scheduler(self):
        calls = []

        def flash_mla_with_kvcache(**kwargs):
            calls.append(kwargs)
            return torch.empty((1, 1, 64, 512), dtype=torch.bfloat16), None

        cache_seqlens = torch.tensor([2048], dtype=torch.int32)
        legacy_metadata = torch.tensor([1], dtype=torch.int32)
        legacy_splits = torch.tensor([2], dtype=torch.int32)
        metadata = SimpleNamespace(
            dsa_cache_seqlens_int32=cache_seqlens,
            flashmla_metadata=DSAFlashMLAMetadata(
                flashmla_metadata=legacy_metadata,
                num_splits=legacy_splits,
            ),
        )
        backend = self._flashmla_backend(kv_cache_dim=656, stores_scaled_fp8=True)

        with patch.dict(
            "sys.modules",
            {
                "sgl_kernel.flash_mla": SimpleNamespace(
                    flash_mla_with_kvcache=flash_mla_with_kvcache
                )
            },
        ):
            backend._forward_flashmla_kv(
                q_all=torch.empty((1, 64, 576), dtype=torch.bfloat16),
                kv_cache=torch.empty((64, 1, 656), dtype=torch.uint8),
                v_head_dim=512,
                sm_scale=576**-0.5,
                layer=SimpleNamespace(tp_q_head_num=64, head_dim=576),
                metadata=metadata,
                page_table_1=torch.zeros((1, 2048), dtype=torch.int32),
            )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["k_cache"].shape, (1, 64, 1, 656))
        self.assertIs(calls[0]["cache_seqlens"], cache_seqlens)
        self.assertIs(calls[0]["tile_scheduler_metadata"], legacy_metadata)
        self.assertIs(calls[0]["num_splits"], legacy_splits)
        self.assertEqual(calls[0]["block_table"].shape, (1, 0))

    def test_base_flashmla_metadata_stays_legacy(self):
        calls = []

        def get_mla_metadata(**kwargs):
            calls.append(kwargs)
            return (
                torch.tensor([1], dtype=torch.int32),
                torch.tensor([2], dtype=torch.int32),
            )

        backend = object.__new__(DeepseekSparseAttnBackend)
        backend.flashmla_kv_num_q_heads = 64
        backend.dsa_index_topk = 2048
        backend.token_to_kv_pool = SimpleNamespace()

        with patch.dict(
            "sys.modules",
            {
                "sgl_kernel.flash_mla": SimpleNamespace(
                    get_mla_metadata=get_mla_metadata
                )
            },
        ):
            metadata = backend._compute_flashmla_metadata(
                cache_seqlens=torch.tensor([2048], dtype=torch.int32),
                seq_len_q=1,
            )

        self.assertEqual(len(calls), 1)
        self.assertNotEqual(calls[0], {})

    def test_flashmla_trims_cp_padding_to_query_rows(self):
        calls = []

        def flash_mla_with_kvcache(**kwargs):
            calls.append(kwargs)
            return torch.empty((6, 1, 64, 512), dtype=torch.bfloat16), None

        metadata = SimpleNamespace(
            dsa_cache_seqlens_int32=torch.tensor(
                [6, 6, 6, 6, 6, 6, 0, 0], dtype=torch.int32
            ),
            flashmla_metadata=DSAFlashMLAMetadata(
                flashmla_metadata=torch.tensor([17], dtype=torch.int32),
                num_splits=torch.arange(9, dtype=torch.int32),
            ),
        )
        backend = self._flashmla_backend(kv_cache_dim=656, stores_scaled_fp8=True)

        with patch.dict(
            "sys.modules",
            {
                "sgl_kernel.flash_mla": SimpleNamespace(
                    flash_mla_with_kvcache=flash_mla_with_kvcache
                )
            },
        ):
            backend._forward_flashmla_kv(
                q_all=torch.empty((6, 64, 576), dtype=torch.bfloat16),
                kv_cache=torch.empty((64, 1, 656), dtype=torch.uint8),
                v_head_dim=512,
                sm_scale=576**-0.5,
                layer=SimpleNamespace(tp_q_head_num=64, head_dim=576),
                metadata=metadata,
                page_table_1=torch.zeros((6, 2048), dtype=torch.int32),
            )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["cache_seqlens"].tolist(), [6] * 6)
        self.assertEqual(calls[0]["num_splits"].tolist(), list(range(7)))

    def test_indexer_keeps_existing_read_buffer_helper(self):
        self.assertTrue(hasattr(Indexer, "_get_index_k_read_buffer"))

    def test_index_write_targets_delegate_to_pool(self):
        expected = ((object(), 3, 8), (object(), 0, 1))
        pool = SimpleNamespace(get_index_k_write_targets=lambda _layer_id: expected)

        self.assertEqual(_get_index_cache_write_targets(pool, 12), expected)

    def test_page_tables_delegate_to_pool(self):
        pool = SimpleNamespace(
            prepare_paged_index_page_table=lambda table: table + 15,
            main_layout=make_shared_dsa_layout(2, 1, 8),
        )
        pool.shared_cache_access = DSASharedCacheAccess(pool)
        table = torch.tensor([[1, -1]], dtype=torch.int32)

        self.assertEqual(
            _prepare_paged_index_page_table(pool, table).tolist(), [[16, 14]]
        )
        with patch(
            "sglang.srt.layers.attention.dsa_backend.translate_owner_sharded_slots",
            return_value=torch.tensor([[8, -1]], dtype=torch.int32),
        ) as translate:
            translated = _translate_pool_main_page_table(pool, table)
        self.assertEqual(translated.tolist(), [[8, -1]])
        translate.assert_called_once()
        self.assertEqual(
            _prepare_pool_index_page_table(pool, table).tolist(), [[16, 14]]
        )

    def test_plain_pool_is_unchanged(self):
        plain_buffer = object()
        pool = SimpleNamespace(
            get_index_k_with_scale_buffer=lambda layer_id: plain_buffer
        )
        table = torch.tensor([[3]], dtype=torch.int32)

        self.assertEqual(
            _get_index_cache_write_targets(pool, 7), ((plain_buffer, 0, 1),)
        )
        self.assertIs(_prepare_paged_index_page_table(pool, table), table)
        self.assertIs(_prepare_pool_index_page_table(pool, table), table)
        self.assertIs(_translate_pool_main_page_table(pool, table), table)

    def test_main_owner_translation_args_follow_shared_layout(self):
        pool = SimpleNamespace(main_layout=make_shared_dsa_layout(8, 64, 1024))
        pool.shared_cache_access = DSASharedCacheAccess(pool)

        self.assertEqual(
            _get_pool_main_owner_translation_args(pool),
            {
                "owner_cp_size": 8,
                "owner_page_size": 64,
                "owner_pages_per_rank": 1024,
            },
        )

    def test_main_owner_translation_does_not_mutate_reused_topk(self):
        pool = SimpleNamespace(main_layout=make_shared_dsa_layout(8, 64, 1024))
        pool.shared_cache_access = DSASharedCacheAccess(pool)
        topk = torch.tensor([[0, 64, -1]], dtype=torch.int32)
        original = topk.clone()

        def translate(slots, *, result, **_kwargs):
            self.assertIsNot(result, slots)
            result.copy_(slots + 100)
            return result

        with patch(
            "sglang.srt.layers.attention.dsa_backend.translate_owner_sharded_slots",
            side_effect=translate,
        ):
            translated = _translate_pool_main_page_table(pool, topk)

        self.assertIsNot(translated, topk)
        torch.testing.assert_close(topk, original)
        torch.testing.assert_close(translated, original + 100)

    def test_plain_pool_has_no_fused_owner_translation(self):
        self.assertIsNone(_get_pool_main_owner_translation_args(SimpleNamespace()))

    def test_indexer_metadata_reuses_prepared_page_table(self):
        logical = torch.tensor([[3]], dtype=torch.int32)
        prepared = torch.tensor([[9]], dtype=torch.int32)
        metadata = DSAIndexerMetadata(
            attn_metadata=SimpleNamespace(
                real_page_table=logical,
                prepared_paged_index_page_table=prepared,
            ),
            topk_transform_method=TopkTransformMethod.PAGED,
        )

        self.assertIs(metadata.get_page_table_64(), logical)
        self.assertIs(metadata.get_prepared_paged_index_page_table(), prepared)


class TestSharedDSAPoolSelection(CustomTestCase):
    def setUp(self):
        self.runner = SimpleNamespace(
            is_draft_worker=False,
            use_mla_backend=True,
            server_args=SimpleNamespace(
                enable_dsa_shared_kv_cache=True,
                enable_dsa_cache_layer_split=False,
                enable_hisparse=False,
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
    @patch(
        "sglang.srt.model_executor.pool_configurator.get_memory",
        return_value=SimpleNamespace(enable_hisparse=False),
    )
    @patch(
        "sglang.srt.model_executor.pool_configurator.get_dsa_index_head_dim",
        return_value=128,
    )
    @patch(
        "sglang.srt.model_executor.pool_configurator.is_deepseek_dsa",
        return_value=True,
    )
    def test_shared_capacity_owner_shards_indexer_cache(self, *_):
        from sglang.srt.model_executor.pool_configurator import (
            DefaultPoolConfigurator,
        )

        self.runner.kv_cache_dtype = torch.float8_e4m3fn
        self.runner.model_config.kv_lora_rank = 512
        self.runner.model_config.qk_rope_head_dim = 64
        configurator = DefaultPoolConfigurator.__new__(DefaultPoolConfigurator)
        configurator.kv_cache_dtype_str = "fp8_e4m3"

        with get_parallel().override(attn_cp_size=8, attn_cp_rank=3, attn_tp_size=8):
            cell_size = configurator._compute_cell_size(self.runner, num_layers=61)

        main_owner_shard = (512 + 64) * 8
        index_row = 128 + 128 // 128 * 4
        owner_sharded_index = index_row * 8
        self.assertEqual(cell_size, main_owner_shard + owner_sharded_index)

    def test_default_pool_capacity_subtracts_fixed_workspace_before_rounding(self):
        from sglang.srt.model_executor.pool_configurator import (
            DefaultPoolConfigurator,
        )

        configurator = DefaultPoolConfigurator.__new__(DefaultPoolConfigurator)
        configurator._cell_size = 128
        configurator._fixed_bytes = 8192
        configurator._indexer_pool_cache_num_layers = 0

        config = configurator.calculate_pool_sizes(
            available_bytes=8192 + 128 * 128,
            page_size=64,
        )

        self.assertEqual(config.max_total_num_tokens, 128)

    @patch(
        "sglang.srt.model_executor.pool_configurator."
        "DefaultPoolConfigurator._compute_cell_size",
        return_value=128,
    )
    @patch(
        "sglang.srt.model_executor.pool_configurator.mambaish_config",
        return_value=None,
    )
    def test_shared_flashmla_demand_workspace_is_a_fixed_pool_bias(self, *_):
        from sglang.srt.model_executor.pool_configurator import (
            DefaultPoolConfigurator,
        )

        spec_algorithm = MagicMock()
        spec_algorithm.is_eagle.return_value = False
        spec_algorithm.is_standalone.return_value = False
        spec_algorithm.is_dflash_family.return_value = False
        kvc = SimpleNamespace(
            model_config=SimpleNamespace(
                context_len=4096,
                hf_config=SimpleNamespace(
                    architectures=["GlmMoeDsaForCausalLM"],
                    index_topk=2048,
                    index_head_dim=128,
                    cli_factor=1,
                ),
            ),
            layer_info=SimpleNamespace(
                num_effective_layers=78,
                start_layer=0,
                end_layer=78,
            ),
            page_size=64,
            kv_cache_dtype_str="fp8_e4m3",
            spec_algorithm=spec_algorithm,
            is_draft_worker=False,
            use_mla_backend=True,
            server_args=SimpleNamespace(
                enable_dsa_shared_kv_cache=True,
                dsa_prefill_backend="flashmla_kv",
                dsa_decode_backend="flashmla_kv",
                max_running_requests=32,
                max_total_tokens=None,
                speculative_num_draft_tokens=4,
            ),
        )

        with get_parallel().override(attn_cp_size=8):
            configurator = DefaultPoolConfigurator(kvc)

        self.assertEqual(getattr(configurator, "_fixed_bytes", 0), 6_988_942_340)

    def test_shared_flashmla_rejects_more_than_four_current_rows(self):
        from sglang.srt.mem_cache.dsa_shared_demand import (
            get_flashmla_shared_demand_workspace_bytes,
        )

        with self.assertRaisesRegex(ValueError, "at most four"):
            get_flashmla_shared_demand_workspace_bytes(
                num_layers=78,
                num_request_slots=32,
                max_current_rows=5,
            )

    def test_shared_flashmla_without_speculation_uses_one_current_row(self):
        from sglang.srt.mem_cache.dsa_shared_demand import (
            resolve_flashmla_shared_max_current_rows,
        )

        self.assertEqual(resolve_flashmla_shared_max_current_rows(None), 1)
        self.assertEqual(resolve_flashmla_shared_max_current_rows(3), 3)
        self.assertEqual(resolve_flashmla_shared_max_current_rows(4), 4)

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


if __name__ == "__main__":
    unittest.main()
