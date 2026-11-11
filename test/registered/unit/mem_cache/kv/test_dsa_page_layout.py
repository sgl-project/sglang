"""The sparse-attention page layout of the unified pool, as a composite unit.

A DSA model's full layers declare FP8 latent rows with block scales laid out
layer by layer, and the layers that select top-k declare their indexer keys as
a second full-lifetime part. The plan composes both into one unit: each page
holds the layers' latent rows, then one index-key blob per indexer layer,
padded to whole rows and whole token entries. The latent rows are the
sub-pool's token KV view; the keys are routed per layer by their own part.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.kv.build import UnifiedBuildContext, build_kv_storage
from sglang.srt.mem_cache.kv.decl import (
    FP8_DS_MLA,
    INDEX_K_PART,
    KV_PART,
    FullLayerParts,
    IndexKeyGeometry,
    MLAGeometry,
    MLARowFormat,
    StateGeometry,
    kv_layer_decls,
)
from sglang.srt.mem_cache.kv.layout.composite import CompositeSubPoolSpec
from sglang.srt.mem_cache.kv.layout.index_key import IndexKeyView
from sglang.srt.mem_cache.kv.layout.mla import LayerMajorMLAView
from sglang.srt.mem_cache.kv.plan import plan_kv_storage
from sglang.srt.mem_cache.kv.sizing import KVSizingInputs, size_plan
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import setup_allocator_context

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_PAGE = 4


def _rows(row_bytes=32, kv_lora_rank=8):
    return MLAGeometry(
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=8,
        kv_cache_dtype=torch.float8_e4m3fn,
        row_format=MLARowFormat(name=FP8_DS_MLA, row_bytes=row_bytes, scale_ue8m0=True),
        layer_major=True,
    )


def _decls(*, full=(0, 2, 4), index=(0, 4), rows=None, keys=None, state=(1, 3)):
    return kv_layer_decls(
        full_layer_ids=full,
        full_geometry=rows or _rows(),
        full_layer_parts=(
            FullLayerParts(
                layer_ids=index,
                geometry=keys or IndexKeyGeometry(index_head_dim=128),
            ),
        ),
        state_layer_ids=state,
        state_geometry=StateGeometry(
            conv_state_shapes=((3, 8),),
            conv_dtype=torch.bfloat16,
            temporal_state_shape=(1, 4, 8),
            temporal_dtype=torch.float32,
        ),
    )


class TestPageArithmetic(CustomTestCase):
    def _k31(self, keys):
        layers = tuple(range(24))
        return plan_kv_storage(
            _decls(
                full=layers,
                index=layers,
                rows=_rows(row_bytes=656, kv_lora_rank=512),
                keys=keys,
                state=(),
            ),
            page_size=64,
        ).owner()

    def test_kimi_k31_page(self):
        """24 layers of 64 656-byte FP8 rows and an 8448-byte index blob per
        layer fill 1,210,368 bytes; the page rounds up to 1848 rows of 656
        bytes, a whole number of 64-token entries."""
        full = self._k31(IndexKeyGeometry(index_head_dim=128))
        spec = full.spec
        self.assertIsInstance(spec, CompositeSubPoolSpec)
        self.assertEqual([p.name for p in full.parts], [KV_PART, INDEX_K_PART])
        self.assertEqual(spec.offsets(), {KV_PART: 0, INDEX_K_PART: 24 * 64 * 656})
        self.assertEqual(spec.page_bytes(), 1_212_288)
        self.assertEqual(spec.page_bytes() % 656, 0)
        self.assertEqual(spec.entry_bytes() * 64, spec.page_bytes())

    def test_fp4_keys(self):
        """FP4 keys take 64 + 4 bytes per token: 4352 bytes per page per layer."""
        keys = IndexKeyGeometry(index_head_dim=128, use_fp4=True)
        self.assertEqual(keys.bytes_per_token, 68)
        self.assertEqual(keys.key_bytes_per_token, 64)
        spec = self._k31(keys).spec
        raw = 24 * 64 * 656 + 24 * 64 * 68
        self.assertEqual(spec.page_bytes(), -(-raw // 2624) * 2624)

    def test_index_layers_are_their_own_part(self):
        full = plan_kv_storage(_decls(), page_size=_PAGE).owner()
        self.assertEqual(full.part(KV_PART).layer_ids, (0, 2, 4))
        self.assertEqual(full.part(INDEX_K_PART).layer_ids, (0, 4))
        self.assertEqual(full.spec.layer_num, 3)

    def test_stored_rows_need_the_layer_major_layout(self):
        rows = MLAGeometry(
            kv_lora_rank=8,
            qk_rope_head_dim=8,
            kv_cache_dtype=torch.float8_e4m3fn,
            row_format=MLARowFormat(name=FP8_DS_MLA, row_bytes=32),
        )
        decls = kv_layer_decls(
            full_layer_ids=(0,),
            full_geometry=rows,
            state_layer_ids=(1,),
            state_geometry=StateGeometry(
                conv_state_shapes=((3, 8),),
                conv_dtype=torch.bfloat16,
                temporal_state_shape=(1, 4, 8),
                temporal_dtype=torch.float32,
            ),
        )
        with self.assertRaisesRegex(ValueError, "layer_major"):
            plan_kv_storage(decls, page_size=_PAGE)


class TestViewsOverTheArena(CustomTestCase):
    def _storage(self):
        setup_allocator_context()
        plan = plan_kv_storage(_decls(), page_size=_PAGE)
        sized = size_plan(
            plan,
            KVSizingInputs(
                full_tokens=64,
                state_slots=4,
                max_running_requests=2,
                context_len=64,
                alignment=1,
            ),
        )
        storage = build_kv_storage(
            plan,
            sized,
            UnifiedBuildContext(
                device="cpu",
                start_layer=0,
                end_layer=5,
                enable_memory_saver=False,
                need_sort=False,
                max_num_reqs=2,
                model_context_len=64,
                state_cache_params=SimpleNamespace(layers=(1, 3)),
            ),
        )
        return plan, storage

    def test_views_and_routes(self):
        plan, storage = self._storage()
        view = storage.view("full")
        self.assertIsInstance(view, LayerMajorMLAView)
        self.assertEqual(view.page_bytes, plan.owner().spec.page_bytes())
        self.assertTrue(view.dsa_kv_cache_store_fp8)
        self.assertTrue(view.kv_scale_ue8m0)
        pool = storage.allocator.get_kvcache()
        self.assertTrue(pool.use_dsa)
        self.assertEqual(pool.index_head_dim, 128)
        self.assertEqual(pool.index_bytes_per_token, 132)
        self.assertIsInstance(pool.layer_router.part_view(INDEX_K_PART), IndexKeyView)
        self.assertEqual(pool.layer_router.part_indices(INDEX_K_PART), {0: 0, 4: 1})
        # Model layer 2 selects no top-k: it keeps no keys.
        with self.assertRaises(KeyError):
            pool.get_index_k_with_scale_buffer(2)

    def test_rows_and_blobs_land_in_their_page(self):
        """Every layer's rows of a page and every index blob occupy disjoint
        bytes inside that page's unit of the arena."""
        plan, storage = self._storage()
        spec = plan.owner().spec
        pool = storage.allocator.get_kvcache()
        arena = storage.view("full")._arena
        raw = arena.raw
        raw.zero_()
        first = arena.first_entry("full") // _PAGE + 1
        stamp, expected = 1, {}
        layer_bytes = _PAGE * 32
        for page in range(first, first + 3):
            for index, layer in enumerate((0, 2, 4)):
                rows = pool.get_key_buffer(layer).view(torch.uint8)
                for slot in range(_PAGE):
                    rows[page, slot] = stamp
                    start = page * spec.page_bytes() + index * layer_bytes + slot * 32
                    expected[stamp] = range(start, start + 32)
                    stamp += 1
            for index, layer in enumerate((0, 4)):
                blob = pool.get_index_k_with_scale_buffer(layer)
                self.assertEqual(blob.stride(0), spec.page_bytes())
                blob[page] = stamp
                start = page * spec.page_bytes() + spec.offsets()[INDEX_K_PART]
                start += index * _PAGE * 132
                expected[stamp] = range(start, start + _PAGE * 132)
                stamp += 1
        for value, span in expected.items():
            self.assertTrue(
                bool((raw[span.start : span.stop] == value).all()), f"stamp {value}"
            )
        self.assertEqual(int((raw != 0).sum()), sum(map(len, expected.values())))


if __name__ == "__main__":
    unittest.main()
