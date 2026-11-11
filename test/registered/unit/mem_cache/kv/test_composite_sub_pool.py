"""Composite sub-pools: the parts layers of one lifetime declare share a unit.

Layers whose parts have the same lifetime and allocation unit share one
sub-pool. When they declare more than one geometry (a layer's KV, and a part
it keeps beside it), the plan composes them into one unit: per-token parts side
by side in each token entry, then per-page parts after the page's entries,
padded to every part's grid. Each part is addressed by physical page, a unit
apart, so a part whose readers take one uniform token stride cannot share a
unit with per-page parts. Plans of one geometry per sub-pool do not change.
"""

import hashlib
import unittest
from types import SimpleNamespace
from typing import ClassVar

import msgspec
import torch

from sglang.srt.mem_cache.kv.build import _bind_unit_parts
from sglang.srt.mem_cache.kv.decl import (
    KV_PART,
    FullLayerParts,
    MHAGeometry,
    MLAGeometry,
    StateGeometry,
    WindowLayers,
    kv_layer_decls,
)
from sglang.srt.mem_cache.kv.layout.composite import (
    CompositeSubPoolSpec,
    LayerMajorRows,
    PageBlobs,
    TokenRows,
)
from sglang.srt.mem_cache.kv.memory.arena import KVArena, plan_bands
from sglang.srt.mem_cache.kv.memory.snapshot import _band_ids, _whole_pages
from sglang.srt.mem_cache.kv.plan import plan_kv_storage, sub_pool_names
from sglang.srt.mem_cache.kv.router import KVLayerRouter
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_PAGE = 4


class _PagedRows(msgspec.Struct, frozen=True, kw_only=True):
    """A token KV geometry whose rows are laid out layer by layer in a unit."""

    row_bytes: int
    part_name: ClassVar[str] = KV_PART

    def requires_unit_layout(self) -> bool:
        return True

    def unit_part(self, *, name, layer_num):
        return LayerMajorRows(name=name, layer_num=layer_num, row_bytes=self.row_bytes)

    def unit_view(self, *, arena, spec, part, page_size):
        return SimpleNamespace(
            buffers=spec.part_views(arena, part.name), layer_ids=part.layer_ids
        )


class _Blobs(msgspec.Struct, frozen=True, kw_only=True):
    """A per-page part some layers keep beside their KV."""

    bytes_per_token: int
    part_name: ClassVar[str] = "blobs"

    def requires_unit_layout(self) -> bool:
        return True

    def unit_part(self, *, name, layer_num):
        return PageBlobs(
            name=name, layer_num=layer_num, bytes_per_token=self.bytes_per_token
        )

    def unit_view(self, *, arena, spec, part, page_size):
        return SimpleNamespace(
            buffers=spec.part_views(arena, part.name), layer_ids=part.layer_ids
        )


def _state():
    return StateGeometry(
        conv_state_shapes=((3, 64),),
        conv_dtype=torch.bfloat16,
        temporal_state_shape=(2, 16, 16),
        temporal_dtype=torch.float32,
    )


def _composite_decls(kv=None):
    return kv_layer_decls(
        full_layer_ids=(0, 2, 4),
        full_geometry=kv or _PagedRows(row_bytes=32),
        full_layer_parts=(
            FullLayerParts(layer_ids=(0, 4), geometry=_Blobs(bytes_per_token=12)),
        ),
        state_layer_ids=(1, 3),
        state_geometry=_state(),
    )


class TestSingleGeometryPlansUnchanged(CustomTestCase):
    """Every sub-pool of one geometry plans exactly as before: the names,
    layers, specs, entry bytes and chain order of representative hybrid
    models, pinned by the plan's repr."""

    PINNED = {
        "mha_window/1": "db2ce5e79a7e2812",
        "mha_window/64": "ccdc9803d16e54f3",
        "mla_state/1": "834b6a3595364a7d",
        "mla_state/64": "350d6510f26a5496",
        "mla_fp8_state/1": "86124c01a0d38362",
        "mla_fp8_state/64": "26e54bee2e19091f",
        "mha_state/1": "cfc7e89429a8d53a",
        "mha_state/64": "4528d4f5af04ca28",
        "mha_two_windows_state/1": "a87645988bb99106",
        "mha_two_windows_state/64": "8c8405e01e07f518",
    }
    ENTRY_BYTES = {
        "mha_window": [("window_4096", 4096), ("full", 8192)],
        "mla_state": [("mamba", 3182592), ("full", 2304)],
        "mla_fp8_state": [("mamba", 1060864), ("full", 576)],
        "mha_state": [("mamba", 1591296), ("full", 1024)],
        "mha_two_windows_state": [
            ("mamba", 530432),
            ("window_512", 2048),
            ("window_128", 1024),
            ("full", 2048),
        ],
    }

    @staticmethod
    def _cases():
        bf16, fp8 = torch.bfloat16, torch.float8_e4m3fn
        state = StateGeometry(
            conv_state_shapes=((3, 1024),),
            conv_dtype=bf16,
            temporal_state_shape=(8, 128, 128),
            temporal_dtype=torch.float32,
        )

        def mha(heads, dim, dtype=bf16):
            return MHAGeometry(
                head_num=heads, head_dim=dim, v_head_dim=dim, kv_cache_dtype=dtype
            )

        def mla(dtype):
            return MLAGeometry(
                kv_lora_rank=512, qk_rope_head_dim=64, kv_cache_dtype=dtype
            )

        return {
            "mha_window": dict(
                full_layer_ids=(0, 2),
                full_geometry=mha(8, 128),
                window_layer_ids=(1, 3),
                window_geometry=mha(4, 128),
                window=4096,
            ),
            "mla_state": dict(
                full_layer_ids=(3, 7),
                full_geometry=mla(bf16),
                state_layer_ids=(0, 1, 2, 4, 5, 6),
                state_geometry=state,
                checkpoint_interval=64,
            ),
            "mla_fp8_state": dict(
                full_layer_ids=(1,),
                full_geometry=mla(fp8),
                state_layer_ids=(0, 2),
                state_geometry=state,
            ),
            "mha_state": dict(
                full_layer_ids=(3,),
                full_geometry=mha(2, 256, fp8),
                state_layer_ids=(0, 1, 2),
                state_geometry=state,
            ),
            "mha_two_windows_state": dict(
                full_layer_ids=(0,),
                full_geometry=mha(4, 128),
                windows=(
                    WindowLayers(layer_ids=(1,), geometry=mha(4, 128), window=512),
                    WindowLayers(layer_ids=(2,), geometry=mha(4, 64), window=128),
                ),
                state_layer_ids=(3,),
                state_geometry=state,
            ),
        }

    def test_plans_are_pinned(self):
        for name, kwargs in self._cases().items():
            for page in (1, 64):
                with self.subTest(case=name, page_size=page):
                    plan = plan_kv_storage(kv_layer_decls(**kwargs), page_size=page)
                    self.assertTrue(all(s.parts == () for s in plan.sub_pools))
                    self.assertEqual(
                        [(s.name, s.spec.entry_bytes()) for s in plan.sub_pools],
                        self.ENTRY_BYTES[name],
                    )
                    digest = hashlib.sha256(repr(plan.sub_pools).encode()).hexdigest()
                    self.assertEqual(digest[:16], self.PINNED[f"{name}/{page}"])


class TestComposition(CustomTestCase):
    def test_parts_of_one_lifetime_share_a_unit(self):
        plan = plan_kv_storage(_composite_decls(), page_size=_PAGE)
        self.assertEqual([s.name for s in plan.sub_pools], ["mamba", "full"])
        full = plan.owner()
        self.assertIsInstance(full.spec, CompositeSubPoolSpec)
        self.assertEqual([p.name for p in full.parts], [KV_PART, "blobs"])
        self.assertEqual(full.part(KV_PART).layer_ids, (0, 2, 4))
        self.assertEqual(full.part("blobs").layer_ids, (0, 4))
        self.assertEqual(full.layer_ids, (0, 2, 4))
        self.assertEqual(full.geometry, _PagedRows(row_bytes=32))
        # The fused-draft placement names sub-pools before planning.
        self.assertEqual(
            sub_pool_names(_composite_decls()),
            tuple(s.name for s in plan.owner_first()),
        )

    def test_unit_bytes(self):
        """3 layers x 4 tokens x 32 B of rows, then 2 layers x 4 tokens x 12 B
        of blobs: 480 B, already a whole number of 32 B rows and 4 entries."""
        spec = plan_kv_storage(_composite_decls(), page_size=_PAGE).owner().spec
        self.assertEqual(spec.offsets(), {KV_PART: 0, "blobs": 384})
        self.assertEqual(spec.page_bytes(), 480)
        self.assertEqual(spec.entry_bytes(), 120)
        self.assertTrue(spec.copies_whole_units)
        self.assertTrue(spec.zero_units_on_handout)

    def test_pages_are_padded_to_every_grid(self):
        spec = CompositeSubPoolSpec(
            name="full",
            layer_num=1,
            grow_direction="down",
            page_size=_PAGE,
            parts=(
                LayerMajorRows(name=KV_PART, layer_num=1, row_bytes=48),
                PageBlobs(name="blobs", layer_num=1, bytes_per_token=4),
            ),
        )
        # 192 + 16 = 208 B, padded to lcm(4, 16, 48) = 48: 240 B.
        self.assertEqual(spec.page_bytes(), 240)
        self.assertEqual(spec.page_bytes() % 48, 0)

    def test_per_token_parts_ride_in_each_entry(self):
        """Per-token parts sit side by side in each token entry; per-page parts
        follow the entries."""
        spec = CompositeSubPoolSpec(
            name="full",
            layer_num=2,
            grow_direction="down",
            page_size=_PAGE,
            parts=(
                TokenRows(
                    name=KV_PART, layer_num=2, row_bytes=16, readers_page_address=True
                ),
                TokenRows(
                    name="draft", layer_num=1, row_bytes=32, readers_page_address=True
                ),
                PageBlobs(name="blobs", layer_num=1, bytes_per_token=4),
            ),
        )
        self.assertEqual(spec.token_bytes(), 64)
        self.assertEqual(spec.offsets(), {KV_PART: 0, "draft": 32, "blobs": 256})
        self.assertEqual(spec.page_bytes(), 272)
        self.assertFalse(spec.uniform())

    def test_a_unit_of_per_token_parts_is_uniform(self):
        spec = CompositeSubPoolSpec(
            name="full",
            layer_num=2,
            grow_direction="down",
            page_size=_PAGE,
            parts=(
                TokenRows(name=KV_PART, layer_num=2, row_bytes=16),
                TokenRows(name="draft", layer_num=1, row_bytes=32),
            ),
        )
        self.assertTrue(spec.uniform())
        self.assertEqual(spec.page_bytes(), _PAGE * spec.token_bytes())
        self.assertFalse(spec.copies_whole_units)

    def test_parts_read_at_a_uniform_stride_are_refused(self):
        """Token-major KV is read as one run at the entry stride: it cannot sit
        beside per-page parts."""
        for kv in (
            MLAGeometry(
                kv_lora_rank=8, qk_rope_head_dim=8, kv_cache_dtype=torch.bfloat16
            ),
            MHAGeometry(
                head_num=1, head_dim=8, v_head_dim=8, kv_cache_dtype=torch.bfloat16
            ),
        ):
            with self.subTest(kv=type(kv).__name__):
                with self.assertRaisesRegex(ValueError, "read by page"):
                    plan_kv_storage(_composite_decls(kv), page_size=_PAGE)

    def test_two_kv_layouts_are_refused(self):
        decls = kv_layer_decls(
            full_layer_ids=(0, 1),
            full_geometry=_PagedRows(row_bytes=32),
            full_layer_parts=(
                FullLayerParts(layer_ids=(1,), geometry=_PagedRows(row_bytes=64)),
            ),
        )
        with self.assertRaisesRegex(ValueError, "more than one geometry"):
            plan_kv_storage(decls, page_size=_PAGE)

    def test_a_state_has_no_unit_part(self):
        decls = kv_layer_decls(
            full_layer_ids=(0,),
            full_geometry=_PagedRows(row_bytes=32),
            full_layer_parts=(FullLayerParts(layer_ids=(0,), geometry=_state()),),
        )
        with self.assertRaisesRegex(ValueError, "no part in a composite unit"):
            plan_kv_storage(decls, page_size=_PAGE)


def _arena(plan, pages=4):
    owner = plan.owner()
    state = plan.sub_pools[0]
    return KVArena(
        bands=plan_bands(plan),
        total_bytes=plan.sink_bytes()
        + pages * owner.unit * owner.spec.entry_bytes()
        + 4 * state.spec.entry_bytes(),
        sink_bytes=plan.sink_bytes(),
        device="cpu",
    )


class TestViewsOverTheArena(CustomTestCase):
    def test_parts_land_in_disjoint_page_regions(self):
        """Every layer's rows of a page and every blob of it sit in that page's
        unit at the part's offset, and no two share a byte."""
        plan = plan_kv_storage(_composite_decls(), page_size=_PAGE)
        spec = plan.owner().spec
        arena = _arena(plan)
        rows = spec.part_views(arena, KV_PART)
        blobs = spec.part_views(arena, "blobs")
        self.assertEqual(len(rows), 3)
        self.assertEqual(len(blobs), 2)
        arena.raw.zero_()
        stamp, spans = 1, {}
        pages = range(rows[0].shape[0] - 3, rows[0].shape[0])
        for page in pages:
            for layer, view in enumerate(rows):
                for slot in range(_PAGE):
                    view[page, slot] = stamp
                    start = page * 480 + layer * _PAGE * 32 + slot * 32
                    spans[stamp] = range(start, start + 32)
                    stamp += 1
            for layer, view in enumerate(blobs):
                view[page] = stamp
                start = page * 480 + 384 + layer * _PAGE * 12
                spans[stamp] = range(start, start + _PAGE * 12)
                stamp += 1
        raw = arena.raw
        for value, span in spans.items():
            self.assertTrue(bool((raw[span.start : span.stop] == value).all()))
        self.assertEqual(int((raw != 0).sum()), sum(map(len, spans.values())))

    def test_extra_parts_are_routed_by_name(self):
        plan = plan_kv_storage(_composite_decls(), page_size=_PAGE)
        arena = _arena(plan)
        router = KVLayerRouter.from_layer_maps(None, [])
        _bind_unit_parts(router, plan, arena)
        self.assertEqual(router.part_indices("blobs"), {0: 0, 4: 1})
        layer = router.layer(4, "blobs")
        self.assertEqual(layer.route.sub_pool, "full")
        self.assertEqual(layer.view_layer_id, 1)
        self.assertIs(router.part_view("blobs"), layer.view)
        self.assertEqual(len(layer.view.buffers), 2)


class TestRetractionCopiesWholeUnits(CustomTestCase):
    def _ids(self, plan, indices):
        arena = _arena(plan)
        allocator = SimpleNamespace(
            arena=arena, roles=SimpleNamespace(state="mamba", requests=())
        )
        band = SimpleNamespace(sub_pool_name="full", page_size=_PAGE)
        return _band_ids(allocator, band, indices, None, None)

    def test_a_composite_unit_is_copied_whole(self):
        plan = plan_kv_storage(_composite_decls(), page_size=_PAGE)
        ids, by_unit = self._ids(plan, torch.tensor([9, 10, 13]))
        self.assertFalse(by_unit)
        self.assertEqual(ids.tolist(), list(range(8, 16)))

    def test_token_entries_are_copied_by_token(self):
        decls = kv_layer_decls(
            full_layer_ids=(0,),
            full_geometry=MLAGeometry(
                kv_lora_rank=8, qk_rope_head_dim=8, kv_cache_dtype=torch.bfloat16
            ),
            state_layer_ids=(1,),
            state_geometry=_state(),
        )
        plan = plan_kv_storage(decls, page_size=_PAGE)
        ids, _ = self._ids(plan, torch.tensor([9, 10, 13]))
        self.assertEqual(ids.tolist(), [9, 10, 13])

    def test_whole_pages(self):
        ids = torch.tensor([8, 9, 10, 11, 12, 13])
        self.assertEqual(_whole_pages(ids, 4).tolist(), list(range(8, 16)))


if __name__ == "__main__":
    unittest.main()
