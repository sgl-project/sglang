"""Composite sub-pools: the parts one lifetime group declares share one paged unit."""

import itertools
import unittest
from types import SimpleNamespace
from typing import ClassVar

import msgspec
import torch

from sglang.srt.mem_cache.kv.build import UnifiedBuildContext, build_kv_storage
from sglang.srt.mem_cache.kv.decl import (
    FP8_DS_MLA,
    KV_PART,
    FullLayerParts,
    FullLifetime,
    IndexKeyGeometry,
    KVLayerDecl,
    KVPartDecl,
    MHAGeometry,
    MLAGeometry,
    MLARowFormat,
    StateGeometry,
    WindowLifetime,
    kv_layer_decls,
)
from sglang.srt.mem_cache.kv.layout.composite import (
    CompositeSubPoolSpec,
    LayerMajorRows,
    PageBlobs,
    TokenRows,
)
from sglang.srt.mem_cache.kv.memory.arena import KVArena, plan_bands
from sglang.srt.mem_cache.kv.memory.snapshot import _band_ids
from sglang.srt.mem_cache.kv.plan import plan_kv_storage, sub_pool_names
from sglang.srt.mem_cache.kv.sizing import KVSizingInputs, size_plan
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import setup_allocator_context

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_PAGE = 4


class _PartView:
    """What a test part's readers take: its per-layer buffers over the unit."""

    dtype = store_dtype = torch.uint8

    def __init__(self, *, arena, spec, part):
        self.buffers = spec.part_views(arena, part.name)
        self.layer_ids = part.layer_ids


class _PagedRows(msgspec.Struct, frozen=True, kw_only=True):
    """A token KV geometry whose rows are laid out layer by layer in a unit."""

    row_bytes: int
    part_name: ClassVar[str] = KV_PART

    def requires_unit_layout(self) -> bool:
        return True

    def unit_part(self, *, name, layer_num):
        return LayerMajorRows(name=name, layer_num=layer_num, row_bytes=self.row_bytes)

    def unit_view(self, *, arena, spec, part, page_size):
        return _PartView(arena=arena, spec=spec, part=part)


class _TokenRows(msgspec.Struct, frozen=True, kw_only=True):
    """Rows inside every token entry, read by page or, when not
    ``page_address``, as one run at the token entry stride."""

    row_bytes: int
    page_address: bool = True
    part_name: str = KV_PART

    def requires_unit_layout(self) -> bool:
        return True

    def unit_part(self, *, name, layer_num):
        return TokenRows(
            name=name,
            layer_num=layer_num,
            row_bytes=self.row_bytes,
            readers_page_address=self.page_address,
        )

    def unit_view(self, *, arena, spec, part, page_size):
        return _PartView(arena=arena, spec=spec, part=part)


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
        return _PartView(arena=arena, spec=spec, part=part)


def _state():
    return StateGeometry(
        conv_state_shapes=((3, 64),),
        conv_dtype=torch.bfloat16,
        temporal_state_shape=(2, 16, 16),
        temporal_dtype=torch.float32,
    )


def _composite_decls():
    return kv_layer_decls(
        full_layer_ids=(0, 2, 4),
        full_geometry=_PagedRows(row_bytes=32),
        full_layer_parts=(
            FullLayerParts(layer_ids=(0, 4), geometry=_Blobs(bytes_per_token=12)),
        ),
        state_layer_ids=(1, 3),
        state_geometry=_state(),
    )


def _build(plan, full_tokens=64):
    """The storage production builds for ``plan``, on CPU."""
    sized = size_plan(
        plan,
        KVSizingInputs(
            full_tokens=full_tokens,
            state_slots=4,
            max_running_requests=2,
            context_len=64,
            alignment=1,
        ),
    )
    layer_ids = [i for s in plan.sub_pools for i in s.layer_ids]
    return build_kv_storage(
        plan,
        sized,
        UnifiedBuildContext(
            device="cpu",
            start_layer=0,
            end_layer=max(layer_ids) + 1,
            enable_memory_saver=False,
            need_sort=False,
            max_num_reqs=2,
            model_context_len=64,
            disable_overlap_schedule=True,
        ),
    )


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

    def test_per_token_parts_ride_in_each_entry(self):
        """Per-token parts sit side by side in each token entry; per-page parts
        follow the entries, and only without them is the unit uniform."""
        token_parts = (
            TokenRows(
                name=KV_PART, layer_num=2, row_bytes=16, readers_page_address=True
            ),
            TokenRows(
                name="draft", layer_num=1, row_bytes=32, readers_page_address=True
            ),
        )

        def spec(*parts):
            return CompositeSubPoolSpec(
                name="full",
                layer_num=2,
                grow_direction="down",
                page_size=_PAGE,
                parts=parts,
            )

        paged = spec(
            *token_parts, PageBlobs(name="blobs", layer_num=1, bytes_per_token=4)
        )
        self.assertEqual(paged.token_bytes(), 64)
        self.assertEqual(paged.offsets(), {KV_PART: 0, "draft": 32, "blobs": 256})
        self.assertEqual(paged.page_bytes(), 272)
        self.assertFalse(paged.uniform())
        self.assertTrue(paged.copies_whole_units)

        uniform = spec(*token_parts)
        self.assertTrue(uniform.uniform())
        self.assertEqual(uniform.page_bytes(), _PAGE * uniform.token_bytes())
        self.assertFalse(uniform.copies_whole_units)

    def test_a_part_name_names_one_part_in_the_plan(self):
        """Two sub-pools keeping a part of one name are refused at planning."""
        decls = [
            KVLayerDecl(
                layer_id=layer_id,
                parts=(
                    KVPartDecl(lifetime=lifetime, geometry=_PagedRows(row_bytes=32)),
                    KVPartDecl(lifetime=lifetime, geometry=_Blobs(bytes_per_token=12)),
                ),
            )
            for layer_id, lifetime in (
                (0, FullLifetime()),
                (1, WindowLifetime(window=64)),
            )
        ]
        with self.assertRaisesRegex(ValueError, "both keep a 'blobs' part"):
            plan_kv_storage(decls, page_size=_PAGE)

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
        """3 layers x 4 tokens x 32 B of rows, then 2 layers x 4 tokens x 12 B
        of blobs: a 480 B unit with the blobs at 384. Every layer's rows of a
        page and every blob of it sit there, and no two share a byte."""
        plan = plan_kv_storage(_composite_decls(), page_size=_PAGE)
        spec = plan.owner().spec
        self.assertEqual(spec.entry_bytes(), 120)
        self.assertTrue(spec.zero_units_on_handout)
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


class TestRetractionCopiesWholeUnits(CustomTestCase):
    def _ids(self, plan, indices):
        arena = _arena(plan)
        allocator = SimpleNamespace(
            arena=arena, roles=SimpleNamespace(state="mamba", requests=())
        )
        band = SimpleNamespace(sub_pool_name="full", page_size=_PAGE)
        return _band_ids(allocator, band, indices, None, None)

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


_KV_GEOMETRIES = {
    "mha": MHAGeometry(
        head_num=1, head_dim=8, v_head_dim=8, kv_cache_dtype=torch.bfloat16
    ),
    "mla": MLAGeometry(
        kv_lora_rank=8, qk_rope_head_dim=8, kv_cache_dtype=torch.bfloat16
    ),
    "layer_major_rows": _PagedRows(row_bytes=32),
    "rows_read_by_page": _TokenRows(row_bytes=32),
    "rows_read_flat": _TokenRows(row_bytes=32, page_address=False),
    "layer_major_latent": MLAGeometry(
        kv_lora_rank=8,
        qk_rope_head_dim=8,
        kv_cache_dtype=torch.float8_e4m3fn,
        row_format=MLARowFormat(name=FP8_DS_MLA, row_bytes=32, scale_ue8m0=True),
        layer_major=True,
    ),
    "layer_major_plain_latent": MLAGeometry(
        kv_lora_rank=8,
        qk_rope_head_dim=8,
        kv_cache_dtype=torch.bfloat16,
        layer_major=True,
    ),
}
_EXTRA_PARTS = {
    "rows_read_by_page": _TokenRows(row_bytes=16, part_name="paged"),
    "rows_read_flat": _TokenRows(row_bytes=16, page_address=False, part_name="flat"),
    "blobs": _Blobs(bytes_per_token=12),
    "index_keys": IndexKeyGeometry(index_head_dim=128),
    "state": _state(),
}


def _accepted(kv: str, extras) -> bool:
    """The compositions the builder can construct."""
    if kv in ("mha", "mla"):
        # Token-major KV: its own layout, alone.
        return not extras
    if kv == "layer_major_plain_latent" or "state" in extras:
        return False
    flat = kv == "rows_read_flat" or "rows_read_flat" in extras
    # Rows read at one token stride need pages of plain token entries.
    by_page = {"layer_major_rows", "layer_major_latent", "blobs", "index_keys"}
    return not (flat and by_page & {kv, *extras})


class TestPlannedIsBuildable(CustomTestCase):
    """The planner accepts exactly the compositions the builder constructs:
    every accepted one builds, and every part of it is routed to a view over
    its layers."""

    @classmethod
    def setUpClass(cls):
        setup_allocator_context()

    def test_every_composition(self):
        for kv, n in itertools.product(_KV_GEOMETRIES, range(len(_EXTRA_PARTS) + 1)):
            for extras in itertools.combinations(_EXTRA_PARTS, n):
                with self.subTest(kv=kv, extras=extras):
                    self._check(kv, extras)

    def _check(self, kv, extras):
        decls = kv_layer_decls(
            full_layer_ids=(0, 2),
            full_geometry=_KV_GEOMETRIES[kv],
            full_layer_parts=tuple(
                FullLayerParts(layer_ids=(2,), geometry=_EXTRA_PARTS[e]) for e in extras
            ),
            state_layer_ids=(1,),
            state_geometry=_state(),
        )
        try:
            plan = plan_kv_storage(decls, page_size=_PAGE)
        except ValueError:
            self.assertFalse(_accepted(kv, extras), "refused")
            return
        self.assertTrue(_accepted(kv, extras), "accepted")
        router = _build(plan).allocator.get_kvcache().layer_router
        owner = plan.owner()
        for part in owner.parts or (SimpleNamespace(name=KV_PART, layer_ids=(0, 2)),):
            for i, layer_id in enumerate(part.layer_ids):
                layer = router.layer(layer_id, part.name)
                self.assertEqual(layer.route.sub_pool, owner.name)
                self.assertEqual(layer.view_layer_id, i)
                if isinstance(layer.view, _PartView):
                    self.assertEqual(len(layer.view.buffers), len(part.layer_ids))


class TestRetractionRoundTrip(CustomTestCase):
    """A request's composite pages survive retraction: copied out whole, then
    restored into other virtual and physical pages, every KV row and every
    per-page blob of its pages reads back as written, and no other byte of the
    buffer changes."""

    @classmethod
    def setUpClass(cls):
        setup_allocator_context()

    def test_restore_into_other_pages(self):
        plan = plan_kv_storage(_composite_decls(), page_size=_PAGE)
        storage = _build(plan)
        allocator = storage.allocator
        router = allocator.get_kvcache().layer_router
        ids = allocator.full_attn_allocator.ids
        rows = router.layer(0, KV_PART).view.buffers
        blobs = router.layer(0, "blobs").view.buffers
        unit = plan.owner().spec.page_bytes()
        raw = allocator.arena.raw
        generator = torch.Generator().manual_seed(0)

        def scramble():
            raw.copy_(torch.randint(0, 256, raw.shape, generator=generator).to(raw))

        # Six tokens over virtual pages 1 and 2: the second page is partial.
        old_tokens = torch.arange(_PAGE, _PAGE + 6)
        ids.bind(torch.tensor([1, 2]), torch.tensor([5, 3]))
        scramble()
        written_rows = [
            torch.stack(
                [
                    view[p, s]
                    for p, s in ((5, 0), (5, 1), (5, 2), (5, 3), (3, 0), (3, 1))
                ]
            )
            for view in rows
        ]
        written_blobs = [torch.stack([view[5], view[3]]) for view in blobs]
        written_units = [raw[p * unit : (p + 1) * unit].clone() for p in (5, 3)]
        saved = allocator.get_cpu_copy(old_tokens)

        # The request comes back on virtual pages 6 and 7, over physical
        # pages 2 and 6 (one of them a page it held before).
        ids.reset()
        ids.bind(torch.tensor([6, 7]), torch.tensor([2, 6]))
        scramble()
        before = raw.clone()
        new_tokens = torch.arange(6 * _PAGE, 6 * _PAGE + 6)
        allocator.load_cpu_copy(saved, new_tokens)

        for view, written in zip(rows, written_rows):
            restored = torch.stack(
                [
                    view[p, s]
                    for p, s in ((2, 0), (2, 1), (2, 2), (2, 3), (6, 0), (6, 1))
                ]
            )
            self.assertTrue(torch.equal(restored, written))
        for view, written in zip(blobs, written_blobs):
            self.assertTrue(torch.equal(torch.stack([view[2], view[6]]), written))
        # Only the two units changed, whole: the partial page's unused entries
        # come back as they were too.
        changed = raw != before
        units = torch.zeros_like(changed)
        for page in (2, 6):
            units[page * unit : (page + 1) * unit] = True
        self.assertFalse(bool((changed & ~units).any()))
        for page, written in zip((2, 6), written_units):
            self.assertTrue(torch.equal(raw[page * unit : (page + 1) * unit], written))


if __name__ == "__main__":
    unittest.main()
