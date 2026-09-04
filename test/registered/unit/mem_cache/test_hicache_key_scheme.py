"""Unit tests for srt/mem_cache/hicache_key_scheme (unified L3 keys)."""

import ctypes
import json
import tempfile
import unittest

import msgspec
import torch

from sglang.srt.mem_cache.hicache_key_scheme import (
    KVCacheNamespace,
    build_unified_suffixes,
    derive_namespace,
    load_namespace_descriptor,
    namespace_digest,
    normalize_dtype,
    plan_unified_kv,
)
from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.srt.mem_cache.pool_host.unified_layout import (
    KVCacheLayoutAdapter,
    UnifiedKVLayoutHostMixin,
    UnifiedKVPageLayout,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _tobytes(tensor) -> bytes:
    """Raw bytes of ``tensor`` in its own dim order."""
    return tensor.contiguous().flatten().view(torch.uint8).numpy().tobytes()


def _chunk_bytes(ptr, size) -> bytes:
    """One chunk's object bytes, whatever descriptor form the layout emitted.

    A contiguous chunk is a plain ``(int, int)``; a strided one is a
    ``(list, list)`` -- the transport's scatter/gather form, whose runs are
    already in unified order, so concatenating them IS the object.
    """
    if isinstance(ptr, list):
        return b"".join(ctypes.string_at(p, n) for p, n in zip(ptr, size))
    return ctypes.string_at(ptr, size)


def _read_chunks(ptrs, sizes) -> list:
    return [_chunk_bytes(p, s) for p, s in zip(ptrs, sizes)]


def _assert_chunk_bytes(case, got, want, label=""):
    """Compare chunk byte lists without letting unittest render a diff.

    ``assertEqual`` on lists of multi-kilobyte bytes spends its time building a
    repr diff nobody can read; report the first differing chunk instead.
    """
    case.assertEqual(len(got), len(want), f"{label}: chunk count")
    for i, (g, w) in enumerate(zip(got, want)):
        if g == w:
            continue
        where = next(
            (j for j, (a, b) in enumerate(zip(g, w)) if a != b), min(len(g), len(w))
        )
        case.fail(
            f"{label}: chunk {i} differs at byte {where} ({len(g)} vs {len(w)} bytes)"
        )


def _gqa_namespace(**overrides) -> KVCacheNamespace:
    """The design doc's worked example: GQA 70B-like, 80 layers, 8 kv heads,
    fleet head grid at lcm(TP 2,4) -> head_group=2. Layer chunks are absolute
    per-stage ranges (layer_group stays 0)."""
    fields = dict(
        model_id="meta-llama/Llama-3-70B",
        dtype="bfloat16",
        page_size=64,
        rank_replicated=False,
        total_kv_heads=8,
        head_group=2,
        object_layout="page_first",
    )
    fields.update(overrides)
    return KVCacheNamespace(**fields)


def _mla_namespace(**overrides) -> KVCacheNamespace:
    fields = dict(
        model_id="deepseek-ai/DeepSeek-V3",
        dtype="bfloat16",
        page_size=64,
        rank_replicated=True,
        total_kv_heads=0,
        head_group=0,
        object_layout="page_first_direct",
    )
    fields.update(overrides)
    return KVCacheNamespace(**fields)


def _gqa_suffixes(
    *,
    tp_rank: int,
    tp_size: int,
    start_layer: int = 0,
    end_layer: int = 40,
    **overrides,
) -> list:
    kwargs = dict(
        attn_tp_rank=tp_rank,
        attn_tp_size=tp_size,
        attn_cp_size=1,
        start_layer=start_layer,
        end_layer=end_layer,
        local_kv_heads=8 // tp_size,
        dtype="bfloat16",
        page_size=64,
        model_id="meta-llama/Llama-3-70B",
        rank_replicated=False,
        object_layout="page_first",
    )
    kwargs.update(overrides)
    return build_unified_suffixes(_gqa_namespace(), **kwargs)


def _mla_suffixes(
    *, tp_rank: int, tp_size: int, start_layer: int, end_layer: int
) -> list:
    return build_unified_suffixes(
        _mla_namespace(),
        attn_tp_rank=tp_rank,
        attn_tp_size=tp_size,
        attn_cp_size=1,
        start_layer=start_layer,
        end_layer=end_layer,
        local_kv_heads=0,
        dtype="bfloat16",
        page_size=64,
        model_id="deepseek-ai/DeepSeek-V3",
        rank_replicated=True,
        object_layout="page_first_direct",
    )


class TestNamespaceDigest(CustomTestCase):
    def test_digest_deterministic_and_field_sensitive(self):
        base = _gqa_namespace()
        self.assertEqual(namespace_digest(base), namespace_digest(_gqa_namespace()))
        for change in (
            {"dtype": "float8_e4m3fn"},
            {"page_size": 32},
            {"head_group": 1},
            {"total_kv_heads": 16},
            {"model_id": "other/model"},
            {"numerics_id": "buildX"},
            {"object_layout": "page_first_direct"},
        ):
            self.assertNotEqual(
                namespace_digest(base),
                namespace_digest(_gqa_namespace(**change)),
                f"digest must change when {change} changes",
            )

    def test_digest_shape(self):
        digest = namespace_digest(_gqa_namespace())
        self.assertRegex(digest, r"^ukv1-[0-9a-f]{16}$")

    def test_descriptor_file_round_trip(self):
        namespace = _gqa_namespace()
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(
                {
                    "schema_version": 1,
                    "model_id": namespace.model_id,
                    "dtype": namespace.dtype,
                    "page_size": namespace.page_size,
                    "rank_replicated": False,
                    "total_kv_heads": 8,
                    "head_group": 2,
                    "object_layout": "page_first",
                },
                f,
            )
            f.flush()
            loaded = load_namespace_descriptor(f.name)
        self.assertEqual(namespace_digest(loaded), namespace_digest(namespace))

    def test_descriptor_rejects_wrong_schema_version(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(
                {
                    "schema_version": 99,
                    "model_id": "m",
                    "dtype": "bfloat16",
                    "page_size": 64,
                    "rank_replicated": True,
                    "total_kv_heads": 0,
                    "head_group": 0,
                    "object_layout": "page_first",
                },
                f,
            )
            f.flush()
            with self.assertRaisesRegex(ValueError, "schema_version"):
                load_namespace_descriptor(f.name)

    def test_descriptor_rejects_unknown_fields(self):
        # A typo'd field must be a decode error, not a silently different
        # keyspace (e.g. "numeric_id" dropping the intended numerics pin).
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(
                {
                    "schema_version": 1,
                    "model_id": "m",
                    "dtype": "bfloat16",
                    "page_size": 64,
                    "rank_replicated": True,
                    "total_kv_heads": 0,
                    "head_group": 0,
                    "object_layout": "page_first",
                    "numeric_id": "buildX",
                },
                f,
            )
            f.flush()
            with self.assertRaises(msgspec.ValidationError):
                load_namespace_descriptor(f.name)

    def test_descriptor_rejects_negative_layer_group(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(
                {
                    "schema_version": 1,
                    "model_id": "m",
                    "dtype": "bfloat16",
                    "page_size": 64,
                    "rank_replicated": True,
                    "total_kv_heads": 0,
                    "layer_group": -1,
                    "head_group": 0,
                    "object_layout": "page_first",
                },
                f,
            )
            f.flush()
            with self.assertRaisesRegex(ValueError, "layer_group"):
                load_namespace_descriptor(f.name)

    def test_grid_validation(self):
        with self.assertRaisesRegex(ValueError, "model_id"):
            derive_namespace(
                model_id="",
                dtype="bfloat16",
                page_size=64,
                rank_replicated=True,
                total_kv_heads=0,
                head_group=0,
                object_layout="page_first",
            )
        with self.assertRaisesRegex(ValueError, "divide"):
            build_unified_suffixes(
                _gqa_namespace(total_kv_heads=8, head_group=3),
                attn_tp_rank=0,
                attn_tp_size=4,
                attn_cp_size=1,
                start_layer=0,
                end_layer=40,
                local_kv_heads=2,
                dtype="bfloat16",
                page_size=64,
                model_id="meta-llama/Llama-3-70B",
                rank_replicated=False,
                object_layout="page_first",
            )


class TestUnifiedSuffixes(CustomTestCase):
    """The design doc's worked example with absolute layer-range coordinates
    and head fan-out at head_group=2 (fleet lcm(TP 2,4) = 4)."""

    def test_single_cell_coordinates(self):
        digest = namespace_digest(_gqa_namespace())
        # TP4 ranks own exactly one head group each.
        self.assertEqual(_gqa_suffixes(tp_rank=0, tp_size=4), [f"{digest}_L0-40_H0"])
        self.assertEqual(_gqa_suffixes(tp_rank=3, tp_size=4), [f"{digest}_L0-40_H3"])
        self.assertEqual(
            _gqa_suffixes(tp_rank=1, tp_size=4, start_layer=40, end_layer=80),
            [f"{digest}_L40-80_H1"],
        )

    def test_head_fan_out_chunks(self):
        # TP2 ranks own two head groups each: the fan-out case.
        digest = namespace_digest(_gqa_namespace())
        self.assertEqual(
            _gqa_suffixes(tp_rank=0, tp_size=2),
            [f"{digest}_L0-40_H0", f"{digest}_L0-40_H1"],
        )
        self.assertEqual(
            _gqa_suffixes(tp_rank=1, tp_size=2),
            [f"{digest}_L0-40_H2", f"{digest}_L0-40_H3"],
        )

    def test_cross_tp_chunks_cover_each_other(self):
        # The cross-topology read: a TP2 rank's chunks are exactly the union
        # of the two corresponding TP4 ranks' chunks — a TP2 reader fetches
        # TP4-written objects (and vice versa) by name alone.
        tp2_rank0 = _gqa_suffixes(tp_rank=0, tp_size=2)
        tp4_rank0 = _gqa_suffixes(tp_rank=0, tp_size=4)
        tp4_rank1 = _gqa_suffixes(tp_rank=1, tp_size=4)
        self.assertEqual(tp2_rank0, tp4_rank0 + tp4_rank1)

    def test_pp_partitions_never_collide(self):
        # Hazard 2 of the design doc: today mooncake keys carry pp_rank but
        # not pp_size, so different PP partitions collide. Absolute layer
        # ranges make differing partitions miss instead.
        self.assertNotEqual(
            _gqa_suffixes(tp_rank=0, tp_size=4, start_layer=0, end_layer=40),
            _gqa_suffixes(tp_rank=0, tp_size=4, start_layer=40, end_layer=80),
        )

    def test_uneven_pp_stages_attach(self):
        # DeepSeek-V3 has 61 layers; the default PP2 split is [0,30)/[30,61).
        # Both stages must derive valid, distinct chunk names.
        digest = namespace_digest(_mla_namespace())
        s0 = _mla_suffixes(tp_rank=0, tp_size=2, start_layer=0, end_layer=30)
        s1 = _mla_suffixes(tp_rank=0, tp_size=2, start_layer=30, end_layer=61)
        self.assertEqual(s0, [f"{digest}_L0-30"])
        self.assertEqual(s1, [f"{digest}_L30-61"])

    def test_mla_cross_tp_size_shares_keys(self):
        # Rank-replicated pools have no head axis: TP2 and TP4 deployments
        # derive identical (single) chunk names.
        digest = namespace_digest(_mla_namespace())
        s_tp2 = _mla_suffixes(tp_rank=1, tp_size=2, start_layer=0, end_layer=61)
        s_tp4 = _mla_suffixes(tp_rank=3, tp_size=4, start_layer=0, end_layer=61)
        self.assertEqual(s_tp2, [f"{digest}_L0-61"])
        self.assertEqual(s_tp2, s_tp4)

    def test_rejects_invalid_layer_range(self):
        with self.assertRaisesRegex(ValueError, "layer range"):
            _gqa_suffixes(tp_rank=0, tp_size=4, start_layer=40, end_layer=40)

    def test_rejects_cp(self):
        with self.assertRaisesRegex(NotImplementedError, "context parallelism"):
            _gqa_suffixes(tp_rank=0, tp_size=4, attn_cp_size=2)

    def test_identity_mismatches_fail_fast(self):
        cases = [
            ({"dtype": "float16"}, "dtype"),
            ({"page_size": 32}, "page_size"),
            ({"model_id": "other/model"}, "model_id"),
            ({"rank_replicated": True}, "rank_replicated"),
            ({"object_layout": "page_first_direct"}, "object_layout"),
        ]
        for deployment_override, expected_msg in cases:
            with self.assertRaisesRegex(ValueError, expected_msg):
                _gqa_suffixes(tp_rank=0, tp_size=4, **deployment_override)

    def test_local_heads_must_tile_head_group(self):
        # 12 heads at TP4 -> 3 heads/rank: 12 % 2 == 0 passes the product
        # check, but 3 % 2 != 0 must hit the tiling branch, not silently
        # floor the chunk count.
        namespace = _gqa_namespace(total_kv_heads=12, head_group=2)
        with self.assertRaisesRegex(ValueError, "tile"):
            build_unified_suffixes(
                namespace,
                attn_tp_rank=0,
                attn_tp_size=4,
                attn_cp_size=1,
                start_layer=0,
                end_layer=40,
                local_kv_heads=3,
                dtype="bfloat16",
                page_size=64,
                model_id="meta-llama/Llama-3-70B",
                rank_replicated=False,
                object_layout="page_first",
            )

    def test_wrong_namespace_head_count_fails(self):
        # Also the kv-head replication case: a truthful namespace for an
        # 8-head model at TP16 reports total_kv_heads=8 != 1 x 16.
        with self.assertRaisesRegex(ValueError, "total_kv_heads"):
            _gqa_suffixes(tp_rank=0, tp_size=16, local_kv_heads=1)


class TestDeriveNamespace(CustomTestCase):
    def test_derived_namespace_admits_own_deployment(self):
        namespace = derive_namespace(
            model_id="m/1B",
            dtype="bfloat16",
            page_size=64,
            rank_replicated=False,
            total_kv_heads=8,
            head_group=4,
            object_layout="page_first",
        )
        suffixes = build_unified_suffixes(
            namespace,
            attn_tp_rank=1,
            attn_tp_size=2,
            attn_cp_size=1,
            start_layer=0,
            end_layer=16,
            local_kv_heads=4,
            dtype="bfloat16",
            page_size=64,
            model_id="m/1B",
            rank_replicated=False,
            object_layout="page_first",
        )
        self.assertEqual(len(suffixes), 1)
        self.assertTrue(suffixes[0].endswith("_L0-16_H1"))

    def test_shared_head_grid_derives_shared_namespace(self):
        # The tp_lcm_size fleet agreement: TP2 (4 heads/rank, split 2) and
        # TP4 (2 heads/rank, split 1) both derive head_group=2 and land in
        # the SAME keyspace — this is what enables the cross-TP read.
        common = dict(
            model_id="m/1B",
            dtype="bfloat16",
            page_size=64,
            rank_replicated=False,
            total_kv_heads=8,
        )
        tp2 = derive_namespace(head_group=2, object_layout="page_first", **common)
        tp4 = derive_namespace(head_group=2, object_layout="page_first", **common)
        self.assertEqual(namespace_digest(tp2), namespace_digest(tp4))

    def test_different_head_grids_derive_disjoint_namespaces(self):
        common = dict(
            model_id="m/1B",
            dtype="bfloat16",
            page_size=64,
            rank_replicated=False,
            total_kv_heads=8,
        )
        self.assertNotEqual(
            namespace_digest(
                derive_namespace(head_group=4, object_layout="page_first", **common)
            ),
            namespace_digest(
                derive_namespace(head_group=2, object_layout="page_first", **common)
            ),
        )

    def test_normalize_dtype(self):
        import torch

        self.assertEqual(normalize_dtype(torch.bfloat16), "bfloat16")
        self.assertEqual(normalize_dtype(torch.float8_e4m3fn), "float8_e4m3fn")
        self.assertEqual(normalize_dtype(torch.float8_e5m2), "float8_e5m2")


class TestUnifiedKVPlan(CustomTestCase):
    """plan_unified_kv: any partition config selects the adapter, which names
    byte-identical objects out of EVERY page-addressable host layout, so the
    layout drops out of the namespace identity (object_layout "unified-v2")."""

    #: The host layouts the adapter can serve, in ADAPTER_LAYOUTS order.
    _LAYOUTS = ("page_first", "page_first_direct")
    #: Layouts that still have no unified page view at all: page_head
    #: interleaves the head axis above the layer axis inside a page block, and
    #: the *_kv_split pools are two tensors rather than one.
    _UNSERVABLE = ("page_head", "page_first_kv_split")

    def _plan(self, **overrides):
        kwargs = dict(
            model_id="meta-llama/Llama-3-70B",
            dtype="bfloat16",
            page_size=64,
            rank_replicated=False,
            local_kv_heads=4,
            attn_tp_rank=0,
            attn_tp_size=2,
            attn_cp_size=1,
            start_layer=0,
            end_layer=61,
            is_final_stage=True,
            # Any of _LAYOUTS would do: the plan is the same for all of them.
            # The layout only reaches the pool, where it decides how many byte
            # runs a chunk takes -- never what the chunk contains.
            pool_layout="page_first_direct",
        )
        kwargs.update(overrides)
        return plan_unified_kv(**kwargs)

    def test_no_configs_keeps_raw_layout(self):
        plan = self._plan()
        self.assertFalse(plan.adapter)
        self.assertEqual(plan.namespace.object_layout, "page_first_direct")
        self.assertEqual(len(plan.suffixes), 1)
        self.assertIsNone(plan.layer_ranges)
        self.assertIsNone(plan.head_ranges)
        # Without a config there is no adapter, so no layout requirement: the
        # raw pool bytes are the object and any host layout may write them.
        raw = self._plan(pool_layout="page_first")
        self.assertFalse(raw.adapter)
        self.assertEqual(raw.namespace.object_layout, "page_first")

    def test_any_config_selects_adapter(self):
        # head config alone: adapter, single layer window, head chunks.
        plan = self._plan(head_group_config=2)
        self.assertTrue(plan.adapter)
        # The host layout is NOT in the identity any more: the adapter emits
        # the same object bytes out of every layout it accepts.
        self.assertEqual(plan.namespace.object_layout, "unified-v2")
        self.assertEqual(plan.layer_ranges, [(0, 61)])
        self.assertEqual(plan.head_ranges, [(0, 2), (2, 4)])
        self.assertEqual(len(plan.suffixes), 2)
        # layer config alone: adapter, the rank's full head span as one chunk.
        plan = self._plan(layer_partition=30)
        self.assertTrue(plan.adapter)
        self.assertEqual(plan.layer_ranges, [(0, 30), (30, 60), (60, 61)])
        self.assertEqual(plan.head_ranges, [(0, 4)])
        self.assertEqual(len(plan.suffixes), 3)
        # both: full cross product, layer-major.
        plan = self._plan(head_group_config=2, layer_partition=30)
        self.assertEqual(len(plan.suffixes), 6)
        # ...and the plan does not depend on which host layout asks for it.
        for layout in self._LAYOUTS:
            other = self._plan(
                head_group_config=2, layer_partition=30, pool_layout=layout
            )
            self.assertEqual(other.namespace, plan.namespace, layout)
            self.assertEqual(other.suffixes, plan.suffixes, layout)

    def test_rank_replicated_layer_partition_uses_adapter(self):
        # MLA fleets may still need the adapter: the layer grid must be
        # byte-uniform regardless of each deployment's host layout.
        plan = self._plan(
            rank_replicated=True,
            local_kv_heads=0,
            pool_layout="page_first_direct",
            layer_partition=30,
        )
        self.assertTrue(plan.adapter)
        self.assertEqual(plan.namespace.object_layout, "unified-v2")
        self.assertIsNone(plan.head_ranges)
        self.assertEqual(plan.layer_ranges, [(0, 30), (30, 60), (60, 61)])
        # "regardless of the host layout" is literal: an MLA deployment on
        # page_first derives the same namespace and the same chunk names.
        for layout in self._LAYOUTS:
            other = self._plan(
                rank_replicated=True,
                local_kv_heads=0,
                pool_layout=layout,
                layer_partition=30,
            )
            self.assertEqual(
                namespace_digest(other.namespace),
                namespace_digest(plan.namespace),
                layout,
            )
            self.assertEqual(other.suffixes, plan.suffixes, layout)
        # head_group alone is a no-op for replicated pools (no head axis).
        plan = self._plan(rank_replicated=True, local_kv_heads=0, head_group_config=2)
        self.assertFalse(plan.adapter)

    def test_global_suffixes_map_to_rank_local_ranges(self):
        plan = self._plan(
            attn_tp_rank=1,
            start_layer=30,
            end_layer=61,
            head_group_config=2,
            layer_partition=30,
        )
        digest = namespace_digest(plan.namespace)
        self.assertEqual(
            plan.suffixes,
            [
                f"{digest}_L30-60_H2",
                f"{digest}_L30-60_H3",
                f"{digest}_L60-61_H2",
                f"{digest}_L60-61_H3",
            ],
        )
        # Suffixes use model-global coordinates; pool views are rank-local.
        self.assertEqual(plan.layer_ranges, [(0, 30), (30, 31)])
        self.assertEqual(plan.head_ranges, [(0, 2), (2, 4)])

    def test_host_layout_partitions_the_keyspace_without_the_adapter(self):
        """WITHOUT the adapter the raw pool bytes are the object, so the three
        layouts serialize a page differently and must not share keys."""
        raw = {
            namespace_digest(self._plan(pool_layout=lay).namespace)
            for lay in self._LAYOUTS
        }
        self.assertEqual(len(raw), len(self._LAYOUTS))
        # ...and adapter chunks never collide with the raw-layout objects an
        # unpartitioned deployment writes for the SAME layout, even when the
        # config is a no-op on the grid.
        self.assertNotEqual(
            namespace_digest(self._plan(head_group_config=4).namespace),
            namespace_digest(self._plan().namespace),
        )

    def test_adapter_namespace_is_independent_of_the_host_layout(self):
        """The feature. Under the adapter every host layout emits BYTE-IDENTICAL
        objects, so the layout must not enter the namespace: a page_first writer
        and a page_first_direct reader have to land in one keyspace and derive
        the very same chunk names. Without the adapter it must still split
        them, because then the raw layout IS the wire format."""
        for config in (
            {"head_group_config": 2},
            {"layer_partition": 30},
            {"head_group_config": 2, "layer_partition": 30},
        ):
            plans = [self._plan(pool_layout=lay, **config) for lay in self._LAYOUTS]
            self.assertTrue(all(p.adapter for p in plans), config)
            self.assertEqual({p.namespace.object_layout for p in plans}, {"unified-v2"})
            self.assertEqual(
                len({namespace_digest(p.namespace) for p in plans}), 1, config
            )
            self.assertEqual(len({tuple(p.suffixes) for p in plans}), 1, config)
        # Same call, config removed: three layouts, three keyspaces.
        self.assertEqual(
            len(
                {
                    namespace_digest(self._plan(pool_layout=lay).namespace)
                    for lay in self._LAYOUTS
                }
            ),
            len(self._LAYOUTS),
        )

    def test_a_whole_shard_rank_shares_by_adopting_the_fleet_grid(self):
        """The reuse question: can a deployment that does not itself need a
        head cut read what a head-cut one wrote?

        Only by planning at the same grid. An unset head_group defaults to this
        rank's kv-head count, which is honest -- a TP2 rank's whole-shard
        object covers heads [0,4) and is NOT the bytes a TP4 rank's [0,2)
        object holds -- but it keys the namespace to one TP size. Setting the
        same head_group everywhere is what shares, and a rank that owns several
        groups simply owns several chunks.
        """
        cut = self._plan(head_group_config=1, layer_partition=30)
        default = self._plan(layer_partition=30)
        self.assertEqual(default.namespace.head_group, 4)  # == local_kv_heads
        self.assertNotEqual(
            namespace_digest(cut.namespace), namespace_digest(default.namespace)
        )

        # Same rank, same heads, now declaring the fleet grid: one keyspace.
        adopts = self._plan(head_group_config=1, layer_partition=30)
        self.assertEqual(
            namespace_digest(adopts.namespace), namespace_digest(cut.namespace)
        )
        self.assertEqual(adopts.suffixes, cut.suffixes)

        # And a narrower rank at a different attn-TP derives the SAME namespace
        # and owns a strict subset of the chunks -- which is what makes the
        # cache reusable across the fleet.
        narrow = self._plan(
            local_kv_heads=2,
            attn_tp_size=4,
            attn_tp_rank=1,
            head_group_config=1,
            layer_partition=30,
        )
        self.assertEqual(
            namespace_digest(narrow.namespace), namespace_digest(cut.namespace)
        )
        self.assertTrue(set(narrow.suffixes) <= set(cut.suffixes))
        self.assertLess(len(narrow.suffixes), len(cut.suffixes))

    def test_adapter_supports_every_page_addressable_layout(self):
        """A chunk is a set of byte RUNS, not one range, so the host layout no
        longer decides whether an object can be served -- only how many
        descriptors it takes. Both page-first layouts are accepted; page_head
        and the split-K/V pools, which have no
        (component, layer, token, head, dim) page view at all, are not."""
        from sglang.srt.mem_cache.hicache_key_scheme import ADAPTER_LAYOUTS

        self.assertEqual(ADAPTER_LAYOUTS, self._LAYOUTS)
        for layout in ADAPTER_LAYOUTS:
            for config in ({"layer_partition": 30}, {"head_group_config": 2}):
                plan = self._plan(pool_layout=layout, **config)
                self.assertTrue(plan.adapter, (layout, config))
                self.assertEqual(plan.namespace.object_layout, "unified-v2")
        for layout in self._UNSERVABLE:
            with self.assertRaisesRegex(ValueError, "does not support") as ctx:
                self._plan(pool_layout=layout, head_group_config=2)
            self.assertIn(layout, str(ctx.exception))

    def test_plan_validation(self):
        for unsupported in self._UNSERVABLE:
            with self.assertRaisesRegex(ValueError, "does not support"):
                self._plan(pool_layout=unsupported, layer_partition=30)
        for supported in self._LAYOUTS:
            self.assertTrue(
                self._plan(pool_layout=supported, layer_partition=30).adapter
            )
        with self.assertRaisesRegex(ValueError, "divide"):
            self._plan(head_group_config=3)
        with self.assertRaisesRegex(ValueError, "positive"):
            self._plan(head_group_config=0)
        with self.assertRaisesRegex(NotImplementedError, "1 kv head"):
            self._plan(local_kv_heads=1, attn_tp_size=8)
        # An explicit head_group attests the sharding and lifts the 1-head
        # ambiguity.
        plan = self._plan(local_kv_heads=1, attn_tp_size=8, head_group_config=1)
        self.assertEqual(plan.head_ranges, [(0, 1)])


class TestControllerGuards(CustomTestCase):
    """Attach-time guards of HiCacheController._build_unified_suffix."""

    class _StubHostPool(UnifiedKVLayoutHostMixin):
        """Real mixin, so set_unified_head_groups() is the production one."""

        def __init__(self, layout: str = "page_first_direct"):
            self.layout = layout
            self.head_num = 4
            self.start_layer = 0
            self.end_layer = 61
            import torch

            self.kv_buffer = torch.zeros(1)

    class _StubPoolGroup:
        """A hybrid stack's host-pool group; ``entries`` is what the guard
        counts. One entry (a dense model's FULL component) is supported."""

        def __init__(self, *host_pools):
            self.entries = list(host_pools)
            self.anchor_entry = self.entries[0]

    class _StubDevicePool:
        def __init__(self):
            import torch

            self.dtype = torch.bfloat16

    def _stub_controller(
        self, controller_cls, backend_type: str, layout: str = "page_first_direct"
    ):
        controller = controller_cls.__new__(controller_cls)
        controller.storage_backend_type = backend_type
        host_pool = self._StubHostPool(layout)
        # A plain controller's host pool is the storage pool itself, and has
        # no `entries`.
        controller.mem_pool_host = host_pool
        controller.storage_host_pool = host_pool
        controller.mem_pool_device = self._StubDevicePool()
        # Head-group-major page blocks are only readable by the pfdhg transfer
        # kernels, i.e. the 'kernel' io backend.
        controller.io_backend = "kernel"
        controller.page_size = 64
        controller.tp_rank, controller.tp_size = 0, 2
        controller.pp_rank, controller.pp_size = 0, 1
        return controller

    def _build_full(self, controller, **overrides):
        from sglang.srt.managers.cache_controller import HiCacheController

        kwargs = dict(
            model_name="m",
            is_rank_replicated=True,
            attn_cp_size=1,
            head_group_config=None,
            layer_partition=None,
        )
        kwargs.update(overrides)
        return HiCacheController._build_unified_suffix(controller, **kwargs)

    def _build(self, controller, **overrides):
        """(suffix, layer_ranges, head_ranges); use _build_full for permute."""
        return self._build_full(controller, **overrides)[:3]

    def _generate_config(self, pool, key_scheme="rank-suffix", extra=None):
        """Drive _generate_storage_config with the runtime context stubbed.

        The head-group reset lives here rather than in _build_unified_suffix,
        and reaching it needs the process-global parallel/memory context, which
        is why it went untested and shipped with a mis-scoped import.
        """
        from unittest import mock

        from sglang.srt.managers.cache_controller import HiCacheController

        controller = HiCacheController.__new__(HiCacheController)
        controller.mem_pool_host = pool
        controller.storage_host_pool = pool
        controller.mem_pool_device = self._StubDevicePool()
        controller.storage_backend_type = "mooncake"
        controller.io_backend = "kernel"
        controller.page_size = 64
        controller.enable_storage_metrics = False
        controller.get_attn_cp_rank_and_size = lambda: (0, 1)

        parallel = mock.Mock(tp_rank=0, tp_size=1, pp_rank=0, pp_size=1)
        memory = mock.Mock(hicache_storage_key_scheme=key_scheme)
        mod = "sglang.srt.managers.cache_controller"
        with (
            mock.patch(f"{mod}.is_dp_attention_enabled", return_value=False),
            mock.patch(f"{mod}.get_parallel", return_value=parallel),
            mock.patch(f"{mod}.get_memory", return_value=memory),
        ):
            return HiCacheController._generate_storage_config(
                controller, model_name="m", storage_backend_extra_config=extra or {}
            )

    def test_head_group_order_is_reset_for_a_backend_that_declares_none(self):
        """Detach/re-attach must not leave a permuted pool behind.

        unified_head_groups survives a detach (correctly -- the pages are still
        permuted). Attaching a backend that declares no head-group order, such
        as rank-suffix, must therefore reset it, or the transfer kernels would
        read head-group-major pages as natural.
        """
        pool = self._StubHostPool()
        pool.unified_head_groups = 2
        pool.slot_used = torch.zeros(8, dtype=torch.bool)

        self._generate_config(pool)

        self.assertEqual(pool.unified_head_groups, 1)

    def test_head_group_reset_refuses_while_pages_are_resident(self):
        """The reset must be loud, not silent, if the pool is not empty."""
        pool = self._StubHostPool()
        pool.unified_head_groups = 2
        pool.slot_used = torch.zeros(8, dtype=torch.bool)
        pool.slot_used[3] = True

        with self.assertRaisesRegex(RuntimeError, "resident"):
            self._generate_config(pool)
        # and it must not have half-applied the change
        self.assertEqual(pool.unified_head_groups, 2)

    def test_unpermuted_pool_needs_no_reset(self):
        """The common path: a pool already in natural order is untouched."""
        pool = self._StubHostPool()
        pool.slot_used = torch.ones(8, dtype=torch.bool)  # resident, but HG == 1

        self._generate_config(pool)

        self.assertEqual(pool.unified_head_groups, 1)

    def test_backend_allowlist_guard(self):
        from sglang.srt.managers.cache_controller import HiCacheController

        nixl = self._stub_controller(HiCacheController, "nixl")
        with self.assertRaisesRegex(NotImplementedError, "file and mooncake"):
            self._build(nixl)

    def test_multi_component_pool_groups_rejected(self):
        """The guard is now about the POOL, not the controller class: one
        unified namespace names one grid, so a host-pool group with several
        components (SWA / Mamba / C128 side pools) has no single grid."""
        from sglang.srt.managers.cache_controller import HiCacheController

        class FakeHybridController(HiCacheController):
            pass

        hybrid = self._stub_controller(FakeHybridController, "mooncake")
        hybrid.mem_pool_host = self._StubPoolGroup(
            hybrid.storage_host_pool, self._StubHostPool()
        )
        with self.assertRaisesRegex(NotImplementedError, "multi-component"):
            self._build(hybrid, is_rank_replicated=False, head_group_config=2)

    def test_kv_only_hybrid_stack_is_allowed(self):
        """A dense model on UnifiedRadixCache is a hybrid controller whose pool
        group holds exactly one (FULL) component. That is the default path and
        must attach: the class no longer decides, the entry count does."""
        from sglang.srt.managers.cache_controller import HiCacheController

        class FakeHybridController(HiCacheController):
            pass

        hybrid = self._stub_controller(FakeHybridController, "mooncake")
        hybrid.mem_pool_host = self._StubPoolGroup(hybrid.storage_host_pool)
        suffix, layer_ranges, head_ranges = self._build(
            hybrid, is_rank_replicated=False, head_group_config=2
        )
        self.assertEqual(len(suffix), 2)
        self.assertEqual(layer_ranges, [(0, 61)])
        self.assertEqual(head_ranges, [(0, 2), (2, 4)])

    def test_partition_configs_require_mooncake(self):
        from sglang.srt.managers.cache_controller import HiCacheController

        file_stub = self._stub_controller(HiCacheController, "file")
        with self.assertRaisesRegex(NotImplementedError, "multi-key"):
            self._build(file_stub, is_rank_replicated=False, head_group_config=2)
        with self.assertRaisesRegex(NotImplementedError, "multi-key"):
            self._build(file_stub, layer_partition=30)

    def test_head_group_on_a_replicated_pool_does_not_require_mooncake(self):
        """head_group is ignored on MLA-family pools -- they have no kv-head
        axis -- so it must not be rejected for them here. The arg hook applies
        the same carve-out, so rejecting it would kill a shared fleet
        extra-config at attach that passed startup. layer_partition IS a real
        fan-out for MLA and stays rejected.
        """
        from sglang.srt.managers.cache_controller import HiCacheController

        file_stub = self._stub_controller(HiCacheController, "file")
        suffix, layer_ranges, head_ranges = self._build(
            file_stub, is_rank_replicated=True, head_group_config=2
        )
        self.assertIsInstance(suffix, str)
        self.assertIsNone(head_ranges)
        with self.assertRaisesRegex(NotImplementedError, "multi-key"):
            self._build(file_stub, is_rank_replicated=True, layer_partition=30)

    def test_adapter_plan_on_page_first_direct(self):
        # head_group on a page_first_direct pool attaches through the adapter
        # (list suffix + local chunk grid).
        from sglang.srt.managers.cache_controller import HiCacheController

        stub = self._stub_controller(HiCacheController, "mooncake")
        suffix, layer_ranges, head_ranges = self._build(
            stub, is_rank_replicated=False, head_group_config=2
        )
        self.assertIsInstance(suffix, list)
        self.assertEqual(len(suffix), 2)
        self.assertEqual(layer_ranges, [(0, 61)])
        self.assertEqual(head_ranges, [(0, 2), (2, 4)])

    def test_adapter_attaches_on_every_page_addressable_layout(self):
        """Attach no longer steers on the host layout: both page-first layouts
        attach, and — the point — they attach to the SAME chunk names, so a
        page_first deployment reads what a page_first_direct one wrote.
        page_head and split-K/V pools, which have no unified page view at all,
        are still refused."""
        from sglang.srt.managers.cache_controller import HiCacheController

        names = set()
        for layout in ("page_first", "page_first_direct"):
            stub = self._stub_controller(HiCacheController, "mooncake", layout=layout)
            suffix, layer_ranges, head_ranges = self._build(
                stub,
                is_rank_replicated=False,
                head_group_config=2,
                layer_partition=30,
            )
            # 3 layer windows ([0,30), [30,60), the short [60,61)) x 2 groups.
            self.assertEqual(layer_ranges, [(0, 30), (30, 60), (60, 61)], layout)
            self.assertEqual(head_ranges, [(0, 2), (2, 4)], layout)
            self.assertEqual(len(suffix), 6, layout)
            names.add(tuple(suffix))
        self.assertEqual(len(names), 1)

        for layout in ("page_head", "page_first_kv_split"):
            stub = self._stub_controller(HiCacheController, "mooncake", layout=layout)
            with self.assertRaisesRegex(ValueError, "does not support"):
                self._build(
                    stub,
                    is_rank_replicated=False,
                    head_group_config=2,
                    layer_partition=30,
                )

    def test_head_cut_permutes_only_on_page_first_direct_with_the_kernel_backend(
        self,
    ):
        """A head cut makes a *page_first_direct* pool's page blocks
        head-group-major, which only the pfdhg kernels can read; the
        copy-engine 'direct' backend moves a page block verbatim. So the
        permutation is taken only when both hold -- otherwise the pool keeps
        its natural order and the same chunk is served as several runs. The
        other layouts keep their own page order either way."""
        from sglang.srt.managers.cache_controller import HiCacheController

        # page_first_direct + kernel: permuted, one descriptor per chunk.
        kernel_stub = self._stub_controller(HiCacheController, "mooncake")
        kernel_stub.io_backend = "kernel"
        self.assertTrue(
            self._build_full(
                kernel_stub, is_rank_replicated=False, head_group_config=2
            )[3]
        )
        # Same grid on the copy engine: allowed now, but NOT permuted.
        direct_stub = self._stub_controller(HiCacheController, "mooncake")
        direct_stub.io_backend = "direct"
        suffix, _, head_ranges, permute = self._build_full(
            direct_stub, is_rank_replicated=False, head_group_config=2
        )
        self.assertFalse(permute)
        self.assertEqual(head_ranges, [(0, 2), (2, 4)])
        self.assertEqual(len(suffix), 2)
        # Same head cut, same io backend, a layout that is not permuted: fine.
        other = self._stub_controller(
            HiCacheController, "mooncake", layout="page_first"
        )
        other.io_backend = "direct"
        suffix, _, head_ranges = self._build(
            other, is_rank_replicated=False, head_group_config=2
        )
        self.assertEqual(head_ranges, [(0, 2), (2, 4)])
        self.assertEqual(len(suffix), 2)
        # A grid that does NOT cut heads keeps the natural order, so any io
        # backend can read it.
        layer_only = self._stub_controller(HiCacheController, "mooncake")
        layer_only.io_backend = "direct"
        suffix, layer_ranges, head_ranges = self._build(
            layer_only, is_rank_replicated=False, layer_partition=30
        )
        self.assertEqual(head_ranges, [(0, 4)])
        self.assertEqual(len(suffix), len(layer_ranges))

    def test_adapter_rejects_split_kv_pools(self):
        from sglang.srt.managers.cache_controller import HiCacheController

        stub = self._stub_controller(HiCacheController, "mooncake")
        stub.storage_host_pool.kv_buffer = (1, 2)
        with self.assertRaisesRegex(NotImplementedError, "split K/V"):
            self._build(stub, is_rank_replicated=False, head_group_config=2)

    def test_adapter_rejects_mtp_draft_pools(self):
        """Draft layers extend the host pool's layer axis but are not named by
        the L3 grid; under a head cut they would sit inside a head group's
        region and be transferred in the wrong byte order."""
        from sglang.srt.managers.cache_controller import HiCacheController

        stub = self._stub_controller(HiCacheController, "mooncake")
        stub.storage_host_pool.mtp_draft_device_pools = (object(),)
        with self.assertRaisesRegex(NotImplementedError, "MTP draft"):
            self._build(stub, is_rank_replicated=False, head_group_config=2)
        # ...and only under the adapter: config-free plans are untouched (one
        # suffix string, no chunk grid), so MTP still works without the grid.
        stub2 = self._stub_controller(HiCacheController, "mooncake")
        stub2.storage_host_pool.mtp_draft_device_pools = (object(),)
        suffix, layer_ranges, head_ranges = self._build(stub2)
        self.assertIsInstance(suffix, str)
        self.assertEqual((layer_ranges, head_ranges), (None, None))

    def test_nonpositive_head_group_rejected(self):
        from sglang.srt.managers.cache_controller import HiCacheController

        stub = self._stub_controller(HiCacheController, "mooncake")
        with self.assertRaisesRegex(ValueError, "positive"):
            self._build(stub, is_rank_replicated=False, head_group_config=0)


class TestFileBackendSuffix(CustomTestCase):
    def _config(self, unified_suffix):
        return HiCacheStorageConfig(
            tp_rank=1,
            tp_size=4,
            pp_rank=1,
            pp_size=2,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=False,
            enable_storage_metrics=False,
            is_page_first_layout=True,
            model_name="meta-llama/Llama-3-70B",
            unified_suffix=unified_suffix,
        )

    def test_file_backend_uses_unified_suffix_verbatim(self):
        from sglang.srt.mem_cache.hicache_storage import HiCacheFile

        with tempfile.TemporaryDirectory() as tmp:
            unified_key = "ukv1-0123456789abcdef_L40-80_H1"
            backend = HiCacheFile(self._config(unified_key), file_path=tmp)
            self.assertEqual(backend.config_suffix, f"_{unified_key}")
            self.assertEqual(
                backend._get_suffixed_key("deadbeef"), f"deadbeef_{unified_key}"
            )

    def test_file_backend_rejects_fan_out_list(self):
        from sglang.srt.mem_cache.hicache_storage import HiCacheFile

        with tempfile.TemporaryDirectory() as tmp:
            chunks = [
                "ukv1-0123456789abcdef_L0-40_H0",
                "ukv1-0123456789abcdef_L0-40_H1",
            ]
            with self.assertRaisesRegex(NotImplementedError, "fan-out"):
                HiCacheFile(self._config(chunks), file_path=tmp)

    def test_file_backend_rank_suffix_unchanged(self):
        from sglang.srt.mem_cache.hicache_storage import HiCacheFile

        with tempfile.TemporaryDirectory() as tmp:
            backend = HiCacheFile(self._config(None), file_path=tmp)
            self.assertEqual(backend.config_suffix, "_meta-llama-Llama-3-70B_1_4_2_1")


class TestLayerPartition(CustomTestCase):
    """Shared layer partition (PP read-back): a stage spanning several
    layer windows owns one chunk per window, so readers consume chunks
    written under a different pipeline split by name alone."""

    def _mla_partition_namespace(self):
        return _mla_namespace(layer_group=30)

    def _suffixes(self, start_layer, end_layer, namespace=None, final=True):
        return build_unified_suffixes(
            namespace or self._mla_partition_namespace(),
            attn_tp_rank=0,
            attn_tp_size=2,
            attn_cp_size=1,
            start_layer=start_layer,
            end_layer=end_layer,
            local_kv_heads=0,
            dtype="bfloat16",
            page_size=64,
            model_id="deepseek-ai/DeepSeek-V3",
            rank_replicated=True,
            object_layout="page_first_direct",
            is_final_stage=final,
        )

    def test_pp_read_back_coverage(self):
        # DeepSeek-V3 (61 layers), layer unit 30, default uneven PP2 split
        # [0,30)/[30,61): the trailing remainder forms a short final chunk
        # (L60-61). PP1 fans out to exactly the stages' union.
        digest = namespace_digest(self._mla_partition_namespace())
        stage0 = self._suffixes(0, 30, final=False)
        stage1 = self._suffixes(30, 61)
        pp1 = self._suffixes(0, 61)
        self.assertEqual(stage0, [f"{digest}_L0-30"])
        self.assertEqual(stage1, [f"{digest}_L30-60", f"{digest}_L60-61"])
        self.assertEqual(pp1, stage0 + stage1)

    def test_misaligned_stage_rejected(self):
        # Start off the layer unit is never legal.
        with self.assertRaisesRegex(ValueError, "start"):
            self._suffixes(45, 61)
        # An off-unit END is legal only on the final stage (short chunk).
        with self.assertRaisesRegex(ValueError, "FINAL"):
            self._suffixes(0, 45, final=False)
        self.assertEqual(len(self._suffixes(0, 45)), 2)  # L0-30 + L30-45

    def _gqa_partition_suffixes(
        self, *, tp_rank, tp_size, start_layer, end_layer, final=True
    ):
        namespace = _gqa_namespace(layer_group=30, object_layout="page_first_direct")
        return build_unified_suffixes(
            namespace,
            attn_tp_rank=tp_rank,
            attn_tp_size=tp_size,
            attn_cp_size=1,
            start_layer=start_layer,
            end_layer=end_layer,
            local_kv_heads=8 // tp_size,
            dtype="bfloat16",
            page_size=64,
            model_id="meta-llama/Llama-3-70B",
            rank_replicated=False,
            object_layout="page_first_direct",
            is_final_stage=final,
        )

    def test_mha_pp_read_back_coverage(self):
        # GQA layer fan-out: TP4 rank 2 (head_group == local, no head
        # fan-out) — PP1's chunks are the union of the PP2 stages', with the
        # H coordinate constant and the short final chunk included.
        stage0 = self._gqa_partition_suffixes(
            tp_rank=2, tp_size=4, start_layer=0, end_layer=30, final=False
        )
        stage1 = self._gqa_partition_suffixes(
            tp_rank=2, tp_size=4, start_layer=30, end_layer=61
        )
        pp1 = self._gqa_partition_suffixes(
            tp_rank=2, tp_size=4, start_layer=0, end_layer=61
        )
        self.assertEqual(len(stage1), 2)  # L30-60 + short L60-61
        self.assertEqual(pp1, stage0 + stage1)
        self.assertTrue(all(sfx.endswith("_H2") for sfx in pp1))

    def test_cross_product_chunks_cover_both_axes(self):
        # The layout adapter: TP2/PP1 owns the full H x L cross product
        # (layer-major, head-minor; short final window included), exactly
        # the union of the four TP4/PP2 members' chunks.
        cross = self._gqa_partition_suffixes(
            tp_rank=0, tp_size=2, start_layer=0, end_layer=61
        )
        self.assertEqual(len(cross), 6)
        members = (
            self._gqa_partition_suffixes(
                tp_rank=0, tp_size=4, start_layer=0, end_layer=30, final=False
            )
            + self._gqa_partition_suffixes(
                tp_rank=1, tp_size=4, start_layer=0, end_layer=30, final=False
            )
            + self._gqa_partition_suffixes(
                tp_rank=0, tp_size=4, start_layer=30, end_layer=61
            )
            + self._gqa_partition_suffixes(
                tp_rank=1, tp_size=4, start_layer=30, end_layer=61
            )
        )
        self.assertEqual(sorted(cross), sorted(members))
        # Order is layer-major, head-minor.
        tails = [
            "_L0-30_H0",
            "_L0-30_H1",
            "_L30-60_H0",
            "_L30-60_H1",
            "_L60-61_H0",
            "_L60-61_H1",
        ]
        for sfx, tail in zip(cross, tails):
            self.assertTrue(sfx.endswith(tail), sfx)

    def test_partition_enters_digest(self):
        self.assertNotEqual(
            namespace_digest(self._mla_partition_namespace()),
            namespace_digest(_mla_namespace(layer_group=61)),
        )
        self.assertNotEqual(
            namespace_digest(self._mla_partition_namespace()),
            namespace_digest(_mla_namespace()),
        )

    def test_negative_layer_unit_rejected(self):
        with self.assertRaises(ValueError):
            derive_namespace(
                model_id="m",
                dtype="bfloat16",
                page_size=64,
                rank_replicated=True,
                total_kv_heads=0,
                head_group=0,
                object_layout="page_first",
                layer_group=-1,
            )


class TestMlaLayoutAdapter(CustomTestCase):
    """MLA unified chunks. The unified MLA order is (layer, token, dim), which
    IS ``page_first_direct``'s page block, so there every chunk is ONE
    contiguous byte range of host pool memory: a get lands straight in the pool
    and a put reads straight out of it. The other layouts store the same bytes
    in another order, so their chunks are several byte runs of pool memory —
    still no staging, just more descriptors."""

    _PS, _LAYERS, _DIM, _PAGES = 4, 6, 8, 3

    def _logical(self):
        torch.manual_seed(0)
        # L[layer, token, 1, dim] per page — the unified MLA order.
        return [
            torch.randn(self._LAYERS, self._PS, 1, self._DIM, dtype=torch.bfloat16)
            for _ in range(self._PAGES)
        ]

    def _pool(self, layout, logical):
        from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost

        pool = MLATokenToKVPoolHost.__new__(MLATokenToKVPoolHost)
        pool.layout = layout
        pool.page_size = self._PS
        pool.kv_cache_dim = self._DIM
        pool.layer_num = self._LAYERS
        pool.dtype = torch.bfloat16
        pool.size = self._PAGES * self._PS
        pool.slot_used = torch.zeros(pool.size, dtype=torch.bool)
        if layout == "page_first":
            pool.kv_buffer = torch.zeros(
                pool.size, self._LAYERS, 1, self._DIM, dtype=pool.dtype
            )
            for p, L in enumerate(logical):
                pool.kv_buffer[p * self._PS : (p + 1) * self._PS] = L.permute(
                    1, 0, 2, 3
                )
        elif layout == "page_first_direct":
            pool.kv_buffer = torch.zeros(
                self._PAGES, self._LAYERS, self._PS, 1, self._DIM, dtype=pool.dtype
            )
            for p, L in enumerate(logical):
                pool.kv_buffer[p] = L
        return pool

    _RANGES = [(0, 2), (2, 6)]

    def test_chunk_metas_address_the_pool_directly(self):
        pool = self._pool("page_first_direct", self._logical())
        itemsize = pool.dtype.itemsize
        layer_stride = pool.page_size * pool.kv_cache_dim * itemsize
        page_stride = pool.layer_num * layer_stride
        base = pool.kv_buffer.data_ptr()
        # Pages 2 and 0, in that order: chunk metas are page-major then slab
        # order, and every pointer is a pool address (no staging buffer).
        indices = torch.tensor([8, 9, 10, 11, 0, 1, 2, 3])
        ptrs, sizes = pool.get_unified_chunk_meta(indices, self._RANGES, None)
        self.assertEqual(
            ptrs,
            [
                base + 2 * page_stride,
                base + 2 * page_stride + 2 * layer_stride,
                base,
                base + 2 * layer_stride,
            ],
        )
        self.assertEqual(sizes, [2 * layer_stride, 4 * layer_stride] * 2)
        # The chunks tile each page block exactly — nothing is left unnamed.
        self.assertEqual(sum(sizes), 2 * page_stride)

    def test_chunk_bytes_are_the_logical_chunk_contents(self):
        """The strongest statement available: the bytes NAMED by chunk_metas
        equal the logical (layer range) chunk taken from the naturally-ordered
        reference tensor by plain indexing."""
        logical = self._logical()
        pool = self._pool("page_first_direct", logical)
        indices = torch.arange(self._PAGES * self._PS)
        ptrs, sizes = pool.get_unified_chunk_meta(indices, self._RANGES, None)
        got = [ctypes.string_at(ptr, size) for ptr, size in zip(ptrs, sizes)]
        want = [_tobytes(L[l0:l1]) for L in logical for l0, l1 in self._RANGES]
        _assert_chunk_bytes(self, got, want)

    def test_direct_get_lands_in_the_pool(self):
        """A get is a direct copy into pool memory: no staging, no scatter.
        Chunks of a page that failed simply leave that page untouched."""
        logical = self._logical()
        writer = self._pool("page_first_direct", logical)
        reader = self._pool("page_first_direct", [torch.zeros_like(L) for L in logical])
        indices = torch.arange(self._PAGES * self._PS)
        src_ptrs, src_sizes = writer.get_unified_chunk_meta(indices, self._RANGES, None)
        dst_ptrs, dst_sizes = reader.get_unified_chunk_meta(indices, self._RANGES, None)
        self.assertEqual(src_sizes, dst_sizes)
        # Emulate the transport writing pool memory; page 1 "failed".
        chunks_per_page = len(self._RANGES)
        for i, (dst, src, size) in enumerate(zip(dst_ptrs, src_ptrs, src_sizes)):
            if i // chunks_per_page == 1:
                continue
            ctypes.memmove(dst, src, size)
        for p, L in enumerate(logical):
            got = reader._unified_page_view(p * self._PS)[0]
            if p == 1:
                self.assertEqual(got.abs().sum().item(), 0)
            else:
                self.assertTrue(torch.equal(got, L))

    def test_every_layout_names_the_same_chunk_bytes(self):
        """The central invariant, for MLA: the object bytes are
        (layer, token, dim) whatever the host layout stores, so all three name
        byte-identical chunks and only the DESCRIPTOR COUNT differs.

        ``page_first`` stores (token, layer, dim) -- layer and token are
        transposed relative to the object order -- so a chunk is one run per
        (layer, token), where ``page_first_direct``'s page block already IS the
        object order and needs exactly one.
        """
        logical = self._logical()
        indices = torch.arange(self._PAGES * self._PS)
        want = [_tobytes(L[l0:l1]) for L in logical for l0, l1 in self._RANGES]
        for layout, runs_per_chunk in (
            ("page_first_direct", [1, 1]),
            ("page_first", [2 * self._PS, 4 * self._PS]),
        ):
            pool = self._pool(layout, logical)
            plan = pool.build_unified_layout(self._RANGES, None)
            self.assertFalse(plan.permuted, layout)
            self.assertEqual([len(s.runs) for s in plan.slabs], runs_per_chunk, layout)
            ptrs, sizes = pool.get_unified_chunk_meta(indices, self._RANGES, None)
            # A contiguous chunk is a plain (int, int); a strided one is the
            # transport's scatter/gather (list, list).
            strided = layout != "page_first_direct"
            self.assertTrue(all(isinstance(p, list) is strided for p in ptrs), layout)
            self.assertTrue(all(isinstance(s, list) is strided for s in sizes), layout)
            _assert_chunk_bytes(self, _read_chunks(ptrs, sizes), want, layout)

    def test_layouts_that_have_no_unified_page_view_are_rejected(self):
        """What is still refused: a layout whose page block is not a
        (component, layer, token, head, dim) rectangle at all."""
        pool = self._pool("page_first_direct", self._logical())
        pool.layout = "page_first_kv_split"
        with self.assertRaisesRegex(ValueError, "does not support") as ctx:
            pool.get_unified_chunk_meta(torch.arange(self._PS), self._RANGES, None)
        self.assertIn("page_first_kv_split", str(ctx.exception))

    def test_grid_must_tile_every_layer_of_the_pool(self):
        """A grid that names only part of the pool's layer axis (an MTP draft
        pool appends unnamed layers) would leave bytes transferred in the other
        order, so the plan is refused."""
        pool = self._pool("page_first_direct", self._logical())
        with self.assertRaisesRegex(ValueError, "layers"):
            pool.build_unified_layout([(0, 2)])


class TestMhaDirectChunks(CustomTestCase):
    """MHA chunk placement inside a ``page_first_direct`` page block.

    With one head group the block is page_first_direct's natural
    (layer, token, head, dim). When the L3 grid cuts the kv-head axis the block
    is stored head-group-major, (head_group, layer, token, head_in_group, dim),
    which keeps every chunk a single contiguous byte range.
    """

    def _pool(self, head_num):
        from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost

        pool = MHATokenToKVPoolHost.__new__(MHATokenToKVPoolHost)
        pool.layout = "page_first_direct"
        pool.page_size = 4
        pool.head_num = head_num
        pool.head_dim = 8
        pool.layer_num = 6
        pool.dtype = torch.bfloat16
        page_num = 3
        pool.size = page_num * pool.page_size
        pool.slot_used = torch.zeros(pool.size, dtype=torch.bool)
        pool.kv_buffer = torch.zeros(
            2,
            page_num,
            pool.layer_num,
            pool.page_size,
            head_num,
            pool.head_dim,
            dtype=pool.dtype,
        )
        return pool

    def test_single_head_layer_chunks_address_the_pool(self):
        pool = self._pool(head_num=1)
        ranges, heads = [(0, 2), (2, 6)], [(0, 1)]
        layout = pool.build_unified_layout(ranges, heads)
        self.assertEqual(layout.head_group_num, 1)
        self.assertFalse(layout.permuted)
        itemsize = pool.dtype.itemsize
        layer_stride = pool.page_size * pool.head_dim * itemsize
        page_stride = pool.layer_num * layer_stride
        v_offset = pool.layer_num * pool.size * pool.head_dim * itemsize
        base = pool.kv_buffer.data_ptr()
        ptrs, sizes = pool.get_unified_chunk_meta(
            torch.tensor([8, 9, 10, 11]), ranges, heads
        )
        k0 = base + 2 * page_stride
        self.assertEqual(
            ptrs,
            [
                k0,
                k0 + v_offset,
                k0 + 2 * layer_stride,
                k0 + 2 * layer_stride + v_offset,
            ],
        )
        self.assertEqual(
            sizes,
            [2 * layer_stride, 2 * layer_stride, 4 * layer_stride, 4 * layer_stride],
        )

    def test_whole_head_shard_chunks_are_one_range(self):
        """A layer-range chunk over the rank's whole head shard is one
        contiguous range of the natural page block — the cross-PP case, which
        needs no permutation at all."""
        pool = self._pool(head_num=2)
        ranges, heads = [(0, 2), (2, 6)], [(0, 2)]
        layout = pool.build_unified_layout(ranges, heads)
        self.assertEqual(layout.head_group_num, 1)
        self.assertFalse(layout.permuted)
        itemsize = pool.dtype.itemsize
        row = pool.page_size * pool.head_num * pool.head_dim * itemsize  # per layer
        self.assertEqual(
            [(s.component, s.runs[0][0], s.nbytes) for s in layout.slabs],
            [
                ("k", 0, 2 * row),
                ("v", 0, 2 * row),
                ("k", 2 * row, 4 * row),
                ("v", 2 * row, 4 * row),
            ],
        )

    def test_head_subgroup_chunks_are_one_range_each(self):
        """Cutting the head axis used to force a staging copy. It no longer
        does: the page block is stored head-group-major, so a chunk is still
        ONE contiguous range per (layer range, head group, component) at
        offset (g * L + l0) * page_size * heads_per_group * head_dim."""
        pool = self._pool(head_num=2)
        ranges, heads = [(0, 2), (2, 6)], [(0, 1), (1, 2)]
        layout = pool.build_unified_layout(ranges, heads)
        self.assertEqual(layout.head_group_num, 2)
        self.assertTrue(layout.permuted)
        itemsize = pool.dtype.itemsize
        hg = 1
        layer_stride = pool.page_size * hg * pool.head_dim * itemsize  # 64 B
        group_stride = pool.layer_num * layer_stride  # 384 B
        self.assertEqual((layer_stride, group_stride), (64, 384))
        self.assertEqual(
            [(s.component, s.runs[0][0], s.nbytes) for s in layout.slabs],
            [
                # layer window [0,2)
                ("k", 0, 2 * layer_stride),
                ("v", 0, 2 * layer_stride),
                ("k", group_stride, 2 * layer_stride),
                ("v", group_stride, 2 * layer_stride),
                # layer window [2,6)
                ("k", 2 * layer_stride, 4 * layer_stride),
                ("v", 2 * layer_stride, 4 * layer_stride),
                ("k", group_stride + 2 * layer_stride, 4 * layer_stride),
                ("v", group_stride + 2 * layer_stride, 4 * layer_stride),
            ],
        )
        # ...and together they tile each component's whole page block.
        block = pool.layer_num * pool.page_size * pool.head_num * pool.head_dim
        self.assertEqual(layout.bytes_per_page, 2 * block * itemsize)

    def test_rejects_split_kv_pools(self):
        pool = self._pool(head_num=2)
        pool.kv_buffer = (pool.kv_buffer, pool.kv_buffer)
        with self.assertRaisesRegex(NotImplementedError, "split K/V"):
            pool.get_unified_chunk_meta(torch.tensor([0, 1, 2, 3]), [(0, 6)], [(0, 2)])

    def test_page_first_names_the_same_bytes_as_page_first_direct(self):
        """``page_first`` stores (token, layer, head, dim): layer and token are
        transposed relative to the object order, so a chunk is one run per
        (layer, token) instead of one range. The BYTES are identical to
        page_first_direct's -- that is exactly what lets the two layouts share
        a keyspace, and it is checked here against the same source tensor."""
        ranges, heads = [(0, 2), (2, 6)], [(0, 2)]
        direct = self._pool(head_num=2)
        page_num = direct.size // direct.page_size
        torch.manual_seed(0)
        # (kv, page, layer, token, head, dim) -- the natural page-block order,
        # which is page_first_direct's buffer as-is.
        logical = torch.randn(
            2,
            page_num,
            direct.layer_num,
            direct.page_size,
            direct.head_num,
            direct.head_dim,
            dtype=direct.dtype,
        )
        direct.kv_buffer.copy_(logical)

        page_first = self._pool(head_num=2)
        page_first.layout = "page_first"
        page_first.kv_buffer = (
            logical.permute(0, 1, 3, 2, 4, 5)  # (kv, page, token, layer, head, dim)
            .contiguous()
            .view(2, direct.size, direct.layer_num, direct.head_num, direct.head_dim)
        )

        indices = torch.arange(direct.size)
        direct_ptrs, direct_sizes = direct.get_unified_chunk_meta(
            indices, ranges, heads
        )
        pf_ptrs, pf_sizes = page_first.get_unified_chunk_meta(indices, ranges, heads)
        # One range each on page_first_direct; one run per (layer, token) on
        # page_first, emitted as the transport's scatter/gather lists.
        self.assertTrue(all(isinstance(p, int) for p in direct_ptrs))
        self.assertTrue(all(isinstance(p, list) for p in pf_ptrs))
        self.assertEqual(
            [len(s.runs) for s in page_first.build_unified_layout(ranges, heads).slabs],
            [2 * direct.page_size, 2 * direct.page_size]
            + [4 * direct.page_size, 4 * direct.page_size],
        )
        # Same chunk sizes, same chunk bytes.
        self.assertEqual([sum(s) for s in pf_sizes], direct_sizes)
        _assert_chunk_bytes(
            self,
            _read_chunks(pf_ptrs, pf_sizes),
            _read_chunks(direct_ptrs, direct_sizes),
            "page_first vs page_first_direct",
        )

    def test_head_ranges_must_tile_the_pools_heads(self):
        pool = self._pool(head_num=4)
        with self.assertRaisesRegex(ValueError, "tile"):
            pool.build_unified_layout([(0, 6)], [(0, 2)])
        with self.assertRaisesRegex(ValueError, "uniform"):
            pool.build_unified_layout([(0, 6)], [(0, 1), (1, 4)])
        # An empty grid is a diagnosed refusal, not an IndexError.
        with self.assertRaises(ValueError):
            pool.build_unified_layout([(0, 6)], [])

    def test_layer_ranges_must_tile_the_pools_layers(self):
        """Total width is not enough: an overlap that double-covers one layer
        while leaving another unnamed sums to the right number, and would
        transfer the unnamed layer's bytes under a name that means something
        else.
        """
        pool = self._pool(head_num=4)
        for ranges, why in (
            ([(0, 2)], "short"),
            ([(0, 2), (2, 8)], "long"),
            ([(0, 4), (2, 4)], "overlap that still sums to 6"),
            ([(0, 2), (3, 6)], "gap"),
            ([(2, 6), (0, 2)], "out of order"),
        ):
            with self.assertRaisesRegex(ValueError, "tile", msg=why):
                pool.build_unified_layout(ranges, [(0, 4)])
        pool.build_unified_layout([(0, 2), (2, 6)], [(0, 4)])  # the valid tiling


class TestUnifiedChunkBytes(CustomTestCase):
    """What the unified layout must actually guarantee: the bytes named by
    ``chunk_metas`` ARE the logical chunk (layer range x head group x
    component), and copying just those bytes into another pool reproduces it.

    The reference is built by plain indexing of a naturally-ordered
    ``(kv, layer, token, head, dim)`` page tensor, never by re-deriving the
    offset formula.
    """

    _PS, _HEADS, _LAYERS, _DIM, _PAGES = 4, 4, 6, 8, 2

    def _logical(self, heads=None, pages=None, seed=0):
        """``logical[p]`` is (kv, layer, token, head, dim): the natural page
        block, independent of any head grid."""
        torch.manual_seed(seed)
        return [
            torch.randn(
                2,
                self._LAYERS,
                self._PS,
                heads or self._HEADS,
                self._DIM,
                dtype=torch.bfloat16,
            )
            for _ in range(pages or self._PAGES)
        ]

    def _pool(self, logical, head_groups=1, layout="page_first_direct"):
        """A host pool of ``layout`` holding ``logical``'s content.

        A ``page_first_direct`` pool stores its page blocks head-group-major
        for ``head_groups`` groups (== the natural order when that is 1), which
        models what the pfdhg transfer kernels write. The other layouts have no
        permuted form -- they store their own order and serve a chunk as
        several runs -- so ``head_groups`` must be 1 for them.
        """
        from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost

        pages, heads = len(logical), logical[0].shape[3]
        pool = MHATokenToKVPoolHost.__new__(MHATokenToKVPoolHost)
        pool.layout = layout
        pool.page_size = self._PS
        pool.head_num = heads
        pool.head_dim = self._DIM
        pool.layer_num = self._LAYERS
        pool.dtype = torch.bfloat16
        pool.size = pages * self._PS
        pool.slot_used = torch.zeros(pool.size, dtype=torch.bool)
        if layout != "page_first_direct":
            assert head_groups == 1, f"{layout} page blocks are never permuted"
        if layout == "page_first":
            # (kv, token, layer, head, dim)
            pool.kv_buffer = torch.zeros(
                2, pool.size, self._LAYERS, heads, self._DIM, dtype=pool.dtype
            )
            for p, page in enumerate(logical):
                pool.kv_buffer[:, p * self._PS : (p + 1) * self._PS] = page.permute(
                    0, 2, 1, 3, 4
                )
            return pool
        pool.kv_buffer = torch.zeros(
            2, pages, self._LAYERS, self._PS, heads, self._DIM, dtype=pool.dtype
        )
        per_group = heads // head_groups
        for p, page in enumerate(logical):
            block = pool.kv_buffer[:, p]
            permuted = (
                page.view(2, self._LAYERS, self._PS, head_groups, per_group, self._DIM)
                .permute(0, 3, 1, 2, 4, 5)  # (kv, group, layer, token, head, dim)
                .contiguous()
            )
            block.copy_(permuted.view(block.shape))
        return pool

    @staticmethod
    def _reference(logical, layer_ranges, head_ranges):
        """The chunk contents, by plain indexing, in chunk_keys order."""
        return [
            _tobytes(page[kv, l0:l1, :, h0:h1, :])
            for page in logical
            for l0, l1 in layer_ranges
            for h0, h1 in head_ranges
            for kv in range(2)
        ]

    _GRIDS = (
        # (label, layer_ranges, head_ranges)
        ("no cut", [(0, 2), (2, 6)], [(0, 4)]),
        ("head cut x2", [(0, 2), (2, 6)], [(0, 2), (2, 4)]),
        ("head cut x4, one layer window", [(0, 6)], [(i, i + 1) for i in range(4)]),
    )

    def test_chunk_bytes_are_the_logical_chunk_contents(self):
        logical = self._logical()
        indices = torch.arange(self._PAGES * self._PS)
        for label, layer_ranges, head_ranges in self._GRIDS:
            pool = self._pool(logical, head_groups=len(head_ranges))
            ptrs, sizes = pool.get_unified_chunk_meta(
                indices, layer_ranges, head_ranges
            )
            got = [ctypes.string_at(ptr, size) for ptr, size in zip(ptrs, sizes)]
            _assert_chunk_bytes(
                self, got, self._reference(logical, layer_ranges, head_ranges), label
            )
            # One chunk per (layer range, head group, component), and together
            # they tile the whole page: nothing staged, nothing left over.
            self.assertEqual(
                len(sizes),
                self._PAGES * len(layer_ranges) * len(head_ranges) * 2,
                label,
            )
            self.assertEqual(
                sum(sizes),
                pool.kv_buffer.numel() * pool.dtype.itemsize,
                label,
            )

    def test_chunk_ranges_are_disjoint_and_inside_the_pool(self):
        logical = self._logical()
        layer_ranges, head_ranges = [(0, 2), (2, 6)], [(0, 2), (2, 4)]
        pool = self._pool(logical, head_groups=2)
        base = pool.kv_buffer.data_ptr()
        span = pool.kv_buffer.numel() * pool.dtype.itemsize
        ptrs, sizes = pool.get_unified_chunk_meta(
            torch.arange(self._PAGES * self._PS), layer_ranges, head_ranges
        )
        covered = sorted(zip(ptrs, sizes))
        cursor = base
        for ptr, size in covered:
            self.assertGreaterEqual(ptr, cursor)
            self.assertLessEqual(ptr + size, base + span)
            cursor = ptr + size
        self.assertEqual(cursor, base + span)

    def test_direct_get_reproduces_the_writers_chunks(self):
        """A put reads the writer's pool, a get writes the reader's pool: the
        two are the same byte ranges. Page 1's chunks are dropped to show a
        failed page is simply left untouched (nothing is published from a
        staging buffer)."""
        logical = self._logical()
        layer_ranges, head_ranges = [(0, 2), (2, 6)], [(0, 2), (2, 4)]
        writer = self._pool(logical, head_groups=2)
        reader = self._pool([torch.zeros_like(page) for page in logical], head_groups=2)
        indices = torch.arange(self._PAGES * self._PS)
        src_ptrs, src_sizes = writer.get_unified_chunk_meta(
            indices, layer_ranges, head_ranges
        )
        dst_ptrs, dst_sizes = reader.get_unified_chunk_meta(
            indices, layer_ranges, head_ranges
        )
        self.assertEqual(src_sizes, dst_sizes)
        per_page = len(layer_ranges) * len(head_ranges) * 2
        for i, (dst, src, size) in enumerate(zip(dst_ptrs, src_ptrs, src_sizes)):
            if i // per_page == 1:
                continue
            ctypes.memmove(dst, src, size)
        got = [ctypes.string_at(ptr, size) for ptr, size in zip(dst_ptrs, dst_sizes)]
        want = self._reference(logical, layer_ranges, head_ranges)
        _assert_chunk_bytes(self, got[:per_page], want[:per_page], "page 0")
        self.assertTrue(
            torch.equal(
                reader.kv_buffer[:, 1], torch.zeros_like(reader.kv_buffer[:, 1])
            )
        )

    def test_cross_tp_pools_name_the_same_chunk_bytes(self):
        """The payoff. A TP2 rank (4 kv heads, head_group=2) owns chunks H0/H1;
        a TP4 rank (2 kv heads, head_group=2 == its whole shard) owns one. The
        head-group-major block makes the TP4 rank's single chunk byte-identical
        to the TP2 rank's H1 chunk, so cross-TP reuse is pure key selection."""
        layer_ranges = [(0, 2), (2, 6)]
        wide = self._logical()  # 4 kv heads
        narrow = [page[:, :, :, 2:4, :].contiguous() for page in wide]
        tp2 = self._pool(wide, head_groups=2)
        tp4 = self._pool(narrow, head_groups=1)
        indices = torch.arange(self._PAGES * self._PS)
        wide_ptrs, wide_sizes = tp2.get_unified_chunk_meta(
            indices, layer_ranges, [(0, 2), (2, 4)]
        )
        narrow_ptrs, narrow_sizes = tp4.get_unified_chunk_meta(
            indices, layer_ranges, [(0, 2)]
        )
        # chunk_keys order is page-major, layer-major, head-minor, K then V:
        # the TP4 rank's chunks are the odd head group of the TP2 rank's.
        wide_h1 = [
            (ptr, size)
            for i, (ptr, size) in enumerate(zip(wide_ptrs, wide_sizes))
            if (i // 2) % 2 == 1
        ]
        self.assertEqual([s for _, s in wide_h1], narrow_sizes)
        _assert_chunk_bytes(
            self,
            [ctypes.string_at(p, s) for p, s in wide_h1],
            [ctypes.string_at(p, s) for p, s in zip(narrow_ptrs, narrow_sizes)],
            "TP2 H1 vs TP4 whole shard",
        )

    #: Host layouts the adapter accepts, and how many byte runs each takes per
    #: chunk as a function of the grid. ``P`` is the page size, ``dl`` the
    #: chunk's layer count, ``cut`` whether the grid cuts the kv-head axis.
    _LAYOUT_RUNS = {
        # page blocks ARE the object order (head-group-major under a cut)
        "page_first_direct": lambda dl, P, cut: 1,
        # (token, layer, head, dim): layer/token transposed -> one run each
        "page_first": lambda dl, P, cut: dl * P,
    }

    def test_every_layout_names_the_same_chunk_bytes(self):
        """THE invariant the shared keyspace rests on: for one logical KV,
        both host layouts name byte-identical L3 objects, on every grid. Only
        the descriptor count differs.

        The reference is plain indexing of the natural
        ``(kv, layer, token, head, dim)`` page tensor -- the run formula is
        never re-derived from the implementation.
        """
        logical = self._logical()
        indices = torch.arange(self._PAGES * self._PS)
        for label, layer_ranges, head_ranges in self._GRIDS:
            want = self._reference(logical, layer_ranges, head_ranges)
            cut = len(head_ranges) > 1
            for layout, runs_of in self._LAYOUT_RUNS.items():
                groups = len(head_ranges) if layout == "page_first_direct" else 1
                pool = self._pool(logical, head_groups=groups, layout=layout)
                plan = pool.build_unified_layout(layer_ranges, head_ranges)
                where = f"{label}/{layout}"
                self.assertEqual(
                    [len(s.runs) for s in plan.slabs],
                    [
                        runs_of(l1 - l0, self._PS, cut)
                        for l0, l1 in layer_ranges
                        for _ in head_ranges
                        for _ in range(2)
                    ],
                    where,
                )
                ptrs, sizes = pool.get_unified_chunk_meta(
                    indices, layer_ranges, head_ranges
                )
                _assert_chunk_bytes(self, _read_chunks(ptrs, sizes), want, where)
                # ...and the chunks still tile the pool exactly.
                self.assertEqual(
                    sum(sum(s) if isinstance(s, list) else s for s in sizes),
                    pool.kv_buffer.numel() * pool.dtype.itemsize,
                    where,
                )

    def test_page_first_direct_without_a_head_cut_stays_one_descriptor(self):
        """The fast path must not regress into scatter/gather. Without a head
        cut a page_first_direct chunk is ONE run and chunk_metas emits a plain
        int pointer, which is what keeps it off the transport's multi-buffer
        path; with a cut the head-group-major block keeps it at one run too."""
        logical = self._logical()
        indices = torch.arange(self._PAGES * self._PS)
        for label, layer_ranges, head_ranges in self._GRIDS:
            pool = self._pool(logical, head_groups=len(head_ranges))
            plan = pool.build_unified_layout(layer_ranges, head_ranges)
            self.assertEqual(plan.permuted, len(head_ranges) > 1, label)
            self.assertEqual(plan.descriptors_per_page, len(plan.slabs), label)
            self.assertTrue(all(s.contiguous for s in plan.slabs), label)
            ptrs, sizes = pool.get_unified_chunk_meta(
                indices, layer_ranges, head_ranges
            )
            self.assertTrue(all(isinstance(p, int) for p in ptrs), label)
            self.assertTrue(all(isinstance(s, int) for s in sizes), label)


class TestLayoutAdapterChunks(CustomTestCase):
    """Backend-neutral KVCacheLayoutAdapter: key fan-out, the slab plan, and
    the pointer round trip a backend performs — no store involved. The adapter
    owns no buffers: both directions address the host pool."""

    def _config(self, suffixes, layer_ranges, head_ranges, extra=None, permute=True):
        return HiCacheStorageConfig(
            tp_rank=0,
            tp_size=2,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=head_ranges is None,
            enable_storage_metrics=False,
            is_page_first_layout=False,
            model_name="m",
            extra_config=extra,
            unified_suffix=suffixes,
            unified_layer_ranges=layer_ranges,
            unified_head_ranges=head_ranges,
            unified_permute_head_groups=permute,
        )

    def test_mla_adapter_owns_no_buffers(self):
        # MLA on page_first_direct: every chunk is pool-contiguous, so the
        # adapter allocates nothing and registers nothing.
        mla = TestMlaLayoutAdapter()
        pool = mla._pool("page_first_direct", mla._logical())
        adapter = KVCacheLayoutAdapter(
            pool,
            self._config(["ns_L0-2", "ns_L2-6"], [(0, 2), (2, 6)], None),
        )
        for gone in (
            "staging_set",
            "staging_get",
            "staging_pages",
            "gather",
            "scatter",
        ):
            self.assertFalse(hasattr(adapter, gone), gone)
        self.assertEqual(adapter.keys_per_page, 2)
        self.assertEqual(
            adapter.chunk_keys(["h1", "h2"]),
            ["h1_ns_L0-2_k", "h1_ns_L2-6_k", "h2_ns_L0-2_k", "h2_ns_L2-6_k"],
        )
        indices = torch.arange(mla._PAGES * mla._PS)
        self.assertEqual(
            adapter.chunk_metas(indices),
            pool.get_unified_chunk_meta(indices, [(0, 2), (2, 6)], None),
        )
        # MLA has no head axis to cut, so its pages keep the natural order.
        self.assertEqual(pool.unified_head_groups, 1)

    def _mha(self):
        return TestUnifiedChunkBytes()

    def test_head_fan_out_round_trip_through_a_keyed_byte_store(self):
        """One chunk per key, straight out of and back into pool memory."""
        gs = self._mha()
        logical = gs._logical()
        layer_ranges, head_ranges = [(0, 2), (2, 6)], [(0, 2), (2, 4)]
        suffixes = [
            f"ns_L{l0}-{l1}_H{j}"
            for l0, l1 in layer_ranges
            for j in range(len(head_ranges))
        ]
        config = self._config(suffixes, layer_ranges, head_ranges)
        writer = KVCacheLayoutAdapter(gs._pool(logical, head_groups=2), config)
        self.assertEqual(writer.keys_per_page, 8)
        # Declaring the grid is what puts the pool in head-group-major order.
        self.assertEqual(writer.pool.unified_head_groups, 2)
        self.assertEqual(
            writer.chunk_keys(["h0"]),
            [
                "h0_ns_L0-2_H0_k",
                "h0_ns_L0-2_H0_v",
                "h0_ns_L0-2_H1_k",
                "h0_ns_L0-2_H1_v",
                "h0_ns_L2-6_H0_k",
                "h0_ns_L2-6_H0_v",
                "h0_ns_L2-6_H1_k",
                "h0_ns_L2-6_H1_v",
            ],
        )
        itemsize = writer.pool.dtype.itemsize
        layer_stride = gs._PS * 2 * gs._DIM * itemsize
        group_stride = gs._LAYERS * layer_stride
        self.assertEqual(
            [(s.component, s.runs[0][0], s.nbytes) for s in writer.layout.slabs],
            [
                ("k", 0, 2 * layer_stride),
                ("v", 0, 2 * layer_stride),
                ("k", group_stride, 2 * layer_stride),
                ("v", group_stride, 2 * layer_stride),
                ("k", 2 * layer_stride, 4 * layer_stride),
                ("v", 2 * layer_stride, 4 * layer_stride),
                ("k", group_stride + 2 * layer_stride, 4 * layer_stride),
                ("v", group_stride + 2 * layer_stride, 4 * layer_stride),
            ],
        )

        page_keys = [f"h{p}" for p in range(gs._PAGES)]
        indices = torch.arange(gs._PAGES * gs._PS)
        store = {}
        ptrs, sizes = writer.chunk_metas(indices)
        for key, ptr, size in zip(writer.chunk_keys(page_keys), ptrs, sizes):
            store[key] = ctypes.string_at(ptr, size)
        self.assertEqual(len(store), gs._PAGES * 8)

        reader = KVCacheLayoutAdapter(
            gs._pool([torch.zeros_like(page) for page in logical], head_groups=2),
            config,
        )
        ptrs, sizes = reader.chunk_metas(indices)
        for key, ptr, size in zip(reader.chunk_keys(page_keys), ptrs, sizes):
            ctypes.memmove(ptr, store[key], size)
        # The reader's pool now holds the writer's logical content.
        got = [ctypes.string_at(ptr, size) for ptr, size in zip(ptrs, sizes)]
        _assert_chunk_bytes(
            self, got, gs._reference(logical, layer_ranges, head_ranges)
        )
        self.assertTrue(torch.equal(reader.pool.kv_buffer, writer.pool.kv_buffer))

    def test_head_group_order_cannot_change_under_live_pages(self):
        """The pool has no per-page provenance, so switching the byte order
        while pages are resident would leave two orders in one pool."""
        gs = self._mha()
        logical = gs._logical()
        pool = gs._pool(logical, head_groups=2)
        pool.slot_used[:2] = True
        layer_ranges, head_ranges = [(0, 6)], [(0, 2), (2, 4)]
        config = self._config(
            [f"ns_L0-6_H{j}" for j in range(2)], layer_ranges, head_ranges
        )
        with self.assertRaisesRegex(RuntimeError, "resident"):
            KVCacheLayoutAdapter(pool, config)
        # An unchanged order is always fine, live pages or not.
        pool.unified_head_groups = 2
        KVCacheLayoutAdapter(pool, config)

    def test_suffix_count_must_match_the_slab_plan(self):
        gs = self._mha()
        pool = gs._pool(gs._logical(), head_groups=2)
        with self.assertRaisesRegex(ValueError, "suffix count"):
            KVCacheLayoutAdapter(
                pool, self._config(["ns_L0-6_H0"], [(0, 6)], [(0, 2), (2, 4)])
            )

    def test_adapter_requires_a_grid(self):
        mla = TestMlaLayoutAdapter()
        pool = mla._pool("page_first_direct", mla._logical())
        with self.assertRaisesRegex(ValueError, "chunk suffixes"):
            KVCacheLayoutAdapter(pool, self._config("ns_L0-6", None, None))


class TestDescriptorFormIsUniform(CustomTestCase):
    """A batch's descriptors must all be the same shape.

    ``MooncakeStore._uses_multi_buffer`` reads ``buffer_ptrs[0]`` and applies
    that verdict to the whole batch, so a plan that emitted a plain ``int`` for
    single-run chunks and a ``list`` for multi-run ones would hand bare
    integers to ``batch_put_from_multi_buffers``. ``plan_unified_kv`` builds
    layer ranges as ``(a, min(a + lg, end))``, so a grid whose partition does
    not divide the layer count leaves a SHORT tail range -- and on the layouts
    whose run count tracks the range width, that tail is the one chunk that
    collapses to a single run.
    """

    _HEADS, _DIM, _TOKENS = 4, 8, 64

    def _layout(self, *, layers, layer_partition, page_size):
        """A ``page_first`` plan: (kv, token, layer, head, dim)."""
        buffer = torch.zeros(
            2, self._TOKENS, layers, self._HEADS, self._DIM, dtype=torch.bfloat16
        )

        def page_view(index):
            page = buffer[:, index : index + page_size]
            return page.permute(0, 2, 1, 3, 4)

        ranges = [
            (a, min(a + layer_partition, layers))
            for a in range(0, layers, layer_partition)
        ]
        return ranges, UnifiedKVPageLayout(
            page_size=page_size,
            dtype=torch.bfloat16,
            page_view=page_view,
            sample=page_view(0),
            components=("k", "v"),
            layer_ranges=ranges,
            head_ranges=((0, self._HEADS),),
            permuted=False,
        )

    def test_a_ragged_layer_partition_does_not_mix_pointer_forms(self):
        # 5 layers by 2 leaves a width-1 tail; at page_size 1 that tail chunk
        # is one run while its siblings are two, which is exactly the mix.
        ranges, layout = self._layout(layers=5, layer_partition=2, page_size=1)
        self.assertEqual(ranges, [(0, 2), (2, 4), (4, 5)])
        self.assertEqual([len(slab.runs) for slab in layout.slabs], [2, 2, 2, 2, 1, 1])
        self.assertFalse(layout.contiguous_chunks)

        ptrs, sizes = layout.chunk_metas(torch.arange(1))
        self.assertTrue(
            all(isinstance(p, list) for p in ptrs),
            f"mixed pointer forms: {[type(p).__name__ for p in ptrs]}",
        )
        self.assertTrue(all(isinstance(n, list) for n in sizes))
        # The single-run chunk is a one-element list, not a bare int.
        self.assertEqual([len(p) for p in ptrs], [2, 2, 2, 2, 1, 1])

    def test_an_all_contiguous_plan_still_uses_plain_pointers(self):
        # page_first_direct with no head cut: every chunk is one run, so the
        # plan stays on the cheap single-descriptor path.
        buffer = torch.zeros(2, 2, 6, 4, self._HEADS, self._DIM, dtype=torch.bfloat16)
        layout = UnifiedKVPageLayout(
            page_size=4,
            dtype=torch.bfloat16,
            page_view=lambda index: buffer[:, index // 4],
            sample=buffer[:, 0],
            components=("k", "v"),
            layer_ranges=[(0, 4), (4, 6)],
            head_ranges=((0, self._HEADS),),
            permuted=False,
        )
        self.assertTrue(layout.contiguous_chunks)
        ptrs, sizes = layout.chunk_metas(torch.arange(4))
        self.assertTrue(all(isinstance(p, int) for p in ptrs))
        self.assertTrue(all(isinstance(n, int) for n in sizes))


if __name__ == "__main__":
    unittest.main()
