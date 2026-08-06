"""Unit tests for srt/mem_cache/hicache_key_scheme (unified L3 keys)."""

import json
import tempfile
import unittest

import msgspec

from sglang.srt.mem_cache.hicache_key_scheme import (
    KVCacheLayoutAdapter,
    KVCacheNamespace,
    build_unified_suffixes,
    derive_namespace,
    load_namespace_descriptor,
    namespace_digest,
    normalize_dtype,
    plan_unified_kv,
)
from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


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
        object_layout="page_head",
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
        object_layout="page_head",
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
            {"object_layout": "page_first"},
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
                    "object_layout": "page_head",
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
                object_layout="page_head",
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
            ({"object_layout": "page_first"}, "object_layout"),
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
                object_layout="page_head",
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
            object_layout="page_head",
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
            object_layout="page_head",
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
        tp2 = derive_namespace(head_group=2, object_layout="page_head", **common)
        tp4 = derive_namespace(head_group=2, object_layout="page_head", **common)
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
                derive_namespace(head_group=4, object_layout="page_head", **common)
            ),
            namespace_digest(
                derive_namespace(head_group=2, object_layout="page_head", **common)
            ),
        )

    def test_normalize_dtype(self):
        import torch

        self.assertEqual(normalize_dtype(torch.bfloat16), "bfloat16")
        self.assertEqual(normalize_dtype(torch.float8_e4m3fn), "float8_e4m3fn")
        self.assertEqual(normalize_dtype(torch.float8_e5m2), "float8_e5m2")


class TestUnifiedKVPlan(CustomTestCase):
    """plan_unified_kv: any partition knob selects the adapter, whose
    unified byte order makes the namespace layout-neutral (unified-v1)."""

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
            pool_layout="page_first",
        )
        kwargs.update(overrides)
        return plan_unified_kv(**kwargs)

    def test_no_knobs_keeps_raw_layout(self):
        plan = self._plan()
        self.assertFalse(plan.adapter)
        self.assertEqual(plan.namespace.object_layout, "page_first")
        self.assertEqual(len(plan.suffixes), 1)
        self.assertIsNone(plan.layer_ranges)
        self.assertIsNone(plan.head_ranges)

    def test_any_knob_selects_adapter(self):
        # head knob alone: adapter, single layer window, head chunks.
        plan = self._plan(head_group_knob=2)
        self.assertTrue(plan.adapter)
        self.assertEqual(plan.namespace.object_layout, "unified-v1")
        self.assertEqual(plan.layer_ranges, [(0, 61)])
        self.assertEqual(plan.head_ranges, [(0, 2), (2, 4)])
        self.assertEqual(len(plan.suffixes), 2)
        # layer knob alone: adapter, the rank's full head span as one chunk.
        plan = self._plan(layer_partition=30)
        self.assertTrue(plan.adapter)
        self.assertEqual(plan.layer_ranges, [(0, 30), (30, 60), (60, 61)])
        self.assertEqual(plan.head_ranges, [(0, 4)])
        self.assertEqual(len(plan.suffixes), 3)
        # both: full cross product, layer-major.
        plan = self._plan(head_group_knob=2, layer_partition=30)
        self.assertEqual(len(plan.suffixes), 6)

    def test_rank_replicated_layer_partition_uses_adapter(self):
        # MLA fleets may still need the adapter: the layer grid must be
        # byte-uniform regardless of each deployment's host layout.
        plan = self._plan(
            rank_replicated=True,
            local_kv_heads=0,
            pool_layout="layer_first",
            layer_partition=30,
        )
        self.assertTrue(plan.adapter)
        self.assertEqual(plan.namespace.object_layout, "unified-v1")
        self.assertIsNone(plan.head_ranges)
        self.assertEqual(plan.layer_ranges, [(0, 30), (30, 60), (60, 61)])
        # head_group alone is a no-op for replicated pools (no head axis).
        plan = self._plan(rank_replicated=True, local_kv_heads=0, head_group_knob=2)
        self.assertFalse(plan.adapter)

    def test_adapter_digest_is_layout_neutral(self):
        digests = {
            namespace_digest(self._plan(layer_partition=30, pool_layout=lay).namespace)
            for lay in ("layer_first", "page_first", "page_first_direct", "page_head")
        }
        self.assertEqual(len(digests), 1)
        # Without a knob the layout stays in the identity: disjoint keyspaces.
        raw = {
            namespace_digest(self._plan(pool_layout=lay).namespace)
            for lay in ("page_first", "page_head")
        }
        self.assertEqual(len(raw), 2)

    def test_plan_validation(self):
        with self.assertRaisesRegex(ValueError, "layout adapter"):
            self._plan(pool_layout="page_first_kv_split", layer_partition=30)
        with self.assertRaisesRegex(ValueError, "divide"):
            self._plan(head_group_knob=3)
        with self.assertRaisesRegex(ValueError, "positive"):
            self._plan(head_group_knob=0)
        with self.assertRaisesRegex(NotImplementedError, "1 kv head"):
            self._plan(local_kv_heads=1, attn_tp_size=8)
        # An explicit head_group attests the sharding and lifts the 1-head
        # ambiguity.
        plan = self._plan(local_kv_heads=1, attn_tp_size=8, head_group_knob=1)
        self.assertEqual(plan.head_ranges, [(0, 1)])


class TestControllerGuards(CustomTestCase):
    """Attach-time guards of HiCacheController._build_unified_suffix."""

    class _StubHostPool:
        def __init__(self, layout: str):
            self.layout = layout
            self.head_num = 4
            self.start_layer = 0
            self.end_layer = 61
            import torch

            self.kv_buffer = torch.zeros(1)

    class _StubDevicePool:
        def __init__(self):
            import torch

            self.dtype = torch.bfloat16

    def _stub_controller(self, controller_cls, backend_type: str, has_draft: bool):
        controller = controller_cls.__new__(controller_cls)
        controller.storage_backend_type = backend_type
        controller.has_draft = has_draft
        controller.mem_pool_host = self._StubHostPool("page_first")
        controller.mem_pool_device = self._StubDevicePool()
        controller.page_size = 64
        controller.tp_rank, controller.tp_size = 0, 2
        controller.pp_rank, controller.pp_size = 0, 1
        return controller

    def _build(self, controller, **overrides):
        from sglang.srt.managers.cache_controller import HiCacheController

        kwargs = dict(
            model_name="m",
            is_rank_replicated=True,
            attn_cp_size=1,
            head_group_knob=None,
            layer_partition=None,
        )
        kwargs.update(overrides)
        return HiCacheController._build_unified_suffix(controller, **kwargs)

    def test_backend_allowlist_and_draft_guards(self):
        from sglang.srt.managers.cache_controller import HiCacheController

        nixl = self._stub_controller(HiCacheController, "nixl", has_draft=False)
        with self.assertRaisesRegex(NotImplementedError, "file and mooncake"):
            self._build(nixl)

        drafty = self._stub_controller(HiCacheController, "file", has_draft=True)
        with self.assertRaisesRegex(NotImplementedError, "draft"):
            self._build(drafty)

    def test_subclass_controllers_rejected(self):
        from sglang.srt.managers.cache_controller import HiCacheController

        class FakeHybridController(HiCacheController):
            pass

        hybrid = self._stub_controller(
            FakeHybridController, "mooncake", has_draft=False
        )
        with self.assertRaisesRegex(NotImplementedError, "hybrid"):
            self._build(hybrid)

    def test_partition_knobs_require_mooncake(self):
        from sglang.srt.managers.cache_controller import HiCacheController

        file_stub = self._stub_controller(HiCacheController, "file", has_draft=False)
        with self.assertRaisesRegex(NotImplementedError, "multi-key"):
            self._build(file_stub, is_rank_replicated=False, head_group_knob=2)
        with self.assertRaisesRegex(NotImplementedError, "multi-key"):
            self._build(file_stub, layer_partition=30)

    def test_adapter_plan_from_any_layout(self):
        # No host-layout requirement: head_group on a page_first pool
        # attaches through the adapter (list suffix + local chunk grid).
        from sglang.srt.managers.cache_controller import HiCacheController

        stub = self._stub_controller(HiCacheController, "mooncake", has_draft=False)
        suffix, layer_ranges, head_ranges = self._build(
            stub, is_rank_replicated=False, head_group_knob=2
        )
        self.assertIsInstance(suffix, list)
        self.assertEqual(len(suffix), 2)
        self.assertEqual(layer_ranges, [(0, 61)])
        self.assertEqual(head_ranges, [(0, 2), (2, 4)])

    def test_adapter_rejects_unsupported_layout_at_attach(self):
        from sglang.srt.managers.cache_controller import HiCacheController

        stub = self._stub_controller(HiCacheController, "mooncake", has_draft=False)
        stub.mem_pool_host = self._StubHostPool("page_first_kv_split")
        with self.assertRaisesRegex(ValueError, "layout adapter"):
            self._build(
                stub,
                is_rank_replicated=False,
                head_group_knob=2,
                layer_partition=30,
            )

    def test_adapter_rejects_split_kv_pools(self):
        from sglang.srt.managers.cache_controller import HiCacheController

        stub = self._stub_controller(HiCacheController, "mooncake", has_draft=False)
        stub.mem_pool_host.kv_buffer = (1, 2)
        with self.assertRaisesRegex(NotImplementedError, "split K/V"):
            self._build(stub, is_rank_replicated=False, head_group_knob=2)

    def test_nonpositive_head_group_rejected(self):
        from sglang.srt.managers.cache_controller import HiCacheController

        stub = self._stub_controller(HiCacheController, "mooncake", has_draft=False)
        with self.assertRaisesRegex(ValueError, "positive"):
            self._build(stub, is_rank_replicated=False, head_group_knob=0)


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
    """MLA layout adapter: page_first_direct is already the unified order,
    so its slabs skip the staging copy (pointer math == the pool address);
    other layouts stage byte-identical unified chunks."""

    _PS, _LAYERS, _DIM, _PAGES = 4, 6, 8, 3

    def _logical(self):
        import torch

        torch.manual_seed(0)
        # L[layer, token, 1, dim] per page — the unified MLA order.
        return [
            torch.randn(self._LAYERS, self._PS, 1, self._DIM, dtype=torch.bfloat16)
            for _ in range(self._PAGES)
        ]

    def _pool(self, layout, logical):
        import torch

        from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost

        pool = MLATokenToKVPoolHost.__new__(MLATokenToKVPoolHost)
        pool.layout = layout
        pool.page_size = self._PS
        pool.kv_cache_dim = self._DIM
        pool.layer_num = self._LAYERS
        pool.dtype = torch.bfloat16
        pool.size = self._PAGES * self._PS
        if layout == "layer_first":
            pool.kv_buffer = torch.zeros(
                self._LAYERS, pool.size, 1, self._DIM, dtype=pool.dtype
            )
            for p, L in enumerate(logical):
                pool.kv_buffer[:, p * self._PS : (p + 1) * self._PS] = L
        elif layout == "page_first":
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

    def test_page_first_direct_is_zero_copy(self):
        import torch

        pool = self._pool("page_first_direct", self._logical())
        self.assertTrue(pool.unified_zero_copy(self._RANGES))
        itemsize = pool.dtype.itemsize
        layer_stride = pool.page_size * pool.kv_cache_dim * itemsize
        page_stride = pool.layer_num * layer_stride
        base = pool.kv_buffer.data_ptr()
        # Pages 2 and 0; no staging needed for direct slabs.
        indices = torch.tensor([8, 9, 10, 11, 0, 1, 2, 3])
        for meta in (pool.gather_unified_chunks, pool.get_unified_chunk_meta):
            ptrs, sizes = meta(indices, self._RANGES, None, None)
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

    def test_all_layouts_gather_identical_unified_bytes(self):
        import ctypes

        import torch

        logical = self._logical()
        indices = torch.arange(self._PAGES * self._PS)
        expected = b"".join(
            bytes(L[l0:l1].contiguous().view(torch.uint8).flatten().tolist())
            for L in logical
            for l0, l1 in self._RANGES
        )
        for layout in ("layer_first", "page_first", "page_first_direct"):
            pool = self._pool(layout, logical)
            staging = torch.zeros(
                self._PAGES * pool.unified_bytes_per_page(self._RANGES),
                dtype=torch.uint8,
            )
            ptrs, sizes = pool.gather_unified_chunks(
                indices, self._RANGES, None, staging
            )
            got = b"".join(
                ctypes.string_at(ptr, size) for ptr, size in zip(ptrs, sizes)
            )
            self.assertEqual(got, expected, layout)

    def test_scatter_inverts_fetch_for_staged_layouts(self):
        import ctypes

        import torch

        logical = self._logical()
        indices = torch.arange(self._PAGES * self._PS)
        writer = self._pool("page_first_direct", logical)
        src_ptrs, src_sizes = writer.gather_unified_chunks(
            indices, self._RANGES, None, None
        )
        reader = self._pool("layer_first", [torch.zeros_like(l) for l in logical])
        self.assertFalse(reader.unified_zero_copy(self._RANGES))
        staging = torch.zeros(
            self._PAGES * reader.unified_bytes_per_page(self._RANGES), dtype=torch.uint8
        )
        dst_ptrs, dst_sizes = reader.get_unified_chunk_meta(
            indices, self._RANGES, None, staging
        )
        self.assertEqual(src_sizes, dst_sizes)
        # Emulate the store fetch, then scatter; page 1 "failed".
        for dst, src, size in zip(dst_ptrs, src_ptrs, src_sizes):
            ctypes.memmove(dst, src, size)
        page_ok = [True, False, True]
        reader.scatter_unified_chunks(indices, self._RANGES, None, staging, page_ok)
        for p, L in enumerate(logical):
            got = reader._page_view_unified(p * self._PS)
            for l0, l1 in self._RANGES:
                if page_ok[p]:
                    self.assertTrue(torch.equal(got[l0:l1], L[l0:l1]))
                else:
                    self.assertEqual(got[l0:l1].abs().sum().item(), 0)


class TestMhaDirectChunks(CustomTestCase):
    """MHA skip-convert: the unified order is (head, layer, token, dim), so
    a multi-head pool always stages — but a single-kv-head page_first_direct
    pool is degenerate-contiguous and goes zero-copy."""

    def _pool(self, head_num):
        import torch

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

    def test_single_head_layer_chunks_are_direct(self):
        import torch

        pool = self._pool(head_num=1)
        ranges, heads = [(0, 2), (2, 6)], [(0, 1)]
        self.assertTrue(pool.unified_zero_copy(ranges, heads))
        itemsize = pool.dtype.itemsize
        layer_stride = pool.page_size * pool.head_dim * itemsize
        page_stride = pool.layer_num * layer_stride
        v_offset = pool.layer_num * pool.size * pool.head_dim * itemsize
        base = pool.kv_buffer.data_ptr()
        ptrs, sizes = pool.get_unified_chunk_meta(
            torch.tensor([8, 9, 10, 11]), ranges, heads, None
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

    def test_multi_head_chunks_stage(self):
        pool = self._pool(head_num=2)
        self.assertFalse(pool.unified_zero_copy([(0, 2)], [(0, 2)]))

    def test_rejects_split_kv_pools(self):
        import torch

        pool = self._pool(head_num=2)
        pool.kv_buffer = (pool.kv_buffer, pool.kv_buffer)
        with self.assertRaisesRegex(NotImplementedError, "split K/V"):
            pool.gather_unified_chunks(
                torch.tensor([0, 1, 2, 3]), [(0, 2)], [(0, 2)], None
            )


class TestLayoutAdapterGatherScatter(CustomTestCase):
    """The layout-neutrality property the adapter exists for: every
    supported layout gathers byte-identical unified chunks ((head, layer,
    token, dim) per K/V half), and the fetch + scatter path inverts them."""

    _PS, _HEADS, _LAYERS, _DIM, _PAGES = 4, 4, 6, 8, 2

    def _logical(self):
        import torch

        # L[kv, head, layer, token, dim] per page.
        torch.manual_seed(0)
        return [
            torch.randn(
                2,
                self._HEADS,
                self._LAYERS,
                self._PS,
                self._DIM,
                dtype=torch.bfloat16,
            )
            for _ in range(self._PAGES)
        ]

    def _pool(self, layout, logical):
        import torch

        from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost

        pool = MHATokenToKVPoolHost.__new__(MHATokenToKVPoolHost)
        pool.layout = layout
        pool.page_size = self._PS
        pool.head_num = self._HEADS
        pool.head_dim = self._DIM
        pool.layer_num = self._LAYERS
        pool.dtype = torch.bfloat16
        pool.size = self._PAGES * self._PS
        if layout == "layer_first":
            pool.kv_buffer = torch.zeros(
                2,
                self._LAYERS,
                pool.size,
                self._HEADS,
                self._DIM,
                dtype=pool.dtype,
            )
            for p, L in enumerate(logical):
                pool.kv_buffer[:, :, p * self._PS : (p + 1) * self._PS] = L.permute(
                    0, 2, 3, 1, 4
                )
        elif layout == "page_first":
            pool.kv_buffer = torch.zeros(
                2,
                pool.size,
                self._LAYERS,
                self._HEADS,
                self._DIM,
                dtype=pool.dtype,
            )
            for p, L in enumerate(logical):
                pool.kv_buffer[:, p * self._PS : (p + 1) * self._PS] = L.permute(
                    0, 3, 2, 1, 4
                )
        elif layout == "page_first_direct":
            pool.kv_buffer = torch.zeros(
                2,
                self._PAGES,
                self._LAYERS,
                self._PS,
                self._HEADS,
                self._DIM,
                dtype=pool.dtype,
            )
            for p, L in enumerate(logical):
                pool.kv_buffer[:, p] = L.permute(0, 2, 3, 1, 4)
        elif layout == "page_head":
            pool.kv_buffer = torch.zeros(
                2,
                self._PAGES,
                self._HEADS,
                self._PS,
                self._LAYERS,
                self._DIM,
                dtype=pool.dtype,
            )
            for p, L in enumerate(logical):
                pool.kv_buffer[:, p] = L.permute(0, 1, 3, 2, 4)
        return pool

    _LAYOUTS = ("layer_first", "page_first", "page_first_direct", "page_head")

    def _grid(self):
        return [(0, 2), (2, 6)], [(0, 2), (2, 4)]  # layer ranges, head ranges

    def test_all_layouts_gather_identical_unified_bytes(self):
        import ctypes

        import torch

        logical = self._logical()
        layer_ranges, head_ranges = self._grid()
        indices = torch.arange(self._PAGES * self._PS)
        # Unified-order bytes computed straight from the logical tensor:
        # page-major, layer-range, head-range, K then V.
        expected = b"".join(
            bytes(L[kv, h0:h1, l0:l1].contiguous().view(torch.uint8).flatten().tolist())
            for L in logical
            for l0, l1 in layer_ranges
            for h0, h1 in head_ranges
            for kv in range(2)
        )
        for layout in self._LAYOUTS:
            pool = self._pool(layout, logical)
            staging = torch.zeros(
                self._PAGES * pool.unified_bytes_per_page(layer_ranges, head_ranges),
                dtype=torch.uint8,
            )
            ptrs, sizes = pool.gather_unified_chunks(
                indices, layer_ranges, head_ranges, staging
            )
            got = b"".join(
                ctypes.string_at(ptr, size) for ptr, size in zip(ptrs, sizes)
            )
            self.assertEqual(got, expected, layout)

    def test_fetch_and_scatter_invert_gather_across_layouts(self):
        import ctypes

        import torch

        logical = self._logical()
        layer_ranges, head_ranges = self._grid()
        indices = torch.arange(self._PAGES * self._PS)
        writer = self._pool("page_head", logical)
        staging_w = torch.zeros(
            self._PAGES * writer.unified_bytes_per_page(layer_ranges, head_ranges),
            dtype=torch.uint8,
        )
        src_ptrs, src_sizes = writer.gather_unified_chunks(
            indices, layer_ranges, head_ranges, staging_w
        )

        # Emulate a fetch into an EMPTY pool of a different layout, then
        # scatter; the covered rectangles must reproduce the writer's values.
        for layout in ("layer_first", "page_first_direct"):
            reader = self._pool(layout, [torch.zeros_like(l) for l in logical])
            staging_r = torch.zeros_like(staging_w)
            dst_ptrs, dst_sizes = reader.get_unified_chunk_meta(
                indices, layer_ranges, head_ranges, staging_r
            )
            self.assertEqual(src_sizes, dst_sizes)
            for dst, src, size in zip(dst_ptrs, src_ptrs, src_sizes):
                ctypes.memmove(dst, src, size)
            reader.scatter_unified_chunks(
                indices, layer_ranges, head_ranges, staging_r, [True] * self._PAGES
            )
            for p, L in enumerate(logical):
                got = reader._page_kv_view_unified(p * self._PS)
                for l0, l1 in layer_ranges:
                    for h0, h1 in head_ranges:
                        self.assertTrue(
                            torch.equal(got[:, h0:h1, l0:l1], L[:, h0:h1, l0:l1])
                        )


class TestLayoutAdapterStaging(CustomTestCase):
    """Backend-neutral KVCacheLayoutAdapter: key fan-out, staging-sized
    sub-batching, and the pointer round trip a backend performs — no store
    involved."""

    class _UnpinnedLayoutAdapter(KVCacheLayoutAdapter):
        # CPU-only test hosts cannot pin memory.
        def _alloc_staging(self, numel):
            import torch

            return torch.empty(numel, dtype=torch.uint8)

    def _config(self, suffixes, layer_ranges, head_ranges, extra=None):
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
        )

    def test_all_direct_pool_allocates_no_staging(self):
        # MLA on page_first_direct: every slab pool-contiguous.
        mla = TestMlaLayoutAdapter()
        pool = mla._pool("page_first_direct", mla._logical())
        registered = []
        adapter = KVCacheLayoutAdapter(
            pool,
            self._config(["ns_L0-2", "ns_L2-6"], [(0, 2), (2, 6)], None),
            register_buffer=registered.append,
        )
        self.assertIsNone(adapter.staging_set)
        self.assertIsNone(adapter.staging_get)
        self.assertEqual(registered, [])
        self.assertEqual(adapter.keys_per_page, 2)
        self.assertEqual(
            adapter.chunk_keys(["h1", "h2"]),
            ["h1_ns_L0-2_k", "h1_ns_L2-6_k", "h2_ns_L0-2_k", "h2_ns_L2-6_k"],
        )

    def test_staged_round_trip_with_single_page_sub_batches(self):
        import ctypes

        import torch

        gs = TestLayoutAdapterGatherScatter()
        logical = gs._logical()
        layer_ranges, head_ranges = gs._grid()
        suffixes = [
            f"ns_L{l0}-{l1}_H{j}"
            for l0, l1 in layer_ranges
            for j in range(len(head_ranges))
        ]
        # staging_buffer_mb=0 floors the staging at ONE page's chunks,
        # forcing a sub-batch per page and buffer reuse across sub-batches.
        config = self._config(
            suffixes, layer_ranges, head_ranges, extra={"staging_buffer_mb": 0}
        )
        writer = self._UnpinnedLayoutAdapter(gs._pool("page_head", logical), config)
        self.assertEqual(writer.staging_pages, 1)
        self.assertEqual(writer.keys_per_page, 8)

        page_keys = [f"h{p}" for p in range(gs._PAGES)]
        indices = torch.arange(gs._PAGES * gs._PS)
        store = {}
        for sub_keys, sub_indices in writer.sub_batches(page_keys, indices):
            self.assertEqual(len(sub_keys), 1)
            ptrs, sizes = writer.gather(sub_indices)
            for key, ptr, size in zip(writer.chunk_keys(sub_keys), ptrs, sizes):
                store[key] = ctypes.string_at(ptr, size)
        self.assertEqual(len(store), gs._PAGES * 8)

        registered = []
        reader = self._UnpinnedLayoutAdapter(
            gs._pool("page_first", [torch.zeros_like(l) for l in logical]),
            config,
            register_buffer=registered.append,
        )
        self.assertEqual(len(registered), 2)
        for sub_keys, sub_indices in reader.sub_batches(page_keys, indices):
            ptrs, sizes = reader.read_metas(sub_indices)
            for key, ptr, size in zip(reader.chunk_keys(sub_keys), ptrs, sizes):
                ctypes.memmove(ptr, store[key], size)
            reader.scatter(sub_indices, [True] * len(sub_keys))
        for p, L in enumerate(logical):
            got = reader.pool._page_kv_view_unified(p * gs._PS)
            for l0, l1 in layer_ranges:
                for h0, h1 in head_ranges:
                    self.assertTrue(
                        torch.equal(got[:, h0:h1, l0:l1], L[:, h0:h1, l0:l1])
                    )


if __name__ == "__main__":
    unittest.main()
