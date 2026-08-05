"""Unit tests for srt/mem_cache/hicache_key_scheme (canonical-grid L3 keys)."""

import json
import tempfile
import unittest

import msgspec

from sglang.srt.mem_cache.hicache_key_scheme import (
    KVCacheNamespace,
    build_canonical_cell_suffixes,
    derive_namespace,
    load_namespace_descriptor,
    namespace_digest,
    normalize_dtype,
)
from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _gqa_namespace(**overrides) -> KVCacheNamespace:
    """The design doc's worked example: GQA 70B-like, 80 layers, 8 kv heads,
    fleet head grid at lcm(TP 2,4) -> head_group=2. Layer cells are absolute
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
    return build_canonical_cell_suffixes(_gqa_namespace(), **kwargs)


def _mla_suffixes(
    *, tp_rank: int, tp_size: int, start_layer: int, end_layer: int
) -> list:
    return build_canonical_cell_suffixes(
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

    def test_descriptor_rejects_uniform_layer_grid(self):
        # layer_group != 0 is the not-yet-implemented uniform-grid fan-out.
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(
                {
                    "schema_version": 1,
                    "model_id": "m",
                    "dtype": "bfloat16",
                    "page_size": 64,
                    "rank_replicated": True,
                    "total_kv_heads": 0,
                    "layer_group": 40,
                    "head_group": 0,
                    "object_layout": "page_first",
                },
                f,
            )
            f.flush()
            with self.assertRaisesRegex(NotImplementedError, "layer_group"):
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
            build_canonical_cell_suffixes(
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


class TestCellSuffixes(CustomTestCase):
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

    def test_head_fan_out_cells(self):
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

    def test_cross_tp_cells_cover_each_other(self):
        # The cross-topology read: a TP2 rank's cells are exactly the union
        # of the two corresponding TP4 ranks' cells — a TP2 reader fetches
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
        # Both stages must derive valid, distinct cell names.
        digest = namespace_digest(_mla_namespace())
        s0 = _mla_suffixes(tp_rank=0, tp_size=2, start_layer=0, end_layer=30)
        s1 = _mla_suffixes(tp_rank=0, tp_size=2, start_layer=30, end_layer=61)
        self.assertEqual(s0, [f"{digest}_L0-30"])
        self.assertEqual(s1, [f"{digest}_L30-61"])

    def test_mla_cross_tp_size_shares_keys(self):
        # Rank-replicated pools have no head axis: TP2 and TP4 deployments
        # derive identical (single) cell names.
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
        # floor the cell count.
        namespace = _gqa_namespace(total_kv_heads=12, head_group=2)
        with self.assertRaisesRegex(ValueError, "tile"):
            build_canonical_cell_suffixes(
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
        suffixes = build_canonical_cell_suffixes(
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


class TestControllerGuards(CustomTestCase):
    """Attach-time guards of HiCacheController._build_canonical_suffix that
    fire before any storage or runtime-context access."""

    class _StubHostPool:
        def __init__(self, layout: str):
            self.layout = layout

    def _stub_controller(self, controller_cls, backend_type: str, has_draft: bool):
        controller = controller_cls.__new__(controller_cls)
        controller.storage_backend_type = backend_type
        controller.has_draft = has_draft
        controller.mem_pool_host = self._StubHostPool("page_first")
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
        return HiCacheController._build_canonical_suffix(controller, **kwargs)

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

    def test_fan_out_requires_page_head_and_mooncake(self):
        from sglang.srt.managers.cache_controller import HiCacheController

        stub = self._stub_controller(HiCacheController, "mooncake", has_draft=False)
        stub.tp_size = 2
        # head_group on a page_first pool: the layout prerequisite fails
        # before any pool-geometry access.
        with self.assertRaisesRegex(ValueError, "page_head"):
            self._build(stub, is_rank_replicated=False, head_group_knob=2)

        file_stub = self._stub_controller(HiCacheController, "file", has_draft=False)
        file_stub.tp_size = 2
        with self.assertRaisesRegex(NotImplementedError, "multi-key"):
            self._build(file_stub, is_rank_replicated=False, head_group_knob=2)

    def test_adapter_rejects_unsupported_layout_at_attach(self):
        from sglang.srt.managers.cache_controller import HiCacheController

        stub = self._stub_controller(HiCacheController, "mooncake", has_draft=False)
        stub.tp_size = 2
        stub.mem_pool_host = self._StubHostPool("layer_first")
        with self.assertRaisesRegex(ValueError, "cell adapter"):
            self._build(
                stub,
                is_rank_replicated=False,
                head_group_knob=2,
                layer_partition=[0, 30, 61],
            )

    def test_nonpositive_head_group_rejected(self):
        from sglang.srt.managers.cache_controller import HiCacheController

        stub = self._stub_controller(HiCacheController, "mooncake", has_draft=False)
        stub.tp_size = 2
        with self.assertRaisesRegex(ValueError, "positive"):
            self._build(stub, is_rank_replicated=False, head_group_knob=0)


class TestFileBackendSuffix(CustomTestCase):
    def _config(self, canonical_suffix):
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
            canonical_suffix=canonical_suffix,
        )

    def test_file_backend_uses_canonical_suffix_verbatim(self):
        from sglang.srt.mem_cache.hicache_storage import HiCacheFile

        with tempfile.TemporaryDirectory() as tmp:
            canonical = "ukv1-0123456789abcdef_L40-80_H1"
            backend = HiCacheFile(self._config(canonical), file_path=tmp)
            self.assertEqual(backend.config_suffix, f"_{canonical}")
            self.assertEqual(
                backend._get_suffixed_key("deadbeef"), f"deadbeef_{canonical}"
            )

    def test_file_backend_rejects_fan_out_list(self):
        from sglang.srt.mem_cache.hicache_storage import HiCacheFile

        with tempfile.TemporaryDirectory() as tmp:
            cells = [
                "ukv1-0123456789abcdef_L0-40_H0",
                "ukv1-0123456789abcdef_L0-40_H1",
            ]
            with self.assertRaisesRegex(NotImplementedError, "fan-out"):
                HiCacheFile(self._config(cells), file_path=tmp)

    def test_file_backend_rank_suffix_unchanged(self):
        from sglang.srt.mem_cache.hicache_storage import HiCacheFile

        with tempfile.TemporaryDirectory() as tmp:
            backend = HiCacheFile(self._config(None), file_path=tmp)
            self.assertEqual(backend.config_suffix, "_meta-llama-Llama-3-70B_1_4_2_1")


class TestLayerPartition(CustomTestCase):
    """Canonical layer partition (PP read-back): a stage spanning several
    canonical ranges owns one cell per range, so readers consume cells
    written under a different pipeline split by name alone."""

    def _mla_partition_namespace(self):
        return _mla_namespace(layer_boundaries=[0, 30, 61])

    def _suffixes(self, start_layer, end_layer, namespace=None):
        return build_canonical_cell_suffixes(
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
        )

    def test_pp_read_back_coverage(self):
        # DeepSeek-V3, partition [0,30,61] (the default uneven PP2 split):
        # PP2 stages own one cell each; a PP1 rank fans out to exactly their
        # union — PP1 reads PP2-written cells and vice versa.
        digest = namespace_digest(self._mla_partition_namespace())
        stage0 = self._suffixes(0, 30)
        stage1 = self._suffixes(30, 61)
        pp1 = self._suffixes(0, 61)
        self.assertEqual(stage0, [f"{digest}_L0-30"])
        self.assertEqual(stage1, [f"{digest}_L30-61"])
        self.assertEqual(pp1, stage0 + stage1)

    def test_misaligned_stage_rejected(self):
        with self.assertRaisesRegex(ValueError, "boundaries"):
            self._suffixes(0, 45)

    def _gqa_partition_suffixes(self, *, tp_rank, tp_size, start_layer, end_layer):
        namespace = _gqa_namespace(
            layer_boundaries=[0, 30, 61], object_layout="page_first_direct"
        )
        return build_canonical_cell_suffixes(
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
        )

    def test_mha_pp_read_back_coverage(self):
        # GQA layer fan-out: TP4 rank 2 (head_group == local, no head
        # fan-out) — PP1's cells are the union of the PP2 stages', with the
        # H coordinate constant.
        stage0 = self._gqa_partition_suffixes(
            tp_rank=2, tp_size=4, start_layer=0, end_layer=30
        )
        stage1 = self._gqa_partition_suffixes(
            tp_rank=2, tp_size=4, start_layer=30, end_layer=61
        )
        pp1 = self._gqa_partition_suffixes(
            tp_rank=2, tp_size=4, start_layer=0, end_layer=61
        )
        self.assertEqual(pp1, stage0 + stage1)
        self.assertTrue(all(sfx.endswith("_H2") for sfx in pp1))

    def test_cross_product_cells_cover_both_axes(self):
        # The cell adapter: TP2/PP1 owns the full H x L cross product
        # (layer-major, head-minor), exactly the union of the four
        # TP4/PP2 members' single cells.
        quad = self._gqa_partition_suffixes(
            tp_rank=0, tp_size=2, start_layer=0, end_layer=61
        )
        self.assertEqual(len(quad), 4)
        singles = (
            self._gqa_partition_suffixes(
                tp_rank=0, tp_size=4, start_layer=0, end_layer=30
            )
            + self._gqa_partition_suffixes(
                tp_rank=1, tp_size=4, start_layer=0, end_layer=30
            )
            + self._gqa_partition_suffixes(
                tp_rank=0, tp_size=4, start_layer=30, end_layer=61
            )
            + self._gqa_partition_suffixes(
                tp_rank=1, tp_size=4, start_layer=30, end_layer=61
            )
        )
        self.assertEqual(sorted(quad), sorted(singles))
        # Order within the quad is layer-major, head-minor.
        self.assertTrue(quad[0].endswith("_L0-30_H0"))
        self.assertTrue(quad[1].endswith("_L0-30_H1"))
        self.assertTrue(quad[2].endswith("_L30-61_H0"))
        self.assertTrue(quad[3].endswith("_L30-61_H1"))

    def test_partition_enters_digest(self):
        self.assertNotEqual(
            namespace_digest(self._mla_partition_namespace()),
            namespace_digest(_mla_namespace(layer_boundaries=[0, 61])),
        )
        self.assertNotEqual(
            namespace_digest(self._mla_partition_namespace()),
            namespace_digest(_mla_namespace()),
        )

    def test_bad_boundaries_rejected(self):
        for bad in ([0], [5, 30], [0, 30, 30]):
            with self.assertRaises(ValueError):
                derive_namespace(
                    model_id="m",
                    dtype="bfloat16",
                    page_size=64,
                    rank_replicated=True,
                    total_kv_heads=0,
                    head_group=0,
                    object_layout="page_first",
                    layer_boundaries=bad,
                )


class TestMlaLayerRangeBufferMeta(CustomTestCase):
    """Pointer math of the layer fan-out zero-copy metas (page_first_direct:
    layer-major page blocks, one contiguous slab per cell)."""

    def _stub_pool(self, layout="page_first_direct"):
        import torch

        from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost

        pool = MLATokenToKVPoolHost.__new__(MLATokenToKVPoolHost)
        pool.layout = layout
        pool.page_size = 4
        pool.kv_cache_dim = 8
        pool.layer_num = 6
        pool.dtype = torch.bfloat16
        pool.kv_buffer = torch.zeros(
            (3, pool.layer_num, pool.page_size, 1, pool.kv_cache_dim),
            dtype=pool.dtype,
        )
        return pool

    def test_pointer_math(self):
        import torch

        pool = self._stub_pool()
        itemsize = pool.dtype.itemsize
        layer_stride = pool.page_size * pool.kv_cache_dim * itemsize
        page_stride = pool.layer_num * layer_stride
        base = pool.kv_buffer.data_ptr()
        # Two pages (host indices 8..11 -> page 2, 0..3 -> page 0), two ranges.
        indices = torch.tensor([8, 9, 10, 11, 0, 1, 2, 3])
        ptrs, sizes = pool.get_layer_range_page_buffer_meta(indices, [(0, 2), (2, 6)])
        self.assertEqual(
            ptrs,
            [
                base + 2 * page_stride,
                base + 2 * page_stride + 2 * layer_stride,
                base,
                base + 2 * layer_stride,
            ],
        )
        self.assertEqual(
            sizes,
            [2 * layer_stride, 4 * layer_stride] * 2,
        )

    def test_rejects_wrong_layout_and_range(self):
        import torch

        with self.assertRaisesRegex(ValueError, "page_first_direct"):
            self._stub_pool(layout="page_first").get_layer_range_page_buffer_meta(
                torch.tensor([0, 1, 2, 3]), [(0, 2)]
            )
        with self.assertRaisesRegex(ValueError, "outside"):
            self._stub_pool().get_layer_range_page_buffer_meta(
                torch.tensor([0, 1, 2, 3]), [(0, 7)]
            )


class TestMhaLayerRangeBufferMeta(CustomTestCase):
    """MHA/GQA layer fan-out metas: one K slab and one V slab per
    (page, range), page-major, range-minor, K then V."""

    def _stub_pool(self, layout="page_first_direct"):
        import torch

        from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost

        pool = MHATokenToKVPoolHost.__new__(MHATokenToKVPoolHost)
        pool.layout = layout
        pool.page_size = 4
        pool.head_num = 2
        pool.head_dim = 8
        pool.layer_num = 6
        pool.dtype = torch.bfloat16
        page_num = 3
        pool.size = page_num * pool.page_size
        pool.kv_buffer = torch.zeros(
            (
                2,
                page_num,
                pool.layer_num,
                pool.page_size,
                pool.head_num,
                pool.head_dim,
            ),
            dtype=pool.dtype,
        )
        return pool

    def test_pointer_math(self):
        import torch

        pool = self._stub_pool()
        itemsize = pool.dtype.itemsize
        layer_stride = pool.page_size * pool.head_num * pool.head_dim * itemsize
        page_stride = pool.layer_num * layer_stride
        v_offset = pool.layer_num * pool.size * pool.head_num * pool.head_dim * itemsize
        base = pool.kv_buffer.data_ptr()
        indices = torch.tensor([8, 9, 10, 11])  # page 2
        ptrs, sizes = pool.get_layer_range_page_buffer_meta(indices, [(0, 2), (2, 6)])
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

    def test_rejects_wrong_layout(self):
        import torch

        with self.assertRaisesRegex(ValueError, "page_first_direct"):
            self._stub_pool(layout="page_head").get_layer_range_page_buffer_meta(
                torch.tensor([0, 1, 2, 3]), [(0, 2)]
            )

    def test_rejects_split_kv_pools(self):
        import torch

        pool = self._stub_pool()
        # Asymmetric MHA pools hold a (k_buffer, v_buffer) tuple.
        pool.kv_buffer = (pool.kv_buffer, pool.kv_buffer)
        with self.assertRaisesRegex(NotImplementedError, "asymmetric"):
            pool.get_layer_range_page_buffer_meta(torch.tensor([0, 1, 2, 3]), [(0, 2)])


class TestCellAdapterGatherScatter(CustomTestCase):
    """The layout-neutrality property the cell adapter exists for: every
    page-first-family layout gathers to byte-identical canonical cells
    ((head, layer, token, dim) per K/V half), and scatter inverts gather."""

    _PS, _HEADS, _LAYERS, _DIM, _PAGES = 4, 4, 6, 8, 2

    def _logical(self):
        import torch

        # L[kv, head, layer, token, dim], distinct value per element, per page.
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
        if layout == "page_first":
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

    def _grid(self):
        return [(0, 2), (2, 6)], [(0, 2), (2, 4)]  # layer ranges, head ranges

    def test_all_layouts_gather_identical_canonical_bytes(self):
        import torch

        logical = self._logical()
        layer_ranges, head_ranges = self._grid()
        indices = torch.arange(self._PAGES * self._PS)
        arenas = {}
        for layout in ("page_first", "page_first_direct", "page_head"):
            pool = self._pool(layout, logical)
            arena = torch.zeros(
                self._PAGES * pool.cell_bytes(layer_ranges, head_ranges),
                dtype=torch.uint8,
            )
            offsets, sizes = pool.gather_cells_canonical(
                indices, layer_ranges, head_ranges, arena
            )
            self.assertEqual(sum(sizes), arena.numel())
            arenas[layout] = arena
        self.assertTrue(torch.equal(arenas["page_first"], arenas["page_first_direct"]))
        self.assertTrue(torch.equal(arenas["page_first"], arenas["page_head"]))
        # And the bytes equal the canonical order computed directly from the
        # logical tensor: page-major, layer-range, head-range, K then V.
        expected = []
        for L in logical:
            for l0, l1 in layer_ranges:
                for h0, h1 in head_ranges:
                    for kv in range(2):
                        expected.append(
                            L[kv, h0:h1, l0:l1].contiguous().view(torch.uint8).flatten()
                        )
        self.assertTrue(torch.equal(arenas["page_first"], torch.cat(expected)))

    def test_scatter_inverts_gather_across_layouts(self):
        import torch

        logical = self._logical()
        layer_ranges, head_ranges = self._grid()
        indices = torch.arange(self._PAGES * self._PS)
        writer = self._pool("page_head", logical)
        arena = torch.zeros(
            self._PAGES * writer.cell_bytes(layer_ranges, head_ranges),
            dtype=torch.uint8,
        )
        writer.gather_cells_canonical(indices, layer_ranges, head_ranges, arena)

        # Scatter into an EMPTY pool of a different layout; the covered
        # rectangles must reproduce the writer's logical values.
        reader = self._pool("page_first_direct", [torch.zeros_like(l) for l in logical])
        reader.scatter_cells_canonical(indices, layer_ranges, head_ranges, arena)
        for p, L in enumerate(logical):
            got = reader._page_kv_view_canonical(p * self._PS)
            for l0, l1 in layer_ranges:
                for h0, h1 in head_ranges:
                    self.assertTrue(
                        torch.equal(got[:, h0:h1, l0:l1], L[:, h0:h1, l0:l1])
                    )


if __name__ == "__main__":
    unittest.main()
