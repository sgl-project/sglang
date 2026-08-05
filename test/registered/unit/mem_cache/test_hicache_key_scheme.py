"""Unit tests for srt/mem_cache/hicache_key_scheme (canonical-grid L3 keys)."""

import json
import tempfile
import unittest

import msgspec

from sglang.srt.mem_cache.hicache_key_scheme import (
    KVCacheNamespace,
    build_canonical_cell_suffix,
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
    per-stage ranges in v1 (layer_group stays 0)."""
    fields = dict(
        model_id="meta-llama/Llama-3-70B",
        dtype="bfloat16",
        page_size=64,
        rank_replicated=False,
        total_kv_heads=8,
        head_group=2,
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
    )
    fields.update(overrides)
    return KVCacheNamespace(**fields)


def _mla_suffix(*, tp_rank: int, tp_size: int, start_layer: int, end_layer: int) -> str:
    return build_canonical_cell_suffix(
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
                local_kv_heads=0,
            )
        with self.assertRaisesRegex(ValueError, "divide"):
            build_canonical_cell_suffix(
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
            )


class TestCellSuffix(CustomTestCase):
    """The section 3.1 worked example of DESIGN_l3_canonical_shard_grid.md,
    with v1's absolute layer-range coordinates (TP4 tiles head_group=2)."""

    def _tp4_pp2_suffix(self, *, tp_rank: int, pp_stage: int) -> str:
        return build_canonical_cell_suffix(
            _gqa_namespace(),
            attn_tp_rank=tp_rank,
            attn_tp_size=4,
            attn_cp_size=1,
            start_layer=40 * pp_stage,
            end_layer=40 * (pp_stage + 1),
            local_kv_heads=2,
            dtype="bfloat16",
            page_size=64,
            model_id="meta-llama/Llama-3-70B",
            rank_replicated=False,
        )

    def test_cell_coordinates_are_canonical(self):
        digest = namespace_digest(_gqa_namespace())
        self.assertEqual(
            self._tp4_pp2_suffix(tp_rank=0, pp_stage=0), f"{digest}_L0-40_H0"
        )
        self.assertEqual(
            self._tp4_pp2_suffix(tp_rank=3, pp_stage=0), f"{digest}_L0-40_H3"
        )
        self.assertEqual(
            self._tp4_pp2_suffix(tp_rank=1, pp_stage=1), f"{digest}_L40-80_H1"
        )

    def test_pp_partitions_never_collide(self):
        # Hazard 2 of the design doc: today mooncake keys carry pp_rank but
        # not pp_size, so different PP partitions collide. Absolute layer
        # ranges make differing partitions miss instead.
        self.assertNotEqual(
            self._tp4_pp2_suffix(tp_rank=0, pp_stage=0),
            self._tp4_pp2_suffix(tp_rank=0, pp_stage=1),
        )

    def test_uneven_pp_stages_attach(self):
        # DeepSeek-V3 has 61 layers; the default PP2 split is [0,30)/[30,61).
        # Both stages must derive valid, distinct cell names (this crashed
        # when layer cells were grid indices instead of ranges).
        digest = namespace_digest(_mla_namespace())
        s0 = _mla_suffix(tp_rank=0, tp_size=2, start_layer=0, end_layer=30)
        s1 = _mla_suffix(tp_rank=0, tp_size=2, start_layer=30, end_layer=61)
        self.assertEqual(s0, f"{digest}_L0-30")
        self.assertEqual(s1, f"{digest}_L30-61")
        self.assertNotEqual(s0, s1)

    def test_mla_cross_tp_size_shares_keys(self):
        # Rank-replicated pools have no head axis: TP2 and TP4 deployments of
        # the same descriptor derive identical cell names.
        digest = namespace_digest(_mla_namespace())
        s_tp2 = _mla_suffix(tp_rank=1, tp_size=2, start_layer=0, end_layer=61)
        s_tp4 = _mla_suffix(tp_rank=3, tp_size=4, start_layer=0, end_layer=61)
        self.assertEqual(s_tp2, f"{digest}_L0-61")
        self.assertEqual(s_tp2, s_tp4)

    def test_v1_rejects_head_fan_out(self):
        # A TP2 rank holds 4 heads = 2 head groups: needs multi-cell fan-out.
        with self.assertRaisesRegex(NotImplementedError, "head fan-out"):
            build_canonical_cell_suffix(
                _gqa_namespace(),
                attn_tp_rank=0,
                attn_tp_size=2,
                attn_cp_size=1,
                start_layer=0,
                end_layer=40,
                local_kv_heads=4,
                dtype="bfloat16",
                page_size=64,
                model_id="meta-llama/Llama-3-70B",
                rank_replicated=False,
            )

    def test_rejects_invalid_layer_range(self):
        with self.assertRaisesRegex(ValueError, "layer range"):
            build_canonical_cell_suffix(
                _gqa_namespace(),
                attn_tp_rank=0,
                attn_tp_size=4,
                attn_cp_size=1,
                start_layer=40,
                end_layer=40,
                local_kv_heads=2,
                dtype="bfloat16",
                page_size=64,
                model_id="meta-llama/Llama-3-70B",
                rank_replicated=False,
            )

    def test_rejects_cp(self):
        with self.assertRaisesRegex(NotImplementedError, "context parallelism"):
            build_canonical_cell_suffix(
                _mla_namespace(),
                attn_tp_rank=0,
                attn_tp_size=2,
                attn_cp_size=2,
                start_layer=0,
                end_layer=61,
                local_kv_heads=0,
                dtype="bfloat16",
                page_size=64,
                model_id="deepseek-ai/DeepSeek-V3",
                rank_replicated=True,
            )

    def test_identity_mismatches_fail_fast(self):
        cases = [
            ({"dtype": "float16"}, "dtype"),
            ({"page_size": 32}, "page_size"),
            ({"model_id": "other/model"}, "model_id"),
        ]
        for deployment_override, expected_msg in cases:
            kwargs = dict(
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
            )
            kwargs.update(deployment_override)
            with self.assertRaisesRegex(ValueError, expected_msg):
                build_canonical_cell_suffix(_gqa_namespace(), **kwargs)
        with self.assertRaisesRegex(ValueError, "rank_replicated"):
            build_canonical_cell_suffix(
                _gqa_namespace(),
                attn_tp_rank=0,
                attn_tp_size=4,
                attn_cp_size=1,
                start_layer=0,
                end_layer=40,
                local_kv_heads=2,
                dtype="bfloat16",
                page_size=64,
                model_id="meta-llama/Llama-3-70B",
                rank_replicated=True,
            )

    def test_wrong_descriptor_head_count_fails(self):
        # Also the kv-head replication case: a truthful descriptor for an
        # 8-head model at TP16 reports total_kv_heads=8 != 1 x 16.
        with self.assertRaisesRegex(ValueError, "total_kv_heads"):
            build_canonical_cell_suffix(
                _gqa_namespace(),
                attn_tp_rank=0,
                attn_tp_size=16,
                attn_cp_size=1,
                start_layer=0,
                end_layer=40,
                local_kv_heads=1,
                dtype="bfloat16",
                page_size=64,
                model_id="meta-llama/Llama-3-70B",
                rank_replicated=False,
            )


class TestDeriveNamespace(CustomTestCase):
    def test_derived_namespace_admits_own_deployment(self):
        namespace = derive_namespace(
            model_id="m/1B",
            dtype="bfloat16",
            page_size=64,
            rank_replicated=False,
            total_kv_heads=8,
            local_kv_heads=4,
        )
        self.assertEqual(namespace.head_group, 4)
        suffix = build_canonical_cell_suffix(
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
        )
        self.assertTrue(suffix.endswith("_L0-16_H1"))

    def test_different_topologies_derive_disjoint_namespaces(self):
        common = dict(
            model_id="m/1B",
            dtype="bfloat16",
            page_size=64,
            rank_replicated=False,
            total_kv_heads=8,
        )
        tp2 = derive_namespace(local_kv_heads=4, **common)
        tp4 = derive_namespace(local_kv_heads=2, **common)
        self.assertNotEqual(namespace_digest(tp2), namespace_digest(tp4))

    def test_normalize_dtype(self):
        import torch

        self.assertEqual(normalize_dtype(torch.bfloat16), "bfloat16")
        self.assertEqual(normalize_dtype(torch.float8_e4m3fn), "float8_e4m3fn")
        self.assertEqual(normalize_dtype(torch.float8_e5m2), "float8_e5m2")


class TestControllerGuards(CustomTestCase):
    """Attach-time guards of HiCacheController._build_canonical_suffix that
    fire before any storage or runtime-context access."""

    def _stub_controller(self, controller_cls, backend_type: str, has_draft: bool):
        controller = controller_cls.__new__(controller_cls)
        controller.storage_backend_type = backend_type
        controller.has_draft = has_draft
        return controller

    def test_backend_allowlist_and_draft_guards(self):
        from sglang.srt.managers.cache_controller import HiCacheController

        nixl = self._stub_controller(HiCacheController, "nixl", has_draft=False)
        with self.assertRaisesRegex(NotImplementedError, "file and mooncake"):
            HiCacheController._build_canonical_suffix(
                nixl, model_name="m", is_rank_replicated=True, attn_cp_size=1
            )

        drafty = self._stub_controller(HiCacheController, "file", has_draft=True)
        with self.assertRaisesRegex(NotImplementedError, "draft"):
            HiCacheController._build_canonical_suffix(
                drafty, model_name="m", is_rank_replicated=True, attn_cp_size=1
            )

    def test_subclass_controllers_rejected(self):
        from sglang.srt.managers.cache_controller import HiCacheController

        class FakeHybridController(HiCacheController):
            pass

        hybrid = self._stub_controller(
            FakeHybridController, "mooncake", has_draft=False
        )
        with self.assertRaisesRegex(NotImplementedError, "hybrid"):
            HiCacheController._build_canonical_suffix(
                hybrid, model_name="m", is_rank_replicated=True, attn_cp_size=1
            )


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

    def test_file_backend_rank_suffix_unchanged(self):
        from sglang.srt.mem_cache.hicache_storage import HiCacheFile

        with tempfile.TemporaryDirectory() as tmp:
            backend = HiCacheFile(self._config(None), file_path=tmp)
            self.assertEqual(backend.config_suffix, "_meta-llama-Llama-3-70B_1_4_2_1")


if __name__ == "__main__":
    unittest.main()
