"""Unit tests for srt/mem_cache/hicache_key_scheme (canonical-grid L3 keys)."""

import json
import tempfile
import unittest

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
    fleet = TP2xPP2 prefill + TP4xPP1 decode -> head_group=2, layer_group=40."""
    fields = dict(
        model_id="meta-llama/Llama-3-70B",
        dtype="bfloat16",
        page_size=64,
        rank_replicated=False,
        total_kv_heads=8,
        layer_group=40,
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
        layer_group=61,
        head_group=0,
    )
    fields.update(overrides)
    return KVCacheNamespace(**fields)


class TestNamespaceDigest(CustomTestCase):
    def test_digest_deterministic_and_field_sensitive(self):
        base = _gqa_namespace()
        self.assertEqual(namespace_digest(base), namespace_digest(_gqa_namespace()))
        for change in (
            {"dtype": "float16"},
            {"page_size": 32},
            {"layer_group": 20},
            {"head_group": 1},
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
                    "layer_group": 40,
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
                    "layer_group": 61,
                    "head_group": 0,
                },
                f,
            )
            f.flush()
            with self.assertRaisesRegex(ValueError, "schema_version"):
                load_namespace_descriptor(f.name)

    def test_grid_validation(self):
        with self.assertRaisesRegex(ValueError, "model_id"):
            derive_namespace(
                model_id="",
                dtype="bfloat16",
                page_size=64,
                rank_replicated=True,
                total_kv_heads=0,
                stage_layer_count=61,
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
    restricted to v1's one-cell-per-rank regime (TP4 tiles head_group=2;
    PP2 stages tile layer_group=40)."""

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
        self.assertEqual(self._tp4_pp2_suffix(tp_rank=0, pp_stage=0), f"{digest}_L0_H0")
        self.assertEqual(self._tp4_pp2_suffix(tp_rank=3, pp_stage=0), f"{digest}_L0_H3")
        self.assertEqual(self._tp4_pp2_suffix(tp_rank=1, pp_stage=1), f"{digest}_L1_H1")

    def test_pp_partitions_never_collide(self):
        # Hazard 2 of the design doc: today mooncake keys carry pp_rank but
        # not pp_size, so different PP partitions collide. Canonical L indices
        # come from the absolute layer range instead.
        self.assertNotEqual(
            self._tp4_pp2_suffix(tp_rank=0, pp_stage=0),
            self._tp4_pp2_suffix(tp_rank=0, pp_stage=1),
        )

    def test_mla_cross_tp_size_shares_keys(self):
        # Rank-replicated pools have no head axis: TP2 and TP4 deployments of
        # the same descriptor derive identical cell names.
        def suffix(tp_size: int) -> str:
            return build_canonical_cell_suffix(
                _mla_namespace(),
                attn_tp_rank=tp_size - 1,
                attn_tp_size=tp_size,
                attn_cp_size=1,
                start_layer=0,
                end_layer=61,
                local_kv_heads=0,
                dtype="bfloat16",
                page_size=64,
                model_id="deepseek-ai/DeepSeek-V3",
                rank_replicated=True,
            )

        digest = namespace_digest(_mla_namespace())
        self.assertEqual(suffix(2), f"{digest}_L0")
        self.assertEqual(suffix(2), suffix(4))

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

    def test_v1_rejects_layer_fan_out_and_non_tiling(self):
        # PP1 deployment (80 layers) against layer_group=40: two cells.
        with self.assertRaisesRegex(NotImplementedError, "layer fan-out"):
            self._suffix_with_layers(start_layer=0, end_layer=80)
        # PP4 stage (20 layers) does not tile layer_group=40 at all.
        with self.assertRaisesRegex(ValueError, "tile"):
            self._suffix_with_layers(start_layer=0, end_layer=20)

    def _suffix_with_layers(self, *, start_layer: int, end_layer: int) -> str:
        return build_canonical_cell_suffix(
            _gqa_namespace(),
            attn_tp_rank=0,
            attn_tp_size=4,
            attn_cp_size=1,
            start_layer=start_layer,
            end_layer=end_layer,
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
        with self.assertRaisesRegex(ValueError, "total_kv_heads"):
            build_canonical_cell_suffix(
                _gqa_namespace(),
                attn_tp_rank=0,
                attn_tp_size=4,
                attn_cp_size=1,
                start_layer=0,
                end_layer=40,
                local_kv_heads=4,  # 4 x 4 = 16 != descriptor's 8
                dtype="bfloat16",
                page_size=64,
                model_id="meta-llama/Llama-3-70B",
                rank_replicated=False,
            )


class TestDeriveNamespace(CustomTestCase):
    def test_derived_grid_equals_deployment(self):
        namespace = derive_namespace(
            model_id="m/1B",
            dtype="bfloat16",
            page_size=64,
            rank_replicated=False,
            total_kv_heads=8,
            stage_layer_count=16,
            local_kv_heads=4,
        )
        self.assertEqual(namespace.layer_group, 16)
        self.assertEqual(namespace.head_group, 4)
        # The derived namespace always admits its own deployment.
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
        self.assertTrue(suffix.endswith("_L0_H1"))

    def test_different_topologies_derive_disjoint_namespaces(self):
        common = dict(
            model_id="m/1B",
            dtype="bfloat16",
            page_size=64,
            rank_replicated=False,
            total_kv_heads=8,
        )
        tp2 = derive_namespace(stage_layer_count=16, local_kv_heads=4, **common)
        tp4 = derive_namespace(stage_layer_count=16, local_kv_heads=2, **common)
        self.assertNotEqual(namespace_digest(tp2), namespace_digest(tp4))

    def test_normalize_dtype(self):
        import torch

        self.assertEqual(normalize_dtype(torch.bfloat16), "bfloat16")
        self.assertEqual(normalize_dtype(torch.float8_e4m3fn), "float8_e4m3fn")


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
            canonical = "ukv1-0123456789abcdef_L1_H1"
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
