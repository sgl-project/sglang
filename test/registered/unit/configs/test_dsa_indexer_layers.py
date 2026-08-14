import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.configs.model_config import resolve_dsa_indexer_layer_ids
from sglang.srt.utils import common
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _glm_dsa_config(**overrides):
    values = {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "index_topk": 2048,
        "index_topk_freq": 4,
        "index_skip_topk_offset": 3,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TestDSAIndexerLayerResolver(CustomTestCase):
    def test_glm52_resolves_only_physical_indexer_layers(self):
        layer_ids = resolve_dsa_indexer_layer_ids(_glm_dsa_config(), 0, 78)

        self.assertEqual(layer_ids[:4], (0, 1, 2, 6))
        self.assertEqual(layer_ids[-1], 74)
        self.assertEqual(len(layer_ids), 21)

    def test_pipeline_stage_uses_absolute_layer_ids(self):
        self.assertEqual(
            resolve_dsa_indexer_layer_ids(_glm_dsa_config(), 20, 40),
            (22, 26, 30, 34, 38),
        )

    def test_nextn_always_owns_an_indexer(self):
        config = _glm_dsa_config(index_topk_pattern="RS")
        self.assertEqual(resolve_dsa_indexer_layer_ids(config, 1, 2), ())
        self.assertEqual(
            resolve_dsa_indexer_layer_ids(config, 1, 2, is_nextn=True),
            (1,),
        )

    def test_cli_factor_matches_module_instantiation_rule(self):
        config = _glm_dsa_config(cli_factor=3)
        self.assertEqual(
            resolve_dsa_indexer_layer_ids(config, 0, 10),
            (0, 3, 6, 9),
        )


class TestAtlasA5Detection(CustomTestCase):
    def test_device_name_normalization(self):
        for device_name in (
            "Ascend950",
            "Ascend 950B",
            "ASCEND_950-Pro",
            "Atlas A5",
        ):
            with self.subTest(device_name=device_name):
                self.assertTrue(common.is_npu_atlas_a5_device_name(device_name))
        self.assertFalse(common.is_npu_atlas_a5_device_name("Ascend 910B"))

    def test_negative_probe_before_registration_is_not_cached(self):
        unavailable_torch = SimpleNamespace()
        available_torch = SimpleNamespace(
            npu=SimpleNamespace(
                is_available=lambda: True,
                get_device_name=lambda _device_id: "Ascend 950B",
            )
        )

        with patch.object(common, "torch", unavailable_torch):
            self.assertFalse(common.is_npu_atlas_a5())
        with patch.object(common, "torch", available_torch):
            self.assertTrue(common.is_npu_atlas_a5())


if __name__ == "__main__":
    unittest.main()
