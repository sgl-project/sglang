"""Unit tests for ``sglang.srt.configs.jet_nemotron.JetNemotronConfig``."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.configs.jet_nemotron import JetNemotronConfig
from sglang.srt.configs.mamba_utils import Mamba2StateDType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestJetNemotronConfig(CustomTestCase):
    def _make_config(
        self,
        *,
        layer_types=None,
        expand_v=2.0,
        num_heads=8,
        head_dim=16,
        conv_size=4,
    ):
        if layer_types is None:
            layer_types = ["attn", "jet", "swa", "jet"]

        return JetNemotronConfig(
            layer_types=layer_types,
            efficient_attention_config={
                "jet": {
                    "mode": "jet",
                    "expand_v": expand_v,
                    "num_heads": num_heads,
                    "head_dim": head_dim,
                    "norm_eps": "1e-5",
                    "conv_size": conv_size,
                    "dconv_generator_reduction": 1,
                    "dconv_implementation": "causal_conv1d",
                }
            },
        )

    def _cache_params(self, config, *, tp_world_size=1, dtype=None):
        if dtype is None:
            dtype = Mamba2StateDType(conv=torch.float32, temporal=torch.float32)

        parallel = SimpleNamespace(attn_tp_size=tp_world_size)
        with (
            patch(
                "sglang.srt.configs.jet_nemotron.get_parallel",
                return_value=parallel,
            ),
            patch(
                "sglang.srt.configs.jet_nemotron.mamba2_state_dtype",
                return_value=dtype,
            ),
        ):
            return config.mamba2_cache_params

    def test_layer_ids_classify_mixed_layer_types(self):
        config = self._make_config(
            layer_types=["attn", "jet", "swa", "mlp", "jet", "unknown"]
        )

        self.assertEqual(config.full_attention_layer_ids, [0, 2])
        self.assertEqual(config.linear_layer_ids, [1, 4])

    def test_layer_ids_are_empty_for_empty_layout(self):
        config = self._make_config(layer_types=[])

        self.assertEqual(config.full_attention_layer_ids, [])
        self.assertEqual(config.linear_layer_ids, [])

    def test_mamba2_cache_params_single_tp(self):
        config = self._make_config()

        params = self._cache_params(config, tp_world_size=1)

        self.assertEqual(params.layers, [1, 3])
        self.assertEqual(params.shape.intermediate_size, 256)
        self.assertEqual(params.shape.conv, [(512, 3)])
        self.assertEqual(params.shape.temporal, (8, 32, 16))
        self.assertEqual(params.shape.num_heads, 8)
        self.assertEqual(params.shape.head_dim, 32)
        self.assertEqual(params.shape.state_size, 16)
        self.assertEqual(params.shape.conv_kernel, 4)
        self.assertEqual(params.shape.num_k_heads_per_tp, 8)

    def test_mamba2_cache_params_two_way_tp(self):
        config = self._make_config()

        params = self._cache_params(config, tp_world_size=2)

        self.assertEqual(params.layers, [1, 3])
        self.assertEqual(params.shape.intermediate_size, 256)
        self.assertEqual(params.shape.conv, [(256, 3)])
        self.assertEqual(params.shape.temporal, (4, 32, 16))
        self.assertEqual(params.shape.num_k_heads_per_tp, 4)

    def test_expand_v_controls_value_head_dimension(self):
        config = self._make_config(expand_v=1.5, num_heads=6, head_dim=16, conv_size=3)

        params = self._cache_params(config, tp_world_size=2)

        self.assertEqual(params.shape.intermediate_size, 144)
        self.assertEqual(params.shape.head_dim, 24)
        self.assertEqual(params.shape.conv, [(168, 2)])
        self.assertEqual(params.shape.temporal, (3, 24, 16))

    def test_mamba2_cache_params_preserve_resolved_dtype(self):
        config = self._make_config()
        expected_dtype = Mamba2StateDType(
            conv=torch.float16,
            temporal=torch.bfloat16,
        )
        parallel = SimpleNamespace(attn_tp_size=1)

        with (
            patch(
                "sglang.srt.configs.jet_nemotron.get_parallel",
                return_value=parallel,
            ),
            patch(
                "sglang.srt.configs.jet_nemotron.mamba2_state_dtype",
                return_value=expected_dtype,
            ) as dtype_mock,
        ):
            params = config.mamba2_cache_params

        self.assertIs(params.dtype, expected_dtype)
        dtype_mock.assert_called_once_with(config)


if __name__ == "__main__":
    unittest.main()
