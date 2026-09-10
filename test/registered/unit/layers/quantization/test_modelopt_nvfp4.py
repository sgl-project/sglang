import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from sglang.srt.layers.linear import MergedColumnParallelLinear, QKVParallelLinear
from sglang.srt.layers.parameter import PerTensorScaleParameter
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp4LinearMethod,
)
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestModelOptExcludeModuleMapping(CustomTestCase):
    """apply_weight_name_mapper expands exclusion patterns to the names sglang builds."""

    class _IdentityMapper:
        def apply_list(self, names):
            return list(names)

    def _mapped(self, exclude_modules):
        config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo=None,
            group_size=16,
            exclude_modules=list(exclude_modules),
        )
        config.apply_weight_name_mapper(self._IdentityMapper())
        return config

    def test_interior_language_model_segment_is_expanded(self):
        """nvidia/GLM-5.3-Flash-NVFP4 nests the decoder under model.language_model.*
        while sglang builds it as model.*, leaving every excluded attention, gate and
        shared-expert layer quantized and crashing the load on a shape mismatch."""
        config = self._mapped(["model.language_model.layers.0.self_attn*"])
        self.assertTrue(config.is_layer_excluded("model.layers.0.self_attn.kv_b_proj"))
        self.assertTrue(config.is_layer_excluded("model.layers.0.self_attn.q_b_proj"))
        # the checkpoint's own spelling keeps matching
        self.assertTrue(
            config.is_layer_excluded(
                "model.language_model.layers.0.self_attn.kv_b_proj"
            )
        )

    def test_interior_expansion_does_not_widen_the_match(self):
        config = self._mapped(["model.language_model.layers.10.mlp.gate"])
        self.assertTrue(config.is_layer_excluded("model.layers.10.mlp.gate"))
        self.assertFalse(config.is_layer_excluded("model.layers.11.mlp.gate"))
        self.assertFalse(config.is_layer_excluded("model.layers.10.mlp.experts"))

    def test_leading_language_model_strip_is_unchanged(self):
        config = self._mapped(["language_model.model.layers.0.self_attn.q_proj"])
        self.assertTrue(config.is_layer_excluded("model.layers.0.self_attn.q_proj"))

    def test_names_without_the_segment_are_untouched(self):
        config = self._mapped(["lm_head", "model.visual*"])
        self.assertEqual(config.exclude_modules, ["lm_head", "model.visual*"])


class TestModelOptNvfp4(CustomTestCase):
    def _make_layer(self):
        return MergedColumnParallelLinear(
            input_size=16,
            output_sizes=[16, 16],
            bias=False,
            tp_rank=0,
            tp_size=1,
        )

    def _make_qkv_layer(self):
        return QKVParallelLinear(
            hidden_size=16,
            head_size=8,
            total_num_heads=2,
            total_num_kv_heads=2,
            bias=False,
            tp_rank=0,
            tp_size=1,
        )

    def test_fused_scalar_scale_load_fills_all_logical_slots(self):
        layer = self._make_layer()
        scale = PerTensorScaleParameter(
            data=torch.empty(2, dtype=torch.float32),
            weight_loader=layer.weight_loader_v2,
        )

        layer.weight_loader_v2(scale, torch.tensor(0.25, dtype=torch.float32))

        torch.testing.assert_close(scale, torch.tensor([0.25, 0.25]))

    def test_fused_scalar_scale_load_rejects_non_scalar(self):
        layer = self._make_layer()
        scale = PerTensorScaleParameter(
            data=torch.empty(2, dtype=torch.float32),
            weight_loader=layer.weight_loader_v2,
        )

        with self.assertRaisesRegex(ValueError, "Expected scalar scale"):
            layer.weight_loader_v2(scale, torch.tensor([0.25, 0.5]))

    def test_fused_qkv_scalar_scale_load_fills_all_logical_slots(self):
        layer = self._make_qkv_layer()
        scale = PerTensorScaleParameter(
            data=torch.empty(3, dtype=torch.float32),
            weight_loader=layer.weight_loader_v2,
        )

        layer.weight_loader_v2(scale, torch.tensor(0.125, dtype=torch.float32))

        torch.testing.assert_close(scale, torch.tensor([0.125, 0.125, 0.125]))

    def test_explicit_shard_scale_loads_stay_independent(self):
        layer = self._make_layer()
        scale = PerTensorScaleParameter(
            data=torch.empty(2, dtype=torch.float32),
            weight_loader=layer.weight_loader_v2,
        )

        layer.weight_loader_v2(scale, torch.tensor(0.25, dtype=torch.float32), 0)
        layer.weight_loader_v2(scale, torch.tensor(0.5, dtype=torch.float32), 1)

        torch.testing.assert_close(scale, torch.tensor([0.25, 0.5]))

    def test_missing_input_scale_defaults_to_one_and_checkpoint_overwrites(self):
        config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            use_per_token_activation=False,
        )
        layer = nn.Module()
        ModelOptFp4LinearMethod(config).create_weights(
            layer,
            input_size_per_partition=16,
            output_partition_sizes=[16],
            input_size=16,
            output_size=16,
            params_dtype=torch.bfloat16,
            weight_loader=default_weight_loader,
        )

        torch.testing.assert_close(layer.input_scale, torch.ones(1))
        default_weight_loader(layer.input_scale, torch.tensor(0.25))
        torch.testing.assert_close(layer.input_scale, torch.tensor([0.25]))

    @patch(
        "sglang.srt.layers.quantization.modelopt_quant.envs."
        "SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION.get",
        return_value=True,
    )
    def test_modelopt_fp4_per_token_activation_contract(self, _):
        # Serialized ModelOpt FP4 retains the existing environment-controlled
        # per-token activation path.
        serialized_config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
        )
        # Online modelopt_fp4 always uses per-tensor activation scaling, even
        # when the serialized-checkpoint environment switch is enabled.
        online_config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=False,
            group_size=16,
        )

        self.assertTrue(serialized_config.use_per_token_activation)
        self.assertFalse(online_config.use_per_token_activation)
        # nvfp4_online is the public interface for online per-token scaling.
        with self.assertRaisesRegex(ValueError, "Use nvfp4_online"):
            ModelOptFp4Config(
                is_checkpoint_nvfp4_serialized=False,
                group_size=16,
                use_per_token_activation=True,
            )


if __name__ == "__main__":
    unittest.main()
